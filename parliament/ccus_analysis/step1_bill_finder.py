import csv
import re
from pathlib import Path

from django.db import connection

from .api_client import OpenParliamentClient
from .keywords import KeywordProvider
from .config import API_BASE_URL, JSON_DIR, CSV_DIR


def _compile_pattern(keywords: list[str]) -> re.Pattern:
    """Compile all keywords into a single alternation regex with word boundaries.

    Sorting by descending length ensures longer phrases (e.g. "carbon capture
    and storage") are attempted before their sub-phrases ("carbon capture"),
    which matters when the match result is used to identify *which* keyword hit.
    A single alternation pattern makes exactly one pass through the text
    regardless of how many keywords there are.
    """
    sorted_kws = sorted(keywords, key=len, reverse=True)
    alternation = "|".join(re.escape(kw) for kw in sorted_kws)
    return re.compile(r"\b(?:" + alternation + r")\b", re.IGNORECASE)


class CCUSBillFinder:
    def __init__(
        self,
        client: OpenParliamentClient,
        keyword_provider: KeywordProvider,
        search_full_text: bool = True,
    ):
        self.client = client
        self.keyword_provider = keyword_provider
        self.search_full_text = search_full_text
        self._pattern = _compile_pattern(keyword_provider.get_keywords())

    def _matches_any(self, text: str) -> bool:
        """Return True if *text* contains any keyword at a word boundary."""
        return bool(self._pattern.search(text))

    # ------------------------------------------------------------------
    # DB-based keyword discovery across all bills
    # ------------------------------------------------------------------

    def find_bills(self) -> list[dict]:
        """Return all bills containing at least one CCUS keyword.

        Uses a two-phase approach:

        Phase 1 — database pre-filter (fast):
          On PostgreSQL, a full-text SearchVector query over ``BillText.text_en``,
          ``BillText.text_fr``, and the ``Bill`` name/title fields reduces the
          candidate set to a small fraction of all bills using GIN indexes.
          On SQLite (development), a plain ``__icontains`` OR filter is used
          as a functionally equivalent fallback.

        Phase 2 — precise regex confirmation (exact):
          The single combined alternation regex (one pass, word-boundary
          anchored) confirms each candidate.  This eliminates false positives
          from the FTS pre-filter, e.g. documents where "carbon" and "capture"
          appear but not adjacent, or where a short acronym like "CCS" appears
          inside an unrelated word.
        """
        from parliament.bills.models import Bill, BillText

        keywords = self.keyword_provider.get_keywords()

        if connection.vendor == "postgresql":
            candidates = self._fts_candidates(keywords)
        else:
            candidates = self._icontains_candidates(keywords)

        matched = []
        for bill in candidates:
            if self._bill_title_matches(bill):
                matched.append(self._bill_to_dict(bill))
                continue
            try:
                bt = bill.get_text_object()
                if self._matches_any(bt.text_en) or self._matches_any(bt.text_fr or ""):
                    matched.append(self._bill_to_dict(bill))
            except Exception:
                pass
        return matched

    def _fts_candidates(self, keywords: list[str]):
        """Phase 1 on PostgreSQL: FTS pre-filter using SearchVector + SearchQuery."""
        from parliament.bills.models import Bill, BillText
        from django.contrib.postgres.search import SearchQuery, SearchVector

        # 'simple' config: lowercases and splits on whitespace only — no
        # stemming, so acronyms and French terms survive intact.
        # search_type='plain' means multi-word phrases require all tokens to
        # appear in the document (broad, fast; regex pass confirms adjacency).
        def _build_query(kws):
            q = SearchQuery(kws[0], search_type="plain", config="simple")
            for kw in kws[1:]:
                q |= SearchQuery(kw, search_type="plain", config="simple")
            return q

        query = _build_query(keywords)

        text_ids = set(
            BillText.objects.annotate(
                search=SearchVector("text_en", "text_fr", config="simple")
            ).filter(search=query).values_list("bill_id", flat=True)
        )
        title_ids = set(
            Bill.objects.annotate(
                search=SearchVector(
                    "name_en", "name_fr", "short_title_en", "short_title_fr",
                    config="simple",
                )
            ).filter(search=query).values_list("id", flat=True)
        )

        return list(
            Bill.objects.filter(id__in=(text_ids | title_ids))
            .select_related("session")
        )

    def _icontains_candidates(self, keywords: list[str]):
        """Phase 1 on SQLite: icontains OR filter as a development fallback."""
        from parliament.bills.models import Bill
        from django.db.models import Q

        q = Q()
        for kw in keywords:
            q |= (
                Q(name_en__icontains=kw)
                | Q(name_fr__icontains=kw)
                | Q(short_title_en__icontains=kw)
                | Q(short_title_fr__icontains=kw)
                | Q(billtext__text_en__icontains=kw)
                | Q(billtext__text_fr__icontains=kw)
            )
        return list(Bill.objects.filter(q).distinct().select_related("session"))

    def _bill_title_matches(self, bill) -> bool:
        return (
            self._matches_any(bill.name_en)
            or self._matches_any(bill.name_fr)
            or self._matches_any(bill.short_title_en)
            or self._matches_any(bill.short_title_fr)
        )

    def _bill_to_dict(self, bill) -> dict:
        """Convert a Bill ORM object to the dict format used throughout the pipeline."""
        return {
            "url": bill.get_absolute_url(),
            "session": bill.session_id,
            "number": bill.number,
            "name": {"en": bill.name_en, "fr": bill.name_fr},
            "introduced": str(bill.introduced) if bill.introduced else None,
            "legisinfo_id": bill.legisinfo_id,
        }

    # ------------------------------------------------------------------
    # Manual bill lookup by number (used by the deployment pipeline)
    # ------------------------------------------------------------------

    def find_by_number(self, number: str) -> list[dict]:
        """Return all bills (across all sessions) matching *number*.

        Uses the API's ``?number=`` filter — efficient, no client-side scan.
        """
        return list(self.client.get_bills(number=number))

    # ------------------------------------------------------------------
    # Keyword checks against API-fetched bill detail and speeches
    # ------------------------------------------------------------------

    def bill_contains_keywords(
        self,
        bill: dict,
        keywords: list[str] | None = None,
    ) -> bool:
        """Fetch the bill detail and check ``full_text`` / ``short_title`` for keywords."""
        pattern = self._pattern if keywords is None else _compile_pattern(keywords)

        bill_url = bill.get("url", "")
        if not bill_url:
            return False
        try:
            detail = self.client.get_bill_detail(bill_url)
        except Exception:
            return False

        full_text = detail.get("full_text") or {}
        texts = [
            full_text.get("en") or "",
            full_text.get("fr") or "",
            (detail.get("short_title") or {}).get("en") or "",
            (detail.get("short_title") or {}).get("fr") or "",
        ]
        return any(bool(pattern.search(t)) for t in texts if t)

    def speeches_contain_keywords(
        self,
        speeches: list[dict],
        keywords: list[str] | None = None,
    ) -> bool:
        """Return True if any speech's content contains at least one keyword."""
        pattern = self._pattern if keywords is None else _compile_pattern(keywords)

        for speech in speeches:
            content_text = speech.get("content_text", {}) or {}
            for lang_text in content_text.values():
                if lang_text and pattern.search(lang_text):
                    return True
        return False


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _write_step1_bills(
    base_url: str = API_BASE_URL,
    output_dir: Path | None = None,
) -> None:
    """
    Resolve the manually-curated CCUS bill list against the API and write the
    matched bill records to ``{output_dir}/step1_bills.json`` and a CSV summary.
    """
    import json
    from .keywords import StaticCCUSKeywordProvider
    from .manual_bills import get_manual_bill_entries

    client = OpenParliamentClient(base_url=base_url)
    keyword_provider = StaticCCUSKeywordProvider()
    finder = CCUSBillFinder(client=client, keyword_provider=keyword_provider, search_full_text=False)

    root = Path(output_dir) if output_dir is not None else JSON_DIR.parent
    json_dir = root / "json"
    csv_dir = root / "csv"
    json_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_path = json_dir / "step1_bills.json"

    records: list[dict] = []
    entries = get_manual_bill_entries()

    print(f"[Step1] Resolving {len(entries)} manual bill entr(y/ies) against the API...", flush=True)

    for number, pinned_session in entries:
        found = finder.find_by_number(number)
        if pinned_session:
            found = [b for b in found if b.get("session") == pinned_session]
        if not found:
            suffix = f" in session {pinned_session}" if pinned_session else ""
            print(
                f"[Step1] WARNING: No bills found for {number}{suffix}",
                flush=True,
            )
            continue
        for bill in found:
            records.append(
                {
                    "manual_number": number,
                    "manual_session": pinned_session,
                    "bill": bill,
                }
            )

    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))

    csv_path = csv_dir / "step1_bills.csv"
    fieldnames = [
        "manual_number",
        "manual_session",
        "session",
        "bill_number",
        "introduced",
        "name_en",
        "name_fr",
        "url",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            bill = rec["bill"]
            name = bill.get("name") or {}
            writer.writerow(
                {
                    "manual_number": rec["manual_number"],
                    "manual_session": rec["manual_session"],
                    "session": bill.get("session", ""),
                    "bill_number": bill.get("number", ""),
                    "introduced": bill.get("introduced", ""),
                    "name_en": name.get("en", ""),
                    "name_fr": name.get("fr", ""),
                    "url": bill.get("url", ""),
                }
            )

    print(
        f"[Step1] Wrote {len(records)} record(s) to {out_path} and {csv_path}",
        flush=True,
    )


def _main_cli() -> None:
    import os
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "parliament.settings")
    django.setup()
    _write_step1_bills()


if __name__ == "__main__":
    _main_cli()
