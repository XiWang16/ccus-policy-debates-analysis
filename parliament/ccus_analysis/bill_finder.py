import re

from .api_client import OpenParliamentClient
from .keywords import KeywordProvider


def _compile_patterns(keywords: list[str]) -> list[re.Pattern]:
    """Compile each keyword into a word-boundary regex pattern.

    Using ``\b`` prevents short acronyms like 'EOR', 'CCUS', 'CCS', 'DAC'
    from matching as substrings inside unrelated words (e.g. 'EOR' inside
    'reORganization', 'CCUS' inside 'aCCUSed', 'CCS' inside 'CCSVI').
    """
    return [
        re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
        for kw in keywords
    ]


class CCUSBillFinder:
    def __init__(
        self,
        client: OpenParliamentClient,
        keyword_provider: KeywordProvider,
        search_full_text: bool = True,
    ):
        self.client = client
        self.keyword_provider = keyword_provider
        # When True, bills not matched by title are also checked against their
        # full legislative text (requires a detail API call per bill, so it is
        # slower than title-only matching).
        self.search_full_text = search_full_text
        self._patterns = _compile_patterns(keyword_provider.get_keywords())

    def _matches_any(self, text: str) -> bool:
        """Return True if *text* contains any keyword at a word boundary."""
        return any(p.search(text) for p in self._patterns)

    # ------------------------------------------------------------------
    # Automatic keyword-based discovery
    # ------------------------------------------------------------------

    def find_bills(self) -> list[dict]:
        """Return all bills containing at least one CCUS keyword.

        Always checks bill titles.  When ``search_full_text`` is True,
        bills that do not match on title are also checked against their
        full legislative text (via the detail API endpoint).
        """
        matched = []
        for bill in self.client.get_bills():
            if self._title_matches(bill):
                matched.append(bill)
            elif self.search_full_text and self.bill_contains_keywords(bill):
                matched.append(bill)
        return matched

    def _title_matches(self, bill: dict) -> bool:
        name_en = bill.get("name", {}).get("en") or ""
        name_fr = bill.get("name", {}).get("fr") or ""
        return self._matches_any(name_en) or self._matches_any(name_fr)

    # ------------------------------------------------------------------
    # Manual bill lookup by number
    # ------------------------------------------------------------------

    def find_by_number(self, number: str) -> list[dict]:
        """Return all bills (across all sessions) matching *number*.

        Uses the API's ``?number=`` filter — efficient, no client-side scan.
        """
        return list(self.client.get_bills(number=number))

    # ------------------------------------------------------------------
    # Full-text / speech keyword checks
    # ------------------------------------------------------------------

    def bill_contains_keywords(
        self,
        bill: dict,
        keywords: list[str] | None = None,
    ) -> bool:
        """Fetch the bill detail and check ``full_text`` / ``short_title`` for keywords.

        Returns False gracefully if the detail is unavailable or has no text.
        Uses the same word-boundary patterns as title matching when no custom
        keyword list is given.
        """
        patterns = self._patterns if keywords is None else _compile_patterns(keywords)

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
            # short_title is only in the detail response
            (detail.get("short_title") or {}).get("en") or "",
            (detail.get("short_title") or {}).get("fr") or "",
        ]
        return any(p.search(t) for p in patterns for t in texts if t)

    def speeches_contain_keywords(
        self,
        speeches: list[dict],
        keywords: list[str] | None = None,
    ) -> bool:
        """Return True if any speech's content contains at least one keyword."""
        patterns = self._patterns if keywords is None else _compile_patterns(keywords)

        for speech in speeches:
            content_text = speech.get("content_text", {}) or {}
            for lang_text in content_text.values():
                if lang_text and any(p.search(lang_text) for p in patterns):
                    return True
        return False
