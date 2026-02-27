import re
import csv 

from pathlib import Path
from lxml.html.clean import Cleaner

from .api_client import OpenParliamentClient
from .config import API_BASE_URL, JSON_DIR, CSV_DIR


_html_cleaner = Cleaner(safe_attrs_only=True, remove_tags=["a", "span", "em", "strong", "b", "i"])


def strip_html(html: str) -> str:
    """Return plain text from an HTML string."""
    if not html:
        return ""
    try:
        import lxml.html
        doc = lxml.html.fromstring(html)
        return doc.text_content()
    except Exception:
        # Fallback: naive tag strip
        return re.sub(r"<[^>]+>", " ", html)


class HansardFetcher:
    def __init__(self, client: OpenParliamentClient):
        self.client = client

    def get_speeches(self, bill: dict) -> list[dict]:
        """Return all speeches for a bill with HTML stripped from content."""
        bill_url = bill.get("url", "")
        speeches = []
        for speech in self.client.get_speeches(bill_url):
            speech = dict(speech)
            content = speech.get("content", {})
            if isinstance(content, dict):
                speech["content_text"] = {
                    lang: strip_html(html) for lang, html in content.items()
                }
            speeches.append(speech)
        return speeches


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _write_step2_speeches(
    base_url: str = API_BASE_URL,
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    """
    Given the step-1 bill list, fetch all associated speeches from the API and
    write them to ``{output_dir}/step2_speeches.json``.

    This is a standalone entry point so that you can run just the "fetch
    speeches" step of the pipeline and inspect the results.
    """
    import json

    root = Path(output_dir) if output_dir is not None else JSON_DIR.parent
    json_dir = root / "json"
    csv_dir = root / "csv"
    json_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    in_path = Path(input_path) if input_path is not None else json_dir / "step1_bills.json"
    if not in_path.exists():
        raise SystemExit(
            f"[Step2] Expected input file not found: {in_path}. "
            "Run parliament.ccus_analysis.bill_finder as a script first."
        )

    records = json.loads(in_path.read_text())

    client = OpenParliamentClient(base_url=base_url)
    fetcher = HansardFetcher(client=client)

    out_path = json_dir / "step2_speeches.json"

    out_records: list[dict] = []

    print(f"[Step2] Fetching speeches for {len(records)} bill entr(y/ies)...", flush=True)

    for rec in records:
        bill = rec["bill"]
        speeches = fetcher.get_speeches(bill)
        rec_out = dict(rec)
        rec_out["speeches"] = speeches
        out_records.append(rec_out)

    out_path.write_text(json.dumps(out_records, ensure_ascii=False, indent=2))

    # CSV summary: one row per bill with speech count.
    csv_path = csv_dir / "step2_speeches.csv"
    fieldnames = [
        "manual_number",
        "manual_session",
        "session",
        "bill_number",
        "speech_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in out_records:
            bill = rec["bill"]
            writer.writerow(
                {
                    "manual_number": rec["manual_number"],
                    "manual_session": rec["manual_session"],
                    "session": bill.get("session", ""),
                    "bill_number": bill.get("number", ""),
                    "speech_count": len(rec.get("speeches", [])),
                }
            )

    print(
        f"[Step2] Wrote speeches for {len(out_records)} bill entr(y/ies) to {out_path} and {csv_path}",
        flush=True,
    )


if __name__ == "__main__":
    _write_step2_speeches()
