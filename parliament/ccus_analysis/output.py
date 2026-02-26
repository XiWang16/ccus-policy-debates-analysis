import csv
import dataclasses
import json
from pathlib import Path

from .models import CCUSAnalysisResult


# ---------------------------------------------------------------------------
# Shared serialisation helper
# ---------------------------------------------------------------------------

def _to_dict(obj):
    """Recursively convert dataclasses to plain dicts for JSON serialisation."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# JSON output (full fidelity dump)
# ---------------------------------------------------------------------------

class JSONOutputWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, result: CCUSAnalysisResult) -> None:
        self._write_bills(result)
        self._write_speeches(result)
        self._write_opinions(result)
        self._write_jurisdictions(result)
        self._write_summary(result)

    def _dump(self, filename: str, data) -> None:
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {path}")

    def _write_bills(self, result: CCUSAnalysisResult) -> None:
        bills = [ba.bill for ba in result.bills]
        self._dump("ccus_bills.json", bills)

    def _write_speeches(self, result: CCUSAnalysisResult) -> None:
        speeches_by_bill = [
            {
                "bill_url": ba.bill.get("url"),
                "bill_name": (ba.bill.get("name") or {}).get("en"),
                "speeches": ba.speeches,
            }
            for ba in result.bills
        ]
        self._dump("ccus_speeches.json", speeches_by_bill)

    def _write_opinions(self, result: CCUSAnalysisResult) -> None:
        opinions_by_bill = []
        for ba in result.bills:
            opinions_data = []
            for opinion in ba.opinions:
                opinions_data.append({
                    "actor": {
                        "name": opinion.actor.name,
                        "politician_url": opinion.actor.politician_url,
                        "party": opinion.actor.party,
                    },
                    "stance": opinion.stance,
                    "confidence": opinion.confidence,
                    "arguments": [
                        {"type": arg.type, "text": arg.text, "quote": arg.quote}
                        for arg in opinion.arguments
                    ],
                })
            opinions_by_bill.append({
                "bill_url": ba.bill.get("url"),
                "bill_name": (ba.bill.get("name") or {}).get("en"),
                "opinions": opinions_data,
            })
        self._dump("ccus_opinions.json", opinions_by_bill)

    def _write_jurisdictions(self, result: CCUSAnalysisResult) -> None:
        jurisdictions_by_bill = [
            {
                "bill_url": ba.bill.get("url"),
                "bill_name": (ba.bill.get("name") or {}).get("en"),
                "jurisdictions": [
                    {"entity": j.entity, "label": j.label, "context": j.context}
                    for j in ba.jurisdictions
                ],
            }
            for ba in result.bills
        ]
        self._dump("ccus_jurisdictions.json", jurisdictions_by_bill)

    def _write_summary(self, result: CCUSAnalysisResult) -> None:
        total_speeches = sum(len(ba.speeches) for ba in result.bills)
        total_actors = sum(len(ba.actors) for ba in result.bills)
        total_opinions = sum(len(ba.opinions) for ba in result.bills)

        stance_counts: dict[str, int] = {}
        for ba in result.bills:
            for op in ba.opinions:
                stance_counts[op.stance] = stance_counts.get(op.stance, 0) + 1

        summary = {
            "generated_at": result.generated_at,
            "bill_count": len(result.bills),
            "total_speeches": total_speeches,
            "total_actors": total_actors,
            "total_opinions": total_opinions,
            "stance_breakdown": stance_counts,
            "bills": [
                {
                    "url": ba.bill.get("url"),
                    "name_en": (ba.bill.get("name") or {}).get("en"),
                    "match_reason": ba.match_reason,
                    "speech_count": len(ba.speeches),
                    "actor_count": len(ba.actors),
                }
                for ba in result.bills
            ],
        }
        self._dump("ccus_summary.json", summary)


# ---------------------------------------------------------------------------
# CSV output — periodic result summary files written to results/ subdirectory
# ---------------------------------------------------------------------------

class CSVOutputWriter:
    """Write analysis results to CSV files in the given directory.

    Produces five files:
    1. ccus_bills.csv          — all CCUS-related bills with metadata
    2. ccus_actors_support.csv — political actors who supported CCUS bills
    3. ccus_actors_opposition.csv — political actors who opposed CCUS bills
    4. ccus_arguments.csv      — arguments made for/against CCUS bills
    5. ccus_jurisdictions.csv  — jurisdictional sources referenced in debates
    """

    def __init__(self, output_dir: Path):
        self.results_dir = output_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def write(self, result: CCUSAnalysisResult) -> None:
        self._write_bills(result)
        self._write_actors_support(result)
        self._write_actors_opposition(result)
        self._write_arguments(result)
        self._write_jurisdictions(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_csv(self, filename: str, fieldnames: list[str], rows: list[dict]) -> None:
        path = self.results_dir / filename
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path}")

    # ------------------------------------------------------------------
    # 1. Bills CSV
    # ------------------------------------------------------------------

    def _write_bills(self, result: CCUSAnalysisResult) -> None:
        fieldnames = [
            "bill_number",
            "session",
            "name_en",
            "name_fr",
            "introduced",
            "status_code",
            "home_chamber",
            "sponsor_politician_url",
            "bill_url",
            "match_reason",
        ]
        rows = []
        for ba in result.bills:
            bill = ba.bill
            rows.append({
                "bill_number": bill.get("number", ""),
                "session": bill.get("session", ""),
                "name_en": (bill.get("name") or {}).get("en", ""),
                "name_fr": (bill.get("name") or {}).get("fr", ""),
                "introduced": bill.get("introduced", ""),
                "status_code": bill.get("status_code", ""),
                "home_chamber": bill.get("home_chamber", ""),
                "sponsor_politician_url": bill.get("sponsor_politician_url", ""),
                "bill_url": bill.get("url", ""),
                "match_reason": ba.match_reason,
            })
        self._write_csv("ccus_bills.csv", fieldnames, rows)

    # ------------------------------------------------------------------
    # 2 & 3. Actors — support and opposition
    # ------------------------------------------------------------------

    def _write_actors_support(self, result: CCUSAnalysisResult) -> None:
        self._write_actors_by_stance(result, stance="support", filename="ccus_actors_support.csv")

    def _write_actors_opposition(self, result: CCUSAnalysisResult) -> None:
        self._write_actors_by_stance(result, stance="oppose", filename="ccus_actors_opposition.csv")

    def _write_actors_by_stance(
        self, result: CCUSAnalysisResult, stance: str, filename: str
    ) -> None:
        fieldnames = [
            "actor_name",
            "party",
            "politician_url",
            "bill_name",
            "bill_number",
            "session",
            "earliest_speech_date",
            "latest_speech_date",
            "speech_count",
        ]
        rows = []
        for ba in result.bills:
            bill = ba.bill
            bill_name = (bill.get("name") or {}).get("en", "")
            bill_number = bill.get("number", "")
            session = bill.get("session", "")

            for opinion in ba.opinions:
                if opinion.stance != stance:
                    continue
                actor = opinion.actor
                speech_dates = sorted(
                    s.get("time", "") or s.get("date", "")
                    for s in actor.speeches
                    if s.get("time") or s.get("date")
                )
                rows.append({
                    "actor_name": actor.name,
                    "party": actor.party or "",
                    "politician_url": actor.politician_url or "",
                    "bill_name": bill_name,
                    "bill_number": bill_number,
                    "session": session,
                    "earliest_speech_date": speech_dates[0] if speech_dates else "",
                    "latest_speech_date": speech_dates[-1] if speech_dates else "",
                    "speech_count": len(actor.speeches),
                })
        self._write_csv(filename, fieldnames, rows)

    # ------------------------------------------------------------------
    # 4. Arguments CSV
    # ------------------------------------------------------------------

    def _write_arguments(self, result: CCUSAnalysisResult) -> None:
        fieldnames = [
            "actor_name",
            "party",
            "bill_name",
            "bill_number",
            "session",
            "argument_label",
            "argument_text",
            "quote",
        ]
        rows = []
        for ba in result.bills:
            bill = ba.bill
            bill_name = (bill.get("name") or {}).get("en", "")
            bill_number = bill.get("number", "")
            session = bill.get("session", "")

            for opinion in ba.opinions:
                for arg in opinion.arguments:
                    rows.append({
                        "actor_name": opinion.actor.name,
                        "party": opinion.actor.party or "",
                        "bill_name": bill_name,
                        "bill_number": bill_number,
                        "session": session,
                        "argument_label": arg.type,
                        "argument_text": arg.text,
                        "quote": arg.quote,
                    })
        self._write_csv("ccus_arguments.csv", fieldnames, rows)

    # ------------------------------------------------------------------
    # 5. Jurisdictions CSV
    # ------------------------------------------------------------------

    def _write_jurisdictions(self, result: CCUSAnalysisResult) -> None:
        fieldnames = [
            "source_name",
            "entity_type",
            "bill_name",
            "bill_number",
            "session",
            "excerpt",
        ]
        rows = []
        for ba in result.bills:
            bill = ba.bill
            bill_name = (bill.get("name") or {}).get("en", "")
            bill_number = bill.get("number", "")
            session = bill.get("session", "")

            for j in ba.jurisdictions:
                rows.append({
                    "source_name": j.entity,
                    "entity_type": j.label,
                    "bill_name": bill_name,
                    "bill_number": bill_number,
                    "session": session,
                    "excerpt": j.context,
                })
        self._write_csv("ccus_jurisdictions.csv", fieldnames, rows)
