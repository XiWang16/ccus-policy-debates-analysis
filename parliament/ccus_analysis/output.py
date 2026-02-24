import dataclasses
import json
from pathlib import Path

from .models import CCUSAnalysisResult


def _to_dict(obj):
    """Recursively convert dataclasses to plain dicts for JSON serialization."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


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
                    "speech_count": len(ba.speeches),
                    "actor_count": len(ba.actors),
                }
                for ba in result.bills
            ],
        }
        self._dump("ccus_summary.json", summary)
