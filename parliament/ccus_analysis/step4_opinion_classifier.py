import csv
import json
from pathlib import Path
from typing import Protocol

from django.conf import settings

from parliament.summaries.llm import get_llm_response, llms

from .config import JSON_DIR, CSV_DIR
from .step2_hansard_fetcher import strip_html
from .models import Argument, Opinion, PoliticalActor


OPINION_SCHEMA = {
    "type": "object",
    "properties": {
        "stance": {
            "type": "string",
            "enum": ["support", "oppose", "neutral", "mixed"],
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "arguments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "economic",
                            "environmental",
                            "ethical",
                            "social",
                            "technical",
                            "jurisdictional",
                        ],
                    },
                    "text": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["type", "text", "quote"],
            },
        },
    },
    "required": ["stance", "confidence", "arguments"],
}

OPINION_INSTRUCTIONS = """You are an expert analyst of Canadian parliamentary debates.

Your task is to classify a politician's stance on Carbon Capture, Utilization and Storage (CCUS)
based on their speeches in Parliament. Analyse both:
1. Their stance on CCUS as a technology (does it work / is it viable?)
2. Their stance on CCUS as a policy tool (should the government support / fund it?)

Classify overall stance as one of:
- support: generally favourable, advocates for CCUS development or funding
- oppose: generally critical, argues against CCUS investment or effectiveness
- neutral: no clear position expressed
- mixed: acknowledges merits and drawbacks, or position is nuanced/conditional

For each distinct argument made, identify:
- type: one of economic | environmental | ethical | social | technical | jurisdictional
- text: a concise summary of the argument (1-2 sentences)
- quote: the most relevant verbatim excerpt from the speeches (keep it under 200 characters)

Respond in JSON matching the schema provided.
"""


class OpinionClassifier(Protocol):
    def classify(self, actor: PoliticalActor, bill: dict) -> Opinion: ...


class LLMOpinionClassifier:
    """Classifies politician stance on CCUS using a configurable LLM (default: Ollama)."""

    def __init__(self, model: str | None = None):
        self.model = model or getattr(settings, "CCUS_LLM_MODEL", llms.OLLAMA)

    def classify(self, actor: PoliticalActor, bill: dict) -> Opinion:
        bill_name = (bill.get("name") or {}).get("en") or bill.get("url", "Unknown bill")
        combined_text = self._combine_speeches(actor.speeches)

        prompt = f"Bill: {bill_name}\nPolitician: {actor.name}\n\n{combined_text}"

        response_text, _ = get_llm_response(
            OPINION_INSTRUCTIONS,
            prompt,
            model=self.model,
            json=OPINION_SCHEMA,
        )

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            data = {"stance": "neutral", "confidence": "low", "arguments": []}

        arguments = [
            Argument(
                type=arg.get("type", "economic"),
                text=arg.get("text", ""),
                quote=arg.get("quote", ""),
            )
            for arg in data.get("arguments", [])
        ]

        return Opinion(
            actor=actor,
            stance=data.get("stance", "neutral"),
            arguments=arguments,
            confidence=data.get("confidence", "low"),
        )

    # Maximum characters of combined speech text to send to the LLM.
    # Keeps prompts reasonable for local models and avoids context-window errors.
    MAX_COMBINED_CHARS = 12_000

    def _combine_speeches(self, speeches: list[dict]) -> str:
        parts = []
        total = 0
        for speech in speeches:
            # Prefer already-stripped text; fall back to stripping HTML
            content_text = speech.get("content_text", {})
            if isinstance(content_text, dict):
                text = content_text.get("en") or content_text.get("fr") or ""
            else:
                content = speech.get("content", {})
                if isinstance(content, dict):
                    text = strip_html(content.get("en") or content.get("fr") or "")
                else:
                    text = ""
            text = text.strip()
            if not text:
                continue
            remaining = self.MAX_COMBINED_CHARS - total
            if remaining <= 0:
                break
            chunk = text[:remaining]
            parts.append(chunk)
            total += len(chunk)
        return "\n\n".join(parts)


# Backward compatibility
GeminiOpinionClassifier = LLMOpinionClassifier


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _write_step4_opinions(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    """
    Given the step-3 actors file, run the LLM classifier for each actor/bill and
    write the resulting opinions and arguments to
    ``{output_dir}/step4_opinions.json`` and a CSV summary.
    """
    from .models import PoliticalActor

    # Resolve output directories.  The default root is the ``output/`` folder
    # next to this file; json_dir and csv_dir fall back to the root itself so
    # that files written by earlier steps (which use the flat layout) are found.
    root = Path(output_dir) if output_dir is not None else JSON_DIR.parent
    # Prefer output/json/ if it exists and is populated, otherwise use root.
    _candidate_json = root / "json"
    json_dir = _candidate_json if _candidate_json.is_dir() else root
    _candidate_csv = root / "csv"
    csv_dir = _candidate_csv if _candidate_csv.is_dir() else root
    json_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    in_path = Path(input_path) if input_path is not None else root / "step3_actors.json"
    if not in_path.exists():
        # Fallback to json/ subdirectory
        in_path = json_dir / "step3_actors.json"
    if not in_path.exists():
        raise SystemExit(
            f"[Step4] Expected input file not found: {in_path}. "
            "Run parliament.ccus_analysis.step3_actor_extractor as a script first."
        )

    records = json.loads(in_path.read_text())
    classifier = LLMOpinionClassifier()

    out_path = json_dir / "step4_opinions.json"

    # Load any existing results so we can resume without redoing work.
    out_records: list[dict] = []
    if out_path.exists():
        try:
            out_records = json.loads(out_path.read_text())
        except Exception:
            out_records = []

    def _key_from_rec(rec: dict) -> tuple[str, str, str, str]:
        bill = rec.get("bill") or {}
        return (
            rec.get("manual_number", ""),
            rec.get("manual_session", "") or "",
            bill.get("session", "") or "",
            bill.get("number", "") or "",
        )

    processed_keys = { _key_from_rec(r) for r in out_records }

    # Helper to rewrite JSON and CSV after each bill so partial progress is saved.
    csv_path = csv_dir / "step4_opinions.csv"
    fieldnames = [
        "manual_number",
        "manual_session",
        "session",
        "bill_number",
        "actor_name",
        "politician_url",
        "party",
        "stance",
        "confidence",
        "argument_count",
    ]

    def _rewrite_outputs() -> None:
        out_path.write_text(json.dumps(out_records, ensure_ascii=False, indent=2))
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in out_records:
                bill = rec["bill"]
                for op in rec.get("opinions", []):
                    actor = op.get("actor", {})
                    writer.writerow(
                        {
                            "manual_number": rec["manual_number"],
                            "manual_session": rec["manual_session"],
                            "session": bill.get("session", ""),
                            "bill_number": bill.get("number", ""),
                            "actor_name": actor.get("name", ""),
                            "politician_url": actor.get("politician_url", ""),
                            "party": actor.get("party") or "",
                            "stance": op.get("stance", ""),
                            "confidence": op.get("confidence", ""),
                            "argument_count": len(op.get("arguments", [])),
                        }
                    )

    records = records[:3]
    print(f"[Step4] Classifying opinions for {len(records)} bill entr(y/ies)...", flush=True)

    for rec in records:
        bill = rec["bill"]
        key = (
            rec.get("manual_number", ""),
            rec.get("manual_session", "") or "",
            bill.get("session", "") or "",
            bill.get("number", "") or "",
        )
        if key in processed_keys:
            # Already processed in a previous run; skip.
            continue

        opinions_out: list[dict] = []
        actors = rec.get("actors", [])[:3]
        for idx, a in enumerate(actors, 1):
            actor = PoliticalActor(
                name=a["name"],
                politician_url=a["politician_url"],
                party=a.get("party"),
                speeches=a.get("speeches", []),
            )
            print(
                f"[Step4]   {bill.get('number','?')} {bill.get('session','')}: "
                f"actor {idx}/{len(actors)} — {actor.name}",
                flush=True,
            )
            op = classifier.classify(actor, bill)
            opinions_out.append(
                {
                    "actor": {
                        "name": op.actor.name,
                        "politician_url": op.actor.politician_url,
                        "party": op.actor.party,
                    },
                    "stance": op.stance,
                    "confidence": op.confidence,
                    "arguments": [
                        {"type": arg.type, "text": arg.text, "quote": arg.quote}
                        for arg in op.arguments
                    ],
                }
            )

        out_records.append(
            {
                "manual_number": rec["manual_number"],
                "manual_session": rec["manual_session"],
                "bill": bill,
                "opinions": opinions_out,
            }
        )
        processed_keys.add(key)
        _rewrite_outputs()

    print(
        f"[Step4] Wrote opinions for {len(out_records)} bill entr(y/ies) to {out_path} and {csv_path}",
        flush=True,
    )


def _main_cli() -> None:
    # Ensure Django settings are available for LLM configuration.
    import os
    import django

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "parliament.settings")
    django.setup()
    _write_step4_opinions()


if __name__ == "__main__":
    _main_cli()
