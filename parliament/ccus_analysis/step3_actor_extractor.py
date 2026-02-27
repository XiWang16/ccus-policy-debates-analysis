import csv
from pathlib import Path

import requests

from .config import API_BASE_URL, JSON_DIR, CSV_DIR
from .models import PoliticalActor


class ActorExtractor:
    def extract(self, speeches: list[dict]) -> list[PoliticalActor]:
        """
        Group CCUS-relevant speeches by politician; one PoliticalActor per unique
        politician.

        If speeches have been annotated by CCUSPassageExtractor (i.e. they carry
        a ``ccus_relevant`` key), only speeches where that flag is True are
        considered.  Unannotated speeches (no ``ccus_relevant`` key) are all
        included so that the extractor remains usable without the annotation step.

        Actors with zero qualifying speeches after filtering are excluded entirely.
        """
        actors: dict[str, PoliticalActor] = {}
        anonymous: list[dict] = []

        for speech in speeches:
            if speech.get("procedural"):
                continue

            # Respect passage-extractor annotations when present.
            if "ccus_relevant" in speech and not speech["ccus_relevant"]:
                continue

            politician_url = speech.get("politician_url") or speech.get("politician")
            if not politician_url:
                anonymous.append(speech)
                continue

            if politician_url not in actors:
                name = self._resolve_name(speech)
                party = self._resolve_party(speech)
                actors[politician_url] = PoliticalActor(
                    name=name,
                    politician_url=politician_url,
                    party=party,
                    speeches=[],
                )
            actors[politician_url].speeches.append(speech)

        result = list(actors.values())

        if anonymous:
            result.append(PoliticalActor(
                name="Unknown",
                politician_url=None,
                party=None,
                speeches=anonymous,
            ))

        return result

    def _resolve_name(self, speech: dict) -> str:
        attribution = speech.get("attribution", {})
        if isinstance(attribution, dict):
            return attribution.get("en") or attribution.get("fr") or "Unknown"
        return str(attribution) if attribution else "Unknown"

    def _resolve_party(self, speech: dict) -> str | None:
        # Party info isn't directly in the speech response; return None for now.
        return None


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _write_step3_actors(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    """
    Given the step-2 speeches file, group speeches into PoliticalActor records
    and write them to ``{output_dir}/step3_actors.json`` and a CSV summary.
    
    This is a standalone entry point so that you can run just the "actor
    extraction" step of the pipeline and inspect the results.
    """
    import json

    root = Path(output_dir) if output_dir is not None else JSON_DIR.parent
    json_dir = root / "json"
    csv_dir = root / "csv"
    json_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    in_path = Path(input_path) if input_path is not None else json_dir / "step2_speeches.json"
    if not in_path.exists():
        raise SystemExit(
            f"[Step3] Expected input file not found: {in_path}. "
            "Run parliament.ccus_analysis.step2_hansard_fetcher as a script first."
        )
    
    records = json.loads(in_path.read_text())
    extractor = ActorExtractor()

    out_path = json_dir / "step3_actors.json"
    
    out_records: list[dict] = []
    party_cache: dict[str, str | None] = {}

    print(f"[Step3] Extracting actors for {len(records)} bill entr(y/ies)...", flush=True)

    def _lookup_party_for_actor(actor_dict: dict) -> str | None:
        """Best-effort lookup of party using politician_membership_url on speeches."""
        for speech in actor_dict.get("speeches", []):
            membership_url = speech.get("politician_membership_url")
            if not membership_url:
                continue
            if membership_url in party_cache:
                return party_cache[membership_url]
            try:
                resp = requests.get(
                    API_BASE_URL.rstrip("/") + membership_url,
                    headers={"Accept": "application/json"},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                party = (
                    (data.get("party") or {})
                    .get("short_name", {})
                    .get("en")
                    or (data.get("party") or {})
                    .get("name", {})
                    .get("en")
                )
            except Exception:
                party = None
            party_cache[membership_url] = party
            if party:
                return party
        return None

    for rec in records:
        speeches = rec.get("speeches", [])
        actors = extractor.extract(speeches)
        actors_out: list[dict] = []
        for a in actors:
            actor_dict = {
                "name": a.name,
                "politician_url": a.politician_url,
                "party": a.party,
                "speeches": a.speeches,
            }
            party = _lookup_party_for_actor(actor_dict)
            if party:
                actor_dict["party"] = party
            actors_out.append(actor_dict)

        out_records.append(
            {
                "manual_number": rec["manual_number"],
                "manual_session": rec["manual_session"],
                "bill": rec["bill"],
                "actors": actors_out,
            }
        )
    
    out_path.write_text(json.dumps(out_records, ensure_ascii=False, indent=2))

    # CSV summary: one row per actor.
    csv_path = csv_dir / "step3_actors.csv"
    fieldnames = [
        "manual_number",
        "manual_session",
        "session",
        "bill_number",
        "actor_name",
        "politician_url",
        "party",
        "speech_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in out_records:
            bill = rec["bill"]
            for a in rec.get("actors", []):
                writer.writerow(
                    {
                        "manual_number": rec["manual_number"],
                        "manual_session": rec["manual_session"],
                        "session": bill.get("session", ""),
                        "bill_number": bill.get("number", ""),
                        "actor_name": a["name"],
                        "politician_url": a["politician_url"],
                        "party": a.get("party") or "",
                        "speech_count": len(a.get("speeches", [])),
                    }
                )
    
    print(
        f"[Step3] Wrote actors for {len(out_records)} bill entr(y/ies) to {out_path} and {csv_path}",
        flush=True,
    )


def _main_cli() -> None:
    import os
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "parliament.settings")
    django.setup()
    _write_step3_actors()


if __name__ == "__main__":
    _main_cli()
