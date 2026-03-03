"""
Speech-level stance analysis: propagate actor opinions to each CCUS-relevant speech,
then run party × stance chi-squared test on the 78 speeches.

Each speech counts toward its party's stance; multiple speeches by the same MP
all contribute to the party total.
"""

import json
from pathlib import Path

from scipy.stats import chi2_contingency


def load_speech_level_stances(
    speeches_path: Path,
    opinions_path: Path,
) -> list[tuple[str, str]]:
    """
    For each CCUS-relevant speech, assign (party, stance) by propagating
    the actor's opinion. Returns list of (party, stance) for speeches with
    a known party.
    """
    speeches_data = json.loads(speeches_path.read_text())
    opinions_data = json.loads(opinions_path.read_text())

    # Build bill key -> opinions index (politician_url -> stance, party)
    def bill_key(rec):
        return (rec["manual_number"], rec.get("manual_session", ""))

    opinions_by_bill = {}
    for rec in opinions_data:
        key = bill_key(rec)
        opinions_by_bill[key] = {}
        for op in rec.get("opinions", []):
            actor = op.get("actor", {})
            pol_url = actor.get("politician_url")
            party = actor.get("party")
            stance = op.get("stance")
            if pol_url and stance:
                opinions_by_bill[key][pol_url] = (party, stance)
        # Anonymous actor: politician_url is None in opinions
        for op in rec.get("opinions", []):
            actor = op.get("actor", {})
            if actor.get("politician_url") is None:
                opinions_by_bill[key][None] = (actor.get("party"), op.get("stance"))
                break

    rows: list[tuple[str, str]] = []
    for rec in speeches_data:
        key = bill_key(rec)
        op_index = opinions_by_bill.get(key, {})
        for speech in rec.get("speeches", []):
            if not speech.get("ccus_relevant"):
                continue
            pol_url = speech.get("politician_url")
            if pol_url not in op_index and None not in op_index:
                continue
            party, stance = op_index.get(pol_url) or op_index.get(None, (None, None))
            if party and stance:
                rows.append((party, stance))

    return rows


def run_chi_squared(rows: list[tuple[str, str]]) -> tuple[float, float, int, str]:
    """Run chi-squared test of independence. Returns (chi2, p, df, result_str)."""
    from collections import defaultdict

    parties = sorted(set(p for p, s in rows))
    stances = sorted(set(s for p, s in rows))

    counts = defaultdict(lambda: defaultdict(int))
    for p, s in rows:
        counts[p][s] += 1

    table = [[counts[p][s] for s in stances] for p in parties]
    chi2, p_value, dof, expected = chi2_contingency(table)

    return chi2, p_value, dof, (
        f"chi2({dof}) = {chi2:.2f}, p = {p_value:.4f}"
    )


def main():
    output_dir = Path(__file__).parent / "output" / "json"
    speeches_path = output_dir / "step2_speeches.json"
    opinions_path = output_dir / "step4_opinions.json"

    rows = load_speech_level_stances(speeches_path, opinions_path)
    print(f"Speech-level records with party + stance: {len(rows)}")

    chi2, p_value, dof, result_str = run_chi_squared(rows)
    print(f"Chi-squared test (party × stance): {result_str}")

    if p_value < 0.05:
        print("Result: Significant difference — stance distribution differs across parties.")
    else:
        print("Result: No significant difference — cannot reject independence.")

    # Write speech-level output for downstream use
    out_path = output_dir.parent / "csv" / "speech_level_stances.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["party", "stance"])
        w.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
