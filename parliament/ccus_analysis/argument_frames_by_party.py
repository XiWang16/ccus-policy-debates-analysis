"""
Argument-frame analysis by party: all arguments categorized by party of the speaker.
Each argument is a unit; run chi-squared test (party × argument frame).
"""

import csv
from pathlib import Path

from scipy.stats import chi2_contingency


def load_argument_frames(csv_path: Path) -> list[tuple[str, str]]:
    """
    Load all arguments from step4_arguments.csv.
    Returns list of (party, arg_type) for arguments with a known party.
    """
    rows: list[tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            party = (row.get("party") or "").strip()
            arg_type = (row.get("arg_type") or "").strip()
            if party and arg_type:
                rows.append((party, arg_type))
    return rows


def run_chi_squared(rows: list[tuple[str, str]]) -> tuple[float, float, int, str]:
    """Run chi-squared test of independence. Returns (chi2, p, df, result_str)."""
    from collections import defaultdict

    parties = sorted(set(p for p, t in rows))
    frames = sorted(set(t for p, t in rows))

    counts = defaultdict(lambda: defaultdict(int))
    for p, t in rows:
        counts[p][t] += 1

    table = [[counts[p][t] for t in frames] for p in parties]
    chi2, p_value, dof, expected = chi2_contingency(table)

    return chi2, p_value, dof, (
        f"chi2({dof}) = {chi2:.2f}, p = {p_value:.4f}"
    )


def main():
    output_dir = Path(__file__).parent / "output" / "csv"
    args_path = output_dir / "step4_arguments.csv"

    rows = load_argument_frames(args_path)
    print(f"Arguments with party + frame: {len(rows)}")

    chi2, p_value, dof, result_str = run_chi_squared(rows)
    print(f"Chi-squared test (party × argument frame): {result_str}")

    if p_value < 0.05:
        print("Result: Significant difference — argument frame distribution differs across parties.")
    else:
        print("Result: No significant difference — cannot reject independence.")

    # Write output for downstream use
    out_path = output_dir / "argument_frames_by_party.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["party", "arg_type"])
        w.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
