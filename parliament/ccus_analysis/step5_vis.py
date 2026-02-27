#!/usr/bin/env python3
"""
master_vis.py
CCUS Parliamentary Debate Visualizations — 10 plots.

Follows the design system in vis_brief.txt.

Dependencies:
    pip install matplotlib seaborn plotly pandas numpy wordcloud scikit-learn scipy
    pip install kaleido   # for Plot 5 PNG/SVG export

Usage:
    python master_vis.py                        # use mock data
    python master_vis.py --real output.json     # use pipeline JSON output
"""

import itertools
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from .config import VIS_DIR, JSON_DIR

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

BASE_DPI = 150
BG       = "#0F0F1A"
PANEL_BG = "#1A1A2E"
GRID     = "#2A2A45"

PARTY_COLORS = {
    "Liberal":        "#E8333C",
    "Conservative":   "#1D6FCA",
    "NDP":            "#F4742B",
    "Bloc Québécois": "#7B5EA7",
    "Green":          "#3DBF6E",
    "Independent":    "#A0A0B0",
}

PARTY_ABBREV = {
    "Liberal":        "Lib.",
    "Conservative":   "CPC",
    "NDP":            "NDP",
    "Bloc Québécois": "BQ",
    "Green":          "Green",
    "Independent":    "Ind.",
}

STANCE_COLORS = {
    "Pro-CCUS":    "#3EC9A7",
    "Anti-CCUS":   "#FF6B6B",
    "Conditional": "#FFD166",
    "Neutral":     "#A0A0B0",
}

FRAME_COLORS = {
    "Economic":       "#F7C59F",
    "Environmental":  "#6EE7B7",
    "Technological":  "#93C5FD",
    "Regional":       "#C4B5FD",
    "Social/Justice": "#FCA5A5",
}

FRAMES   = ["Economic", "Environmental", "Technological", "Regional", "Social/Justice"]
PARTIES  = ["Liberal", "Conservative", "NDP", "Bloc Québécois", "Green"]

# Visualization output directory: <ccus_analysis>/output/vis
OUTPUT_DIR = VIS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_fig(figsize):
    fig = plt.figure(figsize=figsize, dpi=BASE_DPI)
    fig.patch.set_facecolor(BG)
    return fig


def _apply_dark_theme(ax):
    ax.set_facecolor(PANEL_BG)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(colors="#AAAACC", labelsize=9)
    ax.xaxis.label.set_color("#CCCCDD")
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_color("#CCCCDD")
    ax.yaxis.label.set_fontsize(11)


def _save_fig(fig, slug):
    png = OUTPUT_DIR / f"{slug}.png"
    svg = OUTPUT_DIR / f"{slug}.svg"
    fig.savefig(png, dpi=BASE_DPI, bbox_inches="tight", facecolor=BG)
    fig.savefig(svg, bbox_inches="tight", facecolor=BG)
    print(f"  Saved {png}")
    print(f"  Saved {svg}")
    plt.close(fig)


def _save_legend_fig(handles, title: str, slug: str) -> None:
    """Save a standalone legend as its own PNG and SVG."""
    fig_leg = plt.figure(figsize=(3, 2.5), facecolor=BG)
    ax_leg  = fig_leg.add_subplot(111)
    ax_leg.set_facecolor(BG)
    ax_leg.axis("off")
    leg = ax_leg.legend(
        handles=handles, loc="center",
        framealpha=0.3, facecolor=PANEL_BG, edgecolor=GRID,
        labelcolor="#CCCCDD", title=title, title_fontsize=10, fontsize=9,
    )
    leg.get_title().set_color("#CCCCDD")
    png = OUTPUT_DIR / f"{slug}.png"
    svg = OUTPUT_DIR / f"{slug}.svg"
    fig_leg.savefig(png, dpi=BASE_DPI, bbox_inches="tight", facecolor=BG)
    fig_leg.savefig(svg, bbox_inches="tight", facecolor=BG)
    print(f"  Saved {png}")
    print(f"  Saved {svg}")
    plt.close(fig_leg)


def _clean_mp_name(full_name: str) -> str:
    """Extract 'First Last' from 'Mr. First Last (Riding, Party)' attribution strings."""
    import re
    name = re.sub(r"^(Mr\.|Ms\.|Mrs\.|Hon\.|M\.|Mme\.)\s+", "", full_name.strip())
    name = re.sub(r"\s*\(.*\)\s*$", "", name)
    return name.strip() or full_name


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA
# ─────────────────────────────────────────────────────────────────────────────

def generate_mock_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (speeches_df, party_df, frame_df).
    speeches_df: one row per MP × bill combination.
    party_df:    aggregated stance counts by party.
    frame_df:    NxN frame co-occurrence matrix.
    """
    rng = random.Random(42)
    np.random.seed(42)

    # (name, party, province, riding, base_stance)
    mp_roster = [
        ("Greg McLean",             "Conservative",   "Alberta",          "Calgary Centre",               "Pro-CCUS"),
        ("Shannon Stubbs",          "Conservative",   "Alberta",          "Lakeland",                     "Pro-CCUS"),
        ("Warren Steinley",         "Conservative",   "Saskatchewan",     "Regina—Lewvan",                "Pro-CCUS"),
        ("Robert Kitchen",          "Conservative",   "Saskatchewan",     "Souris—Moose Mountain",        "Pro-CCUS"),
        ("Tamara Jansen",           "Conservative",   "British Columbia",  "Cloverdale—Langley City",     "Pro-CCUS"),
        ("Dan Albas",               "Conservative",   "British Columbia",  "Central Okanagan",            "Pro-CCUS"),
        ("Stephanie Kusie",         "Conservative",   "Alberta",          "Calgary Midnapore",            "Pro-CCUS"),
        ("Marc Dalton",             "Conservative",   "British Columbia",  "Pitt Meadows",                "Conditional"),
        ("Patrick Weiler",          "Liberal",        "British Columbia",  "West Vancouver",              "Conditional"),
        ("Jonathan Wilkinson",      "Liberal",        "British Columbia",  "North Vancouver",             "Conditional"),
        ("Mark Gerretsen",          "Liberal",        "Ontario",          "Kingston and the Islands",     "Conditional"),
        ("Kevin Lamoureux",         "Liberal",        "Manitoba",         "Winnipeg North",               "Conditional"),
        ("Francis Drouin",          "Liberal",        "Ontario",          "Glengarry—Prescott—Russell",   "Conditional"),
        ("Nathaniel Erskine-Smith", "Liberal",        "Ontario",          "Beaches—East York",            "Conditional"),
        ("Taylor Bachrach",         "NDP",            "British Columbia",  "Skeena—Bulkley Valley",       "Anti-CCUS"),
        ("Charlie Angus",           "NDP",            "Ontario",          "Timmins—James Bay",            "Anti-CCUS"),
        ("Richard Cannings",        "NDP",            "British Columbia",  "South Okanagan",              "Anti-CCUS"),
        ("Jenny Kwan",              "NDP",            "British Columbia",  "Vancouver East",              "Anti-CCUS"),
        ("Kristina Michaud",        "Bloc Québécois", "Quebec",           "Avignon—La Mitis",             "Anti-CCUS"),
        ("Mario Simard",            "Bloc Québécois", "Quebec",           "Jonquière",                    "Anti-CCUS"),
        ("Monique Pauzé",           "Bloc Québécois", "Quebec",           "Repentigny",                   "Anti-CCUS"),
        ("Alexis Brunelle-Duceppe", "Bloc Québécois", "Quebec",           "Lac-Saint-Jean",               "Conditional"),
        ("Elizabeth May",           "Green",          "British Columbia",  "Saanich—Gulf Islands",        "Anti-CCUS"),
        ("Mike Morrice",            "Green",          "Ontario",          "Kitchener Centre",             "Anti-CCUS"),
    ]

    bills = ["C-59", "C-69", "C-262", "C-50"]
    bill_sessions = {"C-59": "44-1", "C-69": "42-1", "C-262": "43-2", "C-50": "39-2"}

    frame_probs = {
        "Pro-CCUS":    [0.40, 0.15, 0.30, 0.10, 0.05],
        "Anti-CCUS":   [0.20, 0.35, 0.10, 0.10, 0.25],
        "Conditional": [0.30, 0.25, 0.20, 0.15, 0.10],
    }

    text_pool = {
        "Economic": [
            "The CCUS investment tax credit creates thousands of jobs in Alberta and Saskatchewan. "
            "This technology is fundamental to Canada's economic competitiveness.",
            "Carbon capture represents a $100 billion economic opportunity. The private sector "
            "is ready to invest if government provides the right policy framework.",
            "CCUS creates fiscal revenue, employment, and maintains export competitiveness "
            "in the oil sands sector.",
        ],
        "Environmental": [
            "Carbon capture is not a real climate solution. It allows the fossil fuel industry "
            "to continue emissions rather than transitioning to clean energy.",
            "The science shows we cannot rely on CCUS to meet net-zero targets. "
            "We need to reduce emissions at source, not bury them underground.",
            "Every dollar spent on CCUS is a dollar not invested in solar, wind, "
            "and other proven clean technologies.",
        ],
        "Technological": [
            "Canada has world-class geological storage capacity in Western sedimentary basins. "
            "We have the technology and geology to be a global leader in carbon sequestration.",
            "Direct air capture combined with geological storage offers genuine permanent "
            "carbon dioxide removal at industrial scale.",
            "The engineering challenges of CO2 compression, transport, and injection "
            "into saline aquifers have been largely solved.",
        ],
        "Regional": [
            "Alberta and Saskatchewan workers depend on the oil sands. "
            "A just transition must include CCUS investment to protect these communities.",
            "Western provinces have contributed disproportionately to Canada's fiscal health. "
            "Federal CCUS support is the least we can do for these regions.",
            "Quebec has largely decarbonized its electricity grid. "
            "We should not be subsidizing Alberta's emissions through federal CCUS credits.",
        ],
        "Social/Justice": [
            "Indigenous communities near proposed CCUS sites have not given free prior "
            "informed consent. Their rights under UNDRIP must be respected.",
            "Environmental justice demands that we address communities most harmed "
            "by fossil fuel pollution, not extend the lifespan of these industries.",
            "Working families in oil-producing regions face economic anxiety. "
            "CCUS offers a bridge that preserves livelihoods during the energy transition.",
        ],
    }

    rows = []
    start_date = datetime(2022, 1, 1)

    for mp_name, party, province, riding, base_stance in mp_roster:
        n_bills = rng.randint(2, 4)
        mp_bills = rng.sample(bills, min(n_bills, len(bills)))

        for bill in mp_bills:
            stance = base_stance
            if rng.random() < 0.12:
                stance = rng.choice(["Pro-CCUS", "Anti-CCUS", "Conditional"])

            speech_count = rng.randint(1, 8)
            word_count   = speech_count * rng.randint(200, 600)

            probs = frame_probs.get(stance, frame_probs["Conditional"])
            n_frames = rng.randint(1, 3)
            sampled = list(np.random.choice(FRAMES, size=n_frames, replace=False, p=probs))
            primary = sampled[0]

            date    = start_date + timedelta(days=rng.randint(0, 700))
            excerpt = rng.choice(text_pool[primary])

            rows.append({
                "mp_name":      mp_name,
                "party":        party,
                "province":     province,
                "riding":       riding,
                "bill":         bill,
                "parliament":   bill_sessions[bill],
                "date":         date,
                "stance":       stance,
                "speech_count": speech_count,
                "word_count":   word_count,
                "frames":       sampled,
                "primary_frame": primary,
                "text_excerpt": excerpt,
            })

    speeches_df = pd.DataFrame(rows)
    speeches_df["date"] = pd.to_datetime(speeches_df["date"])

    # party_df
    agg = speeches_df.groupby(["party", "stance"])["speech_count"].sum().reset_index()
    agg.columns = ["party", "stance", "count"]
    totals = agg.groupby("party")["count"].sum()
    # Vectorised proportion to avoid apply/DataFrame shape issues:
    agg["proportion"] = agg["count"] / agg["party"].map(totals).replace(0, pd.NA)
    party_df = agg

    # frame_df (co-occurrence)
    fi = {f: i for i, f in enumerate(FRAMES)}
    mat = np.zeros((len(FRAMES), len(FRAMES)))
    for _, row in speeches_df.iterrows():
        fs = row["frames"]
        for f1, f2 in itertools.combinations(fs, 2):
            i, j = fi[f1], fi[f2]
            mat[i][j] += row["speech_count"]
            mat[j][i] += row["speech_count"]
        for f in fs:
            mat[fi[f]][fi[f]] += row["speech_count"]
    frame_df = pd.DataFrame(mat, index=FRAMES, columns=FRAMES)

    return speeches_df, party_df, frame_df


# ─────────────────────────────────────────────────────────────────────────────
# LOAD REAL PIPELINE DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline_data(json_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert pipeline JSON output (written by JSONOutputWriter) into the
    three DataFrames used by all plot functions.

    Stance mapping: support→Pro-CCUS, oppose→Anti-CCUS, mixed→Conditional,
                    neutral→Neutral.
    """
    import json

    STANCE_MAP = {
        "support": "Pro-CCUS",
        "oppose":  "Anti-CCUS",
        "mixed":   "Conditional",
        "neutral": "Neutral",
    }

    with open(json_path) as f:
        data = json.load(f)

    # Accept both {"bills": [...]} (CCUSAnalysisResult) and a bare list (step4 output).
    bill_records = data if isinstance(data, list) else data.get("bills", [])

    rows = []
    for bill_rec in bill_records:
        bill      = bill_rec.get("bill", {})
        bill_num  = bill.get("number", "Unknown")
        session   = bill.get("session", "")
        bill_name = (bill.get("name") or {}).get("en", bill_num)

        opinions  = bill_rec.get("opinions", [])
        for opinion in opinions:
            actor      = opinion.get("actor", {})
            mp_name    = actor.get("name", "Unknown")
            party      = actor.get("party") or "Independent"
            raw_stance = opinion.get("stance", "neutral")
            stance     = STANCE_MAP.get(raw_stance, "Neutral")

            arguments  = opinion.get("arguments", [])
            frames     = list({a.get("type", "economic").capitalize() for a in arguments})
            frame_map  = {"Economic": "Economic", "Environmental": "Environmental",
                          "Technical": "Technological", "Technological": "Technological",
                          "Jurisdictional": "Regional", "Social": "Social/Justice",
                          "Ethical": "Social/Justice"}
            frames     = [frame_map.get(f, "Economic") for f in frames]
            if not frames:
                frames = ["Economic"]
            primary    = frames[0]

            speech_count = len(actor.get("speeches", []))

            rows.append({
                "mp_name":       mp_name,
                "party":         party,
                "province":      "",
                "riding":        "",
                "bill":          bill_num,
                "parliament":    session,
                "date":          pd.Timestamp.now(),
                "stance":        stance,
                "speech_count":  max(speech_count, 1),
                "word_count":    speech_count * 300,
                "frames":        frames,
                "primary_frame": primary,
                "text_excerpt":  " ".join(a.get("quote", "") for a in arguments[:2]),
            })

    speeches_df = pd.DataFrame(rows) if rows else _empty_speeches_df()
    speeches_df["date"] = pd.to_datetime(speeches_df["date"])

    # party_df
    agg = speeches_df.groupby(["party", "stance"])["speech_count"].sum().reset_index()
    agg.columns = ["party", "stance", "count"]
    totals = agg.groupby("party")["count"].sum()
    # Use vectorised division instead of row-wise apply to avoid
    # creating a multi-column frame when assigning a single column.
    denom = agg["party"].map(totals)
    agg["proportion"] = np.where(denom != 0, agg["count"] / denom, np.nan)

    # frame_df
    fi  = {f: i for i, f in enumerate(FRAMES)}
    mat = np.zeros((len(FRAMES), len(FRAMES)))
    for _, row in speeches_df.iterrows():
        fs = [f for f in row["frames"] if f in fi]
        for f1, f2 in itertools.combinations(fs, 2):
            i, j = fi[f1], fi[f2]
            mat[i][j] += row["speech_count"]
            mat[j][i] += row["speech_count"]
        for f in fs:
            mat[fi[f]][fi[f]] += row["speech_count"]

    return speeches_df, agg, pd.DataFrame(mat, index=FRAMES, columns=FRAMES)


def _empty_speeches_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "mp_name", "party", "province", "riding", "bill", "parliament",
        "date", "stance", "speech_count", "word_count", "frames",
        "primary_frame", "text_excerpt",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 01 — Diverging Lollipop: MP Speaking Frequency × Stance
# ─────────────────────────────────────────────────────────────────────────────

def plot_01_mp_lollipop(speeches_df: pd.DataFrame) -> None:
    # If there is no data at all, gracefully skip this plot.
    if speeches_df.empty:
        print("  Skipping Plot 1: no speech data available.")
        return

    agg = (speeches_df.groupby(["mp_name", "party", "stance"])["speech_count"]
           .sum().reset_index())
    pivot = (agg.pivot_table(index=["mp_name", "party"], columns="stance",
                             values="speech_count", fill_value=0)
             .reset_index())
    for col in ["Pro-CCUS", "Anti-CCUS", "Conditional"]:
        if col not in pivot.columns:
            pivot[col] = 0
    if pivot.empty:
        print("  Skipping Plot 1: no aggregated MP speech counts.")
        return

    pivot["total"] = pivot["Pro-CCUS"] + pivot["Anti-CCUS"] + pivot["Conditional"]

    # Sort: party group order first, then descending speech count within each party.
    _PARTY_ORDER = ["Conservative", "Liberal", "NDP", "Bloc Québécois", "Bloc", "Green", "Independent"]
    pivot["_party_rank"] = pivot["party"].map(
        lambda p: _PARTY_ORDER.index(p) if p in _PARTY_ORDER else len(_PARTY_ORDER)
    )
    pivot = (
        pivot.sort_values(["_party_rank", "total"], ascending=[True, False])
        .head(20)
        .reset_index(drop=True)
    )

    mx_candidates = [pivot["Pro-CCUS"].max(), pivot["Anti-CCUS"].max(), 1]
    mx_candidates = [c for c in mx_candidates if pd.notna(c) and np.isfinite(c)]
    if not mx_candidates:
        print("  Skipping Plot 1: no finite speech counts for axis scaling.")
        return
    mx = max(mx_candidates)

    n = len(pivot)
    fig = _make_fig((12, max(8, n * 0.55 + 2)))
    ax  = fig.add_subplot(111)
    _apply_dark_theme(ax)

    for i, row in pivot.iterrows():
        y   = n - 1 - i
        pc  = PARTY_COLORS.get(row["party"], "#A0A0B0")

        if row["Pro-CCUS"] > 0:
            ax.hlines(y, 0, row["Pro-CCUS"],  color=STANCE_COLORS["Pro-CCUS"],  lw=2.5, zorder=3)
            ax.scatter(row["Pro-CCUS"],  y, s=80, color=pc, edgecolors="white", lw=1.2, zorder=5)
        if row["Anti-CCUS"] > 0:
            ax.hlines(y, -row["Anti-CCUS"], 0, color=STANCE_COLORS["Anti-CCUS"], lw=2.5, zorder=3)
            ax.scatter(-row["Anti-CCUS"], y, s=80, color=pc, edgecolors="white", lw=1.2, zorder=5)
        if row["Conditional"] > 0:
            # Conditional stance is shown as a yellow diamond centered at x=0.
            ax.scatter(0, y + 0.33, s=40, color=STANCE_COLORS["Conditional"], marker="D", zorder=4)

    # Put MP labels in the middle (x=0) instead of the left y-axis.
    ax.set_yticks(range(n))
    ax.set_yticklabels([""] * n)
    ax.tick_params(axis="y", length=0)
    for i, row in pivot.iterrows():
        y = n - 1 - i
        pc = PARTY_COLORS.get(row["party"], "#A0A0B0")
        label = f"{_clean_mp_name(row['mp_name'])} ({PARTY_ABBREV.get(row['party'], '?')})"
        ax.text(
            0, y, label,
            ha="center", va="center",
            color=pc, fontsize=10, fontweight="bold",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=PANEL_BG, edgecolor="none", alpha=0.75),
        )

    ax.set_xlim(-mx * 1.5, mx * 1.35)
    ax.axvline(0, color="white", lw=1.5, alpha=0.4, zorder=2)

    # Bold, prominent directional label under the plot.
    ax.set_xlabel(
        "← ANTI   |   CONDITIONAL   |   PRO →",
        fontweight="bold", fontsize=13, labelpad=10,
    )

    ax.set_title(
        "Parliamentary Engagement with Federal CCUS Legislation\nby MP and Stance",
        color="white", fontsize=14, fontweight="bold",
    )

    # Save party legend as a standalone figure.
    present_parties = pivot["party"].unique()
    party_patches = [
        mpatches.Patch(color=c, label=p)
        for p, c in PARTY_COLORS.items()
        if p in present_parties
    ]
    _save_legend_fig(party_patches, "Party", "vis_01_mp_lollipop_legend")

    _save_fig(fig, "vis_01_mp_lollipop")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 02 — Stacked Proportion Bar: Stance by Party
# ─────────────────────────────────────────────────────────────────────────────

def plot_02_party_stance(party_df: pd.DataFrame) -> None:
    pivot = party_df.pivot_table(index="party", columns="stance",
                                 values="proportion", fill_value=0)
    for col in ["Pro-CCUS", "Conditional", "Anti-CCUS"]:
        if col not in pivot.columns:
            pivot[col] = 0
    totals = party_df.groupby("party")["count"].sum()
    pivot  = pivot.reindex(pivot["Pro-CCUS"].sort_values().index)

    fig = _make_fig((11, 6))
    ax  = fig.add_subplot(111)
    _apply_dark_theme(ax)

    left = np.zeros(len(pivot))
    for stance, color in [("Pro-CCUS", STANCE_COLORS["Pro-CCUS"]),
                           ("Conditional", STANCE_COLORS["Conditional"]),
                           ("Anti-CCUS",  STANCE_COLORS["Anti-CCUS"])]:
        vals = pivot[stance].values
        ax.barh(range(len(pivot)), vals, left=left, color=color, height=0.55,
                edgecolor=BG, linewidth=0.8)
        for j, (v, l) in enumerate(zip(vals, left)):
            if v > 0.07:
                ax.text(l + v / 2, j, f"{v:.0%}", va="center", ha="center",
                        color="#0F0F1A", fontsize=9, fontweight="bold")
        left += vals

    for j, party in enumerate(pivot.index):
        n = int(totals.get(party, 0))
        ax.text(1.02, j, f"n={n}", va="center", color="#CCCCDD", fontsize=9)

    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    for tick, party in zip(ax.get_yticklabels(), pivot.index):
        tick.set_color(PARTY_COLORS.get(party, "#A0A0B0"))

    ax.set_xlim(0, 1.13)
    ax.axvline(0.5, color="white", lw=1, alpha=0.3, linestyle=":")
    ax.set_xlabel("Proportion of CCUS Speech Segments")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    patches = [mpatches.Patch(color=STANCE_COLORS[s], label=s)
               for s in ["Pro-CCUS", "Conditional", "Anti-CCUS"]]
    ax.legend(handles=patches, loc="lower right", framealpha=0.2, facecolor=PANEL_BG,
              edgecolor=GRID, labelcolor="#CCCCDD", ncol=3)

    ax.set_title("Party-Level Positioning on Federal CCUS Legislation",
                 color="white", fontsize=14, fontweight="bold")
    ax.text(0.5, -0.13, "Proportion of speech segments by expressed stance",
            transform=ax.transAxes, ha="center", color="#AAAACC", fontsize=9)
    _save_fig(fig, "vis_02_party_stance")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 03 — MP × Bill Heatmap: Engagement Breadth
# ─────────────────────────────────────────────────────────────────────────────

def plot_03_mp_bill_heatmap(speeches_df: pd.DataFrame) -> None:
    if speeches_df.empty:
        print("  Skipping Plot 3: no speech data available.")
        return

    pivot     = speeches_df.pivot_table(index="mp_name", columns="bill",
                                        values="speech_count", aggfunc="sum", fill_value=0)
    if pivot.empty:
        print("  Skipping Plot 3: no MP × bill combinations to plot.")
        return
    party_map = (speeches_df[["mp_name", "party"]].drop_duplicates()
                 .set_index("mp_name")["party"])
    pivot["_party"] = pivot.index.map(party_map)
    pivot["_total"] = pivot.drop(columns="_party").sum(axis=1)
    pivot = pivot.sort_values(["_party", "_total"], ascending=[True, False]).head(25)
    party_col  = pivot["_party"]
    heatmap_data = pivot.drop(columns=["_party", "_total"])
    if heatmap_data.size == 0:
        print("  Skipping Plot 3: empty heatmap matrix.")
        return

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "ccus_heat", ["#1A1A2E", "#1B6CA8", "#3EC9A7"])

    fig = _make_fig((13, 11))
    gs  = fig.add_gridspec(1, 2, width_ratios=[0.04, 0.96], wspace=0.02)
    ax_bar  = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])

    sns.heatmap(
        heatmap_data, ax=ax_main, cmap=cmap, annot=True, fmt="d",
        annot_kws={"fontsize": 9, "color": "white"},
        linewidths=0.5, linecolor="#0F0F1A",
        cbar_kws={"label": "Speech Count", "shrink": 0.8},
    )
    ax_main.set_facecolor(PANEL_BG)
    ax_main.tick_params(colors="#AAAACC")
    ax_main.set_xlabel("")
    ax_main.set_ylabel("")
    ax_main.xaxis.tick_top()
    ax_main.set_xticklabels(ax_main.get_xticklabels(), rotation=0,
                            color="#CCCCDD", fontsize=10)
    y_labels = [
        f"{n.split()[-1]} ({party_col.get(n, '?')[:2]})"
        for n in heatmap_data.index
    ]
    ax_main.set_yticklabels(y_labels, rotation=0, fontsize=8, color="#AAAACC")

    cbar = ax_main.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("#CCCCDD")
    cbar.ax.tick_params(colors="#AAAACC")

    # Party sidebar
    ax_bar.set_facecolor(BG)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, len(party_col))
    ax_bar.axis("off")
    prev, block_start = None, 0
    for i, (mp, party) in enumerate(party_col.items()):
        ax_bar.add_patch(mpatches.Rectangle((0, i), 1, 1,
                         color=PARTY_COLORS.get(party, "#A0A0B0"), ec="none"))
        if party != prev and i > 0:
            ax_bar.axhline(i, color="#0F0F1A", lw=1.5, alpha=0.5)
            mid = (block_start + i) / 2
            ax_bar.text(0.5, mid, PARTY_ABBREV.get(prev, "?"),
                        ha="center", va="center", fontsize=7, color="white",
                        fontweight="bold", rotation=90)
            block_start = i
        prev = party
    mid = (block_start + len(party_col)) / 2
    ax_bar.text(0.5, mid, PARTY_ABBREV.get(prev, "?"),
                ha="center", va="center", fontsize=7, color="white",
                fontweight="bold", rotation=90)

    ax_main.set_title("CCUS Legislative Engagement by MP and Bill/Venue",
                      color="white", fontsize=14, fontweight="bold", pad=30)
    fig.patch.set_facecolor(BG)
    _save_fig(fig, "vis_03_mp_bill_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 04 — Radar / Spider: Frame Prevalence by Stance
# ─────────────────────────────────────────────────────────────────────────────

def plot_04_radar_frames(speeches_df: pd.DataFrame) -> None:
    n      = len(FRAMES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    fig = _make_fig((9, 9))
    ax  = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(PANEL_BG)
    fig.patch.set_facecolor(BG)

    for stance, color in [("Pro-CCUS", STANCE_COLORS["Pro-CCUS"]),
                           ("Anti-CCUS", STANCE_COLORS["Anti-CCUS"])]:
        sub    = speeches_df[speeches_df["stance"] == stance]
        counts = [sub[sub["primary_frame"] == f]["speech_count"].sum() for f in FRAMES]
        mx     = max(counts) or 1
        vals   = [c / mx for c in counts]
        closed = np.append(vals, vals[0])
        ac     = np.append(angles, angles[0])

        ax.plot(ac, closed, color=color, lw=2.5, alpha=0.9, label=stance)
        ax.fill(ac, closed, color=color, alpha=0.25)
        ax.scatter(angles, vals, color=color, s=80, zorder=5)

    for angle, frame in zip(angles, FRAMES):
        ax.plot([angle, angle], [0, 1.0], color=GRID, lw=1, zorder=0)
        ha = "left" if np.cos(angle) > 0.1 else ("right" if np.cos(angle) < -0.1 else "center")
        ax.text(angle, 1.22, frame, ha=ha, va="center", color="#CCCCDD", fontsize=11)

    for r in [0.25, 0.50, 0.75, 1.00]:
        ax.plot(np.append(angles, angles[0]), [r] * (n + 1), color=GRID, lw=0.5, ls="--")
    ax.text(angles[0], 0.56, "50%", ha="center", va="center", color="#AAAACC", fontsize=8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_rlim(0, 1.0)
    ax.spines["polar"].set_color(GRID)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), framealpha=0.2,
              facecolor=PANEL_BG, edgecolor=GRID, labelcolor="#CCCCDD", ncol=2, fontsize=10)
    ax.set_title("Argument Frame Composition\nPro-CCUS vs. Anti-CCUS Legislative Discourse",
                 color="white", fontsize=14, fontweight="bold", pad=35)
    fig.text(0.5, 0.02, "Normalized prevalence across speech segments",
             ha="center", color="#AAAACC", fontsize=9)
    _save_fig(fig, "vis_04_radar_frames")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 05 — Sankey: Party → Stance → Frame
# ─────────────────────────────────────────────────────────────────────────────

def plot_05_sankey(speeches_df: pd.DataFrame) -> None:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        print("  Skipping Plot 5: plotly not installed (pip install plotly kaleido).")
        return

    def hex_rgba(hex_color, alpha=0.45):
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"

    parties_present = [p for p in PARTIES if p in speeches_df["party"].unique()]
    stances_present = [s for s in ["Pro-CCUS", "Anti-CCUS", "Conditional"]
                       if s in speeches_df["stance"].unique()]
    frames_present  = FRAMES

    all_nodes  = parties_present + stances_present + frames_present
    node_idx   = {n: i for i, n in enumerate(all_nodes)}

    sources, targets, values, link_colors = [], [], [], []

    for _, row in speeches_df.groupby(["party", "stance"])["speech_count"].sum().reset_index().iterrows():
        if row["party"] in node_idx and row["stance"] in node_idx:
            sources.append(node_idx[row["party"]])
            targets.append(node_idx[row["stance"]])
            values.append(int(row["speech_count"]))
            link_colors.append(hex_rgba(PARTY_COLORS.get(row["party"], "#A0A0B0")))

    for _, row in speeches_df.groupby(["stance", "primary_frame"])["speech_count"].sum().reset_index().iterrows():
        if row["stance"] in node_idx and row["primary_frame"] in node_idx:
            sources.append(node_idx[row["stance"]])
            targets.append(node_idx[row["primary_frame"]])
            values.append(int(row["speech_count"]))
            link_colors.append(hex_rgba(STANCE_COLORS.get(row["stance"], "#A0A0B0")))

    node_colors = (
        [PARTY_COLORS.get(p, "#A0A0B0")  for p in parties_present] +
        [STANCE_COLORS.get(s, "#A0A0B0") for s in stances_present] +
        [FRAME_COLORS.get(f, "#A0A0B0")  for f in frames_present]
    )

    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, label=all_nodes, color=node_colors,
                  line=dict(color="#0F0F1A", width=0.5)),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
    ))
    fig.update_layout(
        title_text="Flow of CCUS Legislative Discourse: Party → Position → Argument Frame",
        title_font=dict(color="#FFFFFF", size=16),
        paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A",
        font=dict(color="#CCCCDD", size=12, family="Inter"),
        width=1400, height=800,
        margin=dict(l=120, r=120, t=80, b=40),
        annotations=[
            dict(x=0.05, y=1.05, text="Party",          showarrow=False,
                 font=dict(color="white", size=13), xref="paper", yref="paper"),
            dict(x=0.50, y=1.05, text="Stance",         showarrow=False,
                 font=dict(color="white", size=13), xref="paper", yref="paper"),
            dict(x=0.95, y=1.05, text="Dominant Frame", showarrow=False,
                 font=dict(color="white", size=13), xref="paper", yref="paper"),
        ],
    )

    png = str(OUTPUT_DIR / "vis_05_sankey.png")
    svg = str(OUTPUT_DIR / "vis_05_sankey.svg")
    try:
        fig.write_image(png, scale=2)
        fig.write_image(svg)
        print(f"  Saved {png}")
        print(f"  Saved {svg}")
    except Exception as e:
        html = str(OUTPUT_DIR / "vis_05_sankey.html")
        fig.write_html(html)
        print(f"  Warning: image export failed ({e}). Saved HTML → {html}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 06 — Faceted Bar: Frame Frequency by Party
# ─────────────────────────────────────────────────────────────────────────────

def plot_06_faceted_frames(speeches_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    fig.tight_layout(h_pad=4.5, w_pad=3.5)

    for idx, party in enumerate(PARTIES):
        ax = axes[idx // 3][idx % 3]
        _apply_dark_theme(ax)

        sub    = speeches_df[speeches_df["party"] == party]
        counts = (sub.groupby("primary_frame")["speech_count"].sum()
                  .reindex(FRAMES, fill_value=0)
                  .sort_values(ascending=True))
        colors = [FRAME_COLORS[f] for f in counts.index]

        bars = ax.barh(range(len(counts)), counts.values, color=colors, height=0.6, linewidth=0)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    str(int(val)), va="center", color="white", fontsize=8.5)

        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(counts.index, fontsize=9, color="#AAAACC")
        ax.set_title(party, color=PARTY_COLORS.get(party, "#A0A0B0"),
                     fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_color(PARTY_COLORS.get(party, "#A0A0B0"))
        ax.spines["top"].set_linewidth(3)
        if idx // 3 == 1:
            ax.set_xlabel("Speech Count", color="#CCCCDD", fontsize=9)

    # 6th subplot → legend
    ax_leg = axes[1][2]
    ax_leg.set_facecolor(PANEL_BG)
    ax_leg.axis("off")
    patches = [mpatches.Patch(color=FRAME_COLORS[f], label=f) for f in FRAMES]
    ax_leg.legend(handles=patches, loc="center", framealpha=0, labelcolor="#CCCCDD",
                  fontsize=11, title="Argument Frames", title_fontsize=12, labelspacing=1.2)

    fig.suptitle("Argument Frame Distribution by Party",
                 color="white", fontsize=15, fontweight="bold", y=1.01)
    fig.text(0.5, 0.985, "Count of CCUS speech segments per frame, by party",
             ha="center", color="#AAAACC", fontsize=10)
    _save_fig(fig, "vis_06_faceted_frames")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 07 — Co-occurrence Heatmap: Frame × Frame
# ─────────────────────────────────────────────────────────────────────────────

def plot_07_cooccurrence(speeches_df: pd.DataFrame) -> None:
    fi  = {f: i for i, f in enumerate(FRAMES)}
    mat = np.zeros((len(FRAMES), len(FRAMES)))

    for _, row in speeches_df.iterrows():
        fs = [f for f in (row["frames"] if isinstance(row["frames"], list) else [row["frames"]])
              if f in fi]
        for f1, f2 in itertools.combinations(fs, 2):
            i, j = fi[f1], fi[f2]
            mat[i][j] += row["speech_count"]
            mat[j][i] += row["speech_count"]
        for f in fs:
            mat[fi[f]][fi[f]] += row["speech_count"]

    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cooc", ["#1A1A2E", "#7B5EA7", "#F7C59F"])

    fig = _make_fig((9, 8))
    ax  = fig.add_subplot(111)
    ax.set_facecolor(PANEL_BG)

    sns.heatmap(mat, ax=ax, cmap=cmap, mask=mask, annot=True, fmt=".0f",
                annot_kws={"fontsize": 12, "color": "white"},
                square=True, linewidths=1.5, linecolor="#0F0F1A",
                xticklabels=FRAMES, yticklabels=FRAMES,
                cbar_kws={"label": "Co-occurrence Count", "shrink": 0.8})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    for tick, frame in zip(ax.get_xticklabels(), FRAMES):
        tick.set_color(FRAME_COLORS.get(frame, "#AAAACC"))
    for tick, frame in zip(ax.get_yticklabels(), FRAMES):
        tick.set_color(FRAME_COLORS.get(frame, "#AAAACC"))

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("#CCCCDD")
    cbar.ax.tick_params(colors="#AAAACC")

    ax.set_title("Argument Frame Co-occurrence in CCUS Legislative Debates",
                 color="white", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01,
             "Diagonal = total usage frequency. Off-diagonal = co-occurrence frequency.",
             ha="center", color="#AAAACC", fontsize=9, style="italic")
    _save_fig(fig, "vis_07_cooccurrence")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 08 — Temporal Line Chart: Stance Trends Over Time
# ─────────────────────────────────────────────────────────────────────────────

def plot_08_temporal(speeches_df: pd.DataFrame) -> None:
    df = speeches_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    weekly = (df.groupby([pd.Grouper(key="date", freq="W"), "stance"])["speech_count"]
              .sum().unstack(fill_value=0))
    for col in ["Pro-CCUS", "Anti-CCUS", "Conditional"]:
        if col not in weekly.columns:
            weekly[col] = 0
    weekly_smooth = weekly.rolling(4, min_periods=1).mean()

    fig = _make_fig((14, 6))
    ax  = fig.add_subplot(111)
    _apply_dark_theme(ax)

    for stance in ["Pro-CCUS", "Anti-CCUS", "Conditional"]:
        color = STANCE_COLORS[stance]
        ax.plot(weekly_smooth.index, weekly_smooth[stance],
                color=color, lw=2.5, label=stance, zorder=3)
        ax.fill_between(weekly_smooth.index, 0, weekly_smooth[stance],
                        color=color, alpha=0.15, zorder=2)

    events = {
        "C-262 Introduced":               "2022-01-26",
        "Budget 2022 CCUS ITC Announced": "2022-04-07",
        "C-59 First Reading":             "2023-11-30",
        "C-59 Royal Assent":              "2024-06-20",
    }
    y_max = weekly_smooth.max().max() or 1
    t_min, t_max = weekly_smooth.index.min(), weekly_smooth.index.max()
    for label, date_str in events.items():
        ev = pd.Timestamp(date_str)
        if t_min <= ev <= t_max:
            ax.axvline(ev, color="white", lw=1, alpha=0.5, ls="--", zorder=4)
            ax.annotate(label, xy=(ev, y_max * 0.88), xytext=(5, 0),
                        textcoords="offset points", fontsize=8, color="#CCCCDD",
                        rotation=90, va="top")

    sessions = [("43-2", "2020-09-23", "2021-08-15"),
                ("44-1", "2021-11-22", "2025-01-01")]
    for _, start, end in sessions:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        ax.axvspan(max(s, t_min), min(e, t_max), alpha=0.04, color="white")

    ax.set_xlabel("Date")
    ax.set_ylabel("Speech Count (4-week rolling avg)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper right", framealpha=0.2, facecolor=PANEL_BG,
              edgecolor=GRID, labelcolor="#CCCCDD")
    ax.set_title("Temporal Distribution of CCUS Legislative Discourse",
                 color="white", fontsize=14, fontweight="bold")
    fig.text(0.5, -0.02, "Rolling 4-week average speech counts by expressed stance",
             ha="center", color="#AAAACC", fontsize=9)
    _save_fig(fig, "vis_08_temporal")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 09 — Word Clouds by Frame Category
# ─────────────────────────────────────────────────────────────────────────────

def plot_09_wordclouds(speeches_df: pd.DataFrame) -> None:
    try:
        from wordcloud import WordCloud
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("  Skipping Plot 9: wordcloud or scikit-learn not installed.")
        return

    frame_palettes = {
        "Economic":       ["#F7C59F", "#F4A261", "#E07B39"],
        "Environmental":  ["#6EE7B7", "#34D399", "#059669"],
        "Technological":  ["#93C5FD", "#60A5FA", "#3B82F6"],
        "Regional":       ["#C4B5FD", "#A78BFA", "#7C3AED"],
        "Social/Justice": ["#FCA5A5", "#F87171", "#EF4444"],
    }

    def color_func_for(palette):
        def _cf(*args, **kwargs):
            return random.choice(palette)
        return _cf

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.patch.set_facecolor(BG)
    fig.tight_layout(h_pad=3, w_pad=1)

    for idx, frame in enumerate(FRAMES):
        ax     = axes[idx // 3][idx % 3]
        corpus = list(speeches_df[speeches_df["primary_frame"] == frame]["text_excerpt"].dropna())

        if not corpus:
            ax.axis("off")
            continue

        try:
            vec    = TfidfVectorizer(max_features=80, stop_words="english")
            vec.fit(corpus)
            scores = dict(zip(vec.get_feature_names_out(),
                               vec.transform(corpus).mean(axis=0).A1))
        except Exception:
            scores = {w: 1.0 for text in corpus for w in text.split()[:10]}

        # Remove zero-weight words and skip if nothing remains.
        scores = {w: v for w, v in scores.items() if v > 0}
        if not scores:
            ax.axis("off")
            continue

        wc = WordCloud(
            background_color="#1A1A2E",
            color_func=color_func_for(frame_palettes[frame]),
            max_words=60, width=600, height=350,
            prefer_horizontal=0.85, collocations=False,
        ).generate_from_frequencies(scores)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(frame, color=FRAME_COLORS[frame], fontsize=13, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(FRAME_COLORS[frame])
            spine.set_linewidth(2)

    axes[1][2].set_facecolor(PANEL_BG)
    axes[1][2].axis("off")

    fig.suptitle("Key Vocabulary by Argument Frame",
                 color="white", fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, 0.985, "Word size proportional to TF-IDF weight within frame",
             ha="center", color="#AAAACC", fontsize=10)
    _save_fig(fig, "vis_09_wordclouds")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 10 — Dot Strip: MP Ideological Position on CCUS
# ─────────────────────────────────────────────────────────────────────────────

def plot_10_ideological_positions(speeches_df: pd.DataFrame) -> None:
    if speeches_df.empty:
        print("  Skipping Plot 10: no speech data available.")
        return

    agg = (speeches_df.groupby(["mp_name", "party"])
           .apply(lambda g: pd.Series({
               "pro_count":   g[g["stance"] == "Pro-CCUS"]["speech_count"].sum(),
               "anti_count":  g[g["stance"] == "Anti-CCUS"]["speech_count"].sum(),
               "total_count": g["speech_count"].sum(),
           }), include_groups=False)
           .reset_index())

    if agg.empty or not {"pro_count", "anti_count", "total_count"}.issubset(agg.columns):
        print("  Skipping Plot 10: insufficient data to compute ideological scores.")
        return
    agg["score"] = ((agg["pro_count"] - agg["anti_count"]) /
                    agg["total_count"].clip(lower=1)).clip(-1, 1)

    party_y     = {"Conservative": 5, "Liberal": 4, "NDP": 3,
                   "Bloc Québécois": 2, "Green": 1, "Independent": 0}
    party_means = agg.groupby("party")["score"].mean()

    np.random.seed(42)
    fig = _make_fig((14, 7))
    ax  = fig.add_subplot(111)
    _apply_dark_theme(ax)

    ax.axvspan(-1.1, 0,   alpha=0.04, color="#FF6B6B", zorder=0)
    ax.axvspan(0,    1.1, alpha=0.04, color="#3EC9A7", zorder=0)
    ax.axvline(0, color="white", lw=1.5, alpha=0.5, zorder=2)

    for _, row in agg.iterrows():
        py   = party_y.get(row["party"], 0)
        y    = py + np.random.uniform(-0.3, 0.3)
        size = max(40, min(200, int(row["total_count"]) * 8))
        ax.scatter(row["score"], y, s=size,
                   color=PARTY_COLORS.get(row["party"], "#A0A0B0"),
                   edgecolors="white", linewidth=0.8, alpha=0.85, zorder=3)

        pm = party_means.get(row["party"], 0)
        if abs(row["score"] - pm) > 0.3 and row["total_count"] >= 3:
            ax.annotate(row["mp_name"].split()[-1],
                        xy=(row["score"], y), xytext=(8, 5),
                        textcoords="offset points", fontsize=8, color="white",
                        arrowprops=dict(arrowstyle="->", color="white", lw=0.8))

    for party, mean_score in party_means.items():
        if party in party_y:
            py = party_y[party]
            ax.scatter(mean_score, py, s=250, color=PARTY_COLORS.get(party, "#A0A0B0"),
                       marker="D", edgecolors="white", linewidth=2, zorder=5)
            ax.text(mean_score, py + 0.45, PARTY_ABBREV.get(party, "?"),
                    ha="center", color=PARTY_COLORS.get(party, "#A0A0B0"),
                    fontsize=9, fontweight="bold")

    ax.set_yticks(list(party_y.values()))
    ax.set_yticklabels(list(party_y.keys()))
    for tick, party in zip(ax.get_yticklabels(), party_y.keys()):
        tick.set_color(PARTY_COLORS.get(party, "#A0A0B0"))
        tick.set_fontsize(12)
        tick.set_fontweight("bold")

    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel("← Opposes CCUS Legislation     CCUS Position Score     Supports CCUS →")

    for cnt, lbl in [(5, "5 speeches"), (10, "10 speeches"), (20, "20 speeches")]:
        ax.scatter([], [], s=cnt * 8, color="#A0A0B0", edgecolors="white",
                   linewidth=0.8, alpha=0.85, label=lbl)
    ax.legend(loc="lower right", framealpha=0.2, facecolor=PANEL_BG, edgecolor=GRID,
              labelcolor="#CCCCDD", fontsize=8, title="Total Speeches", title_fontsize=9)

    ax.set_title("MP-Level Ideological Positioning on Federal CCUS Legislation",
                 color="white", fontsize=14, fontweight="bold")
    fig.text(0.5, -0.02,
             "Score = (Pro − Anti) / Total speeches. Dot size = total speech volume.",
             ha="center", color="#AAAACC", fontsize=9)
    _save_fig(fig, "vis_10_ideological_positions")


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE HTML DASHBOARD  (plots 1, 2, 4, 5, 6, 7, 9)
# ─────────────────────────────────────────────────────────────────────────────

def _hex_rgba(hex_color: str, alpha: float = 0.45) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _plotly_layout(**extra) -> dict:
    base = dict(
        paper_bgcolor=BG, plot_bgcolor=PANEL_BG,
        font=dict(color="#CCCCDD", size=12, family="Inter"),
        margin=dict(l=60, r=40, t=70, b=50),
    )
    base.update(extra)
    return base


def _write_interactive(fig, slug: str) -> Path:
    """Write a standalone interactive HTML file; return its path."""
    path = OUTPUT_DIR / f"{slug}_interactive.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"  Saved {path}")
    return path


def _iplot_01_lollipop(speeches_df: pd.DataFrame) -> None:
    """Interactive diverging bar: MP speaking frequency × stance."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    if speeches_df.empty:
        return

    agg = speeches_df.groupby(["mp_name", "party", "stance"])["speech_count"].sum().reset_index()
    pivot = (agg.pivot_table(index=["mp_name", "party"], columns="stance",
                             values="speech_count", fill_value=0).reset_index())
    for col in ["Pro-CCUS", "Anti-CCUS", "Conditional"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["total"] = pivot["Pro-CCUS"] + pivot["Anti-CCUS"] + pivot["Conditional"]

    _PARTY_ORDER = ["Conservative", "Liberal", "NDP", "Bloc Québécois", "Bloc", "Green", "Independent"]
    pivot["_pr"] = pivot["party"].map(
        lambda p: _PARTY_ORDER.index(p) if p in _PARTY_ORDER else len(_PARTY_ORDER))
    pivot = pivot.sort_values(["_pr", "total"], ascending=[True, False]).head(20).reset_index(drop=True)
    n = len(pivot)

    fig = go.Figure()
    for i, row in pivot.iterrows():
        y = n - 1 - i
        pc = PARTY_COLORS.get(row["party"], "#A0A0B0")
        label = f"{_clean_mp_name(row['mp_name'])} ({PARTY_ABBREV.get(row['party'], '?')})"

        if row["Pro-CCUS"] > 0:
            fig.add_shape(type="line", x0=0, x1=row["Pro-CCUS"], y0=y, y1=y,
                          line=dict(color=STANCE_COLORS["Pro-CCUS"], width=3))
            fig.add_trace(go.Scatter(x=[row["Pro-CCUS"]], y=[y], mode="markers",
                                     marker=dict(color=pc, size=10, line=dict(color="white", width=1.5)),
                                     hovertemplate=f"<b>{label}</b><br>Pro-CCUS: {row['Pro-CCUS']}<extra></extra>",
                                     showlegend=False))
        if row["Anti-CCUS"] > 0:
            fig.add_shape(type="line", x0=-row["Anti-CCUS"], x1=0, y0=y, y1=y,
                          line=dict(color=STANCE_COLORS["Anti-CCUS"], width=3))
            fig.add_trace(go.Scatter(x=[-row["Anti-CCUS"]], y=[y], mode="markers",
                                     marker=dict(color=pc, size=10, line=dict(color="white", width=1.5)),
                                     hovertemplate=f"<b>{label}</b><br>Anti-CCUS: {row['Anti-CCUS']}<extra></extra>",
                                     showlegend=False))
        if row["Conditional"] > 0:
            fig.add_trace(go.Scatter(x=[0], y=[y + 0.33], mode="markers",
                                     marker=dict(color=STANCE_COLORS["Conditional"], size=8, symbol="diamond"),
                                     hovertemplate=f"<b>{label}</b><br>Conditional: {row['Conditional']}<extra></extra>",
                                     showlegend=False))
        # Centered label annotation
        fig.add_annotation(x=0, y=y, text=label, showarrow=False,
                           font=dict(color=pc, size=11), bgcolor=PANEL_BG,
                           borderpad=3, opacity=0.9, xanchor="center", yanchor="middle")

    fig.add_vline(x=0, line_color="white", line_width=1.5, opacity=0.4)
    mx = max(pivot["Pro-CCUS"].max(), pivot["Anti-CCUS"].max(), 1)
    fig.update_layout(
        **_plotly_layout(
            title=dict(text="Parliamentary Engagement with Federal CCUS Legislation by MP and Stance",
                       font=dict(size=15, color="white")),
            xaxis=dict(range=[-mx * 1.5, mx * 1.35], gridcolor=GRID, zerolinecolor=GRID,
                       title=dict(text="← ANTI   |   CONDITIONAL   |   PRO →",
                                  font=dict(size=13, color="#CCCCDD"))),
            yaxis=dict(visible=False, range=[-0.5, n - 0.5]),
            height=max(500, n * 35 + 100),
        )
    )
    _write_interactive(fig, "vis_01_mp_lollipop")


def _iplot_02_party_stance(party_df: pd.DataFrame) -> None:
    """Interactive stacked bar: stance by party."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    pivot = party_df.pivot_table(index="party", columns="stance",
                                 values="proportion", fill_value=0)
    for col in ["Pro-CCUS", "Conditional", "Anti-CCUS"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot.reindex(pivot["Pro-CCUS"].sort_values().index)
    counts = party_df.groupby("party")["count"].sum()

    fig = go.Figure()
    for stance in ["Pro-CCUS", "Conditional", "Anti-CCUS"]:
        fig.add_trace(go.Bar(
            y=pivot.index.tolist(),
            x=pivot[stance].tolist(),
            name=stance,
            orientation="h",
            marker_color=STANCE_COLORS.get(stance, "#A0A0B0"),
            hovertemplate="%{y}: %{x:.0%}<extra>" + stance + "</extra>",
        ))

    fig.update_layout(
        **_plotly_layout(
            title=dict(text="Party-Level Positioning on Federal CCUS Legislation",
                       font=dict(size=15, color="white")),
            barmode="stack",
            xaxis=dict(tickformat=".0%", range=[0, 1.12], gridcolor=GRID, title="Proportion"),
            yaxis=dict(gridcolor=GRID),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center"),
            height=400,
        )
    )
    _write_interactive(fig, "vis_02_party_stance")


def _iplot_04_radar(speeches_df: pd.DataFrame) -> None:
    """Interactive radar chart: frame prevalence by stance."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    fig = go.Figure()
    for stance, color in [("Pro-CCUS", STANCE_COLORS["Pro-CCUS"]),
                           ("Anti-CCUS", STANCE_COLORS["Anti-CCUS"])]:
        sub    = speeches_df[speeches_df["stance"] == stance]
        counts = [sub[sub["primary_frame"] == f]["speech_count"].sum() for f in FRAMES]
        mx     = max(counts) or 1
        vals   = [c / mx for c in counts]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=FRAMES + [FRAMES[0]],
            fill="toself",
            name=stance,
            line=dict(color=color, width=2.5),
            fillcolor=_hex_rgba(color, 0.2),
        ))

    fig.update_layout(
        **_plotly_layout(
            polar=dict(
                bgcolor=PANEL_BG,
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID, color="#AAAACC"),
                angularaxis=dict(gridcolor=GRID, color="#CCCCDD"),
            ),
            title=dict(text="Argument Frame Composition: Pro-CCUS vs Anti-CCUS",
                       font=dict(size=15, color="white")),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
            height=550,
        )
    )
    _write_interactive(fig, "vis_04_radar_frames")


def _iplot_05_sankey(speeches_df: pd.DataFrame) -> None:
    """Interactive Sankey: party → stance → frame.  (Re-uses existing plot_05 logic.)"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    parties_present = [p for p in PARTIES if p in speeches_df["party"].unique()]
    stances_present = [s for s in ["Pro-CCUS", "Anti-CCUS", "Conditional"]
                       if s in speeches_df["stance"].unique()]
    all_nodes = parties_present + stances_present + FRAMES
    node_idx  = {n: i for i, n in enumerate(all_nodes)}

    sources, targets, values, link_colors = [], [], [], []
    for _, row in speeches_df.groupby(["party", "stance"])["speech_count"].sum().reset_index().iterrows():
        if row["party"] in node_idx and row["stance"] in node_idx:
            sources.append(node_idx[row["party"]])
            targets.append(node_idx[row["stance"]])
            values.append(int(row["speech_count"]))
            link_colors.append(_hex_rgba(PARTY_COLORS.get(row["party"], "#A0A0B0")))
    for _, row in speeches_df.groupby(["stance", "primary_frame"])["speech_count"].sum().reset_index().iterrows():
        if row["stance"] in node_idx and row["primary_frame"] in node_idx:
            sources.append(node_idx[row["stance"]])
            targets.append(node_idx[row["primary_frame"]])
            values.append(int(row["speech_count"]))
            link_colors.append(_hex_rgba(STANCE_COLORS.get(row["stance"], "#A0A0B0")))

    node_colors = (
        [PARTY_COLORS.get(p, "#A0A0B0")  for p in parties_present] +
        [STANCE_COLORS.get(s, "#A0A0B0") for s in stances_present] +
        [FRAME_COLORS.get(f, "#A0A0B0")  for f in FRAMES]
    )
    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, label=all_nodes, color=node_colors,
                  line=dict(color=BG, width=0.5)),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
    ))
    fig.update_layout(
        **_plotly_layout(
            title=dict(text="Flow of CCUS Legislative Discourse: Party → Position → Argument Frame",
                       font=dict(size=15, color="white")),
            height=700, margin=dict(l=120, r=120, t=80, b=40),
        )
    )
    _write_interactive(fig, "vis_05_sankey")


def _iplot_06_faceted_frames(speeches_df: pd.DataFrame) -> None:
    """Interactive faceted bar: frame frequency by party."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    parties_with_data = [p for p in PARTIES if p in speeches_df["party"].unique()]
    ncols = 3
    nrows = (len(parties_with_data) + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=[f"<b>{p}</b>" for p in parties_with_data],
                        horizontal_spacing=0.08, vertical_spacing=0.15)

    for idx, party in enumerate(parties_with_data):
        row_idx = idx // ncols + 1
        col_idx = idx % ncols + 1
        sub    = speeches_df[speeches_df["party"] == party]
        counts = (sub.groupby("primary_frame")["speech_count"].sum()
                  .reindex(FRAMES, fill_value=0).sort_values())
        fig.add_trace(go.Bar(
            y=counts.index.tolist(), x=counts.values.tolist(),
            orientation="h",
            marker_color=[FRAME_COLORS.get(f, "#A0A0B0") for f in counts.index],
            showlegend=False,
            hovertemplate="%{y}: %{x}<extra>" + party + "</extra>",
        ), row=row_idx, col=col_idx)

        # Colour the subplot title to match party colour
        title_idx = idx
        if title_idx < len(fig.layout.annotations):
            fig.layout.annotations[title_idx].font.color = PARTY_COLORS.get(party, "#CCCCDD")

    fig.update_layout(
        **_plotly_layout(
            title=dict(text="Argument Frame Distribution by Party",
                       font=dict(size=15, color="white")),
            height=550,
        )
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    _write_interactive(fig, "vis_06_faceted_frames")


def _iplot_07_cooccurrence(speeches_df: pd.DataFrame) -> None:
    """Interactive co-occurrence heatmap."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    fi  = {f: i for i, f in enumerate(FRAMES)}
    mat = np.zeros((len(FRAMES), len(FRAMES)))
    for _, row in speeches_df.iterrows():
        fs = [f for f in (row["frames"] if isinstance(row["frames"], list) else [row["frames"]])
              if f in fi]
        for f1, f2 in itertools.combinations(fs, 2):
            i, j = fi[f1], fi[f2]
            mat[i][j] += row["speech_count"]
            mat[j][i] += row["speech_count"]
        for f in fs:
            mat[fi[f]][fi[f]] += row["speech_count"]

    # Lower triangle only
    display = np.where(np.tril(np.ones_like(mat, dtype=bool)), mat, None)

    fig = go.Figure(go.Heatmap(
        z=display.tolist(),
        x=FRAMES, y=FRAMES,
        colorscale=[[0, "#1A1A2E"], [0.5, "#7B5EA7"], [1, "#F7C59F"]],
        text=[[f"{int(v)}" if v is not None else "" for v in row] for row in display],
        texttemplate="%{text}",
        hovertemplate="<b>%{y} × %{x}</b><br>Count: %{z}<extra></extra>",
        showscale=True,
        colorbar=dict(title=dict(text="Count", font=dict(color="#CCCCDD")), tickfont=dict(color="#CCCCDD")),
    ))
    fig.update_layout(
        **_plotly_layout(
            title=dict(text="Argument Frame Co-occurrence in CCUS Legislative Debates",
                       font=dict(size=15, color="white")),
            xaxis=dict(side="bottom", tickangle=30,
                       tickfont=dict(color="#CCCCDD"), gridcolor=GRID),
            yaxis=dict(tickfont=dict(color="#CCCCDD"), gridcolor=GRID, autorange="reversed"),
            height=500,
        )
    )
    _write_interactive(fig, "vis_07_cooccurrence")


def _iplot_09_wordcloud_table(speeches_df: pd.DataFrame) -> None:
    """Interactive word-weight table by frame (word clouds can't be interactive in Plotly)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return

    ncols = 3
    nrows = 2
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=[f"<b>{f}</b>" for f in FRAMES],
                        horizontal_spacing=0.06, vertical_spacing=0.18)

    for idx, frame in enumerate(FRAMES):
        row_idx = idx // ncols + 1
        col_idx = idx % ncols + 1
        corpus  = list(speeches_df[speeches_df["primary_frame"] == frame]["text_excerpt"].dropna())
        if not corpus:
            continue
        try:
            vec = TfidfVectorizer(max_features=15, stop_words="english")
            vec.fit(corpus)
            scores = dict(zip(vec.get_feature_names_out(),
                               vec.transform(corpus).mean(axis=0).A1))
            scores = {w: v for w, v in scores.items() if v > 0}
        except Exception:
            scores = {}
        if not scores:
            continue

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:12]
        words, weights = zip(*top) if top else ([], [])
        color = FRAME_COLORS.get(frame, "#CCCCDD")
        fig.add_trace(go.Bar(
            y=list(words), x=list(weights), orientation="h",
            marker_color=color, showlegend=False,
            hovertemplate="%{y}: %{x:.3f}<extra>" + frame + "</extra>",
        ), row=row_idx, col=col_idx)
        fig.layout.annotations[idx].font.color = color

    fig.update_layout(
        **_plotly_layout(
            title=dict(text="Top Keywords by Argument Frame (TF-IDF weight)",
                       font=dict(size=15, color="white")),
            height=600,
        )
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    _write_interactive(fig, "vis_09_wordclouds")


# ─── Dashboard assembly ──────────────────────────────────────────────────────

_DASHBOARD_PLOTS = [
    ("vis_01_mp_lollipop",    "MP Speaking Frequency & Stance",       "01 MP Lollipop"),
    ("vis_02_party_stance",   "Party-Level CCUS Positioning",         "02 Party Stance"),
    ("vis_04_radar_frames",   "Frame Prevalence by Stance (Radar)",   "04 Radar Frames"),
    ("vis_05_sankey",         "Discourse Flow: Party → Stance → Frame","05 Sankey"),
    ("vis_06_faceted_frames", "Frame Distribution by Party",          "06 Faceted Frames"),
    ("vis_07_cooccurrence",   "Frame Co-occurrence Heatmap",          "07 Co-occurrence"),
    ("vis_09_wordclouds",     "Key Vocabulary by Argument Frame",     "09 Word Clouds"),
]

_DASHBOARD_CSS = """
:root {
  --bg:      #0F0F1A;
  --panel:   #1A1A2E;
  --grid:    #2A2A45;
  --text:    #CCCCDD;
  --accent:  #3EC9A7;
  --card-r:  12px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', system-ui, sans-serif;
  min-height: 100vh;
  padding: 2rem;
}
header {
  border-bottom: 1px solid var(--grid);
  padding-bottom: 1.2rem;
  margin-bottom: 2rem;
}
header h1 {
  font-size: 1.5rem;
  font-weight: 700;
  color: #fff;
  letter-spacing: -0.02em;
}
header p {
  font-size: 0.85rem;
  color: #888;
  margin-top: 0.35rem;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 1.25rem;
}
.card {
  background: var(--panel);
  border: 1px solid var(--grid);
  border-radius: var(--card-r);
  overflow: hidden;
  transition: border-color 0.2s, transform 0.2s;
}
.card:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
}
.card a {
  display: block;
  text-decoration: none;
  color: inherit;
}
.thumb-wrap {
  position: relative;
  overflow: hidden;
  height: 200px;
  background: var(--bg);
}
.thumb-wrap img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  opacity: 0.85;
  transition: opacity 0.2s;
}
.card:hover .thumb-wrap img { opacity: 1; }
.thumb-wrap .overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  background: rgba(15,15,26,0.6);
  transition: opacity 0.2s;
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--accent);
}
.card:hover .overlay { opacity: 1; }
.card-body {
  padding: 0.9rem 1rem;
}
.card-body h3 {
  font-size: 0.95rem;
  font-weight: 600;
  color: #fff;
  margin-bottom: 0.25rem;
}
.card-body p {
  font-size: 0.78rem;
  color: #888;
}
footer {
  margin-top: 3rem;
  border-top: 1px solid var(--grid);
  padding-top: 1.2rem;
  font-size: 0.78rem;
  color: #555;
  text-align: center;
}
"""


def plot_interactive_dashboard(
    speeches_df: pd.DataFrame,
    party_df: pd.DataFrame,
    frame_df: pd.DataFrame,
) -> None:
    """
    Build interactive Plotly HTML pages for plots 1, 2, 4, 5, 6, 7, 9, then
    assemble a dark/sleek dashboard landing page at ``output/vis/dashboard.html``.

    Each card shows the static PNG thumbnail (if present) linked to the
    individual interactive HTML page.
    """
    try:
        import plotly.graph_objects as go  # noqa: F401 — check plotly is installed
    except ImportError:
        print("  Skipping dashboard: plotly not installed (pip install plotly).")
        return

    print("  Building interactive HTML pages ...")
    _iplot_01_lollipop(speeches_df)
    _iplot_02_party_stance(party_df)
    _iplot_04_radar(speeches_df)
    _iplot_05_sankey(speeches_df)
    _iplot_06_faceted_frames(speeches_df)
    _iplot_07_cooccurrence(speeches_df)
    _iplot_09_wordcloud_table(speeches_df)

    # ── Build dashboard.html ─────────────────────────────────────────────────
    cards_html = []
    for slug, title, label in _DASHBOARD_PLOTS:
        interactive_file = f"{slug}_interactive.html"
        thumb_file       = f"{slug}.png"
        thumb_exists     = (OUTPUT_DIR / thumb_file).exists()

        thumb_tag = (
            f'<img src="{thumb_file}" alt="{title}" loading="lazy">'
            if thumb_exists
            else f'<div style="height:200px;display:flex;align-items:center;'
                 f'justify-content:center;color:#555;font-size:0.8rem;">'
                 f'No preview</div>'
        )
        cards_html.append(f"""
  <div class="card">
    <a href="{interactive_file}" target="_blank">
      <div class="thumb-wrap">
        {thumb_tag}
        <div class="overlay">Open interactive →</div>
      </div>
      <div class="card-body">
        <h3>{title}</h3>
        <p>{label}</p>
      </div>
    </a>
  </div>""")

    from datetime import datetime
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    dashboard = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CCUS Legislative Debate — Interactive Dashboard</title>
  <style>{_DASHBOARD_CSS}</style>
</head>
<body>
  <header>
    <h1>CCUS Legislative Debate Dashboard</h1>
    <p>Interactive visualizations of Canadian parliamentary discourse on carbon capture, utilization &amp; storage legislation. Generated {generated}.</p>
  </header>
  <main class="grid">{"".join(cards_html)}
  </main>
  <footer>OpenParliament CCUS Analysis Pipeline &nbsp;·&nbsp; Data via openparliament.ca</footer>
</body>
</html>
"""
    dash_path = OUTPUT_DIR / "dashboard.html"
    dash_path.write_text(dashboard, encoding="utf-8")
    print(f"  Saved {dash_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _run_all_plots(speeches_df, party_df, frame_df) -> None:
    print(f"\nOutput directory: {OUTPUT_DIR.resolve()}\n")

    plots = [
        ("01 MP Lollipop",           lambda: plot_01_mp_lollipop(speeches_df)),
        ("02 Party Stance",          lambda: plot_02_party_stance(party_df)),
        ("03 MP Bill Heatmap",       lambda: plot_03_mp_bill_heatmap(speeches_df)),
        ("04 Radar Frames",          lambda: plot_04_radar_frames(speeches_df)),
        ("05 Sankey",                lambda: plot_05_sankey(speeches_df)),
        ("06 Faceted Frames",        lambda: plot_06_faceted_frames(speeches_df)),
        ("07 Co-occurrence",         lambda: plot_07_cooccurrence(speeches_df)),
        ("08 Temporal",              lambda: plot_08_temporal(speeches_df)),
        ("09 Word Clouds",           lambda: plot_09_wordclouds(speeches_df)),
        ("10 Ideological Positions", lambda: plot_10_ideological_positions(speeches_df)),
    ]

    for name, fn in plots:
        print(f"Plot {name} ...")
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()

    print("\nPlot Interactive Dashboard ...")
    try:
        plot_interactive_dashboard(speeches_df, party_df, frame_df)
    except Exception as exc:
        import traceback
        print(f"  ERROR: {exc}")
        traceback.print_exc()

    print("\nDone.")


def _main_cli() -> None:
    """Entry point when run via `python -m parliament.ccus_analysis.step5_vis`."""
    # Default: load real step4 output if present, otherwise mock data.
    default_path = JSON_DIR / "step4_opinions.json"

    if "--real" in sys.argv:
        idx = sys.argv.index("--real")
        json_path = sys.argv[idx + 1]
    elif default_path.exists():
        json_path = str(default_path)
        print(f"Auto-loading step4 output from {json_path} ...")
    else:
        json_path = None

    if json_path:
        speeches_df, party_df, frame_df = load_pipeline_data(json_path)
        print(f"Loaded {len(speeches_df)} opinion records.")
    else:
        print("No step4_opinions.json found — generating mock data ...")
        speeches_df, party_df, frame_df = generate_mock_data()
        print(f"Mock data: {len(speeches_df)} records, "
              f"{speeches_df['mp_name'].nunique()} MPs.")

    _run_all_plots(speeches_df, party_df, frame_df)


if __name__ == "__main__":
    _main_cli()
