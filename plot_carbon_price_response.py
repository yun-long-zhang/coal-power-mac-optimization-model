"""
plot_carbon_price_response.py
Publication-quality figure for Reviewer Comment 8:
Optimal abatement A*(p) as a step function of the carbon price,
with phase annotations and technology composition panel.

Outputs:
  data/fig_carbon_price_response.pdf
  data/fig_carbon_price_response.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from plot_export_utils import save_figure_bundle

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")

CURVE_CSV        = os.path.join(DATA_DIR, "gurobi_original_curve.csv")
INTERVALS_CSV    = os.path.join(DATA_DIR, "carbon_price_response_main_step_intervals.csv")
RESPONSE_CSV     = os.path.join(DATA_DIR, "carbon_price_response_main_response_grid.csv")
PLANT_LEVEL_CSV  = os.path.join(DATA_DIR, "uac_interaction_decomp_plant_level.csv")

OUT_PNG  = os.path.join(DATA_DIR, "fig_carbon_price_response.png")

# ── constants ─────────────────────────────────────────────────────────────────
MAX_ABATEMENT_GT = 4.7          # fleet maximum (GtCO₂)
TOTAL_PLANTS     = 1885

# Reference carbon prices
# Net-zero target: 3.6 GtCO₂ is first triggered at p = $56.04/tCO₂
#   (from step_intervals row 26: p_low=56.036, A_star=3.60 GT)
REF_PRICES = [
    (9,   "China ETS 2025\n($9/tCO\u2082)",     "#c0392b", "-"),
    (56,  "Net-zero target\n(3.5 GtCO\u2082)",    "#27ae60", "-."),
    (83,   "EU ETS 2025\n($83/tCO\u2082)",  "#e67e22", "--"),
]

# Phase boundaries (from curve data analysis)
BC_ONSET_PRICE  = 2.73    # $/tCO₂  — first step in intervals (BC starts)
CC_ONSET_PRICE  = 28.66   # $/tCO₂  — CC first appears (5 plants)
PRICE_XMAX      = 130.0   # x-axis right limit


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_data():
    """Return (response_grid, step_intervals, curve, bc_counts) DataFrames."""
    response  = pd.read_csv(RESPONSE_CSV)
    intervals = pd.read_csv(INTERVALS_CSV)
    curve     = pd.read_csv(CURVE_CSV)
    curve = curve.sort_values("achieved_abatement").reset_index(drop=True)
    # Derive BC plant counts from plant-level decomposition data
    pl = pd.read_csv(PLANT_LEVEL_CSV)
    bc_counts = (
        pl[(pl["tech"] == "BC") & (pl["blend_ratio"] > 0)]
        .groupby("target_id")["plant_id"].count()
        .reset_index()
        .rename(columns={"plant_id": "bc_plants"})
    )
    curve = curve.merge(bc_counts, on="target_id", how="left")
    curve["bc_plants"] = curve["bc_plants"].fillna(0).astype(int)
    return response, intervals, curve


def _tech_at_price(carbon_price: float, intervals: pd.DataFrame, curve: pd.DataFrame) -> dict:
    """Return technology composition for a given carbon price."""
    # Find the matching row in intervals
    mask = (intervals["p_low"] <= carbon_price) & (intervals["p_high"] > carbon_price)
    if not mask.any():
        # edge: use last interval
        row = intervals.iloc[-1]
    else:
        row = intervals[mask].iloc[0]
    # Look up curve row by A_star abatement value (robust to frontier indexing offsets)
    a_star = float(row["A_star"])
    idx = (curve["achieved_abatement"] - a_star).abs().idxmin()
    c = curve.iloc[idx]
    return {
        "abatement_gt": float(c["achieved_abatement"]) / 1e9,
        "ec_plants":    int(round(float(c["ec_adoption_sum"]))),
        "bc_plants":    int(c["bc_plants"]),
        "cc_plants":    int(round(float(c["cc_adoption_sum"]))),
    }


# ── main figure ───────────────────────────────────────────────────────────────

def make_figure():
    response, intervals, curve = _load_data()

    # ── filter response grid to [0, PRICE_XMAX]
    resp = response[(response["carbon_price"] >= 0) &
                    (response["carbon_price"] <= PRICE_XMAX)].copy()
    px = resp["carbon_price"].to_numpy()
    ay = resp["A_star"].to_numpy() / 1e9   # GtCO₂

    # ── layout: 2 rows, ratio 3:1.1
    fig = plt.figure(figsize=(10.5, 8.5), dpi=260)
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1.15], hspace=0.30,
                           left=0.09, right=0.94, top=0.93, bottom=0.09)
    ax_main  = fig.add_subplot(gs[0])
    ax_tech  = fig.add_subplot(gs[1])

    # ═══════════════════════════════════════════════════════════════════════════
    # Panel (a): Step function A*(p)
    # ═══════════════════════════════════════════════════════════════════════════
    ymin, ymax = 0.0, 5.0

    # Phase shading (background, no text labels inside)
    ax_main.axvspan(0, CC_ONSET_PRICE, ymin=0, ymax=1, color="#fff3cd",
                    alpha=0.85, zorder=0)
    ax_main.axvspan(CC_ONSET_PRICE, PRICE_XMAX, ymin=0, ymax=1, color="#d6eaf8",
                    alpha=0.85, zorder=0)

    # Zero-price baseline: A*(0)
    baseline_gt = float(intervals.iloc[0]["A_star"]) / 1e9
    ax_main.axhline(baseline_gt, color="#6c757d", linewidth=1.1, linestyle=":",
                    zorder=1, label=f"Zero-price baseline ({baseline_gt:.1f} GtCO\u2082)")

    # Step function
    ax_main.step(px, ay, where="post", color="#1a5276", linewidth=2.4,
                 zorder=3, label="Optimal abatement $A^*(p)$")

    # BC onset marker
    ax_main.axvline(BC_ONSET_PRICE, color="#7d6608", linewidth=1.0,
                    linestyle="--", alpha=0.7, zorder=2)
    ax_main.text(BC_ONSET_PRICE + 0.8, ymin + 0.18,
                 f"BC onset\n\${BC_ONSET_PRICE:.1f}/tCO\u2082",
                 fontsize=7.2, color="#7d6608", va="bottom", ha="left",
                 style="italic", zorder=5)

    # CC onset marker
    ax_main.axvline(CC_ONSET_PRICE, color="#154360", linewidth=1.2,
                    linestyle="-", alpha=0.5, zorder=2)
    ax_main.text(CC_ONSET_PRICE + 0.8, ymin + 0.18,
                 f"CC onset\n\${CC_ONSET_PRICE:.1f}/tCO\u2082",
                 fontsize=7.2, color="#154360", va="bottom", ha="left",
                 style="italic", zorder=5)

    # Reference price lines + annotations
    ann_y_offsets = {14.0: 0.0, 16.5: 0.42, 56.04: 0.0}
    for p_ref, label, color, ls in REF_PRICES:
        t = _tech_at_price(p_ref, intervals, curve)
        a_gt = t["abatement_gt"]
        pct  = a_gt / MAX_ABATEMENT_GT * 100
        ax_main.axvline(p_ref, color=color, linewidth=1.6, linestyle=ls, zorder=4)
        ax_main.scatter([p_ref], [a_gt], s=55, color=color, zorder=6, clip_on=False)
        x_offset = 2.2 if p_ref < 30 else 2.5
        y_extra  = ann_y_offsets.get(p_ref, 0.0)
        ax_main.annotate(
            f"$p$={p_ref:.0f}: $A^*$={a_gt:.2f} Gt\n({pct:.0f}% of fleet max)",
            xy=(p_ref, a_gt),
            xytext=(p_ref + x_offset, a_gt + 0.12 + y_extra),
            fontsize=7.0,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.9,
                            connectionstyle="arc3,rad=-0.1"),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color,
                      alpha=0.85, lw=0.8),
            zorder=7,
        )

    # Right y-axis: % of max
    ax_r = ax_main.twinx()
    ax_r.set_ylim(ymin, ymax)
    ax_r.set_yticks([1, 2, 3, 4, MAX_ABATEMENT_GT])
    ax_r.set_yticklabels([f"{v/MAX_ABATEMENT_GT*100:.0f}%"
                          for v in [1, 2, 3, 4, MAX_ABATEMENT_GT]], fontsize=8)
    ax_r.set_ylabel("Share of fleet maximum abatement (%)", fontsize=8.5, labelpad=8)
    ax_r.tick_params(axis="y", length=3)

    # Fleet-max horizontal dotted line
    ax_main.axhline(MAX_ABATEMENT_GT, color="#922b21", linewidth=0.9,
                    linestyle=":", alpha=0.6, zorder=1)
    ax_main.text(PRICE_XMAX - 1, MAX_ABATEMENT_GT + 0.05,
                 f"Fleet max = {MAX_ABATEMENT_GT} Gt",
                 ha="right", va="bottom", fontsize=7.5, color="#922b21")

    # Legend
    legend_handles = [
        mpatches.Patch(color="#fff3cd", label="No CC deployed (EC + BC only)", linewidth=0),
        mpatches.Patch(color="#d6eaf8", label="CC deployed (EC + BC + CC)", linewidth=0),
        Line2D([0], [0], color="#1a5276", lw=2.4, label="Optimal abatement $A^*(p)$"),
        Line2D([0], [0], color="#6c757d", lw=1.1, ls=":",
               label=f"Zero-price baseline ({baseline_gt:.1f} GtCO\u2082)"),
        Line2D([0], [0], color="#c0392b", lw=1.5, ls="-",
               label="China ETS 2025 (\$9/tCO\u2082)"),
        Line2D([0], [0], color="#e67e22", lw=1.5, ls="--",
               label="EU ETS 2025 (\$83/tCO\u2082)"),
        Line2D([0], [0], color="#27ae60", lw=1.5, ls="-.",
               label="Net-zero target: 3.6 GtCO\u2082 (\$56/tCO\u2082)"),
    ]
    ax_main.legend(handles=legend_handles, loc="lower right", fontsize=7.4,
                   frameon=True, framealpha=0.92, edgecolor="#cccccc",
                   ncol=2, handlelength=1.8)

    ax_main.set_xlim(0, PRICE_XMAX)
    ax_main.set_ylim(ymin, ymax)
    ax_main.set_xlabel("Carbon price ($/tCO\u2082)", fontsize=10)
    ax_main.set_ylabel("Optimal abatement $A^*(p)$  (GtCO\u2082)", fontsize=10)
    ax_main.set_title(
        "(a)  Optimal fleet abatement as a step function of the carbon price",
        fontsize=10.5, fontweight="bold", pad=10
    )
    ax_main.grid(alpha=0.22, linestyle="--", zorder=0)
    ax_main.tick_params(axis="both", labelsize=8.5)

    # ═══════════════════════════════════════════════════════════════════════════
    # Panel (b): Technology deployment at selected carbon price levels
    # Grouped bars: EC (orange) + BC (green) + CC (blue) — all in plant counts
    # ═══════════════════════════════════════════════════════════════════════════
    key_prices_b = [0.0, 9.0, 28.66, 56.04, 83.0, 124.5]
    bar_labels_b = [
        "$0", "$9",
        "$28.7\n(CC onset)",
        "$56\n(net-zero)",
        "$83\n(EU ETS 2025)",
        "$125\n(fleet max)",
    ]

    tech_rows_b = []
    for p in key_prices_b:
        t = _tech_at_price(p, intervals, curve)
        t["price"] = p
        tech_rows_b.append(t)
    tech_df_b = pd.DataFrame(tech_rows_b)

    n_b   = len(key_prices_b)
    x_b   = np.arange(n_b)
    bw    = 0.26   # width of each sub-bar

    ax_tech.bar(x_b - bw, tech_df_b["ec_plants"], bw,
                label="EC-retrofitted plants", color="#f39c12",
                edgecolor="white", linewidth=0.4)
    ax_tech.bar(x_b,      tech_df_b["bc_plants"], bw,
                label="BC co-firing plants",  color="#2ecc71",
                edgecolor="white", linewidth=0.4)
    ax_tech.bar(x_b + bw, tech_df_b["cc_plants"], bw,
                label="CC-equipped plants",   color="#2980b9",
                edgecolor="white", linewidth=0.4)

    # Total fleet reference
    ax_tech.axhline(TOTAL_PLANTS, color="#922b21", linewidth=0.8,
                    linestyle=":", alpha=0.7)
    ax_tech.text(n_b - 0.55, TOTAL_PLANTS + 18,
                 f"Total fleet: {TOTAL_PLANTS}",
                 ha="right", va="bottom", fontsize=6.5, color="#922b21")

    ax_tech.set_ylim(0, 2280)
    ax_tech.set_ylabel("Number of plants", fontsize=9)
    ax_tech.tick_params(axis="y", labelsize=7.5)

    # One-row legend centred below the bar chart
    handles_b = [
        mpatches.Patch(color="#f39c12", label="EC-retrofitted plants"),
        mpatches.Patch(color="#2ecc71", label="BC co-firing plants"),
        mpatches.Patch(color="#2980b9", label="CC-equipped plants"),
        Line2D([0], [0], color="#922b21", lw=0.8, ls=":",
               label=f"Total fleet ({TOTAL_PLANTS})"),
    ]
    ax_tech.legend(handles=handles_b, fontsize=7.0, loc="upper left",
                   frameon=True, framealpha=0.92, edgecolor="#cccccc",
                   ncol=4, handlelength=1.4, columnspacing=0.9)

    ax_tech.set_xticks(x_b)
    ax_tech.set_xticklabels(bar_labels_b, fontsize=7.8)
    ax_tech.set_xlabel("Carbon price ($/tCO\u2082)", fontsize=9.5)
    ax_tech.set_title(
        "(b)  Technology deployment at selected carbon price levels",
        fontsize=9.5, fontweight="bold"
    )
    ax_tech.grid(axis="y", alpha=0.2, linestyle="--")

    # ── save
    outputs = save_figure_bundle(fig, OUT_PNG, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outputs['png']}")
    print(f"Saved: {outputs['pdf']}")
    print(f"Saved: {outputs['xlsx']}")


if __name__ == "__main__":
    make_figure()
