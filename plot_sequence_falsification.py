"""
plot_sequence_falsification.py
──────────────────────────────────────────────────────────────────────────────
Publication-quality 2-panel figure for the sequence falsification test.

Panel (a): Pareto frontiers (TAC vs. achieved abatement) for all five policies.
Panel (b): Per-target optimality gap (%) relative to unconstrained for each
           constrained policy.

Outputs:
    data/fig_sequence_falsification.pdf
    data/fig_sequence_falsification.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from plot_export_utils import save_figure_bundle

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 260,
    }
)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
CURVES_CSV = os.path.join(DATA_DIR, "sequence_falsification_all_policies_curve.csv")
DETAIL_CSV = os.path.join(DATA_DIR, "sequence_falsification_optimality_gap_detail.csv")

# ── Style ─────────────────────────────────────────────────────────────────────
POLICY_META = {
    "unconstrained": {
        "label": "Unconstrained (global optimum)",
        "color": "#2c3e50",
        "lw": 2.2,
        "ls": "-",
        "zorder": 10,
    },
    "ec_bc_cc": {
        "label": "EC→BC→CC",
        "color": "#27ae60",
        "lw": 1.8,
        "ls": "--",
        "zorder": 9,
    },
    "ec_cc_bc": {
        "label": "EC→CC→BC",
        "color": "#e67e22",
        "lw": 1.3,
        "ls": "-.",
        "zorder": 7,
    },
    "bc_cc": {
        "label": "BC→CC  (no EC)",
        "color": "#e74c3c",
        "lw": 1.3,
        "ls": ":",
        "zorder": 6,
    },
    "cc_bc": {
        "label": "CC→BC  (no EC)",
        "color": "#8e44ad",
        "lw": 1.3,
        "ls": (0, (3, 1, 1, 1)),  # dash-dot-dot
        "zorder": 5,
    },
}

# Policy order for display (reference first, model ordering second, others after)
POLICY_ORDER = ["unconstrained", "ec_bc_cc", "ec_cc_bc", "bc_cc", "cc_bc"]
GAP_POLICIES  = ["ec_bc_cc", "ec_cc_bc", "bc_cc", "cc_bc"]   # Panel (b)

# ── Shading thresholds ────────────────────────────────────────────────────────
NET_ZERO_GT      = 3.56  # GtCO₂ — baseline fleet emissions (net-zero abatement)
FLEET_MAX_GT     = 4.7   # GtCO₂ — full fleet maximum


def _load_data():
    curves = pd.read_csv(CURVES_CSV)
    detail = pd.read_csv(DETAIL_CSV)
    return curves, detail


def _make_figure(curves: pd.DataFrame, detail: pd.DataFrame):
    fig = plt.figure(figsize=(10.5, 8.5), dpi=260)
    gs  = fig.add_gridspec(
        2, 1,
        height_ratios=[1.55, 1],
        hspace=0.38,
        left=0.09, right=0.97, top=0.94, bottom=0.09,
    )
    ax_curve = fig.add_subplot(gs[0])
    ax_gap   = fig.add_subplot(gs[1], sharex=ax_curve)

    X_LIM = (-0.05, 4.85)  # common x range for both panels

    # ══════════════════════════════════════════════════════════════════════════
    # Panel (a): Pareto frontiers
    # ══════════════════════════════════════════════════════════════════════════
    for policy in POLICY_ORDER:
        meta = POLICY_META[policy]
        sub  = curves[
            (curves["policy"] == policy) & curves["status"].isin(["Optimal", "TimeLimit"]) & curves["total_cost"].notna()
        ].sort_values("achieved_abatement")

        ax_curve.plot(
            sub["achieved_abatement"] / 1e9,
            sub["total_cost"] / 1e9,
            color=meta["color"],
            linewidth=meta["lw"],
            linestyle=meta["ls"],
            zorder=meta["zorder"],
            label=meta["label"],
        )

    # Reference lines
    ax_curve.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.35)
    ax_curve.axvline(NET_ZERO_GT, color="#27ae60", linewidth=0.8, linestyle=":",
                     alpha=0.7, zorder=3)
    ax_curve.text(NET_ZERO_GT + 0.04, ax_curve.get_ylim()[1] * 0.95,
                  f"Net-zero\n{NET_ZERO_GT} Gt", fontsize=7.5, color="#27ae60",
                  va="top", ha="left")

    ax_curve.set_xlim(*X_LIM)
    ax_curve.set_xlabel("Achieved annual abatement (GtCO₂)", fontsize=10)
    ax_curve.set_ylabel("Total annual cost (billion USD)", fontsize=10)
    ax_curve.tick_params(labelsize=8.5)
    ax_curve.set_title(
        "(a)  Pareto frontiers under five technology-ordering policies",
        fontsize=10, fontweight="bold",
    )

    # Legend — two columns, inside top-left
    handles_a = [
        Line2D([0], [0], color=POLICY_META[p]["color"],
               linewidth=POLICY_META[p]["lw"],
               linestyle=POLICY_META[p]["ls"],
               label=POLICY_META[p]["label"])
        for p in POLICY_ORDER
    ]
    ax_curve.legend(handles=handles_a, fontsize=8.0, loc="upper left",
                    frameon=True, framealpha=0.92, edgecolor="#cccccc",
                    ncol=1, handlelength=2.0)
    ax_curve.grid(axis="y", alpha=0.15, linestyle="--")

    # ══════════════════════════════════════════════════════════════════════════
    # Panel (b): Absolute optimality gap (billion USD) profile
    # Use absolute gap to avoid denominator inflation when reference cost ≈ 0
    # ══════════════════════════════════════════════════════════════════════════

    # Shade the "ec_bc_cc nearly zero" region (±1 billion USD band)
    ax_gap.axhspan(-1, 1, color="#d5f5e3", alpha=0.45, zorder=0)

    for policy in GAP_POLICIES:
        meta = POLICY_META[policy]
        sub  = detail[
            (detail["policy"] == policy) & detail["both_optimal"]
        ].sort_values("target_abatement").copy()
        sub["gap_abs_bn"] = sub["gap_abs_vs_ref"] / 1e9   # → billion USD

        ax_gap.plot(
            sub["target_abatement"] / 1e9,
            sub["gap_abs_bn"],
            color=meta["color"],
            linewidth=meta["lw"],
            linestyle=meta["ls"],
            zorder=meta["zorder"],
            label=meta["label"],
            marker=".",
            markersize=3.5,
        )

    # Zero reference
    ax_gap.axhline(0, color="black", linewidth=0.8, alpha=0.5, zorder=11)

    # Net-zero reference
    ax_gap.axvline(NET_ZERO_GT, color="#27ae60", linewidth=0.8, linestyle=":",
                   alpha=0.7, zorder=3)
    ax_gap.text(NET_ZERO_GT + 0.05, ax_gap.get_ylim()[1] * 0.95,
                f"Net-zero\n{NET_ZERO_GT} Gt", fontsize=7.5, color="#27ae60",
                va="top", ha="left")

    ax_gap.set_xlim(*X_LIM)
    ax_gap.set_xlabel("Annual abatement (GtCO₂)", fontsize=10)
    ax_gap.set_ylabel("Cost gap vs. unconstrained (billion USD)", fontsize=9.5)
    ax_gap.tick_params(labelsize=8.5)
    ax_gap.set_title(
        "(b)  Per-target cost gap relative to unconstrained global optimum",
        fontsize=10, fontweight="bold",
    )

    # Annotate the EC->BC->CC gap from the data: near zero up to deep targets, then the largest gap.
    seq = detail[(detail["policy"] == "ec_bc_cc") & detail["both_optimal"]].copy()
    if not seq.empty:
        k = seq["gap_abs_vs_ref"].idxmax()
        x_max, g_max = float(seq.loc[k, "target_abatement"]) / 1e9, float(seq.loc[k, "gap_abs_vs_ref"]) / 1e9
        near = seq[seq["gap_abs_vs_ref"].abs() <= 1e8]
        x_near = float(near["target_abatement"].max()) / 1e9 if not near.empty else 0.0
        ax_gap.annotate(
            f"EC→BC→CC: gap < 0.1 B USD up to {x_near:.1f} Gt;\nmax {g_max:.1f} B USD at {x_max:.1f} Gt",
            xy=(x_max, g_max), xycoords="data",
            xytext=(1.0, 12),
            fontsize=7.5, color="#27ae60",
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=0.8),
            ha="left",
        )

    handles_b = (
        [mpatches.Patch(color="#d5f5e3", alpha=0.8, label="±1 B USD band")]
        + [
            Line2D([0], [0], color=POLICY_META[p]["color"],
                   linewidth=POLICY_META[p]["lw"],
                   linestyle=POLICY_META[p]["ls"],
                   marker=".", markersize=4,
                   label=POLICY_META[p]["label"])
            for p in GAP_POLICIES
        ]
    )
    ax_gap.legend(handles=handles_b, fontsize=7.8, loc="upper left",
                  frameon=True, framealpha=0.92, edgecolor="#cccccc",
                  ncol=1, handlelength=2.0)
    ax_gap.grid(axis="y", alpha=0.15, linestyle="--")

    return fig


def main():
    curves, detail = _load_data()
    fig = _make_figure(curves, detail)

    out = os.path.join(DATA_DIR, "fig_sequence_falsification.png")
    outputs = save_figure_bundle(
        fig,
        out,
        dpi=260,
        bbox_inches="tight",
        data_frames={"curves": curves, "detail": detail},
    )
    print(f"Saved: {outputs['png']}")
    print(f"Saved: {outputs['pdf']}")
    print(f"Saved: {outputs['xlsx']}")

    plt.close(fig)


if __name__ == "__main__":
    main()
