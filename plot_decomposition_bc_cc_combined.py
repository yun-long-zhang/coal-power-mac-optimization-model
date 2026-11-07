"""
4-panel combined figure: Top row – stacked-area UAC decomposition (BC & CC);
Bottom row – per-plant hexbin density scatter (BC & CC cost vs. artifact component).

Reads:  data/uac_interaction_decomp_target_summary.csv
        data/uac_interaction_decomp_plant_level.csv
Saves:  data/fig_decomp_4panel.pdf  (+.png)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm

from plot_export_utils import save_figure_bundle

# ── paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

ts = pd.read_csv(os.path.join(DATA, "uac_interaction_decomp_target_summary.csv"))
pl = pd.read_csv(os.path.join(DATA, "uac_interaction_decomp_plant_level.csv"))

# ── target-summary derived quantities ────────────────────────────────────────
ts["target_Gt"] = ts["target_abatement"] / 1e9
# Average over plants where EC actually affects the downstream technology.
_n = ts["n_affected"].where(ts["n_affected"] > 0) if "n_affected" in ts else ts["n_plants"]
ts["avg_delta_uac"] = ts["sum_delta_uac"] / _n
ts["avg_cost"] = ts["sum_cost_component"] / _n
ts["avg_abate"] = ts["sum_abatement_component"] / _n

bc_ts = ts[ts["tech"] == "BC"].sort_values("target_Gt").reset_index(drop=True)
cc_ts = ts[ts["tech"] == "CC"].sort_values("target_Gt").reset_index(drop=True)

# ── plant-level: active rows only ────────────────────────────────────────────
active = pl.dropna(subset=["delta_uac"])
active = active[active["delta_uac"].abs() > 1e-6].copy()
bc_all = active[active["tech"] == "BC"].copy()
cc_all = active[active["tech"] == "CC"].copy()

# ── colours & style ──────────────────────────────────────────────────────────
COL_COST   = "#e74c3c"   # red   – genuine cost channel
COL_ABATE  = "#3498db"   # blue  – accounting artifact
COL_TOTAL  = "#2c3e50"   # dark  – total ΔUAC
COL_BC     = "#27ae60"   # green – BC label colour
COL_CC     = "#8e44ad"   # purple– CC label colour
plt.rcParams.update({"font.size": 9})


# ═══════════════════════════════════════════════════════════════════════════════
# Helper – stacked-area panel  (from plot_uac_decomposition_full.py Fig A)
# ═══════════════════════════════════════════════════════════════════════════════
def _stacked_area_panel(ax, data, title, ylabel):
    x   = data["target_Gt"].values
    tot = data["avg_delta_uac"].values
    cst = data["avg_cost"].values
    abt = data["avg_abate"].values

    pos_cst = np.clip(cst, 0, None)
    neg_cst = np.clip(cst, None, 0)
    pos_abt = np.clip(abt, 0, None)
    neg_abt = np.clip(abt, None, 0)

    ax.stackplot(x, pos_cst, pos_abt,
                 labels=["Engineering component (genuine)", "Accounting component (artifact)"],
                 colors=[COL_COST, COL_ABATE], alpha=0.55, zorder=2)
    ax.stackplot(x, neg_cst, neg_abt,
                 colors=[COL_COST, COL_ABATE], alpha=0.55, zorder=2)
    ax.plot(x, tot, "o-", color=COL_TOTAL, lw=2, ms=3.5, zorder=4, label="Total ΔUAC")
    ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("Fleet abatement target (GtCO₂/yr)", fontsize=9)
    if ylabel:
        ax.set_ylabel("Per-plant avg. ΔUAC  (USD/tCO₂)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper – hexbin density panel  (from original plot_decomposition_bc_cc_combined.py)
# ═══════════════════════════════════════════════════════════════════════════════
def _hexbin_panel(ax, df, cmap, title, show_ylabel, fig_ref):
    if df.empty:
        # No plant with a non-zero change (e.g. BC: all cost and abatement terms scale with biomass input,
        # so the UAC of co-firing is unchanged by prior EC).
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(0.5, 0.5, "$\\Delta UAC_{BC} = 0$ at every plant:\nall BC cost terms and BC abatement\nscale with the EC-reduced biomass input",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return
    xcol = "delta_uac_cost_numerator"
    ycol = "delta_uac_abatement_denominator"
    x = df[xcol].values
    y = df[ycol].values

    hb = ax.hexbin(x, y, gridsize=55, cmap=cmap, norm=LogNorm(vmin=1),
                   linewidths=0.2, mincnt=1, zorder=3)
    cb = fig_ref.colorbar(hb, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("Number of\nplant-scenarios", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    ax.axvline(0, color="black", lw=1.0, ls="--", zorder=4)
    ax.axhline(0, color="black", lw=1.0, ls="--", zorder=4)

    fig_ref.canvas.draw()
    xlim = list(ax.get_xlim())
    ylim = list(ax.get_ylim())
    span_x = xlim[1] - xlim[0]
    span_y = ylim[1] - ylim[0]
    if xlim[0] > -span_x * 0.12: xlim[0] = -span_x * 0.12
    if xlim[1] <  span_x * 0.06: xlim[1] =  span_x * 0.06
    if ylim[0] > -span_y * 0.10: ylim[0] = -span_y * 0.10
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    span_x = xlim[1] - xlim[0]
    span_y = ylim[1] - ylim[0]

    alpha_q = 0.06
    ax.fill_betweenx([0, ylim[1]], xlim[0], 0, color="#3498db", alpha=alpha_q, zorder=1)
    ax.fill_betweenx([0, ylim[1]], 0, xlim[1], color="#9b59b6", alpha=alpha_q, zorder=1)
    ax.fill_betweenx([ylim[0], 0], xlim[0], 0, color="#2ecc71", alpha=alpha_q, zorder=1)
    ax.fill_betweenx([ylim[0], 0], 0, xlim[1], color="#e74c3c", alpha=alpha_q, zorder=1)

    def _qlabel(xf, yf, lines, color):
        for i, line in enumerate(lines):
            ax.text(xf, yf - i * 0.09, line,
                    color=color, fontsize=7.5, ha="center", va="top",
                    style="italic", clip_on=True, transform=ax.transAxes,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    xz = (0 - xlim[0]) / span_x
    yz = (0 - ylim[0]) / span_y
    px = (xz + 1.0) / 2
    mx = xz / 2
    py = (yz + 1.0) / 2
    my = yz / 2

    if (1 - xz) > 0.22 and (1 - yz) > 0.22:
        _qlabel(px, py + 0.03, ["Engineering↑ & Accounting↑", "(both inflate UAC)"],   "#7d3c98")
    if xz > 0.22 and (1 - yz) > 0.22:
        _qlabel(mx, py + 0.19, ["Engineering↓ & Accounting↑", "(accounting dominates)"], "#1a5276")
    if xz > 0.22 and yz > 0.22:
        _qlabel(mx, my - 0.04, ["Engineering↓ & Accounting↓", "(both reduce UAC)"],    "#1e8449")
    if (1 - xz) > 0.22 and yz > 0.22:
        _qlabel(px, my - 0.02, ["Engineering↑ & Accounting↓", "(engineering dominates)"],     "#922b21")

    ax.set_xlabel("Engineering component (ΔUACeng) ($/tCO₂)", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Accounting component (ΔUACacc) ($/tCO₂)", fontsize=9)

    main_title, subtitle = title.split("\n", 1)
    ax.set_title(main_title, fontsize=10, fontweight="bold", pad=25)
    ax.text(0.5, 1.05, subtitle,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, style="italic", color="#333333")
    ax.spines[["top", "right"]].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════════
# 4-panel layout: 2 rows × 2 columns
#   (a) BC stacked-area    | (b) CC stacked-area
#   (c) BC hexbin scatter  | (d) CC hexbin scatter
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(12, 9.5))
fig.subplots_adjust(wspace=0.08, hspace=0.38, bottom=0.13, top=0.90, left=0.07, right=0.96)

# ── Top row: stacked-area decomposition ──────────────────────────────────────
ax_a = fig.add_subplot(2, 2, 1)
ax_b = fig.add_subplot(2, 2, 2)

_stacked_area_panel(ax_a, bc_ts, "(a)  Biomass co-firing (BC)", ylabel=True)
_stacked_area_panel(ax_b, cc_ts, "(b)  Carbon capture (CC)",   ylabel=False)

# Shared legend for top row
legend_handles = [
    mpatches.Patch(color=COL_COST,  alpha=0.7, label="Engineering component (ΔUACeng)"),
    mpatches.Patch(color=COL_ABATE, alpha=0.7, label="Accounting component (ΔUACacc)"),
    plt.Line2D([0],[0], color=COL_TOTAL, marker="o", ms=5, lw=2, label="Total ΔUAC"),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, 0.54))

# ── Bottom row: hexbin scatter ──────────────────────────────────────────────
ax_c = fig.add_subplot(2, 2, 3)
ax_d = fig.add_subplot(2, 2, 4)

_hexbin_panel(ax_c, bc_all, "Greens",
              f"(c)  Biomass co-firing (BC) \n"
              "",
              show_ylabel=True, fig_ref=fig)

_hexbin_panel(ax_d, cc_all, "Purples",
              f"(d)  Carbon capture (CC) \n"
              "",
              show_ylabel=False, fig_ref=fig)

# ── save ─────────────────────────────────────────────────────────────────────
out_png = os.path.join(DATA, "fig_decomp_4panel.png")
outputs = save_figure_bundle(
    fig,
    out_png,
    dpi=300,
    bbox_inches="tight",
    data_frames={
        "target_summary": ts,
        "plant_level": pl,
        "bc_target_series": bc_ts,
        "cc_target_series": cc_ts,
    },
)
print(f"Saved: {outputs['png']}")
print(f"Saved: {outputs['pdf']}")
print(f"Saved: {outputs['xlsx']}")

