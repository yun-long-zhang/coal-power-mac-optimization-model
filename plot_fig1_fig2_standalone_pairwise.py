"""Fig. 1 (standalone unit abatement costs vs. RRH) and Fig. 2 (pairwise interactions) from
data/standalone_pairwise_r2.csv (analysis_standalone_pairwise.py)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_export_utils import save_figure_bundle

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IN_CSV = os.path.join(DATA_DIR, "standalone_pairwise_r2.csv")
OUT_FIG1 = os.path.join(DATA_DIR, "fig1_standalone_uac_r2.png")
OUT_FIG2 = os.path.join(DATA_DIR, "fig2_pairwise_interactions_r2.png")

TECHS = [
    ("CC90_uac", "CC (90% capture)", "#7FD3E0"),
    ("GB10_uac", "GB (10% co-firing)", "#F26B1D"),
    ("GB20_uac", "GB (20% co-firing)", "#B79BC9"),
    ("DB10_uac", "DB (10% co-firing)", "#F2E86B"),
    ("DB20_uac", "DB (20% co-firing)", "#8FB2F0"),
    ("EC10_uac", "EC (10% abatement)", "#F59BE8"),
    ("EC20_uac", "EC (20% abatement)", "#9CE62E"),
]
BC_CC = [
    ("DB10", "DB (10%)", "#3B6FB6"),
    ("DB20", "DB (20%)", "#3FA55B"),
    ("GB10", "GB (10%)", "#C0392B"),
    ("GB20", "GB (20%)", "#8061B5"),
    ("CC90", "CC (90%)", "#C9B062"),
]


def _box(ax, data, positions, colors, width=0.6, whis=(0, 100), showcaps=False):
    bp = ax.boxplot(data, positions=positions, widths=width, whis=whis, showfliers=False, patch_artist=True,
                    showcaps=showcaps, medianprops={"color": "white", "lw": 1.6})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor(c)
    for k in ("whiskers", "caps"):
        for i, line in enumerate(bp.get(k, [])):
            line.set_color(colors[i // 2])
    ax.scatter(positions, [np.mean(d) for d in data], marker="x", color="black", zorder=5, s=30)


def _box_key(ax, vertical=True, color="#9E9E9E"):
    """Schematic key of the box plots: whiskers min-max, box 25th-75th percentile, median line, mean marker."""
    lo, q1, med, q3, hi, mean = 0.0, 0.32, 0.5, 0.68, 1.0, 0.58
    labels = [(hi, "Maximum"), (q3, "75th percentile"), (mean, "Mean (×)"), (med, "Median"), (q1, "25th percentile"), (lo, "Minimum")]
    if vertical:
        ax.add_patch(plt.Rectangle((-0.15, q1), 0.3, q3 - q1, color=color, lw=0))
        ax.plot([-0.15, 0.15], [med, med], color="white", lw=1.6)
        ax.plot([0, 0], [lo, q1], color=color, lw=1.0)
        ax.plot([0, 0], [q3, hi], color=color, lw=1.0)
        ax.scatter([0], [mean], marker="x", color="black", s=30, zorder=5)
        for y, t in labels:
            ax.annotate(t, xy=(0.17 if lo < y < hi else 0.02, y), xytext=(0.32, y), fontsize=8, va="center",
                        arrowprops={"arrowstyle": "-", "color": "grey", "lw": 0.5})
        ax.set_xlim(-0.3, 1.6)
        ax.set_ylim(-0.08, 1.08)
    else:
        ax.add_patch(plt.Rectangle((q1, -0.15), q3 - q1, 0.3, color=color, lw=0))
        ax.plot([med, med], [-0.15, 0.15], color="white", lw=1.6)
        ax.plot([lo, q1], [0, 0], color=color, lw=1.0)
        ax.plot([q3, hi], [0, 0], color=color, lw=1.0)
        ax.scatter([mean], [0], marker="x", color="black", s=30, zorder=5)
        for k, (x, t) in enumerate(sorted(labels)):
            up = k % 2 == 0
            ax.annotate(t, xy=(x, 0.17 if up else -0.17), xytext=(x, 0.62 if up else -0.62), fontsize=7.5,
                        ha="center", va="center", arrowprops={"arrowstyle": "-", "color": "grey", "lw": 0.5})
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(-0.9, 0.9)
    ax.axis("off")


def fig1(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.2)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1], sharey=ax)

    for col, lab, c in TECHS:
        sub = df[[col, "RRH_uac"]].dropna()
        ax.scatter(sub["RRH_uac"], sub[col], s=10, facecolors="none", edgecolors=c, linewidths=0.6, alpha=0.8, label=lab)
    rrh = df["RRH_uac"].dropna()
    xs = np.linspace(0, rrh.max() * 1.02, 10)
    ax.plot(xs, xs, color="black", lw=1.5)
    ax.text(xs[-1] * 0.92, xs[-1] * 0.92 + 8, "x = y", fontsize=10)
    ax.axhline(0, color="grey", lw=0.6)

    # distribution of the RRH UAC (x-axis variable) as a horizontal box inside panel a, above the scatter
    ymax = max(float(df[[c for c, _, _ in TECHS]].max().max()), 0.0)
    y_rrh, h_rrh = ymax + 22, 14
    rrh_col = "#B8B68F"
    ax.boxplot([rrh], positions=[y_rrh], vert=False, whis=(0, 100), showfliers=False, showcaps=False, widths=h_rrh,
               patch_artist=True, manage_ticks=False, boxprops={"facecolor": rrh_col, "edgecolor": rrh_col},
               medianprops={"color": "white", "lw": 1.6}, whiskerprops={"color": rrh_col})
    ax.scatter([rrh.mean()], [y_rrh], marker="x", color="black", zorder=5, s=30)
    ax.scatter([], [], marker="s", s=40, color=rrh_col, label="RRH")
    ax.set_ylim(top=y_rrh + h_rrh)
    ax.set_xlabel("Unit abatement cost of RRH (USD t CO$_2^{-1}$)")
    ax.set_ylabel("Unit abatement cost of retrofit (USD t CO$_2^{-1}$)")
    # two rows, filled row by row (matplotlib fills legends column by column)
    h, l = ax.get_legend_handles_labels()
    h, l = h[-1:] + h[:-1], l[-1:] + l[:-1]  # RRH first
    ncol = (len(h) + 1) // 2
    order = [k for c in range(ncol) for k in (c, c + ncol) if k < len(h)]
    ax.legend([h[k] for k in order], [l[k] for k in order], fontsize=8, ncol=ncol, loc="upper left",
              bbox_to_anchor=(-0.02, -0.12), frameon=False, columnspacing=0.8, handletextpad=0.2, borderaxespad=0, markerscale=1.4)

    data = [df[c].dropna().values for c, _, _ in TECHS]
    _box(axb, data, list(range(len(TECHS))), [c for _, _, c in TECHS])
    axb.set_xticks(range(len(TECHS)))
    axb.set_xticklabels([lab for _, lab, _ in TECHS], rotation=60, ha="right", fontsize=8)
    axb.axhline(0, color="grey", lw=0.6)
    plt.setp(axb.get_yticklabels(), visible=False)
    _box_key(axb.inset_axes([0.6, 0.55, 0.38, 0.42]))
    for a_, t in ((ax, "a"), (axb, "b")):
        a_.text(0.01, 0.995, t, transform=a_.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")
    save_figure_bundle(fig, OUT_FIG1, dpi=300, bbox_inches="tight", data_frames={"plant_uac": df})
    plt.close(fig)


def _unit_gen_cost(df: pd.DataFrame, key: str, lvl) -> pd.Series:
    """Unit generation cost of BC or CC (USD/MWh), without EC (lvl=None) or with the EC{lvl} package.
    BC: investment, O&M, feedstock and transport per MWh of biomass-based electricity (avoided coal cost not
    deducted); CC: capture, transport and storage per MWh of plant generation."""
    if key.startswith("CC"):
        return (df["CC90_cost"] + (0.0 if lvl is None else (df[f"CC90_EC{lvl}_dgen_combo"] - df[f"EC{lvl}_dgen_alone"]) * df["G_mwh"])) / df["G_mwh"]
    col = f"{key}_biopower_cost_per_mwh" if lvl is None else f"{key}_EC{lvl}_biopower_cost_per_mwh"
    return df[col]


def fig2(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 15))
    for row, lvl in enumerate(("10", "20")):
        ax, axb = axes[row]
        # interaction effect on cost: change in the unit generation cost of BC or CC (investment, O&M, feedstock,
        # transport; capture, transport and storage for CC; avoided coal cost not deducted) when EC is added,
        # relative to the same technology without EC
        data, labels, colors = [], [], []
        for key, lab, c in BC_CC:
            d = (_unit_gen_cost(df, key, lvl) / _unit_gen_cost(df, key, None) - 1.0).dropna() * 100.0
            data.append(d.values)
            labels.append(lab)
            colors.append(c)
        _box(ax, data, list(range(len(data))), colors)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_ylabel(f"Change in unit generation cost of BC or CC\nwith EC ({lvl}%) (%)")
        ax.set_title(f"({'a' if lvl == '10' else 'c'}) Unit generation cost of BC or CC: change due to EC {lvl}%", loc="left", fontsize=11)

        data, labels, colors = [], [], []
        for key, lab, c in BC_CC:
            d = (df[f"{key}_EC{lvl}_uac"] - df[f"{key}_uac"]).dropna()
            data.append(d.values)
            labels.append(lab)
            colors.append(c)
        _box(axb, data, list(range(len(data))), colors)
        axb.set_xticks(range(len(data)))
        axb.set_xticklabels(labels, fontsize=9)
        axb.axhline(0, color="grey", lw=0.6)
        axb.set_ylabel(f"$\\Delta$ unit abatement cost with EC ({lvl}%)\n(USD t CO$_2^{{-1}}$)")
        axb.set_title(f"({'b' if lvl == '10' else 'd'}) Unit abatement cost: change due to EC {lvl}%", loc="left", fontsize=11)

    ax = axes[2, 0]
    data = [(df["CC90_uac"] - df[f"CC_DB{t}_uac"]).dropna().values for t in ("10", "20")]
    _box(ax, data, [0, 1], ["#F7C68F", "#9DCE8A"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BC (10% co-firing)", "BC (20% co-firing)"])
    ax.set_ylabel("Reduction in unit abatement cost of CC\n(USD t CO$_2^{-1}$)")
    ax.set_title("(e) BC–CC: reduction in the unit abatement cost of CC", loc="left", fontsize=11)
    _box_key(ax.inset_axes([0.02, 0.5, 0.36, 0.46]))

    ax = axes[2, 1]
    data, labels, colors = [], [], []
    for lvl in ("20", "10"):
        for key, lab, c in BC_CC[:4]:
            d = (df[f"{key}_biopower_cost_per_mwh"] - df[f"{key}_EC{lvl}_biopower_cost_per_mwh"]).dropna()
            data.append(d.values)
            labels.append(f"EC {lvl}% + {lab}")
            colors.append(c)
    _box(ax, data, list(range(len(data))), colors)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Reduction in unit cost of biomass-based electricity\n(USD MWh$^{-1}$)")
    ax.set_title("(f) EC–BC: reduction in the unit cost of biomass-based electricity", loc="left", fontsize=11)

    fig.tight_layout()
    save_figure_bundle(fig, OUT_FIG2, dpi=300, bbox_inches="tight", data_frames={"plant_pairwise": df})
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(IN_CSV)
    fig1(df)
    fig2(df)
    print(f"Saved: {OUT_FIG1}\nSaved: {OUT_FIG2}")


if __name__ == "__main__":
    main()
