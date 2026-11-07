"""Supplementary Fig. S23: provincial reduction in the cost gap of BC and CC due to technology interactions
at a selected node of the fleet-wide optimization (data/interaction_cost_gap_r2.xlsx)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_export_utils import save_figure_bundle

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IN_XLSX = os.path.join(DATA_DIR, "interaction_cost_gap_r2.xlsx")
TARGET_ID = 39  # P36 (3.61 Gt CO2), carbon-neutrality node
OUT_PNG = os.path.join(DATA_DIR, "fig_interaction_cost_gap_provinces.png")


def main() -> None:
    prov = pd.read_excel(IN_XLSX, sheet_name=f"provinces_target{TARGET_ID:02d}")
    by_t = pd.read_excel(IN_XLSX, sheet_name="by_target")
    tot = by_t[by_t["target_id"] == TARGET_ID].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 8.5), sharey=False)
    panels = [
        ("ecbc_gap_reduction", "#2E8B57",
         f"(a) EC–BC: change in the net cost gap of biomass-based electricity\n(negative: EC raises the net cost, as less coal is displaced)\n"
         f"national total {tot['ecbc_gap_reduction_usd'] / 1e9:.2f} billion USD yr$^{{-1}}$ ({int(tot['ecbc_n_plants'])} plants)"),
        ("bccc_gap_reduction", "#2F6DB5",
         f"(b) BC–CC: lower unit cost of carbon capture\n(captured biogenic CO$_2$)\n"
         f"national total {tot['bccc_gap_reduction_usd'] / 1e9:.2f} billion USD yr$^{{-1}}$ ({int(tot['bccc_n_plants'])} plants)"),
    ]
    for ax, (col, color, title) in zip(axes, panels):
        d = prov[["province", col]].copy()
        d[col] = d[col] / 1e6
        d = d[d[col].abs() > 0.05].sort_values(col)
        ax.barh(d["province"], d[col], color=color)
        for y, v in enumerate(d[col]):
            ax.text(v + (2 if v >= 0 else -2), y, f"{v:.0f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Reduction in cost gap (million USD yr$^{-1}$)")
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        f"Interaction-induced reduction in the cost gap of BC and CC by province at P{round(tot['target_abatement'] / 1e8)} "
        f"({tot['target_abatement'] / 1e9:.2f} Gt CO$_2$ yr$^{{-1}}$)",
        fontsize=11.5,
    )
    fig.tight_layout()
    save_figure_bundle(fig, OUT_PNG, dpi=300, bbox_inches="tight", data_frames={"provinces": prov})
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
