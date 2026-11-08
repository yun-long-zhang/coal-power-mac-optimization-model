"""
Supplementary Figs. S9 and S10 from the EC packages used in the fleet model (data/ec_packages_r2.csv).

  S9   abatement contribution (Mt CO2 yr-1) of the six EC process categories in each package
  S10  fleet-average net unit abatement cost (USD t-1, after avoided coal cost) of each package,
       by plant-level savings target and within-category overlap coefficient

Packages: rXX_oYY = at least XX% CO2 reduction per plant with overlap coefficient YY%; no5_ = measure No. 5
excluded (baseline: no5_r10_o00). The category contributions split each plant's package abatement in
proportion to the gross savings of its selected measures (unit savings in Table S3 x plant generation).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_export_utils import save_figure_bundle

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PACKAGES = os.path.join(DATA, "ec_packages_r2.csv")
WORKBOOK = os.path.join(DATA, "1885_coal_power_plants_original.xlsx")
OUT_S9 = os.path.join(DATA, "fig_s9_ec_category_contribution.png")
OUT_S10 = os.path.join(DATA, "fig_s10_ec_package_uac.png")

# unit emission reduction of each measure (kg CO2 MWh-1, Supplementary Table S3)
UNIT_ER = {1: 9.78, 2: 9.28, 3: 2.90, 4: 5.81, 5: 86.40, 6: 0.63, 7: 4.94, 8: 3.28, 9: 4.65, 10: 7.92,
           11: 0.63, 12: 7.15, 13: 5.44, 14: 2.24, 15: 11.56, 16: 6.38, 17: 0.52, 18: 6.40, 19: 1.68, 20: 4.60}
CATEGORY = {
    "Boiler combustion and milling": [2, 5, 6, 8, 11, 19],
    "Steam turbine body and flow path": [1, 13, 18],
    "Cooling and condensing": [3, 9, 10, 15],
    "Air preheater sealing": [4, 7],
    "Flue-gas heat recovery": [12, 16],
    "Electrical auxiliaries and control": [14, 17, 20],
}
COLORS = ["#4C72B0", "#55A868", "#64B5CD", "#8172B3", "#CCB974", "#C44E52"]
ORDER = [("no5_r10_o00", "10%\n(baseline,\nno No. 5)"), ("r10_o00", "10%"), ("r15_o00", "15%"), ("r15_o25", "15%\noverlap 25%"),
         ("r15_o50", "15%\noverlap 50%"), ("r175_o00", "17.5%"), ("r175_o25", "17.5%\noverlap 25%"),
         ("r175_o50", "17.5%\noverlap 50%"), ("r20_o00", "20%")]


def _norm(v) -> str:
    s = str(v).strip()
    try:
        f = float(s)
        return str(int(f)) if abs(f - round(f)) < 1e-9 else s
    except ValueError:
        return s


def main() -> None:
    pk = pd.read_csv(PACKAGES, dtype={"plant_id": str})
    pl = pd.read_excel(WORKBOOK, sheet_name="coal_plant_i")
    gen = pd.Series((pl["capacity_Xi"] * pl["running_hours"]).values, index=pl["plant_i"].map(_norm))
    cat_of = {k: c for c, ks in CATEGORY.items() for k in ks}

    rows = []
    for _, r in pk.iterrows():
        techs = [int(t.split("_")[1]) for t in str(r["selected_tech_id"]).split(",") if t.startswith("TECH_")]
        g = gen.get(_norm(r["plant_id"]), np.nan)
        gross = {k: UNIT_ER[k] * g / 1e3 for k in techs}
        tot = sum(gross.values())
        if not tot or not np.isfinite(tot):
            continue
        for k, v in gross.items():
            rows.append({"package": r["package"], "category": cat_of[k], "abatement_t": r["ec_abatement_ton"] * v / tot})
    cat = pd.DataFrame(rows).pivot_table(index="package", columns="category", values="abatement_t", aggfunc="sum").fillna(0) / 1e6
    cat = cat.reindex([p for p, _ in ORDER])[list(CATEGORY)]

    fig, ax = plt.subplots(figsize=(12, 5.2))
    bottom = np.zeros(len(cat))
    for c, col in zip(CATEGORY, COLORS):
        ax.bar(range(len(cat)), cat[c].values, bottom=bottom, color=col, label=c, width=0.7)
        bottom += cat[c].values
    for x, v in enumerate(bottom):
        ax.text(x, v + 6, f"{v:.0f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(cat)))
    ax.set_xticklabels([lab for _, lab in ORDER], fontsize=8.5)
    ax.set_ylabel("EC abatement (Mt CO$_2$ yr$^{-1}$)")
    ax.set_xlabel("Plant-level EC savings target and within-category overlap coefficient")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    ax.set_ylim(0, bottom.max() * 1.25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure_bundle(fig, OUT_S9, dpi=300, bbox_inches="tight", data_frames={"category_mt": cat.reset_index()})
    plt.close(fig)

    s = pk.groupby("package").agg(abatement=("ec_abatement_ton", "sum"), cost=("ec_net_cost", "sum"))
    s["uac"] = s["cost"] / s["abatement"]
    targets = [("10", "10%"), ("15", "15%"), ("175", "17.5%"), ("20", "20%")]
    overlaps = ["00", "25", "50"]
    grid = np.full((len(overlaps), len(targets)), np.nan)
    for i, o in enumerate(overlaps):
        for j, (t, _) in enumerate(targets):
            key = f"r{t}_o{o}"
            if key in s.index:
                grid[i, j] = s.loc[key, "uac"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    # perceptually uniform colour map; cell labels switch to white on dark cells
    im = ax.imshow(grid, cmap="viridis", vmin=np.nanmin(grid) - 1, vmax=np.nanmax(grid) + 1, aspect="auto")
    for i in range(len(overlaps)):
        for j in range(len(targets)):
            v = grid[i, j]
            # 20%: positive overlaps leave some plants below the target even with all measures; 10%: overlap not run
            missing = "infeasible" if targets[j][0] == "20" else "not evaluated"
            r, g, b_, _ = im.cmap(im.norm(v)) if np.isfinite(v) else (1, 1, 1, 1)
            colour = "white" if 0.299 * r + 0.587 * g + 0.114 * b_ < 0.5 else "black"
            ax.text(j, i, f"{v:.1f}" if np.isfinite(v) else missing, ha="center", va="center", fontsize=9, color=colour)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([lab for _, lab in targets])
    ax.set_yticks(range(len(overlaps)))
    ax.set_yticklabels([f"{int(o)}%" for o in overlaps])
    ax.set_xlabel("Plant-level EC savings target (at least)")
    ax.set_ylabel("Within-category overlap coefficient")
    base = s.loc["no5_r10_o00", "uac"]
    ax.set_title(f"Net unit abatement cost of EC packages (USD t CO$_2^{{-1}}$); baseline without No. 5: {base:.1f}", fontsize=9.5)
    fig.colorbar(im, ax=ax, label="USD t CO$_2^{-1}$")
    fig.tight_layout()
    save_figure_bundle(fig, OUT_S10, dpi=300, bbox_inches="tight", data_frames={"uac": s.reset_index()})
    plt.close(fig)
    print(cat.round(1).to_string())
    print(s.round(2).to_string())


if __name__ == "__main__":
    main()
