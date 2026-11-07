"""
Sensitivity of the fleet MAC curve to the carbon-accounting assumptions (avoided-accounting result set).

  base            avoided accounting: capture energy 2.4 GJ/t + 0.10 MWh/t from turbine extraction, extra coal
                  fired and its flue gas captured; biomass supply chain 10 kg CO2e/GJ + haulage 0.04 kg/t-km
  captured_basis  the previous accounting: captured CO2 counted as abatement, no logistics emissions
  cc_energy_high  conventional MEA capture energy (3.5 GJ/t + 0.23 MWh/t)
  bio_ef_5/15     biomass supply-chain emissions 5 and 15 kg CO2e/GJ (Creutzig et al. 2015 range)

Outputs fig_accounting_sensitivity.png (+ xlsx with the curves and a summary table) in data/.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_export_utils import save_figure_bundle


DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_PNG = os.path.join(DATA, "fig_accounting_sensitivity.png")
CASES = [
    ("base", "Baseline (avoided accounting)", "#1F1F1F", "-", 2.6),
    ("captured_basis", "Captured CO$_2$ counted as abatement, no logistics emissions", "#7F7F7F", "--", 1.8),
    ("cc_energy_high", "MEA capture energy (3.5 GJ t$^{-1}$ + 0.23 MWh t$^{-1}$)", "#D62728", "-.", 1.8),
    ("bio_ef_5", "Biomass supply chain 5 kg CO$_2$e GJ$^{-1}$", "#41AB5D", ":", 1.8),
    ("bio_ef_15", "Biomass supply chain 15 kg CO$_2$e GJ$^{-1}$", "#006D2C", "--", 1.8),
]
REPORT_GT = [1.0, 2.0, 3.0, 3.5, 4.0]


def _curve(name: str) -> pd.DataFrame:
    path = os.path.join(DATA, f"gurobi_r2_{name}_curve.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    return d[d["status"].astype(str).isin(["Optimal", "TimeLimit"])].sort_values("achieved_abatement")


def main() -> None:
    runs = {k: _curve(k) for k, *_ in CASES}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for key, lab, col, ls, lw in CASES:
        d = runs[key]
        if d is None:
            continue
        x = d["achieved_abatement"] / 1e9
        axes[0].plot(x, d["marginal_cost_diff"], color=col, ls=ls, lw=lw, label=lab)
        axes[1].plot(x, d["total_cost"] / 1e9, color=col, ls=ls, lw=lw, label=lab)
    e0 = float(pd.read_excel(os.path.join(DATA, "1885_coal_power_plants_original.xlsx"), sheet_name="coal_plant_i")["carbon_emissions_ton"].sum())
    for ax in axes:
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(e0 / 1e9, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("Abatement (Gt CO$_2$ yr$^{-1}$)")
        ax.grid(alpha=0.2, ls="--")
    axes[0].text(e0 / 1e9, axes[0].get_ylim()[1] * 0.95, " fleet emissions", fontsize=8, color="grey", va="top")
    axes[0].set_ylabel("Marginal abatement cost (USD t CO$_2^{-1}$)")
    axes[0].set_title("(a) Marginal abatement cost", loc="left")
    axes[1].set_ylabel("Total annual abatement cost (billion USD yr$^{-1}$)")
    axes[1].set_title("(b) Total annual cost", loc="left")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()

    rows = []
    for key, lab, *_ in CASES:
        d = runs[key]
        if d is None:
            continue
        row = {"scenario": key, "max_abatement_gt": d["achieved_abatement"].max() / 1e9}
        x = d["achieved_abatement"].to_numpy()
        for g in REPORT_GT:
            ok = g * 1e9 <= x.max()
            row[f"cost_{g:g}Gt_bn"] = float(np.interp(g * 1e9, x, d["total_cost"])) / 1e9 if ok else np.nan
            row[f"mac_{g:g}Gt"] = float(np.interp(g * 1e9, x, d["marginal_cost_diff"].fillna(0))) if ok else np.nan
        neg = d[d["total_cost"] < 0]
        row["min_total_cost_bn"] = d["total_cost"].min() / 1e9
        row["abatement_at_min_cost_gt"] = d.loc[d["total_cost"].idxmin(), "achieved_abatement"] / 1e9
        row["max_abatement_with_negative_total_cost_gt"] = neg["achieved_abatement"].max() / 1e9 if len(neg) else 0.0
        rows.append(row)
    table = pd.DataFrame(rows)
    curves = pd.concat([runs[k].assign(scenario=k) for k, *_ in CASES if runs[k] is not None], ignore_index=True)
    save_figure_bundle(fig, OUT_PNG, dpi=300, bbox_inches="tight", data_frames={"summary": table, "curves": curves})
    plt.close(fig)
    pd.set_option("display.width", 220)
    print(table.round(2).to_string(index=False))
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
