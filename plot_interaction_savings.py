"""
Plot interaction savings results: EC→BC and BC→CC cost savings
across abatement targets.

Three-panel figure:
  (a) Total cost savings from EC→BC and BC→CC (billions $)
  (b) Unit bioelectricity cost of BC with/without EC ($/kWh)
  (c) Unit carbon capture cost of CC with/without BC ($/ton)

Reads from: data/interaction_savings_v10_interaction_savings.xlsx
Output:     fig_interaction_savings_3panel.png  (+.pdf)
            fig_interaction_savings_3panel_data.xlsx
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from plot_export_utils import save_figure_bundle

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
INPUT_PATH = os.path.join(DATA_DIR, "interaction_savings_interaction_savings.xlsx")
OUTPUT_PNG = os.path.join(DATA_DIR, "fig_interaction_savings_3panel.png")

# ── Load ───────────────────────────────────────────────────────────────
summary = pd.read_excel(INPUT_PATH, sheet_name="savings_summary")

# Filter targets with nonzero BC activity
active = summary[summary["total_bc_elec_kwh"] > 0].copy()

# Convert to sensible units
active["target_abatement_Mt"] = active["target_abatement"] / 1e6         # Mt CO₂
active["ec_saving_bc_$B"] = active.get("ec_saving_bc_technical", active.get("ec_saving_bc_total", pd.Series(0.0))) / 1e9
active["bc_saving_cc_$B"] = active.get("bc_saving_cc_total_volume_weighted", active.get("bc_saving_cc_total", pd.Series(0.0))) / 1e9
active["bc_elec_TWh"] = active["total_bc_elec_kwh"] / 1e9                # TWh
active["cc_capture_Mt"] = active["total_cc_capture_ton"] / 1e6           # Mt CO₂
active["cc_capture_no_bc_Mt"] = active["total_cc_capture_no_bc_ton"] / 1e6


# ═══════════════════════════════════════════════════════════════════════════
# Original figure (legacy column names)
# ═══════════════════════════════════════════════════════════════════════════
def make_original_figure(active: pd.DataFrame, out_png: str) -> dict:
    x = active["target_abatement_Mt"].values

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)

    # ── Panel (a): BC unit bioelectricity cost (was panel b) ───────────
    ax = axes[0]
    ax.plot(x, active["weighted_avg_bc_unit_with_ec_per_kwh"], "s-", color="#E67E22",
            lw=1.8, ms=4, label="With EC (reference)")
    ax.plot(x, active["weighted_avg_bc_unit_no_ec_per_kwh"], "o-", color="#95A5A6",
            lw=1.8, ms=4, label="Without EC (counterfactual)")
    avg_save_kwh = (active["weighted_avg_bc_unit_no_ec_per_kwh"] - active["weighted_avg_bc_unit_with_ec_per_kwh"]).mean()
    ax.fill_between(x, active["weighted_avg_bc_unit_with_ec_per_kwh"],
                    active["weighted_avg_bc_unit_no_ec_per_kwh"],
                    alpha=0.15, color="#E67E22",
                    label=f"Average saving ≈ ${avg_save_kwh:.4f}/kWh")
    ax.set_xlabel("Total abatement target (Mt CO₂)")
    ax.set_ylabel("BC unit bioelectricity cost ($/kWh)")
    ax.set_title("(a) BC unit bioelectricity cost", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, None)

    # ── Panel (b): CC unit capture cost (was panel c) ──────────────────
    ax = axes[1]
    ax.plot(x, active["weighted_avg_cc_unit_with_bc_per_ton"], "s-", color="#2980B9",
            lw=1.8, ms=4, label="With BC (reference)")
    ax.plot(x, active["weighted_avg_cc_unit_no_bc_per_ton"], "o-", color="#95A5A6",
            lw=1.8, ms=4, label="Without BC (counterfactual)")
    avg_save_ton = (active["weighted_avg_cc_unit_no_bc_per_ton"] - active["weighted_avg_cc_unit_with_bc_per_ton"]).mean()
    ax.fill_between(x, active["weighted_avg_cc_unit_with_bc_per_ton"],
                    active["weighted_avg_cc_unit_no_bc_per_ton"],
                    alpha=0.15, color="#2980B9",
                    label=f"Average saving ≈ ${avg_save_ton:.2f}/ton")
    ax.set_xlabel("Total abatement target (Mt CO₂)")
    ax.set_ylabel("CCS unit capture abatement cost ($/ton)")
    ax.set_title("(b) CCS unit capture abatement cost", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, None)

    # ── Panel (c): Interaction cost savings (was panel a) ──────────────
    ax = axes[2]
    ec_col = "ec_saving_bc_$B"
    cc_col = "bc_saving_cc_$B"
    ax.plot(x, active[ec_col], "s-", color="#E67E22", lw=1.8, ms=4, label="EC → BC")
    ax.plot(x, active[cc_col], "o-", color="#2980B9", lw=1.8, ms=4, label="BC → CC")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("Total abatement target (Mt CO₂)")
    ax.set_ylabel("Total cost saving (billion $)")
    ax.set_title("(c) Interaction cost savings (volume-weighted)", loc="left", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.set_xlim(0, None)

    plot_df = _build_plot_df(active)
    outputs = save_figure_bundle(
        fig, out_png, dpi=320, bbox_inches="tight",
        data_frames={"plot_data": plot_df, "savings_summary": summary},
    )
    plt.close(fig)
    return outputs


# ═══════════════════════════════════════════════════════════════════════════
# Refined figure (improved metrics from updated analysis script)
# ═══════════════════════════════════════════════════════════════════════════
def make_refined_figure(active: pd.DataFrame, out_png: str) -> dict:
    """Same 3-panel layout but with the updated savings indicators:

    Panel (a): replaces volume-weighted BC→CC saving with pure unit-cost
               effect; adds net EC→BC saving (including coal saving credit).
    Panels (b) and (c): unchanged; additional net-cost unit lines added.
    """
    x = active["target_abatement_Mt"].values

    # Resolve column names (new code renames old columns)
    ec_tech_col = "ec_saving_bc_technical" if "ec_saving_bc_technical" in active.columns else "ec_saving_bc_total"
    ec_net_col  = "ec_saving_bc_net" if "ec_saving_bc_net" in active.columns else ec_tech_col
    cc_vw_col   = ("bc_saving_cc_total_volume_weighted" if "bc_saving_cc_total_volume_weighted" in active.columns
                   else "bc_saving_cc_total")
    cc_uo_col   = "bc_saving_cc_unit_only" if "bc_saving_cc_unit_only" in active.columns else cc_vw_col

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)

    # ── Panel (a): BC costs — technical + net (was panel b) ────────────
    ax = axes[0]
    ax.plot(x, active["weighted_avg_bc_unit_with_ec_per_kwh"], "s-", color="#E67E22",
            lw=1.8, ms=4, label="With EC (technical)")
    ax.plot(x, active["weighted_avg_bc_unit_no_ec_per_kwh"], "o-", color="#95A5A6",
            lw=1.8, ms=4, label="Without EC (counterfactual, tech.)")
    if "bc_unit_net_with_ec" in active.columns and "bc_unit_net_no_ec" in active.columns:
        ax.plot(x, active["bc_unit_net_with_ec"], "s--", color="#D35400",
                lw=1.2, ms=3, label="With EC (net)")
        ax.plot(x, active["bc_unit_net_no_ec"], "o--", color="#7F8C8D",
                lw=1.2, ms=3, label="Without EC (counterfactual, net)")

    avg_save_kwh = (active["weighted_avg_bc_unit_no_ec_per_kwh"] - active["weighted_avg_bc_unit_with_ec_per_kwh"]).mean()
    ax.fill_between(x, active["weighted_avg_bc_unit_with_ec_per_kwh"],
                    active["weighted_avg_bc_unit_no_ec_per_kwh"],
                    alpha=0.12, color="#E67E22",
                    label=f"Avg tech. saving ≈ ${avg_save_kwh:.4f}/kWh")
    ax.set_xlabel("Total abatement target (Mt CO₂)")
    ax.set_ylabel("BC unit bioelectricity cost ($/kWh)")
    ax.set_title("(a) BC unit cost: technical vs. net", loc="left", fontsize=11)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlim(0, None)

    # ── Panel (b): CC costs (was panel c) ──────────────────────────────
    ax = axes[1]
    ax.plot(x, active["weighted_avg_cc_unit_with_bc_per_ton"], "s-", color="#2980B9",
            lw=1.8, ms=4, label="With BC (reference)")
    ax.plot(x, active["weighted_avg_cc_unit_no_bc_per_ton"], "o-", color="#95A5A6",
            lw=1.8, ms=4, label="Without BC (counterfactual)")
    avg_save_ton = (active["weighted_avg_cc_unit_no_bc_per_ton"] - active["weighted_avg_cc_unit_with_bc_per_ton"]).mean()
    ax.fill_between(x, active["weighted_avg_cc_unit_with_bc_per_ton"],
                    active["weighted_avg_cc_unit_no_bc_per_ton"],
                    alpha=0.12, color="#2980B9",
                    label=f"Avg saving ≈ ${avg_save_ton:.2f}/ton")
    ax.set_xlabel("Total abatement target (Mt CO₂)")
    ax.set_ylabel("CCS unit capture abatement cost ($/ton)")
    ax.set_title("(b) CCS unit capture cost", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, None)

    # ── Panel (c): refined savings — two EC→BC metrics + pure unit CC ──
    ax = axes[2]
    ax.plot(x, active[ec_tech_col] / 1e9, "s-", color="#E67E22", lw=1.4, ms=4,
            label="EC→BC (technical, excl. coal saving)")
    ax.plot(x, active[ec_net_col] / 1e9, "s--", color="#D35400", lw=1.8, ms=4,
            label="EC→BC (net, incl. coal saving)")
    ax.plot(x, active[cc_vw_col] / 1e9, "o-", color="#2980B9", lw=1.4, ms=4,
            label="BC→CC (volume-weighted)")
    ax.plot(x, active[cc_uo_col] / 1e9, "o--", color="#1A5276", lw=1.8, ms=4,
            label="BC→CC (unit-cost-only)")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("Total abatement target (Mt CO₂)")
    ax.set_ylabel("Total cost saving (billion $)")
    ax.set_title("(c) Refined interaction cost savings", loc="left", fontsize=11)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.set_xlim(0, None)

    plot_df = _build_plot_df(active)
    outputs = save_figure_bundle(
        fig, out_png, dpi=320, bbox_inches="tight",
        data_frames={"plot_data": plot_df, "savings_summary": summary},
    )
    plt.close(fig)
    return outputs


def _build_plot_df(active: pd.DataFrame) -> pd.DataFrame:
    """Extract available plot columns into a clean DataFrame."""
    plot_cols = [
        "target_abatement",
        "ec_saving_bc_total", "ec_saving_bc_technical", "ec_saving_bc_net",
        "bc_saving_cc_total", "bc_saving_cc_total_volume_weighted", "bc_saving_cc_unit_only",
        "total_bc_elec_kwh",
        "weighted_avg_bc_unit_with_ec_per_kwh", "weighted_avg_bc_unit_no_ec_per_kwh",
        "total_cc_capture_ton", "total_cc_capture_no_bc_ton",
        "weighted_avg_cc_unit_with_bc_per_ton", "weighted_avg_cc_unit_no_bc_per_ton",
    ]
    available = [c for c in plot_cols if c in active.columns]
    out = active[available].copy()
    out["target_abatement_Mt"] = out["target_abatement"] / 1e6
    for c in ["ec_saving_bc_total", "ec_saving_bc_technical", "ec_saving_bc_net"]:
        if c in out.columns:
            out[f"{c}_$B"] = out[c] / 1e9
    for c in ["bc_saving_cc_total", "bc_saving_cc_total_volume_weighted", "bc_saving_cc_unit_only"]:
        if c in out.columns:
            out[f"{c}_$B"] = out[c] / 1e9
    return out


# ── Run both ──────────────────────────────────────────────────────────────
out_original = make_original_figure(active, OUTPUT_PNG)
print(f"Original: {out_original['png']}")

refined_path = os.path.join(DATA_DIR, "fig_interaction_savings_3panel_refined.png")
out_refined = make_refined_figure(active, refined_path)
print(f"Refined: {out_refined['png']}")
print(f"Refined: {out_refined['pdf']}")
print(f"Refined: {out_refined['xlsx']}")
