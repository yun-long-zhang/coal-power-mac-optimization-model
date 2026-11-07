import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
_RNG = np.random.default_rng(20260101)

from plot_export_utils import save_figure_bundle


RMB_PER_USD = 6.8985
YEAR_SCALE = 0.966  # Year conversion factor for fleet LCOE (coal retirements)
DATA_DIR = "data"
INPUT_WORKBOOK = "1885_coal_power_plants_original.xlsx"
RESULTS_WORKBOOK = "gurobi_r2_base_results.xlsx"
OUTPUT_XLSX = "gurobi_r2_lcoe_by_target.xlsx"
OUTPUT_CSV = "gurobi_r2_lcoe_summary.csv"
OUTPUT_PNG = "fig_lcoe_distribution.png"

# Targets for the scatter-boxplot figure
TARGETS_TO_PLOT = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 48]
TARGET_COLORS = plt.cm.viridis(np.linspace(0.15, 0.85, len(TARGETS_TO_PLOT)))


def _norm_id(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if not s:
        return ""
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def load_plant_baseline(data_dir: str) -> pd.DataFrame:
    """Load plant baseline data: capacity, running hours, LCOE, energy demand."""
    path = os.path.join(data_dir, INPUT_WORKBOOK)
    pp = pd.read_excel(path, sheet_name="coal_plant_i")
    pp["plant_id"] = pp["plant_i"].map(_norm_id)
    pp["capacity_MW"] = pd.to_numeric(pp["capacity_Xi"], errors="coerce").fillna(0.0)
    pp["running_hours"] = pd.to_numeric(pp["running_hours"], errors="coerce").fillna(0.0)
    pp["energy_gj"] = pd.to_numeric(pp["energy_demand_GJ"], errors="coerce").fillna(0.0)
    pp["lcoe_rmb_per_kwh"] = pd.to_numeric(pp["LCOE (RMB/kWh)"], errors="coerce").fillna(0.0)
    # Annual electricity generation (kWh)
    pp["generation_kwh"] = pp["capacity_MW"] * pp["running_hours"] * 1000.0
    # Baseline annual total cost (RMB)
    pp["baseline_total_cost_rmb"] = pp["lcoe_rmb_per_kwh"] * pp["generation_kwh"]
    return pp[["plant_id", "capacity_MW", "running_hours", "energy_gj",
               "generation_kwh", "lcoe_rmb_per_kwh",
               "baseline_total_cost_rmb"]].copy()


def compute_lcoe_for_target(
    pm_df: pd.DataFrame,
    baseline: pd.DataFrame,
    target_id: int,
) -> pd.DataFrame:
    """Merge pm detail with baseline and compute new LCOE for each plant."""
    pm = pm_df.copy()
    pm["plant_id"] = pm["plant_id"].astype(str).map(_norm_id)
    merged = pm.merge(baseline, on="plant_id", how="left")

    # Retrofit cost (USD) -> RMB
    merged["retrofit_cost_rmb"] = pd.to_numeric(merged["total_cost"], errors="coerce").fillna(0.0) * RMB_PER_USD

    # New total cost = baseline + retrofit (both in RMB)
    merged["new_total_cost_rmb"] = merged["baseline_total_cost_rmb"] + merged["retrofit_cost_rmb"]

    # New LCOE (RMB/kWh)
    merged["new_lcoe_rmb_per_kwh"] = np.where(
        merged["generation_kwh"] > 1e-6,
        merged["new_total_cost_rmb"] / merged["generation_kwh"],
        0.0,
    )

    # Convert to USD/MWh (1 RMB/kWh = 1000 / 6.8985 USD/MWh)
    # Apply YEAR_SCALE to fleet LCOE (coal retirements affect annualized costs)
    merged["lcoe_usd_per_mwh"] = merged["lcoe_rmb_per_kwh"] * 1000.0 / RMB_PER_USD * YEAR_SCALE
    merged["new_lcoe_usd_per_mwh"] = merged["new_lcoe_rmb_per_kwh"] * 1000.0 / RMB_PER_USD * YEAR_SCALE

    # LCOE change
    merged["lcoe_change_rmb_per_kwh"] = merged["new_lcoe_rmb_per_kwh"] - merged["lcoe_rmb_per_kwh"]
    merged["lcoe_change_usd_per_mwh"] = merged["lcoe_change_rmb_per_kwh"] * 1000.0 / RMB_PER_USD
    merged["lcoe_change_pct"] = np.where(
        merged["lcoe_rmb_per_kwh"] > 1e-6,
        merged["lcoe_change_rmb_per_kwh"] / merged["lcoe_rmb_per_kwh"] * 100.0,
        0.0,
    )

    merged["target_id"] = target_id
    return merged


def compute_fleet_summary(plant_lcoe: pd.DataFrame) -> dict:
    """Compute fleet-level weighted-average LCOE for one target."""
    total_gen = plant_lcoe["generation_kwh"].sum()
    if total_gen <= 0:
        return {
            "weighted_baseline_lcoe": np.nan,
            "weighted_new_lcoe": np.nan,
            "weighted_lcoe_change": np.nan,
            "total_retrofit_cost_rmb": plant_lcoe["retrofit_cost_rmb"].sum(),
            "total_new_cost_rmb": plant_lcoe["new_total_cost_rmb"].sum(),
        }
    weighted_baseline_rmb = (
        (plant_lcoe["lcoe_rmb_per_kwh"] * plant_lcoe["generation_kwh"]).sum() / total_gen
    )
    weighted_new_rmb = (
        (plant_lcoe["new_lcoe_rmb_per_kwh"] * plant_lcoe["generation_kwh"]).sum() / total_gen
    )
    return {
        "weighted_baseline_lcoe": weighted_baseline_rmb,
        "weighted_new_lcoe": weighted_new_rmb,
        "weighted_lcoe_change": weighted_new_rmb - weighted_baseline_rmb,
        "weighted_baseline_lcoe_usd_per_mwh": weighted_baseline_rmb * 1000.0 / RMB_PER_USD * YEAR_SCALE,
        "weighted_new_lcoe_usd_per_mwh": weighted_new_rmb * 1000.0 / RMB_PER_USD * YEAR_SCALE,
        "weighted_lcoe_change_usd_per_mwh": (weighted_new_rmb - weighted_baseline_rmb) * 1000.0 / RMB_PER_USD * YEAR_SCALE,
        "total_retrofit_cost_rmb": plant_lcoe["retrofit_cost_rmb"].sum(),
        "total_new_cost_rmb": plant_lcoe["new_total_cost_rmb"].sum(),
    }


def _compute_all_lcoe(data_dir: str) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """Compute LCOE for all targets; return (summary_df, plant_data_by_target)."""
    print("Loading plant baseline data...")
    baseline = load_plant_baseline(data_dir)
    print(f"  {len(baseline)} plants loaded")

    results_path = os.path.join(data_dir, RESULTS_WORKBOOK)
    xl = pd.ExcelFile(results_path)
    sheet_names = sorted(s for s in xl.sheet_names if s.startswith("pm") and len(s) == 4)
    print(f"  Found {len(sheet_names)} target detail sheets (pm*)")

    plant_by_target: Dict[int, pd.DataFrame] = {}
    summary_rows = []

    for sheet in sheet_names:
        target_id = int(sheet[2:])
        pm = pd.read_excel(results_path, sheet_name=sheet)
        plant_df = compute_lcoe_for_target(pm, baseline, target_id)
        plant_by_target[target_id] = plant_df
        fleet = compute_fleet_summary(plant_df)
        summary_rows.append({
            "target_id": target_id,
            "plants_with_retrofit": int((plant_df["retrofit_cost_rmb"] > 1e-3).sum()),
            "total_retrofit_cost_rmb": fleet["total_retrofit_cost_rmb"],
            "weighted_baseline_lcoe": fleet["weighted_baseline_lcoe"],
            "weighted_new_lcoe": fleet["weighted_new_lcoe"],
            "weighted_lcoe_change": fleet["weighted_lcoe_change"],
            "weighted_baseline_lcoe_usd_per_mwh": fleet["weighted_baseline_lcoe_usd_per_mwh"],
            "weighted_new_lcoe_usd_per_mwh": fleet["weighted_new_lcoe_usd_per_mwh"],
            "weighted_lcoe_change_usd_per_mwh": fleet["weighted_lcoe_change_usd_per_mwh"],
            "fleet_generation_kwh": plant_df["generation_kwh"].sum(),
        })

    summary = pd.DataFrame(summary_rows).sort_values("target_id").reset_index(drop=True)

    # Attach abatement info
    cm_curve = pd.read_excel(results_path, sheet_name="cm")
    summary = summary.merge(
        cm_curve[["target_id", "achieved_abatement", "total_cost", "marginal_cost_diff"]],
        on="target_id", how="left",
    )
    return summary, plant_by_target


def plot_lcoe_box_scatter(
    summary: pd.DataFrame,
    plant_by_target: Dict[int, pd.DataFrame],
    output_png: str,
) -> None:
    """Draw scatter-boxplot + mean/min/max lines for selected targets.
    LCOE is in USD/MWh. Reference lines for onshore wind (27), offshore wind (49),
    and solar PV (36) are overlaid."""
    targets = TARGETS_TO_PLOT

    # Build x positions: achieved abatement / 1e9 (Gt CO2)
    abatement_map = {
        int(r["target_id"]): float(r["achieved_abatement"])
        for _, r in summary.iterrows()
        if int(r["target_id"]) in targets
    }
    x_positions = np.array([abatement_map[t] for t in targets], dtype=float) / 1e9

    # Collect LCOE data arrays (USD/MWh)
    lcoe_arrays: List[np.ndarray] = []
    for t in targets:
        df = plant_by_target[t]
        vals = pd.to_numeric(df["new_lcoe_usd_per_mwh"], errors="coerce").dropna().to_numpy(dtype=float)
        lcoe_arrays.append(vals)

    # Per-target stats in USD/MWh
    means = np.array([np.mean(a) for a in lcoe_arrays])
    mins = np.array([np.min(a) for a in lcoe_arrays])
    maxs = np.array([np.max(a) for a in lcoe_arrays])

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Reference lines for competitive renewable LCOE
    ref_specs = [
        (27.0, "#a233e8", "Onshore wind"),
        (36.0, "#ff7f0e", "Solar PV"),
        (49.0, "#1f77b4", "Offshore wind"),
    ]
    for ref_y, ref_color, ref_label in ref_specs:
        ax.axhline(ref_y, color=ref_color, linewidth=1.5, linestyle="--", alpha=0.7, zorder=3)
        ax.text(
            x_positions[-1] * 1.01, ref_y + 1.2, ref_label,
            fontsize=10, color="#1F1F1F", fontweight="bold", ha="left", va="bottom",
            zorder=10,
        )

    # 1) Scatter points (jittered)
    for idx, (x, arr) in enumerate(zip(x_positions, lcoe_arrays)):
        if len(arr) > 2000:
            step = max(1, len(arr) // 2000)
            arr_sub = arr[::step]
        else:
            arr_sub = arr
        jitter = _RNG.uniform(-0.03, 0.03, size=len(arr_sub))  # fixed seed: reproducible dot positions
        ax.scatter(
            x + jitter, arr_sub,
            s=4, alpha=0.25, color=TARGET_COLORS[idx],
            rasterized=True, zorder=2,
        )

    # 2) Boxplots
    bp_width = 0.05 * (x_positions.max() - x_positions.min()) if x_positions.max() > x_positions.min() else 0.05
    for idx, (x, arr) in enumerate(zip(x_positions, lcoe_arrays)):
        bp = ax.boxplot(
            arr, positions=[x], widths=bp_width,
            showfliers=False, whis=0, patch_artist=True, zorder=4,
        )
        for box in bp["boxes"]:
            box.set(facecolor=TARGET_COLORS[idx], alpha=0.35, edgecolor=TARGET_COLORS[idx], linewidth=1.8)
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_visible(False)

    # 3) Mean, min, max lines
    ax.plot(x_positions, means, color="#d62728", linewidth=2.2, marker="D", markersize=6,
            label="Mean LCOE", zorder=6)
    ax.plot(x_positions, mins, color="#2ca02c", linewidth=1.8, linestyle="--", marker="v",
            markersize=5, label="Min LCOE", zorder=5)
    ax.plot(x_positions, maxs, color="#1f77b4", linewidth=1.8, linestyle="--", marker="^",
            markersize=5, label="Max LCOE", zorder=5)

    # 4) Shaded band between min and max
    ax.fill_between(x_positions, mins, maxs, alpha=0.10, color="#7f7f7f", zorder=1)

    # Solar-plus-storage vertical range line on the right
    storage_y_low = 54.0
    storage_y_high = 82.0
    storage_x = x_positions[-1] + (x_positions[-1] - x_positions[-2]) * 1
    ax.plot(
        [storage_x, storage_x], [storage_y_low, storage_y_high],
        color="#D62728", linewidth=5, solid_capstyle="round", zorder=12,
    )
    ax.text(
        storage_x + 0.03, (storage_y_low + storage_y_high) / 2,
        "Solar +\nstorage",
        fontsize=10, color="#1F1F1F", fontweight="bold",
        ha="left", va="center", zorder=12,
    )

    # Labels & grid
    ax.set_xlabel("Abatement target (Gt CO$_2$)", fontsize=13)
    ax.set_ylabel("LCOE (USD/MWh)", fontsize=13)
    ax.set_title("Fleet-level LCOE distribution under increasing abatement targets", fontsize=14)
    ax.grid(alpha=0.18, linestyle="--")
    ax.legend(fontsize=11, loc="upper left")

    # X ticks
    tick_labels = [f"{x:.1f}" for x in x_positions]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, fontsize=10)

    # Also update the stats to USD/MWh for export
    stats_df = pd.DataFrame({
        "target_id": targets,
        "abatement_gt": x_positions,
        "mean_lcoe_usd_per_mwh": means,
        "min_lcoe_usd_per_mwh": mins,
        "max_lcoe_usd_per_mwh": maxs,
    })

    fig.tight_layout()
    outputs = save_figure_bundle(
        fig, output_png, dpi=320, bbox_inches="tight",
        data_frames={
            "summary": summary[summary["target_id"].isin(targets)].copy(),
            "stats": stats_df,
        },
    )
    plt.close(fig)
    print(f"Output PNG: {outputs['png']}")
    print(f"Output PDF: {outputs['pdf']}")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, DATA_DIR)

    summary, plant_by_target = _compute_all_lcoe(data_dir)

    # Export tables
    out_xlsx = os.path.join(data_dir, OUTPUT_XLSX)
    out_csv = os.path.join(data_dir, OUTPUT_CSV)
    summary.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        for tid in sorted(plant_by_target.keys()):
            df = plant_by_target[tid]
            df_out = df[[
                "target_id", "plant_id", "capacity_MW", "running_hours",
                "generation_kwh", "lcoe_rmb_per_kwh", "lcoe_usd_per_mwh",
                "baseline_total_cost_rmb", "retrofit_cost_rmb",
                "new_total_cost_rmb", "new_lcoe_rmb_per_kwh", "new_lcoe_usd_per_mwh",
                "lcoe_change_rmb_per_kwh", "lcoe_change_usd_per_mwh", "lcoe_change_pct",
            ]].copy()
            df_out.to_excel(writer, sheet_name=f"pm{tid:02d}", index=False)

    print(f"Output XLSX: {out_xlsx}")
    print(f"Output CSV:  {out_csv}")

    # Plot
    out_png = os.path.join(data_dir, OUTPUT_PNG)
    plot_lcoe_box_scatter(summary, plant_by_target, out_png)

    print("\nDone.")


if __name__ == "__main__":
    main()


