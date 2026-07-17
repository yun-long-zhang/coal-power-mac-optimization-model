import argparse
import importlib
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_export_utils import save_figure_bundle, save_plotly_bundle


# ---- User config (edit before running) ----
DATA_DIR_NAME = "data"
INPUT_PREFIX = "fleet_joint"
DEFAULT_CURVE_SUFFIX = "_curve.csv"
DEFAULT_LP_CURVE_SUFFIX = "_lp_curve.csv"
DEFAULT_RESULTS_SUFFIX = "_results.xlsx"
DEFAULT_RESULTS_FALLBACK = "gurobi_original_results.xlsx"
DEFAULT_CURVE_FALLBACK_FILES = [
    "gurobi_original_curve.csv",
    "gurobi_curve.csv",
]

SENSITIVITY_PREFIX = "gurobi"
SENSITIVITY_CASES = [
    ("baseline", "Baseline", "baseline", "_original_curve.csv", "-"),
    ("capex_m30", "CAPEX -30%", "capex", "_capex_0p70_curve.csv", "-"),
    ("capex_p30", "CAPEX +30%", "capex", "_capex_1p30_curve.csv", "--"),
    ("coal_m25", "Coal -25%", "coal", "_coal_m25_curve.csv", "-"),
    ("coal_p25", "Coal +25%", "coal", "_coal_p25_curve.csv", "--"),
    ("af5", "Discount AF 5%", "discount", "_af5_curve.csv", "-"),
    ("af12", "Discount AF 12%", "discount", "_af12_curve.csv", "--"),
]

BIOMASS_SENSITIVITY_PREFIX = "biomass_supply_sensitivity"
BIOMASS_BASELINE_CURVE_NAME = "gurobi_original_curve.csv"
BIOMASS_BASELINE_RESULTS_NAME = "gurobi_original_results.xlsx"
BIOMASS_SENSITIVITY_CASES = [
    ("bio_75", "Biomass supply 75%", "bio75", 0.75, "_bio_supply_75_curve.csv", "_bio_supply_75_results.xlsx", "--"),
    ("bio_50", "Biomass supply 50%", "bio50", 0.50, "_bio_supply_50_curve.csv", "_bio_supply_50_results.xlsx", "-."),
    ("bio_25", "Biomass supply 25%", "bio25", 0.25, "_bio_supply_25_curve.csv", "_bio_supply_25_results.xlsx", ":"),
]

OUTPUT_LOGISTICS_DUALAXIS_PNG = "fleet_joint_logistics_dualaxis.png"
OUTPUT_SCENARIO_SORTED_DIR = "fleet_joint_scenario_sorted_macc"
OUTPUT_STATE_AREA_PNG = "fleet_joint_state_evolution_area.png"
OUTPUT_TECH_SHARE_SPLIT_AREA_PNG = "fleet_joint_tech_share_split_area.png"
OUTPUT_SENSITIVITY_OVERLAY_PNG = "fleet_joint_macc_sensitivity_overlay.png"
OUTPUT_SENSITIVITY_DELTA_PNG = "fleet_joint_macc_sensitivity_delta.png"
OUTPUT_SENSITIVITY_ENVELOPE_PNG = "fleet_joint_macc_sensitivity_envelope.png"
OUTPUT_BIOMASS_SENSITIVITY_COMPARE_PNG = "fleet_joint_biomass_supply_sensitivity_compare.png"
# -------------------------------------------


def _load_curve(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"target_abatement", "total_cost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {label}: {sorted(missing)}")

    df = df.copy()
    df["target_abatement"] = pd.to_numeric(df["target_abatement"], errors="coerce")
    df["total_cost"] = pd.to_numeric(df["total_cost"], errors="coerce")
    df = df.dropna(subset=["target_abatement", "total_cost"]).sort_values("target_abatement").reset_index(drop=True)

    if "marginal_cost_diff" not in df.columns:
        df["marginal_cost_diff"] = df["total_cost"].diff() / df["target_abatement"].diff().replace(0, np.nan)
    else:
        df["marginal_cost_diff"] = pd.to_numeric(df["marginal_cost_diff"], errors="coerce")

    return df


def _resolve_default_curve_path(data_dir: str, input_prefix: str) -> str:
    preferred = os.path.join(data_dir, f"{input_prefix}{DEFAULT_CURVE_SUFFIX}")
    if os.path.exists(preferred):
        return preferred

    # Common fallback files in this project.
    fallback_candidates = [os.path.join(data_dir, name) for name in DEFAULT_CURVE_FALLBACK_FILES]
    for cand in fallback_candidates:
        if os.path.exists(cand):
            print(f"[info] Default curve not found: {preferred}")
            print(f"[info] Falling back to: {cand}")
            return cand

    return preferred


def _build_sensitivity_curve_specs(data_dir: str, prefix: str) -> List[Dict[str, str]]:
    # Curves used for baseline-vs-sensitivity comparison.
    specs = []
    for key, label, category, curve_suffix, linestyle in SENSITIVITY_CASES:
        specs.append(
            {
                "key": key,
                "label": label,
                "category": category,
                "path": os.path.join(data_dir, f"{prefix}{curve_suffix}"),
                "linestyle": linestyle,
            }
        )
    return specs


def _build_biomass_sensitivity_curve_specs(data_dir: str, prefix: str) -> List[Dict[str, str]]:
    # Baseline 100% supply is the original scenario.
    specs = [
        {
            "key": "bio_100",
            "label": "Biomass supply 100%",
            "category": "bio100",
            "supply_ratio": 1.00,
            "path": os.path.join(data_dir, BIOMASS_BASELINE_CURVE_NAME),
            "result_xlsx": os.path.join(data_dir, BIOMASS_BASELINE_RESULTS_NAME),
            "linestyle": "-",
        }
    ]
    for key, label, category, supply_ratio, curve_suffix, result_suffix, linestyle in BIOMASS_SENSITIVITY_CASES:
        specs.append(
            {
                "key": key,
                "label": label,
                "category": category,
                "supply_ratio": supply_ratio,
                "path": os.path.join(data_dir, f"{prefix}{curve_suffix}"),
                "result_xlsx": os.path.join(data_dir, f"{prefix}{result_suffix}"),
                "linestyle": linestyle,
            }
        )
    return specs


def _curve_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(df["target_abatement"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["marginal_cost_diff"], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    uniq_x, uniq_idx = np.unique(x, return_index=True)
    return uniq_x, y[uniq_idx]


def _interp_to_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if x.size < 2:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, x, y, left=np.nan, right=np.nan)


def _build_biomass_share_series(result_xlsx: str, mode: str = "MILP") -> pd.DataFrame:
    xl = pd.ExcelFile(result_xlsx)
    mode_tag = "pm" if str(mode).strip().upper() == "MILP" else "pl"
    plant_sheets = sorted([s for s in xl.sheet_names if s.startswith(mode_tag)], key=_sheet_sort_key)

    rows = []
    for sname in plant_sheets:
        pdf = pd.read_excel(result_xlsx, sheet_name=sname)
        needed = ["target_id", "target_abatement", "bc_abatement_ton", "total_abatement_ton"]
        missing = [c for c in needed if c not in pdf.columns]
        if missing or pdf.empty:
            continue

        target_id = pd.to_numeric(pdf["target_id"], errors="coerce").iloc[0]
        target = pd.to_numeric(pdf["target_abatement"], errors="coerce").iloc[0]
        bc_total = pd.to_numeric(pdf["bc_abatement_ton"], errors="coerce").fillna(0.0).sum()
        total_abate = pd.to_numeric(pdf["total_abatement_ton"], errors="coerce").fillna(0.0).sum()
        share = bc_total / total_abate if total_abate and np.isfinite(total_abate) else np.nan

        # Fuel consumption: bc_use_gj = biomass energy (GJ)
        if "bc_use_gj" in pdf.columns:
            bc_use_gj_sum = float(pd.to_numeric(pdf["bc_use_gj"], errors="coerce").fillna(0.0).sum())
        else:
            bc_use_gj_sum = np.nan

        # EC abatement → energy saving (0.0901 tCO₂/GJ coal emission factor)
        if "ec_abatement_ton" in pdf.columns:
            ec_abate_sum = float(pd.to_numeric(pdf["ec_abatement_ton"], errors="coerce").fillna(0.0).sum())
            ec_saving_gj = ec_abate_sum / 0.0901
        else:
            ec_saving_gj = np.nan

        rows.append(
            {
                "sheet": sname,
                "target_id": int(target_id) if np.isfinite(target_id) else np.nan,
                "target_abatement": float(target) if np.isfinite(target) else np.nan,
                "bc_abatement_ton": float(bc_total),
                "total_abatement_ton": float(total_abate),
                "bc_share": float(share),
                "bc_use_gj": bc_use_gj_sum,
                "ec_saving_gj": ec_saving_gj,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["sheet", "target_id", "target_abatement", "bc_abatement_ton", "total_abatement_ton", "bc_share", "bc_use_gj"])

    out = pd.DataFrame(rows)
    out = out.sort_values(["target_abatement", "target_id"], na_position="last").reset_index(drop=True)
    return out


def _plot_sensitivity_comparison(
    curve_specs: List[Dict[str, str]],
    out_overlay_png: str,
    out_delta_png: str,
    out_envelope_png: str,
) -> Dict[str, str]:
    loaded = []
    for spec in curve_specs:
        p = spec["path"]
        if not os.path.exists(p):
            print(f"[skip] Missing sensitivity curve: {p}")
            continue
        try:
            df = _load_curve(p, spec["label"])
            if "status" in df.columns:
                ok = df["status"].astype(str).str.lower() == "optimal"
                if ok.any():
                    df = df[ok].copy().reset_index(drop=True)
            loaded.append({**spec, "df": df})
        except Exception as exc:
            print(f"[skip] Failed loading {p}: {exc}")

    baseline = [s for s in loaded if s["category"] == "baseline"]
    if not baseline:
        raise FileNotFoundError("Baseline curve for sensitivity comparison is missing.")
    baseline = baseline[0]

    color_map = {
        "baseline": "#0B1F3A",
        "capex": "#E67E22",
        "coal": "#C0392B",
        "discount": "#1E8449",
    }

    # 1) Overlay MAC curves with baseline emphasis.
    fig1, ax1 = plt.subplots(1, 1, figsize=(12.8, 5.4))
    for s in loaded:
        x, y = _curve_xy(s["df"])
        if x.size == 0:
            continue
        cat = s["category"]
        is_base = cat == "baseline"
        ax1.plot(
            x,
            y,
            color=color_map.get(cat, "#7F7F7F"),
            linewidth=3.8 if is_base else 2.0,
            linestyle=s.get("linestyle", "-"),
            alpha=1.0 if is_base else 0.78,
            marker="o" if is_base else None,
            markersize=4.0 if is_base else 0,
            zorder=8 if is_base else 3,
            label=s["label"],
        )
    ax1.set_xlabel("Target abatement (t CO$_2$)")
    ax1.set_ylabel("Marginal abatement cost (USD t CO$_2^{-1}$)")
    ax1.set_title("MAC comparison: baseline vs sensitivity scenarios")
    ax1.grid(alpha=0.2, linestyle="--")
    ax1.legend(frameon=False, ncol=2)
    fig1.tight_layout()
    os.makedirs(os.path.dirname(out_overlay_png), exist_ok=True)
    save_figure_bundle(fig1, out_overlay_png, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # Build baseline interpolation grid for delta/envelope plots.
    xb, yb = _curve_xy(baseline["df"])
    if xb.size < 2:
        raise ValueError("Baseline curve has insufficient valid points for sensitivity comparison.")
    grid = xb
    yb_grid = _interp_to_grid(xb, yb, grid)

    # 2) Delta MAC relative to baseline.
    fig2, ax2 = plt.subplots(1, 1, figsize=(12.8, 5.0))
    ax2.axhline(0.0, color="#555555", linewidth=1.0, alpha=0.75)
    for s in loaded:
        if s["category"] == "baseline":
            continue
        x, y = _curve_xy(s["df"])
        ys = _interp_to_grid(x, y, grid)
        delta = ys - yb_grid
        finite = np.isfinite(delta)
        if not finite.any():
            continue
        ax2.plot(
            grid[finite],
            delta[finite],
            color=color_map.get(s["category"], "#7F7F7F"),
            linewidth=2.0,
            linestyle=s.get("linestyle", "-"),
            alpha=0.9,
            label=s["label"],
        )
    ax2.set_xlabel("Target abatement (t CO$_2$)")
    ax2.set_ylabel("Delta MAC (scenario - baseline)")
    ax2.set_title("Marginal cost deviation from baseline")
    ax2.grid(alpha=0.2, linestyle="--")
    ax2.legend(frameon=False, ncol=2)
    fig2.tight_layout()
    save_figure_bundle(fig2, out_delta_png, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # 3) Envelope by sensitivity category.
    fig3, ax3 = plt.subplots(1, 1, figsize=(12.8, 5.0))
    ax3.plot(grid, yb_grid, color=color_map["baseline"], linewidth=3.5, label="Baseline", zorder=7)
    categories = ["capex", "coal", "discount"]
    for cat in categories:
        members = [s for s in loaded if s["category"] == cat]
        if not members:
            continue
        arr = []
        for s in members:
            x, y = _curve_xy(s["df"])
            ys = _interp_to_grid(x, y, grid)
            arr.append(ys)
        if not arr:
            continue
        mat = np.vstack(arr)
        y_min = np.nanmin(mat, axis=0)
        y_max = np.nanmax(mat, axis=0)
        y_mid = np.nanmean(mat, axis=0)
        finite = np.isfinite(y_min) & np.isfinite(y_max)
        if not finite.any():
            continue
        ax3.fill_between(
            grid[finite],
            y_min[finite],
            y_max[finite],
            color=color_map[cat],
            alpha=0.18,
            label=f"{cat.upper()} range",
            zorder=2,
        )
        mid_f = np.isfinite(y_mid)
        if mid_f.any():
            ax3.plot(
                grid[mid_f],
                y_mid[mid_f],
                color=color_map[cat],
                linewidth=1.8,
                alpha=0.85,
                linestyle="-",
                label=f"{cat.upper()} mean",
                zorder=4,
            )
    ax3.set_xlabel("Target abatement (t CO$_2$)")
    ax3.set_ylabel("Marginal abatement cost (USD t CO$_2^{-1}$)")
    ax3.set_title("Sensitivity envelope by category (with baseline)")
    ax3.grid(alpha=0.2, linestyle="--")
    ax3.legend(frameon=False, ncol=2)
    fig3.tight_layout()
    save_figure_bundle(fig3, out_envelope_png, dpi=300, bbox_inches="tight")
    plt.close(fig3)

    return {
        "overlay": out_overlay_png,
        "delta": out_delta_png,
        "envelope": out_envelope_png,
    }


def _plot_biomass_sensitivity_comparison(
    curve_specs: List[Dict[str, str]],
    out_png: str,
) -> str:
    loaded = []
    for spec in curve_specs:
        p = spec["path"]
        xlsx = spec["result_xlsx"]
        if not os.path.exists(p):
            print(f"[skip] Missing biomass curve: {p}")
            continue
        if not os.path.exists(xlsx):
            print(f"[skip] Missing biomass results workbook: {xlsx}")
            continue
        try:
            curve = _load_curve(p, spec["label"])
            share_df = _build_biomass_share_series(xlsx)
            if curve.empty or share_df.empty:
                continue
            merged = pd.merge(
                curve[["target_abatement", "total_cost", "marginal_cost_diff"]],
                share_df[["target_abatement", "bc_share", "bc_use_gj", "ec_saving_gj"]],
                on="target_abatement",
                how="inner",
            ).sort_values("target_abatement").reset_index(drop=True)
            if merged.empty:
                continue
            loaded.append({**spec, "df": merged})
        except Exception as exc:
            print(f"[skip] Failed loading biomass sensitivity case {p}: {exc}")

    if not loaded:
        raise FileNotFoundError("No biomass sensitivity curves could be loaded.")

    baseline = [s for s in loaded if s["supply_ratio"] >= 0.999]
    if not baseline:
        raise FileNotFoundError("Biomass 100% baseline curve is missing.")
    baseline = baseline[0]

    # Color palette: muted, publication-friendly blues from light to dark.
    color_map = {
        0.25: "#D7E8F6",
        0.50: "#9CC4E4",
        0.75: "#4D91C8",
        1.00: "#163A5F",
    }
    share_edge = {
        0.25: "#7AAAD1",
        0.50: "#4F86B5",
        0.75: "#2E6E9E",
        1.00: "#0F2F4E",
    }

    # Load total baseline energy demand (GJ) to compute coal consumption
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_s = os.path.join(script_dir, DATA_DIR_NAME)
    coord_path = os.path.join(data_dir_s, "1885_coal_power_plants_original.xlsx")
    total_energy_gj = 0.0
    if os.path.exists(coord_path):
        pp_raw = pd.read_excel(coord_path, sheet_name="coal_plant_i")
        energy_col = [c for c in pp_raw.columns if "energy_demand" in str(c).lower() or "energy" == str(c).lower()]
        if energy_col:
            total_energy_gj = float(pd.to_numeric(pp_raw[energy_col[0]], errors="coerce").fillna(0.0).sum())

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3,
        1,
        figsize=(13.2, 12.0),
        sharex=False,
        gridspec_kw={"height_ratios": [1.0, 0.9, 0.9]},
    )

    # Top: MAC overlay.
    for s in loaded:
        x = pd.to_numeric(s["df"]["target_abatement"], errors="coerce").to_numpy()
        y = pd.to_numeric(s["df"]["marginal_cost_diff"], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            continue
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        ratio = float(s["supply_ratio"])
        is_base = ratio >= 0.999
        ax_top.plot(
            x,
            y,
            color=color_map.get(ratio, "#7F7F7F"),
            linewidth=4.2 if is_base else 2.0,
            linestyle=s.get("linestyle", "-"),
            alpha=1.0 if is_base else 0.82,
            marker="o" if is_base else None,
            markersize=4.4 if is_base else 0,
            zorder=10 if is_base else 4,
            label=s["label"],
        )

    ax_top.set_ylabel("Marginal abatement cost (USD t CO$_2^{-1}$)")
    ax_top.set_title("Biomass supply sensitivity: MAC curves and BC abatement share")
    ax_top.grid(alpha=0.2, linestyle="--")
    ax_top.legend(frameon=False, ncol=2)

    # Bottom: BC share bars by target.
    base_df = baseline["df"].copy().sort_values("target_abatement")
    x_base = base_df["target_abatement"].to_numpy(dtype=float)
    valid_x = np.isfinite(x_base)
    x_base = x_base[valid_x]
    if x_base.size == 0:
        raise ValueError("Biomass baseline has no valid target points for share plot.")

    x_min = float(np.nanmin(x_base))
    x_max = float(np.nanmax(x_base))
    x_range = max(x_max - x_min, 1.0)
    dx = np.nanmedian(np.diff(x_base)) if x_base.size > 1 else x_range / 12.0
    dx = dx if np.isfinite(dx) and dx > 0 else x_range / 12.0

    ratios_sorted = sorted([float(s["supply_ratio"]) for s in loaded])
    n = len(ratios_sorted)
    offsets = np.linspace(-0.24, 0.24, n) * dx
    width_map = {
        0.25: 0.10 * dx,
        0.50: 0.14 * dx,
        0.75: 0.18 * dx,
        1.00: 0.22 * dx,
    }

    # Draw a faint reference line for the baseline share trend.
    baseline_share = base_df["bc_share"].to_numpy(dtype=float)
    ax_bot.plot(
        x_base,
        baseline_share,
        color=color_map[1.0],
        linewidth=2.2,
        alpha=0.95,
        marker="o",
        markersize=3.8,
        label="100% supply trend",
        zorder=5,
    )

    for idx, s in enumerate(sorted(loaded, key=lambda d: float(d["supply_ratio"]))):
        ratio = float(s["supply_ratio"])
        sdf = s["df"].sort_values("target_abatement")
        x = sdf["target_abatement"].to_numpy(dtype=float)
        y = sdf["bc_share"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            continue
        width = width_map.get(ratio, 0.22 * dx)
        x_shift = x + offsets[idx]
        ax_bot.bar(
            x_shift,
            y,
            width=width,
            color=color_map.get(ratio, "#7F7F7F"),
            alpha=0.42 if ratio < 1.0 else 0.72,
            edgecolor=share_edge.get(ratio, "#444444"),
            linewidth=0.75 if ratio < 1.0 else 1.0,
            label=s["label"],
            zorder=3 if ratio < 1.0 else 6,
        )

    ax_bot.set_xlabel("Target abatement (t CO$_2$)")
    ax_bot.set_ylabel("BC abatement share")
    ax_bot.set_ylim(0.0, 1.0)
    ax_bot.set_xlim(x_min - 0.8 * dx, x_max + 0.8 * dx)
    ax_bot.grid(alpha=0.2, linestyle="--")
    ax_bot.legend(frameon=False, ncol=2)

    # ── Middle panel: Stacked fuel consumption (biomass + coal + EC saving GJ) ──
    # Use the baseline (100% supply) data for fuel bars
    base_df_sorted = baseline["df"].copy().sort_values("target_abatement")
    x_targets = base_df_sorted["target_abatement"].to_numpy(dtype=float)

    # Biomass use from the baseline share series (bc_use_gj)
    bio_raw = base_df_sorted["bc_use_gj"].to_numpy(dtype=float)
    # EC energy saving: ec_abatement_ton / 0.0901 (tCO₂/GJ coal emission factor)
    ec_raw = base_df_sorted["ec_saving_gj"].to_numpy(dtype=float)
    # Mask away NaN targets
    ok = np.isfinite(bio_raw) & np.isfinite(ec_raw)
    x_targets = x_targets[ok]
    bio_gj = bio_raw[ok]
    ec_save_gj = ec_raw[ok]
    # Coal GJ = total_energy - bio_gj - ec_saving_gj
    coal_gj = total_energy_gj - bio_gj - ec_save_gj
    coal_gj = np.maximum(coal_gj, 0.0)

    bar_width = np.diff(x_targets, prepend=0).mean() * 0.70 if len(x_targets) > 1 else 0.5e9
    ax_mid.bar(x_targets, bio_gj, width=bar_width,
               color="#1F77B4", alpha=0.85, edgecolor="white", linewidth=0.3,
               label="Biomass", zorder=3)
    ax_mid.bar(x_targets, coal_gj, width=bar_width,
               bottom=bio_gj, color="#BDBDBD", alpha=0.70, edgecolor="white", linewidth=0.3,
               label="Coal", zorder=3)
    ax_mid.bar(x_targets, ec_save_gj, width=bar_width,
               bottom=bio_gj + coal_gj, color="#D62728", alpha=0.55, edgecolor="white", linewidth=0.3,
               label="EC energy saving", zorder=3)

    # Horizontal line showing the baseline total energy ceiling
    ax_mid.axhline(total_energy_gj, color="#D62728", linewidth=1.5, linestyle="--", alpha=0.7, zorder=4)
    ax_mid.text(x_targets[-1] * 0.95, total_energy_gj * 1.015,
                f"Baseline total {total_energy_gj/1e9:.2f} GJ",
                fontsize=9, color="#D62728", ha="right", va="bottom")

    ax_mid.set_ylabel("Fuel consumption (GJ)")
    ax_mid.set_title("", fontsize=12)
    ax_mid.grid(alpha=0.2, linestyle="--")
    ax_mid.legend(frameon=False, ncol=3, fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    save_figure_bundle(fig, out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _plot_fuel_standalone(data_dir: str, out_png: str) -> str:
    """Standalone single-panel figure: stacked fuel consumption (biomass + coal + EC saving)."""
    coord_path = os.path.join(data_dir, "1885_coal_power_plants_original.xlsx")
    total_energy_gj = 0.0
    if os.path.exists(coord_path):
        pp_raw = pd.read_excel(coord_path, sheet_name="coal_plant_i")
        energy_col = [c for c in pp_raw.columns if "energy_demand" in str(c).lower() or "energy" == str(c).lower()]
        if energy_col:
            total_energy_gj = float(pd.to_numeric(pp_raw[energy_col[0]], errors="coerce").fillna(0.0).sum())

    # Load baseline share series (has bc_use_gj and ec_saving_gj)
    xlsx = os.path.join(data_dir, "gurobi_original_results.xlsx")
    share_df = _build_biomass_share_series(xlsx)
    if share_df.empty:
        raise ValueError("No share data for fuel standalone plot.")

    base = share_df.sort_values("target_abatement")
    x = base["target_abatement"].to_numpy(dtype=float)
    bio = base["bc_use_gj"].to_numpy(dtype=float)
    ec_save = base["ec_saving_gj"].to_numpy(dtype=float)
    ok = np.isfinite(bio) & np.isfinite(ec_save)
    x = x[ok]
    bio = bio[ok]
    ec_save = ec_save[ok]
    coal = total_energy_gj - bio - ec_save
    coal = np.maximum(coal, 0.0)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.6))
    bar_w = np.diff(x, prepend=0).mean() * 0.65 if len(x) > 1 else 0.5e9

    ax.bar(x, bio, width=bar_w, color="#1F77B4", alpha=0.85, edgecolor="white", linewidth=0.3,
           label="Biomass", zorder=3)
    ax.bar(x, coal, width=bar_w, bottom=bio, color="#BDBDBD", alpha=0.70, edgecolor="white", linewidth=0.3,
           label="Coal", zorder=3)
    ax.bar(x, ec_save, width=bar_w, bottom=bio+coal, color="#D62728", alpha=0.55, edgecolor="white", linewidth=0.3,
           label="EC energy saving", zorder=3)

    ax.axhline(total_energy_gj, color="#D62728", linewidth=1.8, linestyle="--", alpha=0.8, zorder=4)
    ax.text(x[-1] * 0.95, total_energy_gj * 1.015,
            f"Baseline total {total_energy_gj/1e9:.2f} GJ",
            fontsize=10, color="#D62728", ha="right", va="bottom")

    ax.set_xlabel("Target abatement (t CO$_2$)", fontsize=12)
    ax.set_ylabel("Fleet fuel consumption (GJ)", fontsize=12)
    ax.set_title("", fontsize=13)
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(
        frameon=False,
        ncol=1,               
        fontsize=10,
        loc="center left",    
        bbox_to_anchor=(1.01, 0.5)  
    )

    ax.set_xlim(x[0] - 0.5 * bar_w, x[-1] + 0.5 * bar_w)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    save_figure_bundle(fig, out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _plot_biomass_bc_share_simple(
    curve_specs: List[Dict[str, str]],
    out_png: str,
) -> str:
    loaded = []
    for spec in curve_specs:
        p = spec["path"]
        xlsx = spec["result_xlsx"]
        if not os.path.exists(p):
            print(f"[skip] Missing biomass curve: {p}")
            continue
        if not os.path.exists(xlsx):
            print(f"[skip] Missing biomass results workbook: {xlsx}")
            continue
        try:
            curve = _load_curve(p, spec["label"])
            share_df = _build_biomass_share_series(xlsx)
            if curve.empty or share_df.empty:
                continue
            merged = pd.merge(
                curve[["target_abatement", "marginal_cost_diff"]],
                share_df[["target_abatement", "bc_share"]],
                on="target_abatement",
                how="inner",
            ).sort_values("target_abatement").reset_index(drop=True)
            if merged.empty:
                continue
            loaded.append({**spec, "df": merged})
        except Exception as exc:
            print(f"[skip] Failed loading biomass BC-share case {p}: {exc}")

    if not loaded:
        raise FileNotFoundError("No biomass sensitivity curves could be loaded for BC-share simple plot.")

    baseline = [s for s in loaded if s["supply_ratio"] >= 0.999]
    if not baseline:
        raise FileNotFoundError("Biomass 100% baseline curve is missing for BC-share simple plot.")
    baseline = baseline[0]

    color_map = {
        0.25: "#9EC4E8",
        0.50: "#5A97C9",
        0.75: "#2F6FA3",
        1.00: "#12324A",
    }

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12.8, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.05, 0.95]})

    for s in sorted(loaded, key=lambda d: float(d["supply_ratio"])):
        ratio = float(s["supply_ratio"])
        sdf = s["df"].sort_values("target_abatement")
        x = sdf["target_abatement"].to_numpy(dtype=float)
        y = sdf["marginal_cost_diff"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            continue
        is_base = ratio >= 0.999
        ax_top.plot(
            x,
            y,
            color=color_map.get(ratio, "#7F7F7F"),
            linewidth=3.4 if is_base else 2.0,
            marker="o" if is_base else None,
            markersize=4.2 if is_base else 0,
            alpha=1.0 if is_base else 0.84,
            label=s["label"],
        )

    ax_top.set_ylabel("Marginal abatement cost (USD t CO$_2^{-1}$)")
    ax_top.set_title("Biomass supply sensitivity: compact MAC and BC-share view")
    ax_top.grid(alpha=0.2, linestyle="--")
    ax_top.legend(frameon=False, ncol=2)

    for s in sorted(loaded, key=lambda d: float(d["supply_ratio"])):
        ratio = float(s["supply_ratio"])
        sdf = s["df"].sort_values("target_abatement")
        x = sdf["target_abatement"].to_numpy(dtype=float)
        y = sdf["bc_share"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            continue
        is_base = ratio >= 0.999
        ax_bot.plot(
            x,
            y,
            color=color_map.get(ratio, "#7F7F7F"),
            linewidth=3.0 if is_base else 2.1,
            marker="o" if is_base else None,
            markersize=4.0 if is_base else 0,
            alpha=1.0 if is_base else 0.88,
            label=s["label"],
        )

    ax_bot.set_xlabel("Target abatement (t CO$_2$)")
    ax_bot.set_ylabel("BC abatement share")
    ax_bot.set_ylim(0.0, 1.0)
    ax_bot.grid(alpha=0.2, linestyle="--")
    ax_bot.legend(frameon=False, ncol=2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    save_figure_bundle(fig, out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _plot_biomass_dualaxis_selected_points(
    curve_specs: List[Dict[str, str]],
    out_png: str,
    selected_targets: List[float],
) -> str:
    loaded = []
    for spec in curve_specs:
        p = spec["path"]
        xlsx = spec["result_xlsx"]
        if not os.path.exists(p):
            print(f"[skip] Missing biomass curve: {p}")
            continue
        if not os.path.exists(xlsx):
            print(f"[skip] Missing biomass results workbook: {xlsx}")
            continue
        try:
            curve = _load_curve(p, spec["label"])
            share_df = _build_biomass_share_series(xlsx)
            if curve.empty or share_df.empty:
                continue
            merged = pd.merge(
                curve[["target_abatement", "marginal_cost_diff"]],
                share_df[["target_abatement", "bc_share"]],
                on="target_abatement",
                how="inner",
            ).sort_values("target_abatement").reset_index(drop=True)
            if merged.empty:
                continue
            loaded.append({**spec, "df": merged})
        except Exception as exc:
            print(f"[skip] Failed loading biomass dual-axis case {p}: {exc}")

    if not loaded:
        raise FileNotFoundError("No biomass sensitivity curves could be loaded for dual-axis plot.")

    color_map = {
        0.25: "#9EC4E8",
        0.50: "#5A97C9",
        0.75: "#2F6FA3",
        1.00: "#12324A",
    }
    width_scale = {
        0.25: 0.22,
        0.50: 0.36,
        0.75: 0.52,
        1.00: 0.68,
    }

    fig, ax1 = plt.subplots(1, 1, figsize=(12.6, 5.8))
    ax2 = ax1.twinx()

    # Left axis: MAC curves (same visual language as panel-a style).
    for s in sorted(loaded, key=lambda d: float(d["supply_ratio"])):
        ratio = float(s["supply_ratio"])
        sdf = s["df"].sort_values("target_abatement")
        x = sdf["target_abatement"].to_numpy(dtype=float)
        y = sdf["marginal_cost_diff"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            continue
        is_base = ratio >= 0.999
        ax1.plot(
            x,
            y,
            color=color_map.get(ratio, "#7F7F7F"),
            linewidth=3.5 if is_base else 2.0,
            linestyle=s.get("linestyle", "-"),
            alpha=1.0 if is_base else 0.86,
            marker="o" if is_base else None,
            markersize=3.8 if is_base else 0,
            zorder=8 if is_base else 5,
            label=s["label"],
        )

    # Right axis: BC share bars at selected targets, overlapped at same x center.
    selected = np.array(sorted(selected_targets), dtype=float)
    if selected.size == 0:
        raise ValueError("selected_targets cannot be empty for dual-axis plot.")

    x_span = max(float(np.nanmax(selected) - np.nanmin(selected)), 1.0)
    base_width = x_span / max(30.0, 4.0 * float(selected.size))

    # Draw wider bars first, then narrower bars on top so all are visible.
    by_ratio_desc = sorted(loaded, key=lambda d: float(d["supply_ratio"]), reverse=True)
    for s in by_ratio_desc:
        ratio = float(s["supply_ratio"])
        sdf = s["df"].sort_values("target_abatement")
        x_all = sdf["target_abatement"].to_numpy(dtype=float)
        y_all = sdf["bc_share"].to_numpy(dtype=float)
        mask_all = np.isfinite(x_all) & np.isfinite(y_all)
        x_all = x_all[mask_all]
        y_all = y_all[mask_all]
        if x_all.size == 0:
            continue

        y_pick = []
        for x0 in selected:
            idx = int(np.argmin(np.abs(x_all - x0)))
            y_pick.append(float(y_all[idx]) * 100.0)
        y_pick = np.array(y_pick, dtype=float)

        bar_w = base_width * width_scale.get(ratio, 0.3)
        ax2.bar(
            selected,
            y_pick,
            width=bar_w,
            color=color_map.get(ratio, "#7F7F7F"),
            alpha=0.2 + 0.18 * ratio,
            edgecolor=color_map.get(ratio, "#4F4F4F"),
            linewidth=0.8,
            zorder=2,
            label=f"BC share ({int(round(ratio * 100))}%)",
        )

    ax1.set_xlabel("Target abatement (t CO$_2$)")
    ax1.set_ylabel("Marginal abatement cost (USD t CO$_2^{-1}$)")
    ax2.set_ylabel("BC abatement share (%)")
    ax2.set_ylim(0.0, 100.0)
    ax1.set_title("Biomass supply sensitivity: MAC with selected BC-share points")
    ax1.grid(alpha=0.22, linestyle="--")

    # Show only requested target nodes as x ticks for clean paper-style layout.
    ax1.set_xticks(selected)
    ax1.set_xticklabels([f"{v/1e9:.1f}" for v in selected])
    ax1.set_xlabel("Target abatement (GtCO$_2$)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, ncol=2, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    save_figure_bundle(fig, out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _plot(milp: pd.DataFrame, lp: pd.DataFrame = None) -> None:
    x_m = milp["target_abatement"].to_numpy()
    c_m = milp["total_cost"].to_numpy()
    mac_m = milp["marginal_cost_diff"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(
        x_m,
        c_m,
        color="#1F77B4",
        linewidth=2.8,
        marker="o",
        markersize=4,
        label="MILP total cost",
    )
    if lp is not None:
        axes[0].plot(
            lp["target_abatement"].to_numpy(),
            lp["total_cost"].to_numpy(),
            color="#D62728",
            linewidth=2.2,
            linestyle="--",
            marker="s",
            markersize=3.5,
            label="LP-relax total cost",
        )

    axes[0].set_ylabel("Total sectoral annualized cost")
    axes[0].set_title("Cost-abatement frontier from fleet-wide joint optimization")
    axes[0].grid(alpha=0.2, linestyle="--")
    axes[0].legend(frameon=False)

    axes[1].plot(
        x_m,
        mac_m,
        color="#1F77B4",
        linewidth=2.8,
        marker="o",
        markersize=4,
        label="MILP MAC (difference quotient)",
    )
    if lp is not None:
        axes[1].plot(
            lp["target_abatement"].to_numpy(),
            lp["marginal_cost_diff"].to_numpy(),
            color="#D62728",
            linewidth=2.2,
            linestyle="--",
            marker="s",
            markersize=3.5,
            label="LP-relax MAC",
        )

    axes[1].set_ylabel("Marginal abatement cost (USD t CO$_2^{-1}$)")
    axes[1].set_xlabel("Target abatement (GtCO$_2$)")
    axes[1].set_title("Standard MAC curve (difference quotient)")
    axes[1].grid(alpha=0.2, linestyle="--")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    plt.close(fig)


def _plot_logistics_dualaxis(milp: pd.DataFrame, out_png: str) -> None:
    required = {
        "target_abatement",
        "total_co2_stored_ton",
        "weighted_co2_distance_km",
        "weighted_transport_distance_km",
    }
    missing = [c for c in required if c not in milp.columns]
    if missing:
        raise ValueError(
            "Cannot plot logistics dual-axis figure. "
            f"Missing columns in MILP curve CSV: {sorted(missing)}"
        )

    x = pd.to_numeric(milp["target_abatement"], errors="coerce").to_numpy()
    y_store = pd.to_numeric(milp["total_co2_stored_ton"], errors="coerce").to_numpy()
    y_co2_dist = pd.to_numeric(milp["weighted_co2_distance_km"], errors="coerce").to_numpy()
    y_bio_dist = pd.to_numeric(milp["weighted_transport_distance_km"], errors="coerce").to_numpy()

    valid = np.isfinite(x) & np.isfinite(y_store) & np.isfinite(y_co2_dist) & np.isfinite(y_bio_dist)
    x = x[valid]
    y_store = y_store[valid]
    y_co2_dist = y_co2_dist[valid]
    y_bio_dist = y_bio_dist[valid]

    if x.size == 0:
        raise ValueError("No valid rows available to plot logistics dual-axis figure.")

    order = np.argsort(x)
    x = x[order]
    y_store = y_store[order]
    y_co2_dist = y_co2_dist[order]
    y_bio_dist = y_bio_dist[order]

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5.4))
    ax2 = ax1.twinx()

    l1 = ax1.plot(
        x,
        y_store,
        color="#2CA02C",
        linewidth=2.8,
        marker="o",
        markersize=4,
        label="Total CO2 stored",
    )
    l2 = ax2.plot(
        x,
        y_co2_dist,
        color="#D62728",
        linewidth=2.4,
        linestyle="--",
        marker="s",
        markersize=3.8,
        label="Avg CO2 transport distance",
    )
    l3 = ax2.plot(
        x,
        y_bio_dist,
        color="#1F77B4",
        linewidth=2.4,
        linestyle=":",
        marker="^",
        markersize=4,
        label="Avg biomass transport distance",
    )

    ax1.set_xlabel("Target abatement (GtC$_2$)")
    ax1.set_ylabel("Total CO2 stored (ton)", color="#2CA02C")
    ax2.set_ylabel("Average transport distance (km)")
    ax1.tick_params(axis="y", labelcolor="#2CA02C")
    ax1.set_title("Target abatement vs storage and transport distances")
    ax1.grid(alpha=0.2, linestyle="--")

    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, frameon=False, loc="best")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    save_figure_bundle(fig, out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _sheet_sort_key(name: str) -> Tuple[int, str]:
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        return int(digits), name
    return (10**9), name


def _build_scenario_sorted_points(plant_df: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "plant_id",
        "ec_abatement_ton",
        "ec_cost",
        "bc_abatement_ton",
        "bc_cost",
        "cc_capture_ton",
        "cc_cost",
    ]
    missing = [c for c in needed if c not in plant_df.columns]
    if missing:
        raise ValueError(f"Plant scenario sheet missing columns: {sorted(missing)}")

    rows = []
    tech_map = [
        ("EC", "ec_abatement_ton", "ec_cost"),
        ("BC", "bc_abatement_ton", "bc_cost"),
        ("CC", "cc_capture_ton", "cc_cost"),
    ]
    for _, r in plant_df.iterrows():
        plant_id = r["plant_id"]
        for tech, a_col, c_col in tech_map:
            abate = pd.to_numeric(r[a_col], errors="coerce")
            cost = pd.to_numeric(r[c_col], errors="coerce")
            if not np.isfinite(abate) or not np.isfinite(cost):
                continue
            if abate <= 0:
                continue
            unit = cost / abate
            rows.append(
                {
                    "plant_id": plant_id,
                    "tech": tech,
                    "abatement_ton": float(abate),
                    "cost": float(cost),
                    "unit_cost": float(unit),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["plant_id", "tech", "abatement_ton", "cost", "unit_cost", "cum_abatement_ton"])

    out = pd.DataFrame(rows).sort_values(["unit_cost", "abatement_ton"], ascending=[True, False]).reset_index(drop=True)
    out["cum_abatement_ton"] = out["abatement_ton"].cumsum()
    return out


def _plot_all_scenario_sorted_macc(result_xlsx: str, out_dir: str, mode: str = "MILP") -> List[str]:
    xl = pd.ExcelFile(result_xlsx)
    mode_tag = "pm" if str(mode).strip().upper() == "MILP" else "pl"
    plant_sheets = sorted([s for s in xl.sheet_names if s.startswith(mode_tag)], key=_sheet_sort_key)
    if not plant_sheets:
        return []

    os.makedirs(out_dir, exist_ok=True)
    outputs = []

    for sname in plant_sheets:
        pdf = pd.read_excel(result_xlsx, sheet_name=sname)
        sorted_pts = _build_scenario_sorted_points(pdf)
        if sorted_pts.empty:
            continue

        target_id = int(pd.to_numeric(pdf.get("target_id", pd.Series([np.nan])).iloc[0], errors="coerce")) if len(pdf) else np.nan
        target = float(pd.to_numeric(pdf.get("target_abatement", pd.Series([np.nan])).iloc[0], errors="coerce")) if len(pdf) else np.nan

        widths = sorted_pts["abatement_ton"].to_numpy()
        heights = sorted_pts["unit_cost"].to_numpy()
        x_right = np.cumsum(widths)
        x_left = x_right - widths

        fig, ax = plt.subplots(1, 1, figsize=(11.5, 5.2))

        # Step MACC where each point contributes a rectangle:
        # width = abatement contribution, height = unit abatement cost.
        tech_color = {"EC": "#2CA02C", "BC": "#1F77B4", "CC": "#D62728"}
        face_colors = sorted_pts["tech"].map(tech_color).fillna("#7F7F7F").to_numpy()
        ax.bar(
            x_left,
            heights,
            width=widths,
            align="edge",
            color=face_colors,
            edgecolor="white",
            linewidth=0.15,
            alpha=0.55,
            zorder=2,
        )

        # Draw step envelope and fill the staircase area under it.
        x_step = np.concatenate(([0.0], x_right))
        y_step = np.concatenate(([heights[0]], heights))
        ax.step(x_step, y_step, where="post", color="#1F77B4", linewidth=1.8, zorder=4)
        ax.fill_between(x_step, 0.0, y_step, step="post", color="#1F77B4", alpha=0.12, zorder=1)

        ax.axhline(0.0, color="#444444", linewidth=0.9, alpha=0.7)
        ax.grid(alpha=0.2, linestyle="--")
        ax.set_xlabel("Cumulative abatement (step width = abatement contribution)")
        ax.set_ylabel("Unit abatement cost (USD t CO$_2^{-1}$)")
        ax.set_title(
            f"Scenario sorted unit-cost curve ({mode_tag.upper()}{target_id:02d}) | target={target:,.0f}"
            if np.isfinite(target_id) and np.isfinite(target)
            else f"Scenario sorted unit-cost curve ({sname})"
        )

        handles = [
            plt.Line2D([], [], color="#2CA02C", marker="o", linestyle="", markersize=5, label="EC points"),
            plt.Line2D([], [], color="#1F77B4", marker="o", linestyle="", markersize=5, label="BC points"),
            plt.Line2D([], [], color="#D62728", marker="o", linestyle="", markersize=5, label="CC points"),
        ]
        ax.legend(handles=handles, frameon=False, loc="best")

        out_png = os.path.join(out_dir, f"{sname}_sorted_unit_cost_curve.png")
        fig.tight_layout()
        save_figure_bundle(fig, out_png, dpi=260, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_png)

    return outputs


def _build_scenario_tech_summary(result_xlsx: str, mode: str = "MILP") -> pd.DataFrame:
    xl = pd.ExcelFile(result_xlsx)
    mode_tag = "pm" if str(mode).strip().upper() == "MILP" else "pl"
    plant_sheets = sorted([s for s in xl.sheet_names if s.startswith(mode_tag)], key=_sheet_sort_key)
    rows = []

    for sname in plant_sheets:
        pdf = pd.read_excel(result_xlsx, sheet_name=sname)
        needed = ["target_id", "target_abatement", "ec_abatement_ton", "bc_abatement_ton", "cc_capture_ton"]
        missing = [c for c in needed if c not in pdf.columns]
        if missing or pdf.empty:
            continue

        target_id = pd.to_numeric(pdf["target_id"], errors="coerce").iloc[0]
        target = pd.to_numeric(pdf["target_abatement"], errors="coerce").iloc[0]
        ec_total = pd.to_numeric(pdf["ec_abatement_ton"], errors="coerce").fillna(0.0).sum()
        bc_total = pd.to_numeric(pdf["bc_abatement_ton"], errors="coerce").fillna(0.0).sum()
        cc_total = pd.to_numeric(pdf["cc_capture_ton"], errors="coerce").fillna(0.0).sum()

        rows.append(
            {
                "sheet": sname,
                "target_id": int(target_id) if np.isfinite(target_id) else np.nan,
                "target_abatement": float(target) if np.isfinite(target) else np.nan,
                "EC": float(ec_total),
                "BC": float(bc_total),
                "CC": float(cc_total),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["sheet", "target_id", "target_abatement", "EC", "BC", "CC"])

    out = pd.DataFrame(rows)
    out = out.sort_values(["target_abatement", "target_id"], na_position="last").reset_index(drop=True)
    return out


def _plot_tech_sankey(result_xlsx: str, out_html: str, mode: str = "MILP", normalize: bool = False) -> str:
    try:
        go = importlib.import_module("plotly.graph_objects")
    except Exception as exc:
        raise RuntimeError(
            "Sankey plotting requires plotly. Install with: pip install plotly"
        ) from exc

    summary = _build_scenario_tech_summary(result_xlsx, mode=mode)
    if summary.empty:
        raise ValueError(f"No scenario plant sheets found for mode={mode} in {result_xlsx}")

    scenario_labels = []
    for _, r in summary.iterrows():
        tid = r["target_id"]
        tgt = r["target_abatement"]
        if np.isfinite(tid) and np.isfinite(tgt):
            scenario_labels.append(f"S{int(tid):02d} | {tgt:,.0f}")
        else:
            scenario_labels.append(str(r["sheet"]))

    tech_labels = ["EC", "BC", "CC"]
    labels = scenario_labels + tech_labels
    tech_color = {"EC": "#2CA02C", "BC": "#1F77B4", "CC": "#D62728"}

    src = []
    dst = []
    vals = []
    link_colors = []
    n_s = len(scenario_labels)

    for i, r in summary.iterrows():
        ec = max(0.0, float(r["EC"]))
        bc = max(0.0, float(r["BC"]))
        cc = max(0.0, float(r["CC"]))
        total = ec + bc + cc
        if total <= 0:
            continue

        flows = {"EC": ec, "BC": bc, "CC": cc}
        if normalize:
            flows = {k: (v / total) for k, v in flows.items()}

        for j, tech in enumerate(tech_labels):
            v = flows[tech]
            if v <= 0:
                continue
            src.append(i)
            dst.append(n_s + j)
            vals.append(v)
            base = tech_color[tech]
            link_colors.append(base.replace("#", "rgba(") if False else base)

    node_colors = ["#BDBDBD"] * n_s + [tech_color[t] for t in tech_labels]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=16,
                    thickness=16,
                    line=dict(color="rgba(0,0,0,0.2)", width=0.5),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(
                    source=src,
                    target=dst,
                    value=vals,
                    color=["rgba(44,160,44,0.45)" if c == "#2CA02C" else "rgba(31,119,180,0.45)" if c == "#1F77B4" else "rgba(214,39,40,0.45)" for c in link_colors],
                ),
            )
        ]
    )
    metric = "Share of scenario abatement" if normalize else "Abatement amount (ton)"
    fig.update_layout(
        title=f"Technology evolution Sankey across scenarios ({mode.upper()})",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=60, b=20),
        annotations=[
            dict(
                x=0.5,
                y=-0.08,
                xref="paper",
                yref="paper",
                text=f"Link value metric: {metric}",
                showarrow=False,
                font=dict(size=11, color="#555555"),
            )
        ],
    )

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def _save_plotly_png(fig, out_png: str, width: int = 1500, height: int = 900, scale: int = 2) -> str:
    try:
        save_plotly_bundle(fig, out_png, width=width, height=height, scale=scale)
    except Exception as exc:
        raise RuntimeError(
            "PNG export requires kaleido. Install with: pip install kaleido"
        ) from exc
    return out_png


def _state_label_from_row(row: pd.Series, bc_threshold: float, binary_threshold: float) -> str:
    ec = int(float(pd.to_numeric(row.get("x_ec", 0.0), errors="coerce") or 0.0) >= binary_threshold)
    cc = int(float(pd.to_numeric(row.get("x_cc", 0.0), errors="coerce") or 0.0) >= binary_threshold)
    br = float(pd.to_numeric(row.get("blend_ratio", 0.0), errors="coerce") or 0.0)
    bc = int(br > bc_threshold)
    return f"E{ec}C{cc}B{bc}"


def _load_state_assignments(
    result_xlsx: str,
    mode: str = "MILP",
    bc_threshold: float = 1e-8,
    binary_threshold: float = 0.5,
) -> pd.DataFrame:
    xl = pd.ExcelFile(result_xlsx)
    mode_tag = "pm" if str(mode).strip().upper() == "MILP" else "pl"
    plant_sheets = sorted([s for s in xl.sheet_names if s.startswith(mode_tag)], key=_sheet_sort_key)

    rows = []
    for sname in plant_sheets:
        pdf = pd.read_excel(result_xlsx, sheet_name=sname)
        need = ["plant_id", "x_ec", "x_cc", "blend_ratio", "target_id", "target_abatement"]
        missing = [c for c in need if c not in pdf.columns]
        if missing or pdf.empty:
            continue

        target_id = pd.to_numeric(pdf["target_id"], errors="coerce").iloc[0]
        target = pd.to_numeric(pdf["target_abatement"], errors="coerce").iloc[0]
        if not np.isfinite(target):
            continue

        sub = pdf[["plant_id", "x_ec", "x_cc", "blend_ratio"]].copy()
        sub["plant_id"] = sub["plant_id"].astype(str)
        sub["state"] = sub.apply(
            lambda r: _state_label_from_row(r, bc_threshold=bc_threshold, binary_threshold=binary_threshold),
            axis=1,
        )
        sub["target_abatement"] = float(target)
        sub["target_id"] = int(target_id) if np.isfinite(target_id) else np.nan
        rows.append(sub[["plant_id", "target_id", "target_abatement", "state"]])

    if not rows:
        return pd.DataFrame(columns=["plant_id", "target_id", "target_abatement", "state"])

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["target_abatement", "plant_id"]).reset_index(drop=True)
    return out


def _plot_state_evolution_area(state_df: pd.DataFrame, out_png: str, mode: str = "MILP") -> str:
    if state_df.empty:
        raise ValueError("No state records found for state evolution area plot.")

    states = [
        "E0C0B0", "E1C0B0", "E0C1B0", "E1C1B0",
        "E0C0B1", "E1C0B1", "E0C1B1", "E1C1B1",
    ]
    colors = {
        "E0C0B0": "#BDBDBD",
        "E1C0B0": "#8C564B",
        "E0C1B0": "#D62728",
        "E1C1B0": "#9467BD",
        "E0C0B1": "#1F77B4",
        "E1C0B1": "#17BECF",
        "E0C1B1": "#FF7F0E",
        "E1C1B1": "#2CA02C",
    }

    def _display_state_label(s: str) -> str:
        if len(s) == 6 and s[0] == "E" and s[2] == "C" and s[4] == "B":
            return f"{s[0:2]}-{s[2:4]}-{s[4:6]}"
        return s

    counts = (
        state_df.groupby(["target_abatement", "state"]).size().unstack(fill_value=0)
    )
    for s in states:
        if s not in counts.columns:
            counts[s] = 0
    counts = counts[states].sort_index()

    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    x = shares.index.to_numpy(dtype=float)
    ys = [shares[s].to_numpy(dtype=float) for s in states]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.6))
    ax.stackplot(x, ys, labels=states, colors=[colors[s] for s in states], alpha=0.92)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Target abatement (t CO$_2$)")
    ax.set_ylabel("Plant share by state")
    ax.set_title(f"Plant retrofit state evolution ({mode.upper()})")
    ax.grid(alpha=0.2, linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    labels = [_display_state_label(s) for s in labels]
    ax.legend(handles, labels, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    save_figure_bundle(fig, out_png, dpi=280, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _build_tech_share_split_series(
    result_xlsx: str,
    mode: str = "MILP",
    bc_threshold: float = 1e-8,
    bc_split_ratio_threshold: float = 0.2,
) -> pd.DataFrame:
    xl = pd.ExcelFile(result_xlsx)
    mode_tag = "pm" if str(mode).strip().upper() == "MILP" else "pl"
    plant_sheets = sorted([s for s in xl.sheet_names if s.startswith(mode_tag)], key=_sheet_sort_key)

    rows = []
    for sname in plant_sheets:
        pdf = pd.read_excel(result_xlsx, sheet_name=sname)
        need = ["target_id", "target_abatement", "ec_abatement_ton", "bc_abatement_ton", "cc_capture_ton", "blend_ratio"]
        missing = [c for c in need if c not in pdf.columns]
        if missing or pdf.empty:
            continue

        target_id = pd.to_numeric(pdf["target_id"], errors="coerce").iloc[0]
        target = pd.to_numeric(pdf["target_abatement"], errors="coerce").iloc[0]
        if not np.isfinite(target):
            continue

        ec = pd.to_numeric(pdf["ec_abatement_ton"], errors="coerce").fillna(0.0)
        cc = pd.to_numeric(pdf["cc_capture_ton"], errors="coerce").fillna(0.0)
        bc = pd.to_numeric(pdf["bc_abatement_ton"], errors="coerce").fillna(0.0)
        br = pd.to_numeric(pdf["blend_ratio"], errors="coerce").fillna(0.0)

        db_mask = (br > bc_threshold) & (br <= bc_split_ratio_threshold)
        gb_mask = br > bc_split_ratio_threshold

        ec_total = float(ec.sum())
        db_total = float(bc[db_mask].sum())
        gb_total = float(bc[gb_mask].sum())
        cc_total = float(cc.sum())

        rows.append(
            {
                "sheet": sname,
                "target_id": int(target_id) if np.isfinite(target_id) else np.nan,
                "target_abatement": float(target),
                "EC": ec_total,
                "DB": db_total,
                "GB": gb_total,
                "CC": cc_total,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["sheet", "target_id", "target_abatement", "EC", "DB", "GB", "CC"])

    out = pd.DataFrame(rows)
    out = out.sort_values(["target_abatement", "target_id"], na_position="last").reset_index(drop=True)

    total = out[["EC", "DB", "GB", "CC"]].sum(axis=1)
    total = total.replace(0.0, np.nan)
    out["share_EC"] = (out["EC"] / total).fillna(0.0)
    out["share_DB"] = (out["DB"] / total).fillna(0.0)
    out["share_GB"] = (out["GB"] / total).fillna(0.0)
    out["share_CC"] = (out["CC"] / total).fillna(0.0)
    return out


def _plot_tech_share_split_area(
    split_df: pd.DataFrame,
    out_png: str,
    mode: str = "MILP",
    bc_split_ratio_threshold: float = 0.2,
) -> str:
    if split_df.empty:
        raise ValueError("No valid scenario data for technology share split plot.")

    x = split_df["target_abatement"].to_numpy(dtype=float)
    y_ec = split_df["share_EC"].to_numpy(dtype=float)
    y_db = split_df["share_DB"].to_numpy(dtype=float)
    y_gb = split_df["share_GB"].to_numpy(dtype=float)
    y_cc = split_df["share_CC"].to_numpy(dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(12.2, 5.8))
    labels = ["EC", "DB", "GB", "CC"]
    colors = ["#2CA02C", "#4E79A7", "#1F3A5F", "#D62728"]
    ax.stackplot(x, y_ec, y_db, y_gb, y_cc, labels=labels, colors=colors, alpha=0.92)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Target abatement (t CO$_2$)")
    ax.set_ylabel("Abatement share")
    ax.set_title(f"Technology abatement share split ({mode.upper()}): EC, DB, GB, CC")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.10))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    save_figure_bundle(fig, out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _plot_state_transition_sankey(state_df: pd.DataFrame, out_html: str, mode: str = "MILP") -> str:
    if state_df.empty:
        raise ValueError("No state records found for state transition Sankey.")

    try:
        go = importlib.import_module("plotly.graph_objects")
    except Exception as exc:
        raise RuntimeError("State transition Sankey requires plotly. Install with: pip install plotly") from exc

    targets = sorted(state_df["target_abatement"].dropna().unique().tolist())
    if len(targets) < 2:
        raise ValueError("Need at least two scenarios to build state transition Sankey.")

    states = [
        "E0C0B0", "E1C0B0", "E0C1B0", "E1C1B0",
        "E0C0B1", "E1C0B1", "E0C1B1", "E1C1B1",
    ]
    state_color = {
        "E0C0B0": "rgba(189,189,189,0.55)",
        "E1C0B0": "rgba(140,86,75,0.55)",
        "E0C1B0": "rgba(214,39,40,0.55)",
        "E1C1B0": "rgba(148,103,189,0.55)",
        "E0C0B1": "rgba(31,119,180,0.55)",
        "E1C0B1": "rgba(23,190,207,0.55)",
        "E0C1B1": "rgba(255,127,14,0.55)",
        "E1C1B1": "rgba(44,160,44,0.55)",
    }

    labels = []
    node_index = {}
    for t in targets:
        for s in states:
            idx = len(labels)
            lab = f"{t:,.0f} | {s}"
            labels.append(lab)
            node_index[(t, s)] = idx

    source = []
    target = []
    value = []
    link_color = []

    for left_t, right_t in zip(targets[:-1], targets[1:]):
        left = state_df[state_df["target_abatement"] == left_t][["plant_id", "state"]].rename(columns={"state": "state_left"})
        right = state_df[state_df["target_abatement"] == right_t][["plant_id", "state"]].rename(columns={"state": "state_right"})
        trans = left.merge(right, on="plant_id", how="inner")
        flow = trans.groupby(["state_left", "state_right"]).size().reset_index(name="count")

        for _, r in flow.iterrows():
            s_left = str(r["state_left"])
            s_right = str(r["state_right"])
            cnt = int(r["count"])
            if cnt <= 0:
                continue
            source.append(node_index[(left_t, s_left)])
            target.append(node_index[(right_t, s_right)])
            value.append(cnt)
            link_color.append(state_color.get(s_left, "rgba(120,120,120,0.45)"))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=10,
                    thickness=14,
                    line=dict(color="rgba(0,0,0,0.2)", width=0.3),
                    label=labels,
                    color=[state_color.get(lab.split(" | ")[-1], "rgba(160,160,160,0.6)") for lab in labels],
                ),
                link=dict(source=source, target=target, value=value, color=link_color),
            )
        ]
    )
    fig.update_layout(
        title=f"Plant state transitions across adjacent scenarios ({mode.upper()})",
        font=dict(size=10),
        margin=dict(l=15, r=15, t=55, b=15),
    )
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def _build_tech_sankey_figure(result_xlsx: str, mode: str = "MILP", normalize: bool = False):
    go = importlib.import_module("plotly.graph_objects")
    summary = _build_scenario_tech_summary(result_xlsx, mode=mode)
    if summary.empty:
        raise ValueError(f"No scenario plant sheets found for mode={mode} in {result_xlsx}")

    scenario_labels = []
    for _, r in summary.iterrows():
        tid = r["target_id"]
        tgt = r["target_abatement"]
        if np.isfinite(tid) and np.isfinite(tgt):
            scenario_labels.append(f"S{int(tid):02d} | {tgt:,.0f}")
        else:
            scenario_labels.append(str(r["sheet"]))

    tech_labels = ["EC", "BC", "CC"]
    labels = scenario_labels + tech_labels
    tech_color = {"EC": "#2CA02C", "BC": "#1F77B4", "CC": "#D62728"}

    src = []
    dst = []
    vals = []
    link_colors = []
    n_s = len(scenario_labels)

    for i, r in summary.iterrows():
        ec = max(0.0, float(r["EC"]))
        bc = max(0.0, float(r["BC"]))
        cc = max(0.0, float(r["CC"]))
        total = ec + bc + cc
        if total <= 0:
            continue

        flows = {"EC": ec, "BC": bc, "CC": cc}
        if normalize:
            flows = {k: (v / total) for k, v in flows.items()}

        for j, tech in enumerate(tech_labels):
            v = flows[tech]
            if v <= 0:
                continue
            src.append(i)
            dst.append(n_s + j)
            vals.append(v)
            link_colors.append(tech_color[tech])

    node_colors = ["#BDBDBD"] * n_s + [tech_color[t] for t in tech_labels]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=16,
                    thickness=16,
                    line=dict(color="rgba(0,0,0,0.2)", width=0.5),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(
                    source=src,
                    target=dst,
                    value=vals,
                    color=["rgba(44,160,44,0.45)" if c == "#2CA02C" else "rgba(31,119,180,0.45)" if c == "#1F77B4" else "rgba(214,39,40,0.45)" for c in link_colors],
                ),
            )
        ]
    )
    metric = "Share of scenario abatement" if normalize else "Abatement amount (ton)"
    fig.update_layout(
        title=f"Technology evolution Sankey across scenarios ({mode.upper()})",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=60, b=20),
        annotations=[
            dict(
                x=0.5,
                y=-0.08,
                xref="paper",
                yref="paper",
                text=f"Link value metric: {metric}",
                showarrow=False,
                font=dict(size=11, color="#555555"),
            )
        ],
    )
    return fig


def _build_state_transition_sankey_figure(state_df: pd.DataFrame, mode: str = "MILP"):
    go = importlib.import_module("plotly.graph_objects")
    if state_df.empty:
        raise ValueError("No state records found for state transition Sankey.")

    targets = sorted(state_df["target_abatement"].dropna().unique().tolist())
    if len(targets) < 2:
        raise ValueError("Need at least two scenarios to build state transition Sankey.")

    states = [
        "E0C0B0", "E1C0B0", "E0C1B0", "E1C1B0",
        "E0C0B1", "E1C0B1", "E0C1B1", "E1C1B1",
    ]
    state_color = {
        "E0C0B0": "rgba(189,189,189,0.55)",
        "E1C0B0": "rgba(140,86,75,0.55)",
        "E0C1B0": "rgba(214,39,40,0.55)",
        "E1C1B0": "rgba(148,103,189,0.55)",
        "E0C0B1": "rgba(31,119,180,0.55)",
        "E1C0B1": "rgba(23,190,207,0.55)",
        "E0C1B1": "rgba(255,127,14,0.55)",
        "E1C1B1": "rgba(44,160,44,0.55)",
    }

    def _display_state_label(s: str) -> str:
        # Render state tags in paper-friendly form, e.g. E0-C0-B0.
        if len(s) == 6 and s[0] == "E" and s[2] == "C" and s[4] == "B":
            return f"{s[0:2]}-{s[2:4]}-{s[4:6]}"
        return s

    labels = []
    node_index = {}
    node_state_keys = []
    for t in targets:
        for s in states:
            idx = len(labels)
            lab = f"{t:,.0f} | {_display_state_label(s)}"
            labels.append(lab)
            node_index[(t, s)] = idx
            node_state_keys.append(s)

    source = []
    target = []
    value = []
    link_color = []

    for left_t, right_t in zip(targets[:-1], targets[1:]):
        left = state_df[state_df["target_abatement"] == left_t][["plant_id", "state"]].rename(columns={"state": "state_left"})
        right = state_df[state_df["target_abatement"] == right_t][["plant_id", "state"]].rename(columns={"state": "state_right"})
        trans = left.merge(right, on="plant_id", how="inner")
        flow = trans.groupby(["state_left", "state_right"]).size().reset_index(name="count")

        for _, r in flow.iterrows():
            s_left = str(r["state_left"])
            s_right = str(r["state_right"])
            cnt = int(r["count"])
            if cnt <= 0:
                continue
            source.append(node_index[(left_t, s_left)])
            target.append(node_index[(right_t, s_right)])
            value.append(cnt)
            link_color.append(state_color.get(s_left, "rgba(120,120,120,0.45)"))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=10,
                    thickness=14,
                    line=dict(color="rgba(0,0,0,0.2)", width=0.3),
                    label=labels,
                    color=[state_color.get(s, "rgba(160,160,160,0.6)") for s in node_state_keys],
                ),
                link=dict(source=source, target=target, value=value, color=link_color),
            )
        ]
    )
    fig.update_layout(
        title=f"Plant state transitions across adjacent scenarios ({mode.upper()})",
        font=dict(size=10),
        margin=dict(l=15, r=15, t=55, b=15),
    )
    return fig


def _plot_tech_sankey(result_xlsx: str, out_html: str, mode: str = "MILP", normalize: bool = False) -> str:
    try:
        importlib.import_module("plotly.graph_objects")
    except Exception as exc:
        raise RuntimeError(
            "Sankey plotting requires plotly. Install with: pip install plotly"
        ) from exc
    fig = _build_tech_sankey_figure(result_xlsx=result_xlsx, mode=mode, normalize=normalize)
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def _plot_state_transition_sankey(state_df: pd.DataFrame, out_html: str, mode: str = "MILP") -> str:
    try:
        importlib.import_module("plotly.graph_objects")
    except Exception as exc:
        raise RuntimeError("State transition Sankey requires plotly. Install with: pip install plotly") from exc
    fig = _build_state_transition_sankey_figure(state_df=state_df, mode=mode)
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot standardized MAC curve from fleet-wide joint optimization outputs.")
    parser.add_argument("--curve", default=None, help="MILP curve CSV path. Default: data/<input-prefix>_curve.csv")
    parser.add_argument("--curve-lp", default=None, help="Optional LP-relax curve CSV path.")
    parser.add_argument("--input-prefix", default=INPUT_PREFIX, help="Output prefix used by optimization_fleet_joint_milp.py")
    parser.add_argument("--output", default=None, help="Output PNG path for 2-panel figure.")
    parser.add_argument("--output-panel-macc", default=None, help="Output PNG path for MAC-only panel.")
    parser.add_argument("--output-logistics-dualaxis", default=None, help="Output PNG path for dual-axis logistics figure.")
    parser.add_argument("--result-xlsx", default=None, help="Optimization results workbook path (for per-scenario sorted curves).")
    parser.add_argument("--output-scenario-sorted-dir", default=None, help="Output directory for per-scenario sorted unit-cost curves.")
    parser.add_argument("--scenario-mode", choices=["MILP", "LP"], default="MILP", help="Choose scenario sheets to plot: MILP uses pmXX, LP uses plXX.")
    parser.add_argument("--skip-standard", action="store_true", help="Skip standard curve plots and only generate per-scenario sorted curves.")
    parser.add_argument("--plot-sankey", action="store_true", help="Generate Sankey of scenario -> technology abatement flows.")
    parser.add_argument("--sankey-normalize", action="store_true", help="Normalize each scenario to shares (0-1) before Sankey plotting.")
    parser.add_argument("--output-sankey-html", default=None, help="Output HTML path for Sankey figure.")
    parser.add_argument("--output-sankey-png", default=None, help="Optional output PNG path for technology Sankey.")
    parser.add_argument("--plot-state-evolution", action="store_true", help="Generate plant state evolution figures (stacked area + transition Sankey).")
    parser.add_argument("--output-state-area-png", default=None, help="Output PNG path for plant-state evolution stacked area chart.")
    parser.add_argument("--output-state-transition-sankey-html", default=None, help="Output HTML path for plant-state transition Sankey.")
    parser.add_argument("--output-state-transition-sankey-png", default=None, help="Optional output PNG path for plant-state transition Sankey.")
    parser.add_argument("--plot-tech-share-split", action="store_true", help="Plot technology abatement share stacked area with BC split into DB/GB by blend_ratio.")
    parser.add_argument("--output-tech-share-split-png", default=None, help="Output PNG path for technology abatement share split stacked area chart.")
    parser.add_argument("--bc-split-ratio-threshold", type=float, default=0.2, help="BC split threshold: DB if bc_threshold < blend_ratio <= this value, GB if blend_ratio > this value.")
    parser.add_argument("--state-bc-threshold", type=float, default=1e-8, help="Threshold for classifying BC state from blend_ratio.")
    parser.add_argument("--state-binary-threshold", type=float, default=0.5, help="Threshold for classifying x_ec/x_cc to binary states.")
    parser.add_argument("--plot-sensitivity-compare", action="store_true", help="Plot MAC comparison across baseline/capex/coal/discount sensitivity cases.")
    parser.add_argument("--sensitivity-prefix", default=SENSITIVITY_PREFIX, help="Curve file prefix for sensitivity outputs (e.g., gurobi).")
    parser.add_argument("--output-sensitivity-overlay", default=None, help="Output PNG path for MAC overlay sensitivity comparison.")
    parser.add_argument("--output-sensitivity-delta", default=None, help="Output PNG path for delta-MAC vs baseline.")
    parser.add_argument("--output-sensitivity-envelope", default=None, help="Output PNG path for category envelope plot.")
    parser.add_argument("--plot-biomass-sensitivity-compare", action="store_true", help="Plot MAC + BC-share comparison for biomass supply sensitivity (100/75/50/25%).")
    parser.add_argument("--biomass-sensitivity-prefix", default=BIOMASS_SENSITIVITY_PREFIX, help="Curve/result prefix for biomass supply sensitivity outputs.")
    parser.add_argument("--output-biomass-sensitivity-compare", default=None, help="Output PNG path for biomass supply sensitivity comparison figure.")
    parser.add_argument("--plot-biomass-share-simple", action="store_true", help="Plot a simplified biomass BC-share figure for paper text citation.")
    parser.add_argument("--output-biomass-share-simple", default=None, help="Output PNG path for simplified biomass BC-share figure.")
    parser.add_argument("--plot-biomass-dualaxis-selected", action="store_true", help="Plot dual-y figure: MAC curves + selected-point BC-share bars.")
    parser.add_argument("--output-biomass-dualaxis-selected", default=None, help="Output PNG path for dual-y selected-point biomass figure.")
    parser.add_argument("--plot-fuel-standalone", action="store_true", help="Plot standalone stacked fuel consumption breakdown figure.")
    parser.add_argument("--output-fuel-standalone", default="fleet_joint_fuel_breakdown.png", help="Output PNG path for standalone fuel breakdown figure.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, DATA_DIR_NAME)

    curve_path = args.curve or _resolve_default_curve_path(data_dir, args.input_prefix)
    curve_lp_path = args.curve_lp
    if curve_lp_path is None:
        default_lp = os.path.join(data_dir, f"{args.input_prefix}{DEFAULT_LP_CURVE_SUFFIX}")
        if os.path.exists(default_lp):
            curve_lp_path = default_lp

    out_logistics_dualaxis = args.output_logistics_dualaxis or os.path.join(data_dir, OUTPUT_LOGISTICS_DUALAXIS_PNG)
    result_xlsx = args.result_xlsx or os.path.join(data_dir, DEFAULT_RESULTS_FALLBACK)
    out_scenario_sorted_dir = args.output_scenario_sorted_dir or os.path.join(data_dir, OUTPUT_SCENARIO_SORTED_DIR)
    out_state_area_png = args.output_state_area_png or os.path.join(data_dir, OUTPUT_STATE_AREA_PNG)
    out_tech_share_split_png = args.output_tech_share_split_png or os.path.join(data_dir, OUTPUT_TECH_SHARE_SPLIT_AREA_PNG)
    out_biomass_sensitivity_compare = args.output_biomass_sensitivity_compare or os.path.join(data_dir, OUTPUT_BIOMASS_SENSITIVITY_COMPARE_PNG)

    if not args.skip_standard:
        if not os.path.exists(curve_path):
            curve_files = sorted(
                [f for f in os.listdir(data_dir) if f.endswith("_curve.csv")]
            )
            raise FileNotFoundError(
                f"MILP curve CSV not found: {curve_path}\n"
                f"Available *_curve.csv files under data/: {curve_files}\n"
                f"Use --curve <path> or --input-prefix <prefix>."
            )

        milp = _load_curve(curve_path, "MILP curve")
        lp = None
        if curve_lp_path is not None and os.path.exists(curve_lp_path):
            lp = _load_curve(curve_lp_path, "LP-relax curve")

        _plot(milp, lp)
        _plot_logistics_dualaxis(milp, out_logistics_dualaxis)

    scenario_outputs = []
    if os.path.exists(result_xlsx):
        scenario_outputs = _plot_all_scenario_sorted_macc(result_xlsx, out_scenario_sorted_dir, mode=args.scenario_mode)

    state_area_out = None
    if args.plot_state_evolution:
        if not os.path.exists(result_xlsx):
            raise FileNotFoundError(f"Results xlsx not found for state evolution plotting: {result_xlsx}")
        state_df = _load_state_assignments(
            result_xlsx=result_xlsx,
            mode=args.scenario_mode,
            bc_threshold=args.state_bc_threshold,
            binary_threshold=args.state_binary_threshold,
        )
        state_area_out = _plot_state_evolution_area(
            state_df=state_df,
            out_png=out_state_area_png,
            mode=args.scenario_mode,
        )

    tech_share_split_out = None
    if args.plot_tech_share_split:
        if not os.path.exists(result_xlsx):
            raise FileNotFoundError(f"Results xlsx not found for technology share split plotting: {result_xlsx}")
        split_df = _build_tech_share_split_series(
            result_xlsx=result_xlsx,
            mode=args.scenario_mode,
            bc_threshold=args.state_bc_threshold,
            bc_split_ratio_threshold=args.bc_split_ratio_threshold,
        )
        tech_share_split_out = _plot_tech_share_split_area(
            split_df=split_df,
            out_png=out_tech_share_split_png,
            mode=args.scenario_mode,
            bc_split_ratio_threshold=args.bc_split_ratio_threshold,
        )

    sensitivity_out = None
    out_sensitivity_overlay = os.path.join(data_dir, OUTPUT_SENSITIVITY_OVERLAY_PNG)
    out_sensitivity_delta = os.path.join(data_dir, OUTPUT_SENSITIVITY_DELTA_PNG)
    out_sensitivity_envelope = os.path.join(data_dir, OUTPUT_SENSITIVITY_ENVELOPE_PNG)
    sensitivity_specs = _build_sensitivity_curve_specs(data_dir=data_dir, prefix=SENSITIVITY_PREFIX)
    try:
        sensitivity_out = _plot_sensitivity_comparison(
            curve_specs=sensitivity_specs,
            out_overlay_png=out_sensitivity_overlay,
            out_delta_png=out_sensitivity_delta,
            out_envelope_png=out_sensitivity_envelope,
        )
    except Exception as exc:
        print(f"[warn] Sensitivity comparison plot failed: {exc}")

    biomass_sensitivity_out = None
    if args.plot_biomass_sensitivity_compare:
        biomass_specs = _build_biomass_sensitivity_curve_specs(data_dir=data_dir, prefix=args.biomass_sensitivity_prefix)
        biomass_sensitivity_out = _plot_biomass_sensitivity_comparison(
            curve_specs=biomass_specs,
            out_png=out_biomass_sensitivity_compare,
        )

    # Always output standalone fuel breakdown figure first (fast, no solver needed)
    fuel_out_path = os.path.join(data_dir, args.output_fuel_standalone)
    try:
        fuel_out = _plot_fuel_standalone(data_dir, fuel_out_path)
        print(f"Fuel breakdown figure: {fuel_out}")
    except Exception as exc:
        print(f"[warn] Could not generate fuel breakdown figure: {exc}")

    print("Done.")
    if not args.skip_standard:
        print(f"Input MILP curve: {curve_path}")
        if 'lp' in locals() and lp is not None:
            print(f"Input LP-relax curve: {curve_lp_path}")
        print(f"Output logistics dual-axis figure: {out_logistics_dualaxis}")
    if os.path.exists(result_xlsx):
        print(f"Input results xlsx: {result_xlsx}")
        print(f"Scenario sorted curves: {len(scenario_outputs)} files in {out_scenario_sorted_dir}")
    else:
        print(f"Results xlsx not found, skip per-scenario sorted curves: {result_xlsx}")
    if state_area_out is not None:
        print(f"State evolution area output: {state_area_out}")
    if tech_share_split_out is not None:
        print(f"Technology share split area output: {tech_share_split_out}")
    if sensitivity_out is not None:
        print(f"Sensitivity overlay output: {sensitivity_out['overlay']}")
        print(f"Sensitivity delta output: {sensitivity_out['delta']}")
        print(f"Sensitivity envelope output: {sensitivity_out['envelope']}")
    if biomass_sensitivity_out is not None:
        print(f"Biomass sensitivity comparison output: {biomass_sensitivity_out}")


if __name__ == "__main__":
    main()
