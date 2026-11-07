"""
Standalone MAC curve from EC / BC-only / CC-only max-abatement results.
No interaction constraints — simply sorts all plant×technology pairs by
unit abatement cost and picks cheapest units per target.

Inputs:
  data/1885_coal_power_plants_original.xlsx → EC sheet (carbon_reduction, unit_cost)
  gurobi_r2_bc_only_results.xlsx           → last pm sheet (max target: BC abatement & cost)
  gurobi_r2_cc_only[_max]_results.xlsx     → last pm sheet (max target: CC abatement & cost)
  Both are read from the result-set folder (R2_SET_DIR) when it holds them, else from data/.
  Abatement is net (BC minus biomass logistics emissions, CC net of the capture-energy coal), as in the
  fleet model; under the captured accounting the two deductions are zero.
"""

import os
from pathlib import Path
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_export_utils import save_figure_bundle
from r2_result_columns import add_net_abatement_columns


DATA_DIR = Path(__file__).resolve().parent / "data"
WORKBOOK = DATA_DIR / "1885_coal_power_plants_original.xlsx"
_SET_DIR = Path(os.environ.get("R2_SET_DIR", str(DATA_DIR)))


def _set_file(*names: str) -> Path:
    for folder in (_SET_DIR, DATA_DIR):
        for name in names:
            if (folder / name).exists():
                return folder / name
    return DATA_DIR / names[0]


BC_RESULTS = _set_file("gurobi_r2_bc_only_results.xlsx")
CC_RESULTS = _set_file("gurobi_r2_cc_only_results.xlsx", "gurobi_r2_cc_only_max_results.xlsx")
EC_PACKAGES = DATA_DIR / "ec_packages_r2.csv"
EC_PACKAGE = "r20_o00" if os.environ.get("R2_SET", "ec10") == "ec20" else "no5_r10_o00"  # baseline EC package of the result set
OUTPUT_CSV = DATA_DIR / "standalone_mac_curve.csv"
OUTPUT_PNG = DATA_DIR / "fig_standalone_mac.png"
OUTPUT_RAW_PNG = DATA_DIR / "fig_standalone_mac_raw.png"
TARGET_STEP = 0.01e9  # 100 Mt CO₂TARGET_MATCH_MODE = "min_overshoot"  # "min_overshoot" | "cost_only"
TARGET_MODE_SWITCH_ABATEMENT = 1.1e9  # ≤ threshold uses min_overshoot

TECH_COLORS = {"EC": "#4DBBD5", "BC": "#00A087", "CC": "#E64B35"}
TECH_NAMES = {"EC": "Energy conservation", "BC": "Biomass co-firing", "CC": "Carbon capture"}


def _norm_id(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if not s:
        return ""
    try:
        f = float(s)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def load_ec_data() -> pd.DataFrame:
    """Load the baseline EC package (net annual cost, i.e. after avoided coal cost), as in the fleet model."""
    pk = pd.read_csv(EC_PACKAGES, dtype={"plant_id": str})
    pk = pk[pk["package"] == EC_PACKAGE]
    out = pd.DataFrame({
        "plant_id": pk["plant_id"].map(_norm_id),
        "tech": "EC",
        "abatement_ton": pk["ec_abatement_ton"].astype(float),
        "total_cost": pk["ec_net_cost"].astype(float),
    })
    out["unit_cost_per_ton"] = out["total_cost"] / out["abatement_ton"].replace(0, np.nan)
    return out


def _last_pm_sheet(path) -> str:
    sheets = [s for s in pd.ExcelFile(path).sheet_names if s.startswith("pm")]
    return max(sheets, key=lambda x: int(x[2:]))  # maximum target


def load_bc_data() -> pd.DataFrame:
    """Load BC per-plant abatement from the max-target pm sheet.
    Includes coal savings: bc_total_cost = bc_investment - bc_coal_saving + bc_feedstock + bc_transport
    This matches the actual net cost of BC in the joint optimization model."""
    pdf = add_net_abatement_columns(pd.read_excel(BC_RESULTS, sheet_name=_last_pm_sheet(BC_RESULTS)))
    out = pd.DataFrame({
        "plant_id": pdf["plant_id"].astype(str),
        "tech": "BC",
        "abatement_ton": pd.to_numeric(pdf["bc_net_abatement_ton"], errors="coerce").fillna(0.0),
        "total_cost": pd.to_numeric(pdf["bc_total_cost" if "bc_total_cost" in pdf.columns else "total_cost"], errors="coerce").fillna(0.0),
    })
    out["unit_cost_per_ton"] = out["total_cost"] / out["abatement_ton"].replace(0, np.nan)
    return out


def load_cc_data() -> pd.DataFrame:
    """Load CC per-plant capture from the max-target pm sheet."""
    pdf = add_net_abatement_columns(pd.read_excel(CC_RESULTS, sheet_name=_last_pm_sheet(CC_RESULTS)))
    out = pd.DataFrame({
        "plant_id": pdf["plant_id"].astype(str),
        "tech": "CC",
        "abatement_ton": pd.to_numeric(pdf["cc_net_abatement_ton"], errors="coerce").fillna(0.0),
        # capture cost plus CO2 transport and storage, as in the fleet model
        "total_cost": pd.to_numeric(pdf["total_cost"], errors="coerce").fillna(0.0),
    })
    out["unit_cost_per_ton"] = out["total_cost"] / out["abatement_ton"].replace(0, np.nan)
    return out


def build_mac_curve(
    units: pd.DataFrame,
    step: float = TARGET_STEP,
    raw: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build MAC curve by greedy least-cost selection.

    Parameters
    ----------
    units : DataFrame with columns [tech, plant_id, abatement_ton, total_cost, unit_cost_per_ton]
    step  : target spacing (ignored when raw=True)
    raw   : if True, output every unit as a point (no target-grid compression)

    Returns
    -------
    (compressed curve DF, sorted units DF)
    """
    # Filter valid entries and sort by unit cost ascending
    valid = units.dropna(subset=["unit_cost_per_ton", "abatement_ton"]).copy()
    valid = valid[valid["abatement_ton"] > 0.001].copy()
    valid = valid.sort_values("unit_cost_per_ton").reset_index(drop=True)

    cum_abate = valid["abatement_ton"].cumsum()
    cum_cost = valid["total_cost"].cumsum()

    if raw:
        # Raw mode: one row per unit, no target grid
        rows = []
        for idx in range(len(valid)):
            unit = valid.iloc[idx]
            rows.append({
                "target_abatement": float(cum_abate.iloc[idx]),
                "unit_cost": float(unit["unit_cost_per_ton"]),
                "cum_abatement": float(cum_abate.iloc[idx]),
                "cum_cost": float(cum_cost.iloc[idx]),
                "tech": unit["tech"],
                "plant_id": unit["plant_id"],
            })
        df = pd.DataFrame(rows)
        df["marginal_cost_diff"] = df["unit_cost"]
        return df, valid

    total_max = float(cum_abate.iloc[-1])

    # Target grid: 0, step, 2*step, ..., total_max
    n_steps = max(1, int(np.ceil(total_max / step)))
    targets = np.array([float(i) * step for i in range(n_steps + 1)])
    targets = targets[targets <= total_max]
    targets = np.concatenate([targets, [total_max]]) if targets[-1] < total_max else targets

    # For each target, find how many cheapest units fill it
    rows = []
    for t in targets:
        mask = cum_abate >= t
        if not mask.any():
            continue
        idx = int(mask.idxmax())  # first index where cum_abate >= t
        achieved = float(cum_abate.iloc[idx])
        cost = float(cum_cost.iloc[idx])
        rows.append({"target_abatement": t, "achieved_abatement": achieved, "total_cost": cost})

    df = pd.DataFrame(rows).sort_values("target_abatement").reset_index(drop=True)

    # MAC = dCost / dAbatement, using *actual achieved* abatement as denominator
    cost_diff = df["total_cost"].diff()
    abate_diff = df["achieved_abatement"].diff()
    df["marginal_cost_diff"] = cost_diff / abate_diff.replace(0, np.nan)

    df["marginal_cost_diff_target"] = cost_diff / df["target_abatement"].diff().replace(0, np.nan)
    df["avg_cost_from_origin"] = df["total_cost"] / df["achieved_abatement"].replace(0, np.nan)
    return df, valid


def plot_mac(df: pd.DataFrame, units: pd.DataFrame, out_path: str) -> None:
    """Plot the standalone MAC curve colored by technology share."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Main MAC line
    x = df["target_abatement"].to_numpy(dtype=float)
    y = df["marginal_cost_diff"].to_numpy(dtype=float)
    ax.plot(x / 1e9, y, color="#1F1F1F", linewidth=2.2, zorder=6, label="MAC curve")

    # Fill between: technology share as stacked bars at each target
    bar_width = 0.04 * x[-1] / 1e9 if len(x) > 1 else 0.01
    for idx in range(len(x) - 1):
        t_low = x[idx]
        t_high = x[idx + 1]
        # Which units are between these targets?
        mask = (units["abatement_ton"].cumsum() > t_low) & (units["abatement_ton"].cumsum() <= t_high)
        segment = units[mask]
        if segment.empty:
            continue
        x_pos = t_high / 1e9
        stacked = segment.groupby("tech")["abatement_ton"].sum()
        total = stacked.sum()
        if total == 0:
            continue
        bottom = 0
        for tech, color in TECH_COLORS.items():
            share = stacked.get(tech, 0) / total
            if share > 0.01:
                ax.bar(
                    x_pos, share * y[idx + 1] if np.isfinite(y[idx + 1]) else 0,
                    width=bar_width, bottom=bottom, color=color,
                    edgecolor="white", linewidth=0.3, alpha=0.75, zorder=3,
                )
                bottom += share * y[idx + 1] if np.isfinite(y[idx + 1]) else 0

    ax.axhline(0, color="black", linewidth=1.5, alpha=0.9, zorder=5)
    ax.set_xlabel("Cumulative abatement (Gt CO₂)")
    ax.set_ylabel("Marginal abatement cost (USD/tCO₂)")
    ax.set_title("Standalone MAC curve (EC + BC + CC, cheapest-first dispatch)")
    ax.grid(alpha=0.2, linestyle="--")

    # Legend
    handles = [
        plt.Line2D([], [], color="#1F1F1F", linewidth=2.2, label="MAC"),
    ]
    for tech, color in TECH_COLORS.items():
        handles.append(plt.Line2D([], [], color=color, marker="s", linestyle="", markersize=6, label=TECH_NAMES[tech]))
    ax.legend(handles=handles, frameon=True, fontsize=9, loc="upper left")

    fig.tight_layout()
    save_figure_bundle(
        fig,
        out_path,
        dpi=320,
        bbox_inches="tight",
        data_frames={"curve": df, "sorted_units": units},
    )
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def plot_raw_mac(units: pd.DataFrame, out_path: str) -> None:
    """Plot raw MAC: every unit as a step function, colored by technology."""
    valid = units.dropna(subset=["unit_cost_per_ton", "abatement_ton"]).copy()
    valid = valid[valid["abatement_ton"] > 0.001].copy()
    valid = valid.sort_values("unit_cost_per_ton").reset_index(drop=True)

    cum_abate = valid["abatement_ton"].cumsum() / 1e9  # Gt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Step plot: each unit as a horizontal segment
    x_starts = np.concatenate([[0.0], cum_abate.to_numpy()[:-1]])
    x_ends = cum_abate.to_numpy()
    y_vals = valid["unit_cost_per_ton"].to_numpy()

    for i in range(len(valid)):
        color = TECH_COLORS.get(valid.iloc[i]["tech"], "#7F7F7F")
        ax.hlines(y_vals[i], x_starts[i], x_ends[i], color=color, linewidth=2.0, zorder=3)

    # Also draw the cumulative cost curve (step function, different axis)
    cum_cost = valid["total_cost"].cumsum() / 1e9  # $B
    ax2 = ax.twinx()
    ax2.step(cum_abate, cum_cost, where="post", color="#2F2F2F", linewidth=1.8, alpha=0.7, zorder=5, label="Cumulative cost")
    ax2.set_ylabel("Cumulative cost (Billion USD)")
    ax2.legend(frameon=True, fontsize=9, loc="lower right")

    ax.axhline(0, color="black", linewidth=1.5, alpha=0.9, zorder=4)
    ax.set_xlabel("Cumulative abatement (Gt CO₂)")
    ax.set_ylabel("Unit abatement cost (USD/tCO₂)")
    ax.set_title("Raw MAC curve: every unit as a step (sorted by cost)")
    ax.grid(alpha=0.2, linestyle="--")

    handles = [plt.Line2D([], [], color=c, linewidth=2, label=TECH_NAMES[t]) for t, c in TECH_COLORS.items()]
    ax.legend(handles=handles, frameon=True, fontsize=9, loc="upper left")

    fig.tight_layout()
    save_figure_bundle(
        fig,
        out_path,
        dpi=320,
        bbox_inches="tight",
        data_frames={"raw_units": units},
    )
    plt.close(fig)
    print(f"Raw plot saved: {out_path}")


def main() -> None:
    print("Loading EC data...")
    ec = load_ec_data()
    print(f"  EC: {len(ec)} plants, total abatement={ec['abatement_ton'].sum()/1e9:.3f} Gt")

    print("Loading BC data...")
    bc = load_bc_data()
    print(f"  BC: {len(bc)} plants, total abatement={bc['abatement_ton'].sum()/1e9:.3f} Gt")

    print("Loading CC data...")
    cc = load_cc_data()
    print(f"  CC: {len(cc)} plants, total abatement={cc['abatement_ton'].sum()/1e9:.3f} Gt")

    # Combine
    all_units = pd.concat([ec, bc, cc], ignore_index=True)
    print(f"\nTotal: {len(all_units)} unit-records")
    print(f"Total abatement potential: {all_units['abatement_ton'].sum()/1e9:.3f} Gt")

    # Build MAC
    print("Building MAC curve...")
    curve, sorted_units = build_mac_curve(all_units, step=TARGET_STEP)
    print(f"  Targets: {len(curve)} points from 0 to {curve['target_abatement'].iloc[-1]/1e9:.3f} Gt")

    # Save CSV
    curve.to_csv(OUTPUT_CSV, index=False)
    print(f"Curve saved: {OUTPUT_CSV}")

    # Plot compressed and raw versions
    plot_mac(curve, sorted_units, str(OUTPUT_PNG))
    raw_curve, _ = build_mac_curve(all_units, step=TARGET_STEP, raw=True)
    raw_curve.to_csv(DATA_DIR / "standalone_mac_raw_units.csv", index=False)
    plot_raw_mac(all_units, str(OUTPUT_RAW_PNG))

    # Print summary
    print("\n=== MAC Curve Summary ===")
    for _, r in curve.iterrows():
        print(f"  Target={r['target_abatement']/1e9:.2f}Gt  achieved={r['achieved_abatement']/1e9:.2f}Gt  "
              f"cost=${r['total_cost']:,.0f}  MAC=${r['marginal_cost_diff']:,.1f}/t"
              if pd.notna(r['marginal_cost_diff']) else "")


if __name__ == "__main__":
    main()
