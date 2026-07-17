import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_export_utils import save_figure_bundle


DATA_DIR_NAME = "data"
BC1_CURVE = "gurobi_subsidy_scenarios_1_bc_subsidy_curve.csv"
CC1_CURVE = "gurobi_subsidy_scenarios_1_cc_subsidy_curve.csv"
BC2_CURVE = "gurobi_subsidy_scenarios_2_bc_subsidy_curve.csv"
CC2_CURVE = "gurobi_subsidy_scenarios_2_cc_subsidy_curve.csv"
BASELINE_CURVE = "gurobi_original_curve.csv"
OUTPUT_PNG_NAME = "fig_mac_subsidy_scenarios.png"


def _load_curve(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"target_abatement", "total_cost", "marginal_cost_diff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label}: missing required columns {sorted(missing)}")

    out = df.copy()
    for c in ["target_abatement", "total_cost", "marginal_cost_diff"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["target_abatement", "marginal_cost_diff"]).copy()
    out = out.sort_values("target_abatement").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot subsidy scenario MAC curves vs baseline on one figure."
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--bc1-curve", default=BC1_CURVE)
    parser.add_argument("--cc1-curve", default=CC1_CURVE)
    parser.add_argument("--bc2-curve", default=BC2_CURVE)
    parser.add_argument("--cc2-curve", default=CC2_CURVE)
    parser.add_argument("--baseline-curve", default=BASELINE_CURVE)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(script_dir, DATA_DIR_NAME)

    bc1_path = os.path.join(data_dir, args.bc1_curve)
    cc1_path = os.path.join(data_dir, args.cc1_curve)
    bc2_path = os.path.join(data_dir, args.bc2_curve)
    cc2_path = os.path.join(data_dir, args.cc2_curve)
    base_path = os.path.join(data_dir, args.baseline_curve)

    curve_specs = [
        (bc1_path, "BC-subsidy-0.25"),
        (cc1_path, "CC-subsidy-35"),
        (bc2_path, "BC-subsidy-0.15"),
        (cc2_path, "CC-subsidy-105"),
        (base_path, "baseline"),
    ]
    for p, name in curve_specs:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} curve not found: {p}")

    output_png = args.output or os.path.join(data_dir, OUTPUT_PNG_NAME)

    bc1_df = _load_curve(bc1_path, "BC-subsidy-0.25")
    cc1_df = _load_curve(cc1_path, "CC-subsidy-35")
    bc2_df = _load_curve(bc2_path, "BC-subsidy-0.15")
    cc2_df = _load_curve(cc2_path, "CC-subsidy-105")
    base_df = _load_curve(base_path, "baseline")

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(12.8, 7.2))

    # Baseline in thick black
    x_base = base_df["target_abatement"].to_numpy(dtype=float) / 1e9
    y_base = base_df["marginal_cost_diff"].to_numpy(dtype=float)
    ax.plot(
        x_base, y_base,
        color="#1a1a1a", linewidth=3.2, linestyle="-", alpha=0.95,
        label="Baseline (no subsidy)", zorder=9,
    )

    # BC subsidy 0.15 RMB/kWh in light green
    x_bc1 = bc1_df["target_abatement"].to_numpy(dtype=float) / 1e9
    y_bc1 = bc1_df["marginal_cost_diff"].to_numpy(dtype=float)
    ax.plot(
        x_bc1, y_bc1,
        color="#2ca02c", linewidth=3, linestyle="--", alpha=0.92,
        label="BC electricity subsidy (0.15 RMB/kWh)", zorder=8,
    )

    # BC subsidy 0.25 RMB/kWh in dark green
    x_bc2 = bc2_df["target_abatement"].to_numpy(dtype=float) / 1e9
    y_bc2 = bc2_df["marginal_cost_diff"].to_numpy(dtype=float)
    ax.plot(
        x_bc2, y_bc2,
        color="#7fbf7f", linewidth=3, linestyle="--", alpha=0.92,
        label="BC electricity subsidy (0.25 RMB/kWh)", zorder=8,
    )

    # CC subsidy 35 RMB/tCO2 in light blue
    x_cc1 = cc1_df["target_abatement"].to_numpy(dtype=float) / 1e9
    y_cc1 = cc1_df["marginal_cost_diff"].to_numpy(dtype=float)
    ax.plot(
        x_cc1, y_cc1,
        color="#87ceeb", linewidth=3, linestyle="--", alpha=0.92,
        label="CC storage subsidy (35 RMB/tCO\u2082)", zorder=8,
    )

    # CC subsidy 105 RMB/tCO2 in dark blue
    x_cc2 = cc2_df["target_abatement"].to_numpy(dtype=float) / 1e9
    y_cc2 = cc2_df["marginal_cost_diff"].to_numpy(dtype=float)
    ax.plot(
        x_cc2, y_cc2,
        color="#1f77b4", linewidth=3, linestyle="--", alpha=0.92,
        label="CC storage subsidy (105 RMB/tCO\u2082)", zorder=8,
    )

    ax.axhline(0.0, color="#444444", linewidth=0.9, alpha=0.7)
    ax.grid(alpha=0.22, linestyle="--")
    ax.set_xlabel("Abatement target (Gt CO$_2$)", fontsize=14)
    ax.set_ylabel("Marginal abatement cost (USD/tCO$_2$)", fontsize=14)
    ax.set_title("MAC curves under subsidy scenarios", fontsize=14)

    x_max = max(
        np.nanmax(x_base) if len(x_base) > 0 else 0,
        np.nanmax(x_bc1) if len(x_bc1) > 0 else 0,
        np.nanmax(x_bc2) if len(x_bc2) > 0 else 0,
        np.nanmax(x_cc1) if len(x_cc1) > 0 else 0,
        np.nanmax(x_cc2) if len(x_cc2) > 0 else 0,
    )
    ax.set_xlim(0.0, max(0.1, float(x_max) * 1.03))

    # Legend
    handles = [
        Line2D([0], [0], color="#1a1a1a", lw=3.2, ls="-", label="Baseline (no subsidy)"),
        Line2D([0], [0], color="#2ca02c", lw=3, ls="--", label="BC electricity subsidy (0.25 RMB/kWh)"),
        Line2D([0], [0], color="#7fbf7f", lw=3, ls="--", label="BC electricity subsidy (0.15 RMB/kWh)"),
        Line2D([0], [0], color="#87ceeb", lw=3, ls="--", label="CC storage subsidy (35 RMB/tCO\u2082)"),
        Line2D([0], [0], color="#1f77b4", lw=3, ls="--", label="CC storage subsidy (105 RMB/tCO\u2082)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=14)

    fig.tight_layout()
    exported_frames: Dict[str, pd.DataFrame] = {
        "baseline_curve": base_df,
        "bc_subsidy_025_curve": bc1_df,
        "bc_subsidy_015_curve": bc2_df,
        "cc_subsidy_035_curve": cc1_df,
        "cc_subsidy_105_curve": cc2_df,
    }
    outputs = save_figure_bundle(
        fig,
        output_png,
        dpi=320,
        bbox_inches="tight",
        data_frames=exported_frames,
    )
    plt.close(fig)

    print("Done.")
    print(f"Output PNG: {outputs['png']}")
    print(f"Output PDF: {outputs['pdf']}")
    print(f"Output XLSX: {outputs['xlsx']}")


if __name__ == "__main__":
    main()
