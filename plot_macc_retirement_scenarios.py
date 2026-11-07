import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_export_utils import save_figure_bundle


DATA_DIR_NAME = "data"
BASELINE_CURVE_NAME = "gurobi_r2_base_curve.csv"
DEFAULT_YEARS = [2030, 2035, 2040, 2045]
DEFAULT_AGE_LIMITS = [40, 30]
DEFAULT_PREFIX = "gurobi_r2"
OUTPUT_PNG_NAME = "fig_macc_retirement_scenarios.png"


def _load_curve(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"target_abatement", "total_cost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label}: missing required columns {sorted(missing)}")

    out = df.copy()
    out["target_abatement"] = pd.to_numeric(out["target_abatement"], errors="coerce")
    out["total_cost"] = pd.to_numeric(out["total_cost"], errors="coerce")

    if "marginal_cost_diff" in out.columns:
        out["marginal_cost_diff"] = pd.to_numeric(out["marginal_cost_diff"], errors="coerce")
    else:
        out = out.sort_values("target_abatement").reset_index(drop=True)
        out["marginal_cost_diff"] = out["total_cost"].diff() / out["target_abatement"].diff().replace(0, np.nan)

    out = out.dropna(subset=["target_abatement", "marginal_cost_diff"]).copy()
    out = out.sort_values("target_abatement").reset_index(drop=True)
    return out


def _scenario_curve_path(data_dir: str, prefix: str, age_limit: int, year: int) -> str:
    return os.path.join(data_dir, f"{prefix}_t{age_limit}_{year}_curve.csv")


def _build_style_maps(years: List[int]) -> Tuple[Dict[int, str], Dict[int, str]]:
    year_colors = {
        years[0]: "#1f77b4",  # blue
        years[1]: "#17becf",  # cyan
        years[2]: "#2ca02c",  # green
        years[3]: "#9467bd",  # purple
    }
    age_linestyles = {
        40: "-",
        30: "-",
    }
    age_markers = {
        40: "o",
        30: "x",
    }

    return year_colors, age_linestyles, age_markers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot baseline and age-retirement scenario MAC curves on one figure."
    )
    parser.add_argument("--data-dir", default=None, help="Data directory. Defaults to script_dir/data")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Scenario file prefix, e.g. gurobi")
    parser.add_argument("--baseline", default=BASELINE_CURVE_NAME, help="Baseline curve csv filename")
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS, help="Scenario years, e.g. 2030 2035 2040 2045")
    parser.add_argument("--age-limits", nargs="+", type=int, default=DEFAULT_AGE_LIMITS, help="Age limits to compare, e.g. 40 30")
    parser.add_argument("--output", default=None, help="Output png path")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(script_dir, DATA_DIR_NAME)

    years = sorted(list(dict.fromkeys(args.years)))
    age_limits = list(dict.fromkeys(args.age_limits))
    if len(years) < 2:
        raise ValueError("Need at least 2 years for comparison.")

    baseline_path = os.path.join(data_dir, args.baseline)
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline curve not found: {baseline_path}")

    output_png = args.output or os.path.join(data_dir, OUTPUT_PNG_NAME)

    baseline = _load_curve(baseline_path, "baseline")

    curves = []
    missing_files = []
    for age_limit in age_limits:
        for year in years:
            p = _scenario_curve_path(data_dir, args.prefix, age_limit, year)
            if not os.path.exists(p):
                missing_files.append(p)
                continue
            df = _load_curve(p, f"age{age_limit}_{year}")
            df["age_limit"] = age_limit
            df["year"] = year
            curves.append(df)

    if missing_files:
        print("[warn] Missing scenario files:")
        for p in missing_files:
            print(f"  - {p}")

    if not curves:
        raise FileNotFoundError("No scenario curves found. Please check --prefix/--years/--age-limits.")

    year_colors, age_linestyles, age_markers = _build_style_maps(years)

    fig, ax = plt.subplots(1, 1, figsize=(12.8, 7.2))

    # Baseline first, in thick black.
    x_base = baseline["target_abatement"].to_numpy(dtype=float) / 1e9
    y_base = baseline["marginal_cost_diff"].to_numpy(dtype=float)
    ax.plot(
        x_base,
        y_base,
        color="#1a1a1a",
        linewidth=3.2,
        linestyle="-",
        alpha=0.95,
        label="Baseline (no age-based retirement)",
        zorder=9,
    )

    exported_frames: Dict[str, pd.DataFrame] = {"baseline_curve": baseline}

    for df in curves:
        age_limit = int(df["age_limit"].iloc[0])
        year = int(df["year"].iloc[0])
        x = df["target_abatement"].to_numpy(dtype=float) / 1e9
        y = df["marginal_cost_diff"].to_numpy(dtype=float)
        color = year_colors.get(year, "#7f7f7f")
        ls = age_linestyles.get(age_limit, "-")
        marker = age_markers.get(age_limit)

        markerfacecolor = "white" if age_limit == 40 else color

        ax.plot(
            x,
            y,
            color=color,
            linestyle=ls,
            linewidth=2.0,

            marker=marker,
            markersize=4,

            markerfacecolor=markerfacecolor,
            markeredgecolor=color,
            markeredgewidth=1.2,

            markevery=1,

            alpha=0.92,
            zorder=5,
        )
        exported_frames[f"age{age_limit}_{year}"] = df

    ax.axhline(0.0, color="#444444", linewidth=0.9, alpha=0.7)
    ax.grid(alpha=0.22, linestyle="--")
    ax.set_xlabel("Abatement target (Gt CO$_2$)")
    ax.set_ylabel("Marginal abatement cost (USD/tC$_2$)")
    ax.set_title("MAC curves under plant retirement scenarios by lifetime")

    x_max = max(
        [np.nanmax(x_base)]
        + [np.nanmax(df["target_abatement"].to_numpy(dtype=float) / 1e9) for df in curves if len(df) > 0]
    )
    ax.set_xlim(0.0, max(0.1, float(x_max) * 1.03))

    # Two clean legends: one for years (color), one for age-limit rule (line style).
    year_handles = [
        Line2D([0], [0], color=year_colors.get(y, "#7f7f7f"), lw=2.4, ls="-", label=f"Year {y}")
        for y in years
    ]
    age_handles = [
    Line2D(
        [0], [0],
        color="#333333",
        lw=2.4,
        ls="-",
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="#333333",
        markeredgewidth=1.2,
        label="Lifetime = 40 years",
    ),

    Line2D(
        [0], [0],
        color="#333333",
        lw=2.4,
        ls="-",
        marker="x",
        markersize=6,
        label="Lifetime = 30 years",
    ),
]
    baseline_handle = [
        Line2D([0], [0], color="#1a1a1a", lw=3.2, ls="-", label="Baseline")
    ]

    leg1 = ax.legend(handles=baseline_handle + year_handles, loc="upper left", bbox_to_anchor=(0.00, 1.00), frameon=True, fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=age_handles, loc="upper left", bbox_to_anchor=(0.00, 0.80), frameon=True, fontsize=9)

    fig.tight_layout()
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
