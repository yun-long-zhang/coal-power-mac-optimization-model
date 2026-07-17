import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---- User config (edit before running) ----
DATA_DIR_NAME = "data"
INPUT_CURVE_NAME = "gurobi_original_curve.csv"
OUTPUT_PREFIX = "carbon_price_response"
DEFAULT_REFERENCE_PRICES = [9.0, 56.04, 83.0]
# ------------------------------------------


def _load_curve(path: str, abatement_col: str, drop_zero_abatement: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"total_cost"}
    if abatement_col not in df.columns:
        raise ValueError(f"Abatement column '{abatement_col}' is missing in {path}")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")

    work = df.copy()
    if "status" in work.columns:
        work = work[work["status"].astype(str).str.lower() == "optimal"].copy()

    work[abatement_col] = pd.to_numeric(work[abatement_col], errors="coerce")
    work["total_cost"] = pd.to_numeric(work["total_cost"], errors="coerce")
    work = work.dropna(subset=[abatement_col, "total_cost"]).copy()

    # Multiple targets can map to near-identical achieved abatement; keep the cheapest one.
    work["abatement"] = work[abatement_col].astype(float)
    work = work.sort_values(["abatement", "total_cost"]).groupby("abatement", as_index=False).first()

    if drop_zero_abatement:
        work = work[work["abatement"] > 1e-6].copy()

    work = work.sort_values("abatement").reset_index(drop=True)
    if work.empty:
        raise ValueError("No valid points remain after filtering curve data.")
    return work[["abatement", "total_cost"]]


def _argmin_line(points: pd.DataFrame, p: float) -> int:
    vals = points["total_cost"].to_numpy() - p * points["abatement"].to_numpy()
    return int(np.argmin(vals))


def _default_price_bounds(points: pd.DataFrame) -> Tuple[float, float]:
    if len(points) < 2:
        return 0.0, 200.0

    dA = points["abatement"].diff().to_numpy()[1:]
    dC = points["total_cost"].diff().to_numpy()[1:]
    slopes = dC / np.where(np.abs(dA) < 1e-12, np.nan, dA)
    slopes = slopes[np.isfinite(slopes)]
    if len(slopes) == 0:
        return 0.0, 200.0

    p_min = min(0.0, float(np.nanmin(slopes)) * 1.2)
    p_max = max(200.0, float(np.nanmax(slopes)) * 1.2)
    if p_max <= p_min:
        p_max = p_min + 200.0
    return p_min, p_max


def build_price_step_intervals(points: pd.DataFrame, p_min: float, p_max: float) -> pd.DataFrame:
    A = points["abatement"].to_numpy()
    C = points["total_cost"].to_numpy()

    cands: List[float] = []
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dA = A[j] - A[i]
            if abs(dA) < 1e-12:
                continue
            p_ij = (C[j] - C[i]) / dA
            if np.isfinite(p_ij):
                cands.append(float(p_ij))

    boundaries = [p_min] + sorted(set(x for x in cands if p_min < x < p_max)) + [p_max]

    rows = []
    current_idx = None
    seg_start = p_min

    for k in range(len(boundaries) - 1):
        left = boundaries[k]
        right = boundaries[k + 1]
        mid = (left + right) / 2.0
        idx = _argmin_line(points, mid)

        if current_idx is None:
            current_idx = idx
            seg_start = left
            continue

        if idx != current_idx:
            rows.append(
                {
                    "p_low": seg_start,
                    "p_high": left,
                    "point_index": current_idx,
                    "A_star": float(A[current_idx]),
                    "TAC_star": float(C[current_idx]),
                }
            )
            current_idx = idx
            seg_start = left

    rows.append(
        {
            "p_low": seg_start,
            "p_high": p_max,
            "point_index": current_idx,
            "A_star": float(A[current_idx]),
            "TAC_star": float(C[current_idx]),
        }
    )

    out = pd.DataFrame(rows)
    out = out[out["p_high"] - out["p_low"] > 1e-12].reset_index(drop=True)
    return out


def build_price_response_grid(points: pd.DataFrame, p_min: float, p_max: float, p_step: float) -> pd.DataFrame:
    if p_step <= 0:
        raise ValueError("p_step must be positive.")

    grid = np.arange(p_min, p_max + 1e-12, p_step)
    A = points["abatement"].to_numpy()
    C = points["total_cost"].to_numpy()

    rows = []
    for p in grid:
        idx = _argmin_line(points, float(p))
        tac = float(C[idx])
        a_star = float(A[idx])
        net_obj = tac - float(p) * a_star
        rows.append(
            {
                "carbon_price": float(p),
                "point_index": idx,
                "A_star": a_star,
                "TAC_star": tac,
                "objective_T_minus_pA": net_obj,
            }
        )
    return pd.DataFrame(rows)


def _attach_reference_prices(
    points: pd.DataFrame,
    ref_prices: List[float],
) -> pd.DataFrame:
    A = points["abatement"].to_numpy()
    C = points["total_cost"].to_numpy()
    rows = []
    for p in ref_prices:
        idx = _argmin_line(points, p)
        rows.append(
            {
                "carbon_price": float(p),
                "point_index": int(idx),
                "A_star": float(A[idx]),
                "TAC_star": float(C[idx]),
                "objective_T_minus_pA": float(C[idx] - p * A[idx]),
            }
        )
    return pd.DataFrame(rows)


def plot_price_response(
    response: pd.DataFrame,
    ref: pd.DataFrame,
    out_png: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=260)

    x = response["carbon_price"].to_numpy()
    y = (response["A_star"].to_numpy() / 1e9)
    ax.step(x, y, where="post", linewidth=2.6, color="#1f77b4", label="Optimal abatement A*(p)")

    if not ref.empty:
        xr = ref["carbon_price"].to_numpy()
        yr = ref["A_star"].to_numpy() / 1e9
        ax.scatter(xr, yr, s=36, color="#d62728", zorder=4, label="Reference carbon prices")
        for _, r in ref.iterrows():
            ax.axvline(float(r["carbon_price"]), color="#999999", linewidth=0.9, linestyle="--", alpha=0.6)

    ax.set_xlabel("Carbon price ($/tCO2)")
    ax.set_ylabel("Optimal abatement A*(p) (GtCO2)")
    ax.set_title("Carbon price to optimal abatement response")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build carbon-price response A*(p) from a TAC curve by solving min_k [TAC_k - p * A_k]."
        )
    )
    parser.add_argument("--curve", default=None, help="Input curve CSV path. Default: data/gurobi_original_curve.csv")
    parser.add_argument(
        "--abatement-col",
        choices=["achieved_abatement", "target_abatement"],
        default="achieved_abatement",
        help="Which curve column to use as abatement axis.",
    )
    parser.add_argument("--price-min", type=float, default=None, help="Minimum carbon price for response grid.")
    parser.add_argument("--price-max", type=float, default=None, help="Maximum carbon price for response grid.")
    parser.add_argument("--price-step", type=float, default=0.5, help="Carbon price step for response grid.")
    parser.add_argument("--drop-zero-abatement", action="store_true", help="Drop A≈0 point before response computation.")
    parser.add_argument(
        "--reference-prices",
        type=float,
        nargs="*",
        default=DEFAULT_REFERENCE_PRICES,
        help="Reference carbon prices to annotate and report.",
    )
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX, help="Output prefix under data/.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, DATA_DIR_NAME)

    curve_path = args.curve or os.path.join(data_dir, INPUT_CURVE_NAME)
    if not os.path.isabs(curve_path):
        curve_path = os.path.join(data_dir, curve_path)

    points = _load_curve(
        path=curve_path,
        abatement_col=args.abatement_col,
        drop_zero_abatement=args.drop_zero_abatement,
    )

    p_min_def, p_max_def = _default_price_bounds(points)
    p_min = p_min_def if args.price_min is None else float(args.price_min)
    p_max = p_max_def if args.price_max is None else float(args.price_max)
    if p_max <= p_min:
        raise ValueError("price-max must be greater than price-min.")

    intervals = build_price_step_intervals(points, p_min, p_max)
    response = build_price_response_grid(points, p_min, p_max, args.price_step)
    ref = _attach_reference_prices(points, list(args.reference_prices))

    out_points = os.path.join(data_dir, f"{args.output_prefix}_frontier_points.csv")
    out_intervals = os.path.join(data_dir, f"{args.output_prefix}_step_intervals.csv")
    out_response = os.path.join(data_dir, f"{args.output_prefix}_response_grid.csv")
    out_ref = os.path.join(data_dir, f"{args.output_prefix}_reference_prices.csv")
    out_png = os.path.join(data_dir, f"{args.output_prefix}_response.png")
    out_xlsx = os.path.join(data_dir, f"{args.output_prefix}_response.xlsx")

    points.to_csv(out_points, index=False)
    intervals.to_csv(out_intervals, index=False)
    response.to_csv(out_response, index=False)
    ref.to_csv(out_ref, index=False)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        points.to_excel(writer, sheet_name="frontier_points", index=False)
        intervals.to_excel(writer, sheet_name="step_intervals", index=False)
        response.to_excel(writer, sheet_name="response_grid", index=False)
        ref.to_excel(writer, sheet_name="reference_prices", index=False)
    plot_price_response(response=response, ref=ref, out_png=out_png)

    print("Done.")
    print(f"Input curve: {curve_path}")
    print(f"Abatement column: {args.abatement_col}")
    print(f"Carbon price range: [{p_min:.3f}, {p_max:.3f}] with step {args.price_step}")
    print(f"Output points: {out_points}")
    print(f"Output intervals: {out_intervals}")
    print(f"Output response grid: {out_response}")
    print(f"Output reference prices: {out_ref}")
    print(f"Output figure: {out_png}")
    print(f"Output workbook: {out_xlsx}")


if __name__ == "__main__":
    main()
