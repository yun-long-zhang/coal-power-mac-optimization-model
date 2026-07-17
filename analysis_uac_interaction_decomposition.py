import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import optimization_fleet_joint_milp_persistent as base


# ---- User config (edit before running) ----
DATA_DIR_NAME = "data"
INPUT_RESULTS_NAME = "gurobi_original_results.xlsx"
INPUT_WORKBOOK_NAME = "1885_coal_power_plants_original.xlsx"
INPUT_BIOMASS_DISTANCE_NAME = "Dji_200.csv"
INPUT_CO2_DISTANCE_NAME = "Dki_4000.csv"
OUTPUT_PREFIX = "uac_interaction_decomp"
# -------------------------------------------


def _safe_uac(cost: float, abate: float) -> float:
    if not np.isfinite(cost) or not np.isfinite(abate) or abs(abate) < 1e-12:
        return np.nan
    return cost / abate


def _shapley_two_factor(c0: float, a0: float, c1: float, a1: float) -> Tuple[float, float, float]:
    u0 = _safe_uac(c0, a0)
    u1 = _safe_uac(c1, a1)
    if not np.isfinite(u0) or not np.isfinite(u1):
        return np.nan, np.nan, np.nan

    # Exact two-factor decomposition of Delta(C/A):
    # DeltaU = cost_effect + abatement_baseline_effect.
    cost_effect = (c1 - c0) * 0.5 * ((1.0 / a0) + (1.0 / a1))
    abatement_effect = (0.5 * (c0 + c1)) * ((1.0 / a1) - (1.0 / a0))
    delta_u = u1 - u0
    return delta_u, cost_effect, abatement_effect


def _collect_pm_sheets(path: str) -> List[str]:
    xl = pd.ExcelFile(path)
    pm = [s for s in xl.sheet_names if re.fullmatch(r"pm\d+", str(s).strip().lower())]
    return sorted(pm, key=lambda x: int(x[2:]))


def _load_plant_and_cc_maps(workbook_path: str, dji_path: str, dki_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    plants, _, _, cc, _, _, _, _ = base.load_inputs(workbook_path, dji_path, dki_path)
    plants = plants.copy()
    cc = cc.copy()

    for c in ["energy_gj", "base_emission", "power", "af", "coal_price"]:
        plants[c] = pd.to_numeric(plants[c], errors="coerce").fillna(0.0)

    for c in ["cc_capex_per_year", "cc_opex_per_t", "cc_ele1_per_t", "cc_ele2_per_t"]:
        cc[c] = pd.to_numeric(cc[c], errors="coerce").fillna(0.0)

    return plants, cc


def _compute_bc_counterfactual_rows(df: pd.DataFrame, plants_map: pd.DataFrame) -> pd.DataFrame:
    left = df.copy()
    right = plants_map.copy()
    left["plant_id"] = left["plant_id"].map(base._norm_id)
    right["plant_id"] = right["plant_id"].map(base._norm_id)
    merged = left.merge(right[["plant_id", "base_emission", "power", "af", "coal_price", "energy_gj"]], on="plant_id", how="left")

    req_cols = [
        "target_id",
        "target_abatement",
        "achieved_abatement",
        "status",
        "plant_id",
        "x_ec",
        "blend_ratio",
        "high_ratio",
        "bc_feedstock_cost",
    ]
    miss = [c for c in req_cols if c not in merged.columns]
    if miss:
        raise ValueError(f"Missing required columns for BC decomposition: {miss}")

    out_rows = []
    for _, r in merged.iterrows():
        x_ec = float(pd.to_numeric(r.get("x_ec"), errors="coerce") or 0.0)
        blend = float(pd.to_numeric(r.get("blend_ratio"), errors="coerce") or 0.0)
        high = float(pd.to_numeric(r.get("high_ratio"), errors="coerce") or 0.0)

        base_emission = float(pd.to_numeric(r.get("base_emission"), errors="coerce") or 0.0)
        power = float(pd.to_numeric(r.get("power"), errors="coerce") or 0.0)
        af = float(pd.to_numeric(r.get("af"), errors="coerce") or 0.0)
        coal_price = float(pd.to_numeric(r.get("coal_price"), errors="coerce") or 0.0)
        energy = float(pd.to_numeric(r.get("energy_gj"), errors="coerce") or 0.0)
        feedstock = float(pd.to_numeric(r.get("bc_feedstock_cost"), errors="coerce") or 0.0)

        # With prior EC: effective blend ratio shrinks by 20% when EC selected.
        eff_with = blend * (1.0 - 0.2 * x_ec)
        eff_noec = blend

        a_with = base_emission * eff_with
        a_noec = base_emission * eff_noec

        c_with = (
            1.02 * (1153000.0 * eff_with * power + 4628000.0 * high * power) * af / 6.8589
            - coal_price * energy * eff_with
            + feedstock
        )
        c_noec = (
            1.02 * (1153000.0 * eff_noec * power + 4628000.0 * high * power) * af / 6.8589
            - coal_price * energy * eff_noec
            + feedstock
        )

        u_with = _safe_uac(c_with, a_with)
        u_noec = _safe_uac(c_noec, a_noec)
        delta_u, cost_effect, abatement_effect = _shapley_two_factor(c_noec, a_noec, c_with, a_with)

        out_rows.append(
            {
                "tech": "BC",
                "target_id": r.get("target_id"),
                "target_abatement": r.get("target_abatement"),
                "achieved_abatement": r.get("achieved_abatement"),
                "status": r.get("status"),
                "plant_id": r.get("plant_id"),
                "x_ec": x_ec,
                "blend_ratio": blend,
                "high_ratio": high,
                "A_noec": a_noec,
                "A_with_prior": a_with,
                "C_noec": c_noec,
                "C_with_prior": c_with,
                "U_noec": u_noec,
                "U_with_prior": u_with,
                "delta_uac": delta_u,
                "delta_uac_cost_numerator": cost_effect,
                "delta_uac_abatement_denominator": abatement_effect,
            }
        )

    return pd.DataFrame(out_rows)


def _compute_cc_counterfactual_rows(df: pd.DataFrame, plants_map: pd.DataFrame, cc_map: pd.DataFrame) -> pd.DataFrame:
    left = df.copy()
    pmap = plants_map.copy()
    cmap = cc_map.copy()
    left["plant_id"] = left["plant_id"].map(base._norm_id)
    pmap["plant_id"] = pmap["plant_id"].map(base._norm_id)
    cmap["plant_id"] = cmap["plant_id"].map(base._norm_id)
    merged = (
        left.merge(pmap[["plant_id", "energy_gj"]], on="plant_id", how="left")
        .merge(cmap[["plant_id", "cc_capex_per_year", "cc_opex_per_t", "cc_ele1_per_t", "cc_ele2_per_t"]], on="plant_id", how="left")
    )

    req_cols = [
        "target_id",
        "target_abatement",
        "achieved_abatement",
        "status",
        "plant_id",
        "x_cc",
        "x_ec_cc",
        "blend_ratio",
    ]
    miss = [c for c in req_cols if c not in merged.columns]
    if miss:
        raise ValueError(f"Missing required columns for CC decomposition: {miss}")

    out_rows = []
    for _, r in merged.iterrows():
        x_cc = float(pd.to_numeric(r.get("x_cc"), errors="coerce") or 0.0)
        x_ec_cc = float(pd.to_numeric(r.get("x_ec_cc"), errors="coerce") or 0.0)
        blend = float(pd.to_numeric(r.get("blend_ratio"), errors="coerce") or 0.0)

        energy = float(pd.to_numeric(r.get("energy_gj"), errors="coerce") or 0.0)
        capex = float(pd.to_numeric(r.get("cc_capex_per_year"), errors="coerce") or 0.0)
        opex = float(pd.to_numeric(r.get("cc_opex_per_t"), errors="coerce") or 0.0)
        ele1 = float(pd.to_numeric(r.get("cc_ele1_per_t"), errors="coerce") or 0.0)
        ele2 = float(pd.to_numeric(r.get("cc_ele2_per_t"), errors="coerce") or 0.0)

        # Use the same capture structure as the optimization model.
        br_xcc = blend * x_cc
        br_xec_cc = blend * x_ec_cc

        a_base = 0.9 * energy * (0.0901 * x_cc + (0.112 - 0.0901) * br_xcc)
        a_with = a_base - 0.2 * 0.9 * energy * (0.0901 * x_ec_cc + (0.112 - 0.0901) * br_xec_cc)
        a_noec = a_base

        cap_no_ec_part_with = 0.9 * energy * (
            0.0901 * (x_cc - x_ec_cc) + (0.112 - 0.0901) * (br_xcc - br_xec_cc)
        )
        cap_with_ec_part = 0.8 * 0.9 * energy * (
            0.0901 * x_ec_cc + (0.112 - 0.0901) * br_xec_cc
        )

        c_with = capex * x_cc + opex * a_with + ele1 * cap_no_ec_part_with + ele2 * cap_with_ec_part
        c_noec = capex * x_cc + opex * a_noec + ele1 * a_noec

        u_with = _safe_uac(c_with, a_with)
        u_noec = _safe_uac(c_noec, a_noec)
        delta_u, cost_effect, abatement_effect = _shapley_two_factor(c_noec, a_noec, c_with, a_with)

        out_rows.append(
            {
                "tech": "CC",
                "target_id": r.get("target_id"),
                "target_abatement": r.get("target_abatement"),
                "achieved_abatement": r.get("achieved_abatement"),
                "status": r.get("status"),
                "plant_id": r.get("plant_id"),
                "x_cc": x_cc,
                "x_ec_cc": x_ec_cc,
                "blend_ratio": blend,
                "A_noec": a_noec,
                "A_with_prior": a_with,
                "C_noec": c_noec,
                "C_with_prior": c_with,
                "U_noec": u_noec,
                "U_with_prior": u_with,
                "delta_uac": delta_u,
                "delta_uac_cost_numerator": cost_effect,
                "delta_uac_abatement_denominator": abatement_effect,
            }
        )

    return pd.DataFrame(out_rows)


def _summarize_per_target(plant_df: pd.DataFrame) -> pd.DataFrame:
    work = plant_df.copy()
    work = work[np.isfinite(work["delta_uac"])].copy()
    if work.empty:
        return pd.DataFrame()

    grouped = []
    for (tech, target_id), g in work.groupby(["tech", "target_id"], dropna=False):
        grouped.append(
            {
                "tech": tech,
                "target_id": target_id,
                "target_abatement": float(pd.to_numeric(g["target_abatement"], errors="coerce").median()),
                "n_plants": int(len(g)),
                "median_delta_uac": float(np.nanmedian(g["delta_uac"])),
                "p90_abs_delta_uac": float(np.nanpercentile(np.abs(g["delta_uac"]), 90)),
                "sum_cost_component": float(np.nansum(g["delta_uac_cost_numerator"])),
                "sum_abatement_component": float(np.nansum(g["delta_uac_abatement_denominator"])),
                "sum_delta_uac": float(np.nansum(g["delta_uac"])),
            }
        )

    return pd.DataFrame(grouped).sort_values(["tech", "target_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose downstream UAC change caused by prior EC into cost-numerator and abatement-baseline components "
            "at plant level using existing optimization detail sheets."
        )
    )
    parser.add_argument("--results", default=None, help="Results workbook path. Default: data/gurobi_original_results.xlsx")
    parser.add_argument("--workbook", default=None, help="Input workbook path. Default: data/1885_coal_power_plants_original.xlsx")
    parser.add_argument("--distance-csv", default=None, help="Biomass distance CSV path. Default: data/Dji_200.csv")
    parser.add_argument("--co2-distance-csv", default=None, help="CO2 distance CSV path. Default: data/Dki_4000.csv")
    parser.add_argument("--tech", choices=["BC", "CC", "BOTH"], default="BOTH", help="Which downstream technology to decompose.")
    parser.add_argument("--target-sheets", nargs="*", default=None, help="Optional pm sheets to process (e.g., pm20 pm30).")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX, help="Output file prefix under data/.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, DATA_DIR_NAME)

    results_path = args.results or os.path.join(data_dir, INPUT_RESULTS_NAME)
    workbook_path = args.workbook or os.path.join(data_dir, INPUT_WORKBOOK_NAME)
    dji_path = args.distance_csv or os.path.join(data_dir, INPUT_BIOMASS_DISTANCE_NAME)
    dki_path = args.co2_distance_csv or os.path.join(data_dir, INPUT_CO2_DISTANCE_NAME)

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results workbook not found: {results_path}")

    plants_map, cc_map = _load_plant_and_cc_maps(workbook_path, dji_path, dki_path)

    sheets = _collect_pm_sheets(results_path)
    if args.target_sheets:
        keep = set(args.target_sheets)
        sheets = [s for s in sheets if s in keep]
    if not sheets:
        raise ValueError("No pm sheets selected for decomposition.")

    plant_rows_all = []
    for s in sheets:
        df = pd.read_excel(results_path, sheet_name=s)
        if df.empty:
            continue
        if "status" in df.columns:
            df = df[df["status"].astype(str).str.lower() == "optimal"].copy()
        if df.empty:
            continue

        if args.tech in {"BC", "BOTH"}:
            bc_df = _compute_bc_counterfactual_rows(df, plants_map)
            bc_df["sheet"] = s
            plant_rows_all.append(bc_df)

        if args.tech in {"CC", "BOTH"}:
            cc_df = _compute_cc_counterfactual_rows(df, plants_map, cc_map)
            cc_df["sheet"] = s
            plant_rows_all.append(cc_df)

    if not plant_rows_all:
        raise ValueError("No decomposition rows were generated. Check selected sheets and fields.")

    plant_df = pd.concat(plant_rows_all, ignore_index=True)
    target_summary = _summarize_per_target(plant_df)

    out_plant = os.path.join(data_dir, f"{args.output_prefix}_plant_level.csv")
    out_target = os.path.join(data_dir, f"{args.output_prefix}_target_summary.csv")
    out_xlsx = os.path.join(data_dir, f"{args.output_prefix}_decomposition.xlsx")

    plant_df.to_csv(out_plant, index=False)
    target_summary.to_csv(out_target, index=False)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        plant_df.to_excel(writer, sheet_name="plant_level", index=False)
        target_summary.to_excel(writer, sheet_name="target_summary", index=False)

    print("Done.")
    print(f"Input results: {results_path}")
    print(f"Input workbook: {workbook_path}")
    print(f"Sheets processed: {len(sheets)}")
    print(f"Output plant-level: {out_plant}")
    print(f"Output target summary: {out_target}")
    print(f"Output workbook: {out_xlsx}")


if __name__ == "__main__":
    main()
