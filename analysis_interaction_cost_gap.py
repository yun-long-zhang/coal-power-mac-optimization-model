"""
Interaction-induced changes in the cost gap of biomass co-firing (BC) and carbon capture (CC),
post-processed from the fleet-model solutions (no re-optimization).

For each plant in a target's plant sheet (pmXX), all optimized decisions are held fixed and the
downstream technology is evaluated with and without the upstream technology:

  EC -> BC (plants with x_ec = 1 and theta > 0), same nominal co-firing ratio theta:
    with EC:    biomass input theta (1 - s_i) P;  without EC: theta P
    net BC cost C = investment (DB and GB, per kW of capacity at the nominal ratio) x (AF + fixed O&M share)
                    + feedstock + biomass transport - avoided coal cost (the latter three follow biomass input)
    biomass-based electricity kWh = generation x theta (EC does not change generation)
    unit cost gap u = C / kWh  (USD per kWh of biomass electricity, net of avoided coal cost)

  BC -> CC (plants with x_cc = 1 and theta > 0), same capture decisions and EC:
    with BC:    capture Q includes biogenic CO2;  without BC: coal CO2 only (theta = 0)
    CC cost C = capex + opex Q + ele1 Q_noEC + ele2 Q_EC + CO2 transport and storage (per tonne) Q
    unit cost gap u = C / Q  (USD per tonne captured)
    Under the avoided accounting Q includes the flue gas of the extra coal fired for the capture energy
    (Q_noEC x f0, Q_EC x fec, as in the fleet model).

Reported per plant and aggregated:
  gap_reduction = (u_without - u_with) x volume_with
      the reduction in the remaining unit cost gap, applied to the deployed volume
  total_cost_change = C_with - C_without
      the change in the technology's total annual cost

These are cost differences at fixed deployment. They are not subsidy outlays: under a uniform per-unit
subsidy (as in the subsidy scenarios), the required rate is set by the marginal plant.
"""

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd

import optimization_1_fleet_joint_milp_persistent as base


DATA_DIR_NAME = "data"
INPUT_RESULTS_NAME = "gurobi_r2_base_results.xlsx"
PROVINCE_MAP_NAME = "plant_province_r2.csv"
OUTPUT_PREFIX = "interaction_cost_gap_r2"
DEFAULT_TARGET_ID = 39  # P36 (3.61 Gt CO2), the carbon-neutrality node of the baseline (first target with abatement >= fleet emissions)
BC_INV_DB_RMB_PER_MW = 1153000.0
BC_INV_GB_EXTRA_RMB_PER_MW = 4628000.0


def _plant_params(workbook_path: str) -> pd.DataFrame:
    plants_raw = pd.read_excel(workbook_path, sheet_name="coal_plant_i")
    cc_raw = pd.read_excel(workbook_path, sheet_name="CC")
    p = pd.DataFrame(
        {
            "plant_id": plants_raw["plant_i"].map(base._norm_id),
            "energy_gj": plants_raw["energy_demand_GJ"].astype(float),
            "power": plants_raw["capacity_Xi"].astype(float),
            "af": plants_raw["annuity_factor"].astype(float),
            "coal_price": plants_raw["coal_price"].astype(float),
            "generation_kwh": plants_raw["capacity_Xi"].astype(float) * plants_raw["running_hours"].astype(float) * 1000.0,
        }
    )
    c = pd.DataFrame(
        {
            "plant_id": cc_raw["plant_i"].map(base._norm_id),
            "cc_capex_per_year": cc_raw["capex_dollar_per_year"].astype(float),
            "cc_opex_per_t": cc_raw["opex_dollar_per_t"].astype(float),
            "cc_ele1_per_t": cc_raw["electricity1_dollar_per_t"].astype(float),
            "cc_ele2_per_t": cc_raw["electricity2_dollar_per_t"].astype(float),
        }
    )
    return p.merge(c, on="plant_id", how="left")


def _bc_net_cost(theta: float, theta_h: float, theta_eff: float, r: pd.Series, bio_cost: float) -> float:
    # investment at the nominal co-firing ratio; avoided coal at the effective (EC-adjusted) ratio
    inv = (BC_INV_DB_RMB_PER_MW * theta + BC_INV_GB_EXTRA_RMB_PER_MW * theta_h) * r["power"]
    return inv * (r["bc_af"] + base.BC_FIXED_OM_SHARE) / base.RMB_PER_USD + bio_cost - r["coal_price"] * r["energy_gj"] * theta_eff


def plant_level(sheet: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    for col in ["ec_saving_ratio", "cc_capture_rate", "bc_af", "cc_fixed_cost_if_adopted", "cc_ele1_per_t", "cc_ele2_per_t", "cc_opex_per_t"]:
        if col not in sheet.columns:
            raise ValueError(f"Results sheet lacks '{col}'; rerun the fleet model with the current code.")
    d = sheet.copy()
    d["plant_id"] = d["plant_id"].map(base._norm_id)
    d = d.merge(params[["plant_id", "energy_gj", "power", "coal_price", "generation_kwh"]], on="plant_id", how="left", suffixes=("", "_p"))
    ef_c = base.EF_COAL_T_PER_GJ
    ef_d = base.EF_BIOMASS_T_PER_GJ - base.EF_COAL_T_PER_GJ

    rows = []
    for _, r in d.iterrows():
        x_ec, x_cc, y = float(r["x_ec"]), float(r["x_cc"]), float(r["x_ec_cc"])
        theta, theta_h = float(r["blend_ratio"]), float(r["high_ratio"])
        s, rate, E = float(r["ec_saving_ratio"]), float(r["cc_capture_rate"]), float(r["energy_gj"])
        out = {"plant_id": r["plant_id"]}

        # EC -> BC
        if x_ec > 0.5 and theta > 1e-9:
            th1 = theta * (1.0 - s)
            bio1 = float(r["bc_feedstock_cost"]) + float(r["biomass_transport_cost"])
            bio0 = bio1 * theta / th1
            c1 = _bc_net_cost(theta, theta_h, th1, r, bio1)
            c0 = _bc_net_cost(theta, theta_h, theta, r, bio0)
            kwh1 = kwh0 = r["generation_kwh"] * theta
            u1, u0 = c1 / kwh1, c0 / kwh0
            out.update(
                {
                    "ecbc_kwh_with": kwh1, "ecbc_kwh_without": kwh0,
                    "ecbc_cost_with": c1, "ecbc_cost_without": c0,
                    "ecbc_unit_with_per_kwh": u1, "ecbc_unit_without_per_kwh": u0,
                    "ecbc_gap_reduction": (u0 - u1) * kwh1,
                    "ecbc_total_cost_change": c1 - c0,
                }
            )

        # BC -> CC
        if x_cc > 0.5 and theta > 1e-9:
            f0 = float(r.get("cc_penalty_capture_factor", 1.0) or 1.0)
            if f0 > 1.0 + 1e-12:
                p = (1.0 - 1.0 / f0) / (rate * ef_c)  # extra coal per tonne captured
                fec = 1.0 / (1.0 - rate * ef_c * p * (1.0 - s))
            else:
                f0 = fec = 1.0
            q1_noec = f0 * rate * E * (ef_c * (x_cc - y) + ef_d * theta * (x_cc - y))
            q1_ec = fec * (1.0 - s) * rate * E * (ef_c * y + ef_d * theta * y)
            q1 = q1_noec + q1_ec
            q0_noec = f0 * rate * E * ef_c * (x_cc - y)
            q0_ec = fec * (1.0 - s) * rate * E * ef_c * y
            q0 = q0_noec + q0_ec
            ts_per_t = (float(r["co2_transport_cost"]) + float(r["co2_storage_cost"])) / q1 if q1 > 1e-9 else 0.0
            capex = r["cc_fixed_cost_if_adopted"] * x_cc  # annualized capex + fixed O&M

            def cc_cost(q_noec: float, q_ec: float) -> float:
                q = q_noec + q_ec
                return capex + r["cc_opex_per_t"] * q + r["cc_ele1_per_t"] * q_noec + r["cc_ele2_per_t"] * q_ec + ts_per_t * q

            c1, c0 = cc_cost(q1_noec, q1_ec), cc_cost(q0_noec, q0_ec)
            u1, u0 = c1 / q1, c0 / q0
            out.update(
                {
                    "bccc_capture_with": q1, "bccc_capture_without": q0,
                    "bccc_cost_with": c1, "bccc_cost_without": c0,
                    "bccc_unit_with_per_t": u1, "bccc_unit_without_per_t": u0,
                    "bccc_gap_reduction": (u0 - u1) * q1,
                    "bccc_total_cost_change": c1 - c0,
                }
            )
        rows.append(out)
    return pd.DataFrame(rows)


def summarize(pl: pd.DataFrame) -> dict:
    s = {}
    for pre, vol_with, unit_w, unit_wo in [
        ("ecbc", "ecbc_kwh_with", "ecbc_unit_with_per_kwh", "ecbc_unit_without_per_kwh"),
        ("bccc", "bccc_capture_with", "bccc_unit_with_per_t", "bccc_unit_without_per_t"),
    ]:
        g = pl.dropna(subset=[f"{pre}_gap_reduction"]) if f"{pre}_gap_reduction" in pl else pl.iloc[0:0]
        s[f"{pre}_n_plants"] = int(len(g))
        s[f"{pre}_gap_reduction_usd"] = float(g[f"{pre}_gap_reduction"].sum()) if len(g) else 0.0
        s[f"{pre}_total_cost_change_usd"] = float(g[f"{pre}_total_cost_change"].sum()) if len(g) else 0.0
        if len(g):
            w = g[vol_with]
            s[f"{pre}_volume_with"] = float(w.sum())
            s[f"{pre}_mean_unit_with"] = float((g[unit_w] * w).sum() / w.sum())
            s[f"{pre}_mean_unit_without"] = float((g[unit_wo] * w).sum() / w.sum())
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--target-id", type=int, default=DEFAULT_TARGET_ID, help="Target sheet used for plant and provincial tables.")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_DIR_NAME)
    results_path = args.results_path or os.path.join(data_dir, INPUT_RESULTS_NAME)
    params = _plant_params(os.path.join(data_dir, base.INPUT_WORKBOOK_NAME))
    province = pd.read_csv(os.path.join(data_dir, PROVINCE_MAP_NAME), dtype={"plant_id": str})
    province["plant_id"] = province["plant_id"].map(base._norm_id)

    xl = pd.ExcelFile(results_path)
    sheets = sorted([s for s in xl.sheet_names if s.startswith("pm")], key=lambda x: int(x[2:]))
    summary_rows, pinned = [], None
    for sh in sheets:
        sheet = pd.read_excel(xl, sheet_name=sh)
        tid = int(sheet["target_id"].iloc[0])
        pl = plant_level(sheet, params)
        row = {"target_id": tid, "target_abatement": float(sheet["target_abatement"].iloc[0])}
        row.update(summarize(pl))
        summary_rows.append(row)
        if tid == args.target_id:
            pinned = pl.merge(province[["plant_id", "province"]], on="plant_id", how="left")
        print(
            f"{sh}: EC->BC gap reduction {row['ecbc_gap_reduction_usd']/1e9:.3f} bn ({row['ecbc_n_plants']} plants), "
            f"BC->CC gap reduction {row['bccc_gap_reduction_usd']/1e9:.3f} bn, total CC cost change "
            f"{row['bccc_total_cost_change_usd']/1e9:+.3f} bn ({row['bccc_n_plants']} plants)"
        )

    summary = pd.DataFrame(summary_rows)
    if pinned is None:
        raise ValueError(f"Target id {args.target_id} not found in {results_path}")
    cols = ["ecbc_gap_reduction", "ecbc_total_cost_change", "bccc_gap_reduction", "bccc_total_cost_change"]
    prov = pinned.groupby("province")[cols].sum(min_count=1).fillna(0.0).sort_values("bccc_gap_reduction", ascending=False)
    total = summary.loc[summary["target_id"] == args.target_id].iloc[0]
    for c in ["ecbc_gap_reduction", "bccc_gap_reduction"]:
        if not np.isclose(prov[c].sum(), total[f"{c}_usd"], rtol=1e-9, atol=1.0):
            raise AssertionError(f"Provincial sum of {c} does not equal the national total.")

    out = os.path.join(data_dir, f"{args.output_prefix}.xlsx")
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        summary.to_excel(w, sheet_name="by_target", index=False)
        pinned.to_excel(w, sheet_name=f"plants_target{args.target_id:02d}", index=False)
        prov.reset_index().to_excel(w, sheet_name=f"provinces_target{args.target_id:02d}", index=False)
    print(f"\nPinned target {args.target_id} (A = {total['target_abatement']/1e9:.2f} Gt):")
    print(prov[["ecbc_gap_reduction", "bccc_gap_reduction"]].div(1e6).round(1).head(8).to_string())
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
