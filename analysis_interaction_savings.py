"""
Post-processing: Compute EC→BC and BC→CC interaction cost savings
from existing joint-model results without re-running MILP.

For each plant i in each target sheet (pm*):

  EC→BC (same θ, WITH EC vs WITHOUT EC):
    Current (with EC):  bc_total  = inv + feedstock + transport  (coal saving excluded)
    Counterfactual (no EC): same inv, feedstock+transport scale by θ/θ_eff
      bc_total_no_ec  = inv + (feedstock+transport) × (θ/θ_eff)
    Denominator: bc_elec = blend_ratio × total_coal_elec (same in both)
    ec_saving_bc = bc_total_no_ec - bc_total_with_ec  (+ = EC lowers BC cost)

  BC→CC (same x_CC, WITH BC vs WITHOUT BC (θ=0)):
    Current (with BC):  cc_total  = capex + opex + energy + transport + storage
                         cc_capture includes biomass CO₂ + coal CO₂
    Counterfactual (no BC): same capex; capture drops to coal-only;
      opex, energy, transport, storage scale with capture ratio
    Savings = (unit_cost_no_bc - unit_cost_with_bc) × old_cc_capture
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DATA_DIR_NAME = "data"
OUTPUT_SUFFIX = "_interaction_savings.xlsx"
INPUT_RESULTS_NAME = "gurobi_original_results.xlsx"
INPUT_CURVE_NAME = "gurobi_original_curve.csv"
OUTPUT_PREFIX = "interaction_savings"


def _sheet_sort_key(name: str) -> Tuple[int, str]:
    """Sort sheet names like pm01, pm02, ..."""
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        return int(digits), name
    return (10 ** 9), name


def compute_interaction_savings(
    plant_df: pd.DataFrame,
    model_data_path: str,
) -> pd.DataFrame:
    """
    Add counterfactual saving columns to a single target's plant detail DataFrame.

    Parameters
    ----------
    plant_df : pd.DataFrame
        Detail plant sheet (e.g. pm01) from existing results.
    model_data_path : str
        Path to input workbook (for coal_price, power, AF, etc.).

    Returns
    -------
    pd.DataFrame with additional columns:
        bc_cost_no_ec_cf, bc_abatement_no_ec_cf, ec_saving_on_bc, ec_saving_on_bc_per_abate
        cc_cost_no_bc_cf, cc_capture_no_bc_cf, bc_saving_on_cc, bc_saving_on_cc_per_capture
    """
    df = plant_df.copy()
    needed = {"plant_id", "energy_gj", "base_emission", "power", "af", "coal_price",
              "cc_capex_per_year", "cc_opex_per_t", "cc_ele1_per_t", "cc_ele2_per_t"}
    # Load plant-level parameters from input workbook
    # Use the SAME column-name resolution as the MILP model for consistency.
    plants_raw = pd.read_excel(model_data_path, sheet_name="coal_plant_i")
    pcols = list(plants_raw.columns)
    pid_col = [c for c in pcols if str(c).strip().lower() in {"plant_i", "plantid", "id"}][0]
    en_col = [c for c in pcols if str(c).strip().lower() in {"energy_demand_gj", "energydemandgj", "energy_demand", "energy"}][0]
    em_col = [c for c in pcols if str(c).strip().lower() in {"carbon_emissions_ton", "carbonemissionston", "co2", "carbon_emissions"}][0]
    pw_col = [c for c in pcols if str(c).strip().lower() in {"power", "power_xi", "capacity_xi", "capacity"}][0]
    af_col = [c for c in pcols if str(c).strip().lower() in {"af", "annuity_factor"}][0]
    co_col = [c for c in pcols if str(c).strip().lower() in {"coal_price", "coalprice"}][0]
    eff_col = [c for c in pcols if str(c).strip().lower() in {"efficiency", "eff"}][0]
    rh_col = [c for c in pcols if str(c).strip().lower() in {"running_hours", "runninghours", "hours"}][0]

    # Build a plant_id → parameter dict keyed by string (to match saved result's plant_id)
    params_dict = {}
    for _, r in plants_raw.iterrows():
        pid = str(int(float(r[pid_col])))  # e.g. "1"
        params_dict[pid] = {
            "energy_gj": float(pd.to_numeric(r[en_col], errors="coerce") or 0.0),
            "base_emission": float(pd.to_numeric(r[em_col], errors="coerce") or 0.0),
            "power": float(pd.to_numeric(r[pw_col], errors="coerce") or 0.0),
            "af": float(pd.to_numeric(r[af_col], errors="coerce") or 0.0),
            "coal_price": float(pd.to_numeric(r[co_col], errors="coerce") or 0.0),
            "efficiency": float(pd.to_numeric(r[eff_col], errors="coerce") or 0.0) if eff_col else 0.0,
            "running_hours": float(pd.to_numeric(r[rh_col], errors="coerce") or 0.0) if rh_col else 0.0,
        }

    # Load CC decomposed coefficients, same approach
    cc_raw = pd.read_excel(model_data_path, sheet_name="CC")
    ccols = list(cc_raw.columns)
    cc_pid_col = [c for c in ccols if str(c).strip().lower() in {"plant_i", "plantid", "id"}][0]
    cc_col_map = {
        "cc_capex_per_year": "capex_dollar_per_year",
        "cc_opex_per_t": "opex_dollar_per_t",
        "cc_ele1_per_t": "electricity1_dollar_per_t",
        "cc_ele2_per_t": "electricity2_dollar_per_t",
        "cc_total_cost": "unit _cost",  # fallback: CC sheet has unit_cost, not total_cost
    }
    def _find_cc_col(search_key: str) -> str:
        """Find the exact column in cc_raw whose lowercase name contains the key."""
        for c in ccols:
            if search_key in str(c).strip().lower():
                return c
        return ""
    cc_params_dict = {}
    for _, r in cc_raw.iterrows():
        pid = str(int(float(r[cc_pid_col])))
        cc_params_dict[pid] = {}
        for out_name, search_key in cc_col_map.items():
            col = _find_cc_col(search_key)
            if col:
                cc_params_dict[pid][out_name] = float(pd.to_numeric(r[col], errors="coerce") or 0.0)
            else:
                cc_params_dict[pid][out_name] = 0.0

    # Map parameters into the DataFrame using plant_id (string key)
    df["plant_id_str"] = pd.to_numeric(df["plant_id"], errors="coerce").astype("Int64").astype(str)
    df["energy_gj"] = df["plant_id_str"].map(lambda pid: params_dict.get(pid, {}).get("energy_gj", 0.0))
    df["base_emission"] = df["plant_id_str"].map(lambda pid: params_dict.get(pid, {}).get("base_emission", 0.0))
    df["power"] = df["plant_id_str"].map(lambda pid: params_dict.get(pid, {}).get("power", 0.0))
    df["af"] = df["plant_id_str"].map(lambda pid: params_dict.get(pid, {}).get("af", 0.0))
    df["coal_price"] = df["plant_id_str"].map(lambda pid: params_dict.get(pid, {}).get("coal_price", 0.0))
    df["efficiency"] = df["plant_id_str"].map(lambda pid: params_dict.get(pid, {}).get("efficiency", 0.0))
    df["running_hours"] = df["plant_id_str"].map(lambda pid: params_dict.get(pid, {}).get("running_hours", 0.0))
    for col in ["cc_capex_per_year", "cc_opex_per_t", "cc_ele1_per_t", "cc_ele2_per_t", "cc_total_cost"]:
        df[col] = df["plant_id_str"].map(lambda pid: cc_params_dict.get(pid, {}).get(col, 0.0))

    results = []
    for _, row in df.iterrows():
        # Extract decision variables from the saved result
        x_ec = float(row.get("x_ec", 0.0))
        x_cc = float(row.get("x_cc", 0.0))
        y_ec_cc = float(row.get("x_ec_cc", 0.0))
        blend_ratio = float(row.get("blend_ratio", 0.0))
        effective_ratio = float(row.get("effective_blend_ratio", blend_ratio))
        high_ratio = float(row.get("high_ratio", 0.0))

        # Plant parameters
        energy = float(row.get("energy_gj", 0.0))
        base_em = float(row.get("base_emission", 0.0))
        power = float(row.get("power", 0.0))
        af = float(row.get("af", 0.0))
        coal_px = float(row.get("coal_price", 0.0))
        efficiency = float(row.get("efficiency", 0.0))
        running_hours = float(row.get("running_hours", 0.0))

        # BC cost components from saved result
        bc_cost = float(row.get("bc_cost", 0.0))
        bc_abatement = float(row.get("bc_abatement_ton", 0.0))
        bc_feedstock = float(row.get("bc_feedstock_cost", 0.0))
        bc_investment = float(row.get("bc_investment_cost", 0.0))
        bc_coal_saving = float(row.get("bc_coal_saving_credit", 0.0))
        bc_use_gj = float(row.get("bc_use_gj", 0.0))
        bc_transport = float(row.get("biomass_transport_cost", 0.0))

        # CC cost components from saved result
        cc_cost = float(row.get("cc_cost", 0.0))
        cc_capture = float(row.get("cc_capture_ton", 0.0))
        capex = float(row.get("cc_capex_per_year", 0.0))
        opex = float(row.get("cc_opex_per_t", 0.0))
        ele1 = float(row.get("cc_ele1_per_t", 0.0))
        ele2 = float(row.get("cc_ele2_per_t", 0.0))
        cc_legacy = float(row.get("cc_total_cost", 0.0))
        cc_has_decomposed = (capex != 0 or opex != 0 or ele1 != 0 or ele2 != 0)
        co2_transport_cost_val = float(row.get("co2_transport_cost", 0.0))
        co2_storage_cost_val = float(row.get("co2_storage_cost", 0.0))

        # ═══════════════════════════════════════════════════════════════
        #  EC → BC : same blend_ratio θ, WITH EC vs WITHOUT EC
        # ═══════════════════════════════════════════════════════════════
        # Counterfactual: fix θ_i at joint solution, set EC interaction off.
        # Without EC (x_EC=0 → brxEC=0), effective_ratio = θ_i.
        # All BC cost components recomputed at effective_ratio_noEC = θ_i.
        # Transport cost excluded from BC comparison (consistent with how
        # bc_internal_cost excludes it — it's added in total_cost_component).
        # ============================================================
        eff_no_ec = blend_ratio
        bio_scale = (eff_no_ec / effective_ratio) if effective_ratio > 1e-12 else 1.0

        # Counterfactual BC cost components (no EC, same θ)
        bc_inv_no_ec = 1.02 * (1153000 * eff_no_ec * power + 4628000 * high_ratio * power) * af / 6.8589
        bc_coal_save_no_ec = coal_px * energy * eff_no_ec
        bc_feed_no_ec = bc_feedstock * bio_scale

        # Current BC cost (excludes transport — consistent with model's bc_internal_cost)
        bc_current_tech = bc_investment + bc_feedstock
        bc_current_net  = bc_current_tech - bc_coal_saving

        bc_cf_tech = bc_inv_no_ec + bc_feed_no_ec
        bc_cf_net  = bc_cf_tech - bc_coal_save_no_ec

        ec_saving_bc_technical = bc_cf_tech - bc_current_tech   # + = EC reduces BC tech cost
        ec_saving_bc_net       = bc_cf_net - bc_current_net     # + = EC reduces BC net cost

        # Unit costs (same bc_use_gj denominator in both)
        total_elec_kwh = power * running_hours * 1000.0 if power > 0 and running_hours > 0 else 0.0
        bc_elec_kwh = blend_ratio * total_elec_kwh
        bc_unit_cost_with_ec = bc_current_tech / bc_use_gj if bc_use_gj > 1e-12 else 0.0
        bc_unit_cost_no_ec   = bc_cf_tech / bc_use_gj if bc_use_gj > 1e-12 else 0.0
        bc_unit_net_with_ec  = bc_current_net / bc_use_gj if bc_use_gj > 1e-12 else 0.0
        bc_unit_net_no_ec    = bc_cf_net / bc_use_gj if bc_use_gj > 1e-12 else 0.0

        # ═══════════════════════════════════════════════════════════════
        #  BC → CC : same x_CC, WITH BC vs WITHOUT BC (θ=0)
        # ═══════════════════════════════════════════════════════════════
        # Current (with BC): cc_total = capex + opex + ele1 + transport + storage
        #                     cc_capture includes biomass CO₂ + coal CO₂
        # Counterfactual (no BC, θ=0, same x_CC, same EC):
        #   - Capex unchanged (capacity × x_cc)
        #   - Capture volume = coal-only CO₂, but EC still reduces coal by 20%
        #     cc_capture_no_bc = 0.9 × E × 0.0901 × (x_cc − 0.2 × y_ec_cc)
        #   - Opex, ele1, transport, storage all scale with capture volume
        #   - Denominator: cc_capture_no_bc for unit cost
        # Savings = (unit_cost_without_BC - unit_cost_with_BC) × old_capture_volume
        # Note: EC is NOT involved in CC cost components (no ele1/ele2 split),
        #       but its physical coal-reduction effect affects capture volume.
        cc_capture_no_bc = 0.9 * energy * 0.0901 * max(0.0, x_cc - 0.2 * y_ec_cc)  # coal-only, EC-adjusted
        cc_scale = (cc_capture_no_bc / cc_capture) if cc_capture > 1e-12 else 0.0

        cc_total_cost_with_bc = cc_cost + co2_transport_cost_val + co2_storage_cost_val
        if cc_has_decomposed:
            cc_capex_part = capex * x_cc  # fixed, unchanged
            cc_var_part_no_bc = (opex + ele1) * cc_capture_no_bc  # variable scales with coal-only capture
        else:
            cc_capex_part = 0.0
            cc_var_part_no_bc = cc_legacy * x_cc * cc_scale
        cc_transport_no_bc = co2_transport_cost_val * cc_scale
        cc_storage_no_bc = co2_storage_cost_val * cc_scale
        cc_total_cost_no_bc = cc_capex_part + cc_var_part_no_bc + cc_transport_no_bc + cc_storage_no_bc

        cc_unit_cost_with_bc = cc_total_cost_with_bc / cc_capture if cc_capture > 1e-12 else 0.0
        cc_unit_cost_no_bc = cc_total_cost_no_bc / cc_capture_no_bc if cc_capture_no_bc > 1e-12 else 0.0
        bc_saving_cc_total = (cc_unit_cost_no_bc - cc_unit_cost_with_bc) * cc_capture if cc_capture > 1e-12 else 0.0  # + = BC lowers CC cost
        bc_saving_cc_unit_only = (cc_unit_cost_no_bc - cc_unit_cost_with_bc) * cc_capture_no_bc if cc_capture_no_bc > 1e-12 else 0.0  # pure unit-cost effect

        # Output all intermediate results
        row_out = row.to_dict()
        row_out.update({
            # BC unit-cost comparison
            "bc_use_gj": bc_use_gj,
            "bc_elec_kwh": bc_elec_kwh,
            "bc_investment_cost": bc_investment,
            "bc_feedstock_cost": bc_feedstock,
            "bc_transport_cost": bc_transport,
            "bc_coal_saving_credit": bc_coal_saving,
            "bc_total_cost_with_ec": bc_current_tech,
            "bc_total_cost_no_ec": bc_cf_tech,
            "bc_unit_cost_with_ec_per_gj": bc_unit_cost_with_ec,
            "bc_unit_cost_no_ec_per_gj": bc_unit_cost_no_ec,
            "ec_saving_bc_technical": ec_saving_bc_technical,
            "ec_saving_bc_net": ec_saving_bc_net,
            # CC unit-cost comparison
            "cc_capex_part": capex * x_cc,
            "cc_var_with_bc": cc_cost - capex * x_cc if cc_has_decomposed else cc_cost,
            "cc_transport_cost": co2_transport_cost_val,
            "cc_storage_cost": co2_storage_cost_val,
            "cc_total_cost_with_bc": cc_total_cost_with_bc,
            "cc_unit_cost_with_bc_per_ton": cc_unit_cost_with_bc,
            "cc_capture_no_bc_ton": cc_capture_no_bc,
            "cc_total_cost_no_bc": cc_total_cost_no_bc,
            "cc_unit_cost_no_bc_per_ton": cc_unit_cost_no_bc,
            "bc_saving_cc_total_volume_weighted": bc_saving_cc_total,
            "bc_saving_cc_unit_only": bc_saving_cc_unit_only,
        })
        results.append(row_out)

    out = pd.DataFrame(results)
    return out


def process_results(
    results_path: str,
    workbook_path: str,
    curve_path: str,
    output_path: str,
) -> str:
    """Read existing results, compute savings for all targets, output new Excel."""
    xl = pd.ExcelFile(results_path)

    # Identify plant detail sheets
    plant_sheets = sorted(
        [s for s in xl.sheet_names if s.startswith("pm")],
        key=_sheet_sort_key,
    )
    if not plant_sheets:
        raise ValueError(f"No 'pm*' sheets found in {results_path}")

    # Load curve for target mapping
    curve = pd.read_csv(curve_path)
    target_map = dict(zip(
        pd.to_numeric(curve["target_id"], errors="coerce").astype(int),
        pd.to_numeric(curve["target_abatement"], errors="coerce"),
    ))

    print(f"Processing {len(plant_sheets)} target sheets from {os.path.basename(results_path)}...")

    # Aggregate savings across all targets
    agg_rows = []
    all_detail = []

    for sname in plant_sheets:
        pdf = pd.read_excel(results_path, sheet_name=sname)
        if pdf.empty:
            continue
        target_id = int(pd.to_numeric(pdf["target_id"], errors="coerce").iloc[0])
        target_abatement = float(pd.to_numeric(pdf["target_abatement"], errors="coerce").iloc[0])

        result_df = compute_interaction_savings(pdf, workbook_path)

        # Aggregate
        total_ec_saving_tech = float(result_df["ec_saving_bc_technical"].sum())
        total_ec_saving_net = float(result_df["ec_saving_bc_net"].sum())
        total_abatement = float(pd.to_numeric(result_df["total_abatement_ton"], errors="coerce").sum())
        total_bc_elec = float(result_df["bc_use_gj"].sum())
        total_bc_elec_kwh = float(result_df["bc_elec_kwh"].sum())
        weighted_avg_bc_unit_with_ec = (
            float((result_df["bc_total_cost_with_ec"].sum() / result_df["bc_elec_kwh"].sum()))
            if total_bc_elec_kwh > 1e-12 else 0.0
        )
        weighted_avg_bc_unit_no_ec = (
            float((result_df["bc_total_cost_no_ec"].sum() / result_df["bc_elec_kwh"].sum()))
            if total_bc_elec_kwh > 1e-12 else 0.0
        )

        # CC metrics
        total_cc_capture = float(result_df["cc_capture_ton"].sum())
        total_cc_capture_no_bc = float(result_df["cc_capture_no_bc_ton"].sum())
        weighted_avg_cc_unit_with_bc = (
            float((result_df["cc_total_cost_with_bc"].sum() / result_df["cc_capture_ton"].sum()))
            if total_cc_capture > 1e-12 else 0.0
        )
        weighted_avg_cc_unit_no_bc = (
            float((result_df["cc_total_cost_no_bc"].sum() / result_df["cc_capture_no_bc_ton"].sum()))
            if total_cc_capture_no_bc > 1e-12 else 0.0
        )
        total_bc_saving_cc = float(result_df["bc_saving_cc_total_volume_weighted"].sum())
        total_bc_saving_cc_unit = float(result_df["bc_saving_cc_unit_only"].sum())
        n_ec_benefit = int((result_df["ec_saving_bc_technical"] > 1e-6).sum())

        agg_rows.append({
            "target_id": target_id,
            "target_abatement": target_abatement,
            "total_abatement": total_abatement,
            # BC metrics
            "total_bc_elec_gj": total_bc_elec,
            "total_bc_elec_kwh": total_bc_elec_kwh,
            "weighted_avg_bc_unit_with_ec_per_kwh": weighted_avg_bc_unit_with_ec,
            "weighted_avg_bc_unit_no_ec_per_kwh": weighted_avg_bc_unit_no_ec,
            "ec_saving_bc_technical": total_ec_saving_tech,
            "ec_saving_bc_net": total_ec_saving_net,
            # CC metrics
            "total_cc_capture_ton": total_cc_capture,
            "total_cc_capture_no_bc_ton": total_cc_capture_no_bc,
            "weighted_avg_cc_unit_with_bc_per_ton": weighted_avg_cc_unit_with_bc,
            "weighted_avg_cc_unit_no_bc_per_ton": weighted_avg_cc_unit_no_bc,
            "bc_saving_cc_total": total_bc_saving_cc,
            "bc_saving_cc_unit_only": total_bc_saving_cc_unit,
            "n_plants_ec_benefit_bc": n_ec_benefit,
        })

        # Store target id in each detail row
        result_df["target_id"] = target_id
        result_df["target_abatement"] = target_abatement
        all_detail.append(result_df)

        print(f"  {sname}: target={target_abatement:,.0f}  "
              f"EC→BC tech=${total_ec_saving_tech:,.2f} net=${total_ec_saving_net:,.2f}  "
              f"BC→CC vol-wt=${total_bc_saving_cc:,.2f} unit-only=${total_bc_saving_cc_unit:,.2f}  "
              f"BC elec: {total_bc_elec:,.0f} GJ"
              f" | CC capture: {total_cc_capture:,.0f}→{total_cc_capture_no_bc:,.0f} tons")

    # Write output
    summary = pd.DataFrame(agg_rows).sort_values("target_id").reset_index(drop=True)
    detail = pd.concat(all_detail, ignore_index=True)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="savings_summary", index=False)
        detail.to_excel(writer, sheet_name="savings_detail", index=False)

    print(f"\nDone. Output: {output_path}")
    print(f"  Summary: {len(summary)} targets")
    print(f"  Detail: {len(detail)} plant-records")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process existing joint MILP results to compute EC→BC and BC→CC interaction savings."
    )
    parser.add_argument("--results-path", default=None,
                        help="Path to existing results .xlsx. Default: data/gurobi_original_results.xlsx")
    parser.add_argument("--workbook-path", default=None,
                        help="Path to input workbook with parameters. Default: data/1885_coal_power_plants_original.xlsx")
    parser.add_argument("--curve-path", default=None,
                        help="Path to curve CSV. Default: data/gurobi_original_curve.csv")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX,
                        help="Output file prefix.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, DATA_DIR_NAME)

    results_path = args.results_path or os.path.join(data_dir, INPUT_RESULTS_NAME)
    workbook_path = args.workbook_path or os.path.join(data_dir, "1885_coal_power_plants_original.xlsx")
    curve_path = args.curve_path or os.path.join(data_dir, INPUT_CURVE_NAME)
    output_path = os.path.join(data_dir, f"{args.output_prefix}{OUTPUT_SUFFIX}")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results not found: {results_path}")
    if not os.path.exists(workbook_path):
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")
    if not os.path.exists(curve_path):
        raise FileNotFoundError(f"Curve not found: {curve_path}")

    process_results(results_path, workbook_path, curve_path, output_path)


if __name__ == "__main__":
    main()
