import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from pulp import (
    GUROBI_CMD,
    LpBinary,
    LpContinuous,
    LpMaximize,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)


# ---- User config (edit before running) ----
DATA_DIR_NAME = "data"
INPUT_WORKBOOK_NAME = "1885_coal_power_plants_original.xlsx"
INPUT_BIOMASS_DISTANCE_NAME = "Dji_200.csv"
INPUT_CO2_DISTANCE_NAME = "Dki_4000.csv"
OUTPUT_CURVE_SUFFIX = "_curve.csv"
OUTPUT_RESULTS_SUFFIX = "_results.xlsx"
OUTPUT_LP_CURVE_SUFFIX = "_lp_curve.csv"
OUTPUT_SUMMARY_SUFFIX = "_summary.csv"
N_TARGETS = 48
#EXTRA_TARGETS = [1300000000.0, 3700000000.0]
EXTRA_TARGETS = []
OUTPUT_PREFIX = "gurobi_subsidy_scenarios_1"
TARGET_MATCH_MODE = "cost_only"  # "min_overshoot" | "cost_only"
TARGET_MODE_SWITCH_ABATEMENT = 1.8e9  # e.g. 1e9; <= threshold uses min_overshoot, > threshold uses cost_only
BLEND_CAP_RATIO = 1.0
TRANSPORT_COST_PER_GJ_KM = 0.0017 * 1.5
BIOMASS_FEEDSTOCK_COST_PER_GJ = 2.474
MAX_LINKS_PER_PLANT = 500
CO2_STORAGE_COST_PER_TON = 5.0
CO2_TRANSPORT_COST_PER_TON_KM = 0.1
MAX_SINK_LINKS_PER_PLANT = 10
BC_COST_MODE = "tiered_static"  # "sheet_linear" | "tiered_static"
TIER_SPLIT_RATIO = 0.2
FIXED_MAX_ABATEMENT = 4.7e9  # e.g. 3.27692078378e9; None means compute internally
RUN_LP_DIAGNOSTIC = False  # True to run LP-relax benchmark sweep
SOLVER_TIME_LIMIT_SEC = 7200.0
SOLVER_MIP_GAP = 0.001
SOLVER_THREADS = 0  # 0 means let solver decide

# Subsidy scenario settings (input in RMB; converted internally to USD).
RMB_PER_USD = 6.8985
BC_SUBSIDY_RMB_PER_KWH = 0.25
CC_SUBSIDY_RMB_PER_TON = 35.0
# -------------------------------------------


def _canon(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _find_col(columns: List[str], candidates: List[str]) -> str:
    canon_to_raw = {_canon(c): c for c in columns}
    for cand in candidates:
        key = _canon(cand)
        if key in canon_to_raw:
            return canon_to_raw[key]
    raise KeyError(f"Could not find any of columns {candidates} in {columns}")


def _find_col_optional(columns: List[str], candidates: List[str]) -> str:
    try:
        return _find_col(columns, candidates)
    except KeyError:
        return ""


def _norm_id(v) -> str:
    # Harmonize IDs across xlsx/csv where one side can be 1 and the other 1.0.
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


def load_inputs(
    workbook_path: str,
    biomass_distance_csv: str,
    co2_distance_csv: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plants_raw = pd.read_excel(workbook_path, sheet_name="coal_plant_i")
    ec_raw = pd.read_excel(workbook_path, sheet_name="EC")
    bc_raw = pd.read_excel(workbook_path, sheet_name="BC")
    cc_raw = pd.read_excel(workbook_path, sheet_name="CC")
    biomass_raw = pd.read_excel(workbook_path, sheet_name="biomass_point_j")
    sinks_raw = pd.read_excel(workbook_path, sheet_name="carbon_sink_k")
    biomass_dist_raw = pd.read_csv(biomass_distance_csv)
    co2_dist_raw = pd.read_csv(co2_distance_csv)

    pcols = list(plants_raw.columns)
    pid = _find_col(pcols, ["plant_i", "plantid", "id"])
    p_energy = _find_col(pcols, ["energy_demand_GJ", "energydemandgj", "energy_demand", "energy"])
    p_emission = _find_col(pcols, ["carbon_emissions_ton", "carbonemissionston", "co2", "carbon_emissions"])
    p_eff = _find_col_optional(pcols, ["efficiency", "eta", "thermal_efficiency", "效率"])
    p_power = _find_col_optional(
        pcols,
        ["power", "power_xi", "capacity_xi", "capacity", "电厂功率xi", "installed_power"],
    )
    p_hours = _find_col_optional(pcols, ["running_hours", "runninghours", "hours", "年利用小时"])
    p_af = _find_col_optional(pcols, ["AF", "annuity_factor", "年金因子"])
    p_coal = _find_col_optional(pcols, ["coal_price", "燃煤价格", "coalprice"])

    plants = pd.DataFrame(
        {
            "plant_id": plants_raw[pid].map(_norm_id),
            "energy_gj": pd.to_numeric(plants_raw[p_energy], errors="coerce").fillna(0.0),
            "base_emission": pd.to_numeric(plants_raw[p_emission], errors="coerce").fillna(0.0),
            "efficiency": pd.to_numeric(plants_raw[p_eff], errors="coerce").fillna(0.0) if p_eff else 0.0,
            "power": pd.to_numeric(plants_raw[p_power], errors="coerce").fillna(0.0) if p_power else 0.0,
            "running_hours": pd.to_numeric(plants_raw[p_hours], errors="coerce").fillna(0.0) if p_hours else 0.0,
            "af": pd.to_numeric(plants_raw[p_af], errors="coerce").fillna(0.0) if p_af else 0.0,
            "coal_price": pd.to_numeric(plants_raw[p_coal], errors="coerce").fillna(0.0) if p_coal else 0.0,
        }
    )

    def _extract_tech(df: pd.DataFrame, tech_name: str) -> pd.DataFrame:
        cols = list(df.columns)
        c_pid = _find_col(cols, ["plant_i", "plantid", "id"])
        c_abate = _find_col(cols, ["carbon_reduction", "co2_reduction", "abatement", "reduction"])
        c_total = None
        c_unit = None
        try:
            c_total = _find_col(cols, ["total_cost", "totalcost", "cost_total"])
        except KeyError:
            c_total = None
        try:
            c_unit = _find_col(cols, ["unit_cost", "unit cost", "unit_abatement_cost", "cost", "marginal_cost"])
        except KeyError:
            c_unit = None

        out = pd.DataFrame(
            {
                "plant_id": df[c_pid].map(_norm_id),
                f"{tech_name}_abatement": pd.to_numeric(df[c_abate], errors="coerce").fillna(0.0),
            }
        )
        if c_total is not None:
            out[f"{tech_name}_total_cost"] = pd.to_numeric(df[c_total], errors="coerce")
        else:
            out[f"{tech_name}_total_cost"] = np.nan

        if c_unit is not None:
            unit = pd.to_numeric(df[c_unit], errors="coerce")
        else:
            unit = pd.Series(np.nan, index=out.index)

        need_total = out[f"{tech_name}_total_cost"].isna()
        out.loc[need_total, f"{tech_name}_total_cost"] = (
            unit[need_total] * out.loc[need_total, f"{tech_name}_abatement"]
        )
        out[f"{tech_name}_total_cost"] = out[f"{tech_name}_total_cost"].fillna(0.0)

        out = out.groupby("plant_id", as_index=False).first()
        return out

    ec = _extract_tech(ec_raw, "ec")
    bc = _extract_tech(bc_raw, "bc")

    ccols = list(cc_raw.columns)
    cc_pid = _find_col(ccols, ["plant_i", "plantid", "id"])
    cc_abate_col = _find_col(ccols, ["carbon_reduction", "co2_reduction", "abatement", "reduction"])
    cc_total_col = _find_col_optional(ccols, ["total_cost", "totalcost", "cost_total"])
    cc_unit_col = _find_col_optional(ccols, ["unit_cost", "unit cost", "unit_abatement_cost", "cost", "marginal_cost"])
    cc_capex_col = _find_col_optional(ccols, ["capex_dollar_per_year", "capex", "capex_per_year"])
    cc_opex_col = _find_col_optional(ccols, ["opex_dollar_per_t", "opex", "opex_per_t"])
    cc_ele1_col = _find_col_optional(ccols, ["electricity1_dollar_per_t", "electricity1", "ele1_per_t"])
    cc_ele2_col = _find_col_optional(ccols, ["electricity2_dollar_per_t", "electricity2", "ele2_per_t"])

    cc = pd.DataFrame(
        {
            "plant_id": cc_raw[cc_pid].map(_norm_id),
            "cc_abatement": pd.to_numeric(cc_raw[cc_abate_col], errors="coerce").fillna(0.0),
            "cc_total_cost": pd.to_numeric(cc_raw[cc_total_col], errors="coerce") if cc_total_col else np.nan,
            "cc_unit_cost": pd.to_numeric(cc_raw[cc_unit_col], errors="coerce") if cc_unit_col else np.nan,
            "cc_capex_per_year": pd.to_numeric(cc_raw[cc_capex_col], errors="coerce").fillna(0.0) if cc_capex_col else 0.0,
            "cc_opex_per_t": pd.to_numeric(cc_raw[cc_opex_col], errors="coerce").fillna(0.0) if cc_opex_col else 0.0,
            "cc_ele1_per_t": pd.to_numeric(cc_raw[cc_ele1_col], errors="coerce").fillna(0.0) if cc_ele1_col else 0.0,
            "cc_ele2_per_t": pd.to_numeric(cc_raw[cc_ele2_col], errors="coerce").fillna(0.0) if cc_ele2_col else 0.0,
        }
    )
    # Backfill legacy total cost if only unit cost exists.
    cc_need_total = cc["cc_total_cost"].isna()
    cc.loc[cc_need_total, "cc_total_cost"] = cc.loc[cc_need_total, "cc_unit_cost"] * cc.loc[cc_need_total, "cc_abatement"]
    cc["cc_total_cost"] = cc["cc_total_cost"].fillna(0.0)
    cc = cc.groupby("plant_id", as_index=False).first()

    bcols = list(biomass_raw.columns)
    b_id = _find_col(bcols, ["biomass_point_j", "biomass_j", "j", "point_j"])
    b_supply = _find_col(bcols, ["bioenergy_GJ", "bioenergy", "supply", "available_biomass"])
    biomass = pd.DataFrame(
        {
            "biomass_id": biomass_raw[b_id].map(_norm_id),
            "available_gj": pd.to_numeric(biomass_raw[b_supply], errors="coerce").fillna(0.0),
        }
    ).groupby("biomass_id", as_index=False).sum()

    dcols = list(biomass_dist_raw.columns)
    d_j = _find_col(dcols, ["biomass_j", "biomass_point_j", "j"])
    d_i = _find_col(dcols, ["plant_i", "i", "plantid"])
    d_d = _find_col(dcols, ["Dji", "distance", "dji"])
    biomass_dist = pd.DataFrame(
        {
            "biomass_id": biomass_dist_raw[d_j].map(_norm_id),
            "plant_id": biomass_dist_raw[d_i].map(_norm_id),
            "distance_km": pd.to_numeric(biomass_dist_raw[d_d], errors="coerce").fillna(np.inf),
        }
    )
    biomass_dist = biomass_dist[np.isfinite(biomass_dist["distance_km"])].copy()

    scols = list(sinks_raw.columns)
    s_id = _find_col(scols, ["sink_k", "sink", "k"])
    s_cap = _find_col(scols, ["Estimated_storage_potential_Mt", "estimatedstoragepotentialmt", "capacity", "storage"])
    sinks = pd.DataFrame(
        {
            "sink_id": sinks_raw[s_id].map(_norm_id),
            "capacity_raw": pd.to_numeric(sinks_raw[s_cap], errors="coerce").fillna(0.0),
        }
    ).groupby("sink_id", as_index=False).sum()
    # Storage capacity in this workbook is Mt CO2; convert to ton CO2 if values are in Mt scale.
    if len(sinks) > 0 and sinks["capacity_raw"].max() < 1e5:
        sinks["capacity_ton"] = sinks["capacity_raw"] * 1e6
    else:
        sinks["capacity_ton"] = sinks["capacity_raw"]

    kcols = list(co2_dist_raw.columns)
    k_i = _find_col(kcols, ["plant_i", "i", "plantid"])
    k_k = _find_col(kcols, ["sink_k", "k", "sink"])
    k_d = _find_col(kcols, ["Dki", "distance", "dki"])
    co2_dist = pd.DataFrame(
        {
            "plant_id": co2_dist_raw[k_i].map(_norm_id),
            "sink_id": co2_dist_raw[k_k].map(_norm_id),
            "distance_km": pd.to_numeric(co2_dist_raw[k_d], errors="coerce").fillna(np.inf),
        }
    )
    co2_dist = co2_dist[np.isfinite(co2_dist["distance_km"])].copy()

    return plants, ec, bc, cc, biomass, biomass_dist, sinks, co2_dist


def build_model_data(
    plants: pd.DataFrame,
    ec: pd.DataFrame,
    bc: pd.DataFrame,
    cc: pd.DataFrame,
    biomass: pd.DataFrame,
    biomass_dist: pd.DataFrame,
    sinks: pd.DataFrame,
    co2_dist: pd.DataFrame,
    blend_cap_ratio: float,
    max_links_per_plant: int,
    max_sink_links_per_plant: int,
    transport_cost_per_gj_km: float,
    co2_transport_cost_per_ton_km: float,
    co2_storage_cost_per_ton: float,
    biomass_feedstock_cost_per_gj: float,
) -> Dict[str, object]:
    df = plants.merge(ec, on="plant_id", how="left").merge(bc, on="plant_id", how="left").merge(cc, on="plant_id", how="left")
    for c in [
        "ec_abatement",
        "ec_total_cost",
        "bc_abatement",
        "bc_total_cost",
        "cc_abatement",
        "cc_total_cost",
        "cc_capex_per_year",
        "cc_opex_per_t",
        "cc_ele1_per_t",
        "cc_ele2_per_t",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Approximate BC as continuous from 0 to blend cap: use BC sheet totals as full-cap anchor.
    df["bc_cap_gj"] = blend_cap_ratio * df["energy_gj"]
    df["bc_abate_per_gj"] = np.where(df["bc_cap_gj"] > 0, df["bc_abatement"] / df["bc_cap_gj"], 0.0)
    df["bc_base_cost_per_gj"] = np.where(df["bc_cap_gj"] > 0, df["bc_total_cost"] / df["bc_cap_gj"], 0.0)

    plants_used = sorted(df["plant_id"].astype(str).unique().tolist())
    biomass_used = sorted(biomass["biomass_id"].astype(str).unique().tolist())

    dist2 = biomass_dist[biomass_dist["plant_id"].isin(plants_used) & biomass_dist["biomass_id"].isin(biomass_used)].copy()
    if max_links_per_plant is not None and max_links_per_plant > 0:
        dist2 = (
            dist2.sort_values(["plant_id", "distance_km"]).groupby("plant_id", as_index=False).head(max_links_per_plant)
        )

    arcs = [(r["biomass_id"], r["plant_id"]) for _, r in dist2.iterrows()]
    distance = {(r["biomass_id"], r["plant_id"]): float(r["distance_km"]) for _, r in dist2.iterrows()}

    sink_ids = sorted(sinks["sink_id"].astype(str).unique().tolist())
    co2_dist2 = co2_dist[co2_dist["plant_id"].isin(plants_used) & co2_dist["sink_id"].isin(sink_ids)].copy()
    if max_sink_links_per_plant is not None and max_sink_links_per_plant > 0:
        co2_dist2 = (
            co2_dist2.sort_values(["plant_id", "distance_km"]).groupby("plant_id", as_index=False).head(max_sink_links_per_plant)
        )
    sink_arcs = [(r["plant_id"], r["sink_id"]) for _, r in co2_dist2.iterrows()]
    sink_distance = {(r["plant_id"], r["sink_id"]): float(r["distance_km"]) for _, r in co2_dist2.iterrows()}

    supply = {r["biomass_id"]: float(r["available_gj"]) for _, r in biomass.iterrows()}
    energy_gj = {r["plant_id"]: float(r["energy_gj"]) for _, r in df.iterrows()}
    base_emission = {r["plant_id"]: float(r["base_emission"]) for _, r in df.iterrows()}
    ec_abate = {r["plant_id"]: float(r["ec_abatement"]) for _, r in df.iterrows()}
    ec_cost = {r["plant_id"]: float(r["ec_total_cost"]) for _, r in df.iterrows()}
    cc_abate = {r["plant_id"]: float(r["cc_abatement"]) for _, r in df.iterrows()}
    cc_cost = {r["plant_id"]: float(r["cc_total_cost"]) for _, r in df.iterrows()}
    cc_capex = {r["plant_id"]: float(r["cc_capex_per_year"]) for _, r in df.iterrows()}
    cc_opex_per_t = {r["plant_id"]: float(r["cc_opex_per_t"]) for _, r in df.iterrows()}
    cc_ele1_per_t = {r["plant_id"]: float(r["cc_ele1_per_t"]) for _, r in df.iterrows()}
    cc_ele2_per_t = {r["plant_id"]: float(r["cc_ele2_per_t"]) for _, r in df.iterrows()}
    bc_abate_per_gj = {r["plant_id"]: float(r["bc_abate_per_gj"]) for _, r in df.iterrows()}
    bc_base_cost_per_gj = {r["plant_id"]: float(r["bc_base_cost_per_gj"]) for _, r in df.iterrows()}
    bc_cap_gj = {r["plant_id"]: float(r["bc_cap_gj"]) for _, r in df.iterrows()}
    efficiency = {r["plant_id"]: float(r["efficiency"]) for _, r in df.iterrows()}
    power = {r["plant_id"]: float(r["power"]) for _, r in df.iterrows()}
    running_hours = {r["plant_id"]: float(r["running_hours"]) for _, r in df.iterrows()}
    generation_kwh = {
        i: max(0.0, power[i] * running_hours[i] * 1000.0)
        for i in plants_used
    }
    af = {r["plant_id"]: float(r["af"]) for _, r in df.iterrows()}
    coal_price = {r["plant_id"]: float(r["coal_price"]) for _, r in df.iterrows()}

    transport_cost = {
        (j, i): transport_cost_per_gj_km * distance[(j, i)]
        for (j, i) in arcs
    }
    sink_capacity_ton = {r["sink_id"]: float(r["capacity_ton"]) for _, r in sinks.iterrows()}
    co2_transport_cost = {
        (i, k): co2_transport_cost_per_ton_km * sink_distance[(i, k)]
        for (i, k) in sink_arcs
    }

    return {
        "plants": plants_used,
        "biomass": biomass_used,
        "arcs": arcs,
        "supply": supply,
        "distance": distance,
        "transport_cost": transport_cost,
        "energy_gj": energy_gj,
        "base_emission": base_emission,
        "ec_abate": ec_abate,
        "ec_cost": ec_cost,
        "cc_abate": cc_abate,
        "cc_cost": cc_cost,
        "cc_capex": cc_capex,
        "cc_opex_per_t": cc_opex_per_t,
        "cc_ele1_per_t": cc_ele1_per_t,
        "cc_ele2_per_t": cc_ele2_per_t,
        "bc_abate_per_gj": bc_abate_per_gj,
        "bc_base_cost_per_gj": bc_base_cost_per_gj,
        "bc_cap_gj": bc_cap_gj,
        "efficiency": efficiency,
        "power": power,
        "running_hours": running_hours,
        "generation_kwh": generation_kwh,
        "af": af,
        "coal_price": coal_price,
        "sinks": sink_ids,
        "sink_arcs": sink_arcs,
        "sink_distance": sink_distance,
        "sink_capacity_ton": sink_capacity_ton,
        "co2_transport_cost": co2_transport_cost,
        "co2_storage_cost_per_ton": co2_storage_cost_per_ton,
        "biomass_feedstock_cost_per_gj": biomass_feedstock_cost_per_gj,
    }


def _solve_model(
    model,
    use_gurobi: bool,
    time_limit_sec: float,
    mip_gap: float,
    threads: int,
) -> str:
    if not use_gurobi:
        raise RuntimeError("Persistent script supports Gurobi only. Remove --no-gurobi.")
    model.Params.TimeLimit = float(time_limit_sec)
    model.Params.MIPGap = float(mip_gap)
    if int(threads) > 0:
        model.Params.Threads = int(threads)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        return "Optimal"
    if model.Status == GRB.TIME_LIMIT:
        return "TimeLimit"
    if model.Status == GRB.INFEASIBLE:
        return "Infeasible"
    if model.Status == GRB.INTERRUPTED:
        return "Interrupted"
    return f"Status_{int(model.Status)}"


def _value_or_zero(var) -> float:
    # Support both PuLP vars and gurobipy vars in one extraction path.
    if hasattr(var, "X"):
        try:
            return float(var.X)
        except Exception:
            return 0.0
    v = value(var)
    return 0.0 if v is None else float(v)


def _extract_solution_tables(
    model_data: Dict[str, object],
    vars_pack: Dict[str, object],
    target_id: int,
    target: float,
    achieved: float,
    status: str,
    relaxation: str,
    bc_cost_mode: str,
) -> Dict[str, pd.DataFrame]:
    t_start = time.time()
    print(f"  → Extracting detail tables for target {target_id}...", end="", flush=True)
    plants = model_data["plants"]
    arcs = model_data["arcs"]
    sink_arcs = model_data["sink_arcs"]

    x_ec = {i: _value_or_zero(v) for i, v in vars_pack["x_ec"].items()}
    x_cc = {i: _value_or_zero(v) for i, v in vars_pack["x_cc"].items()}
    bc_use = {i: _value_or_zero(v) for i, v in vars_pack["bc_use"].items()}
    blend_ratio = {i: _value_or_zero(v) for i, v in vars_pack["blend_ratio"].items()}
    high_ratio = {i: _value_or_zero(v) for i, v in vars_pack["high_ratio"].items()}
    br_xec = {i: _value_or_zero(v) for i, v in vars_pack["br_xec"].items()}
    y_ec_cc = {i: _value_or_zero(v) for i, v in vars_pack["y_ec_cc"].items()}
    br_xcc = {i: _value_or_zero(v) for i, v in vars_pack["br_xcc"].items()}
    br_xec_cc = {i: _value_or_zero(v) for i, v in vars_pack["br_xec_cc"].items()}
    b_flow = {(j, i): _value_or_zero(v) for (j, i), v in vars_pack["b_flow"].items()}
    co2_flow = {(i, k): _value_or_zero(v) for (i, k), v in vars_pack["co2_flow"].items()}

    common = {
        "target_id": target_id,
        "target_abatement": target,
        "achieved_abatement": achieved,
        "status": status,
        "relaxation": relaxation,
    }

    biomass_rows = []
    biomass_transport_cost_by_plant = {i: 0.0 for i in plants}
    biomass_feedstock_cost_by_plant = {i: 0.0 for i in plants}
    for (j, i) in arcs:
        flow = b_flow[(j, i)]
        if flow <= 1e-9:
            continue
        transport_cost = model_data["transport_cost"][(j, i)] * flow
        feedstock_cost = model_data["biomass_feedstock_cost_per_gj"] * flow
        biomass_transport_cost_by_plant[i] += transport_cost
        biomass_feedstock_cost_by_plant[i] += feedstock_cost
        biomass_rows.append(
            {
                **common,
                "biomass_id": j,
                "plant_id": i,
                "biomass_flow_gj": flow,
                "distance_km": model_data["distance"][(j, i)],
                "transport_cost": transport_cost,
                "feedstock_cost": feedstock_cost,
            }
        )

    co2_rows = []
    co2_transport_cost_by_plant = {i: 0.0 for i in plants}
    co2_storage_cost_by_plant = {i: 0.0 for i in plants}
    co2_stored_by_plant = {i: 0.0 for i in plants}
    for (i, k) in sink_arcs:
        flow = co2_flow[(i, k)]
        if flow <= 1e-9:
            continue
        transport_cost = model_data["co2_transport_cost"][(i, k)] * flow
        storage_cost = model_data["co2_storage_cost_per_ton"] * flow
        co2_transport_cost_by_plant[i] += transport_cost
        co2_storage_cost_by_plant[i] += storage_cost
        co2_stored_by_plant[i] += flow
        co2_rows.append(
            {
                **common,
                "plant_id": i,
                "sink_id": k,
                "co2_flow_ton": flow,
                "distance_km": model_data["sink_distance"][(i, k)],
                "transport_cost": transport_cost,
                "storage_cost": storage_cost,
                "total_cost": transport_cost + storage_cost,
            }
        )

    plant_rows = []
    for i in plants:
        ec_abatement = model_data["ec_abate"][i] * x_ec[i]
        energy_i = model_data["energy_gj"][i]
        effective_ratio = (bc_use[i] / energy_i) if energy_i > 0 else 0.0
        effective_ratio = max(0.0, min(1.0, effective_ratio))
        # BC abatement in both model and output now follows the physical definition.
        bc_abatement_formula = model_data["base_emission"][i] * effective_ratio
        bc_abatement_model = bc_abatement_formula
        # Keep the old sheet-linear reference for auditing only.
        bc_abatement_sheet_reference = model_data["bc_abate_per_gj"][i] * bc_use[i]
        cc_capture = 0.9 * model_data["energy_gj"][i] * (
            0.0901 * x_cc[i] + (0.112 - 0.0901) * br_xcc[i]
        ) - 0.2 * 0.9 * model_data["energy_gj"][i] * (
            0.0901 * y_ec_cc[i] + (0.112 - 0.0901) * br_xec_cc[i]
        )
        cc_capture_no_ec = 0.9 * model_data["energy_gj"][i] * (
            0.0901 * (x_cc[i] - y_ec_cc[i]) + (0.112 - 0.0901) * (br_xcc[i] - br_xec_cc[i])
        )
        cc_capture_with_ec = 0.8 * 0.9 * model_data["energy_gj"][i] * (
            0.0901 * y_ec_cc[i] + (0.112 - 0.0901) * br_xec_cc[i]
        )

        ec_cost_component = model_data["ec_cost"][i] * x_ec[i]
        cc_is_decomposed = not (
            model_data["cc_capex"][i] == 0
            and model_data["cc_opex_per_t"][i] == 0
            and model_data["cc_ele1_per_t"][i] == 0
            and model_data["cc_ele2_per_t"][i] == 0
        )
        cc_cost_component = (
            model_data["cc_capex"][i] * x_cc[i]
            + model_data["cc_opex_per_t"][i] * cc_capture
            + model_data["cc_ele1_per_t"][i] * cc_capture_no_ec
            + model_data["cc_ele2_per_t"][i] * cc_capture_with_ec
        )
        cc_legacy_cost_component = model_data["cc_cost"][i] * x_cc[i] if not cc_is_decomposed else 0.0

        # BC cost decomposition for reporting (always computed with the formula requested by user).
        bc_investment_cost = (
            1.02
            * (
                1153000.0 * effective_ratio * model_data["power"][i]
                + 4628000.0 * high_ratio[i] * model_data["power"][i]
            )
            * model_data["af"][i]
            / 6.8589
        )
        bc_coal_saving_credit = model_data["coal_price"][i] * energy_i * effective_ratio
        bc_feedstock_cost = biomass_feedstock_cost_by_plant[i]
        bc_internal_cost_formula = bc_investment_cost - bc_coal_saving_credit + bc_feedstock_cost
        bc_internal_cost_model = model_data["bc_base_cost_per_gj"][i] * bc_use[i]
        if bc_cost_mode == "tiered_static":
            bc_internal_cost = bc_internal_cost_formula
        else:
            bc_internal_cost = bc_internal_cost_model

        total_cost_component = (
            ec_cost_component
            + cc_cost_component
            + cc_legacy_cost_component
            + bc_internal_cost
            + biomass_transport_cost_by_plant[i]
            + co2_transport_cost_by_plant[i]
            + co2_storage_cost_by_plant[i]
        )

        plant_rows.append(
            {
                **common,
                "plant_id": i,
                "x_ec": x_ec[i],
                "x_cc": x_cc[i],
                "x_ec_cc": y_ec_cc[i],
                "blend_ratio": blend_ratio[i],
                "effective_blend_ratio": effective_ratio,
                "high_ratio": high_ratio[i],
                "bc_use_gj": bc_use[i],
                "ec_abatement_ton": ec_abatement,
                "bc_abatement_ton": bc_abatement_formula,
                "bc_abatement_ton_formula": bc_abatement_formula,
                "bc_abatement_ton_model": bc_abatement_model,
                "bc_abatement_ton_sheet_reference": bc_abatement_sheet_reference,
                "cc_capture_ton": cc_capture,
                "total_abatement_ton": ec_abatement + bc_abatement_formula + cc_capture,
                "total_abatement_ton_model": ec_abatement + bc_abatement_model + cc_capture,
                "co2_stored_ton": co2_stored_by_plant[i],
                "ec_cost": ec_cost_component,
                "cc_cost": cc_cost_component + cc_legacy_cost_component,
                "bc_cost": bc_internal_cost,
                "bc_cost_formula": bc_internal_cost_formula,
                "bc_cost_model_linear": bc_internal_cost_model,
                "bc_investment_cost": bc_investment_cost,
                "bc_feedstock_cost": bc_feedstock_cost,
                "bc_coal_saving_credit": bc_coal_saving_credit,
                "biomass_transport_cost": biomass_transport_cost_by_plant[i],
                "co2_transport_cost": co2_transport_cost_by_plant[i],
                "co2_storage_cost": co2_storage_cost_by_plant[i],
                "total_cost": total_cost_component,
            }
        )

    plant_df = pd.DataFrame(plant_rows).sort_values("plant_id").reset_index(drop=True)
    biomass_df = pd.DataFrame(biomass_rows)
    co2_df = pd.DataFrame(co2_rows)

    if biomass_df.empty:
        biomass_df = pd.DataFrame(columns=[*common.keys(), "biomass_id", "plant_id", "biomass_flow_gj", "distance_km", "transport_cost", "feedstock_cost"])
    else:
        biomass_df = biomass_df.sort_values(["plant_id", "biomass_id"]).reset_index(drop=True)

    if co2_df.empty:
        co2_df = pd.DataFrame(columns=[*common.keys(), "plant_id", "sink_id", "co2_flow_ton", "distance_km", "transport_cost", "storage_cost", "total_cost"])
    else:
        co2_df = co2_df.sort_values(["plant_id", "sink_id"]).reset_index(drop=True)

    t_elapsed = time.time() - t_start
    print(f" done ({t_elapsed:.2f}s)")
    return {"plant": plant_df, "biomass": biomass_df, "co2": co2_df}


def _write_detail_sheets(writer: pd.ExcelWriter, detail_records: List[Dict[str, object]]) -> None:
    t_start = time.time()
    print(f"Writing {len(detail_records)} detail scenarios to Excel (batch mode)...", flush=True)
    sheet_writes = []
    for record in detail_records:
        target_id = int(record["target_id"])
        mode_raw = str(record["mode"]).strip().upper()
        mode_tag = "m" if mode_raw == "MILP" else "l"
        tables = record["tables"]
        sheet_writes.append((f"p{mode_tag}{target_id:02d}", tables["plant"]))
        sheet_writes.append((f"b{mode_tag}{target_id:02d}", tables["biomass"]))
        sheet_writes.append((f"c{mode_tag}{target_id:02d}", tables["co2"]))
    
    for idx, (sheet_name, df) in enumerate(sheet_writes, start=1):
        t_sheet_start = time.time()
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        t_sheet_elapsed = time.time() - t_sheet_start
        if idx % 3 == 0:
            print(f"  [{idx}/{len(sheet_writes)}] Scenario group {idx//3} done ({t_sheet_elapsed:.2f}s for sheet)")
    t_total = time.time() - t_start
    print(f"Detail sheets written (total {t_total:.2f}s)")


def _write_detail_tables_for_target(
    writer: pd.ExcelWriter,
    target_id: int,
    relaxation: str,
    tables: Dict[str, pd.DataFrame],
) -> None:
    mode_raw = str(relaxation).strip().upper()
    mode_tag = "m" if mode_raw == "MILP" else "l"
    tables["plant"].to_excel(writer, sheet_name=f"p{mode_tag}{target_id:02d}", index=False)
    tables["biomass"].to_excel(writer, sheet_name=f"b{mode_tag}{target_id:02d}", index=False)
    tables["co2"].to_excel(writer, sheet_name=f"c{mode_tag}{target_id:02d}", index=False)


def _build_problem(
    model_data: Dict[str, object],
    target: float,
    objective_mode: str,
    relax_integrality: bool,
    bc_cost_mode: str,
    tier_split_ratio: float,
    bc_subsidy_rmb_per_kwh: float,
    cc_subsidy_rmb_per_ton: float,
    rmb_per_usd: float,
    abatement_upper: float = None,
) -> Tuple[gp.Model, Dict[str, object], object, object, object, object]:
    plants = model_data["plants"]
    biomass = model_data["biomass"]
    arcs = model_data["arcs"]
    sinks = model_data["sinks"]
    sink_arcs = model_data["sink_arcs"]

    model = gp.Model(f"Fleet_Joint_{'LP' if relax_integrality else 'MILP'}")

    binary_vtype = GRB.CONTINUOUS if relax_integrality else GRB.BINARY
    x_ec = {i: model.addVar(lb=0.0, ub=1.0, vtype=binary_vtype, name=f"x_ec__{i}") for i in plants}
    x_cc = {i: model.addVar(lb=0.0, ub=1.0, vtype=binary_vtype, name=f"x_cc__{i}") for i in plants}
    b_flow = {(j, i): model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"B__{j}__{i}") for (j, i) in arcs}
    bc_use = {i: model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"BC__{i}") for i in plants}
    co2_flow = {(i, k): model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"CO2__{i}__{k}") for (i, k) in sink_arcs}

    # Optional tiered BC-cost approximation, aligned with the static graded script.
    blend_ratio = {i: model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"BR__{i}") for i in plants}
    high_ratio = {i: model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"HR__{i}") for i in plants}
    y_high = {i: model.addVar(lb=0.0, ub=1.0, vtype=binary_vtype, name=f"YH__{i}") for i in plants}
    br_xec = {i: model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"BRXEC__{i}") for i in plants}
    y_ec_cc = {i: model.addVar(lb=0.0, ub=1.0, vtype=binary_vtype, name=f"Y_EC_CC__{i}") for i in plants}
    br_xcc = {i: model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"BRXCC__{i}") for i in plants}
    br_xec_cc = {i: model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"BRXECCC__{i}") for i in plants}

    model.update()

    if bc_cost_mode == "sheet_linear":
        bc_cost_expr = gp.quicksum(model_data["bc_base_cost_per_gj"][i] * bc_use[i] for i in plants)
    elif bc_cost_mode == "tiered_static":
        # Reference form:
        # 1.02*(1153000*ratio*power + 4628000*high_ratio*power)*AF/6.8589 - coal_price*energy*ratio
        # + feedstock_cost*biomass_flow
        # Use EC-adjusted effective blend ratio (ratio_eff = ratio - 0.2*ratio*x_ec).
        bc_cost_expr = (
            gp.quicksum(
                1.02
                * (
                    1153000.0 * (blend_ratio[i] - 0.2 * br_xec[i]) * model_data["power"][i]
                    + 4628000.0 * high_ratio[i] * model_data["power"][i]
                )
                * model_data["af"][i]
                / 6.8589
                - model_data["coal_price"][i] * model_data["energy_gj"][i] * (blend_ratio[i] - 0.2 * br_xec[i])
                for i in plants
            )
            + gp.quicksum(model_data["biomass_feedstock_cost_per_gj"] * b_flow[(j, i)] for (j, i) in arcs)
        )
    else:
        raise ValueError(f"Unsupported bc_cost_mode: {bc_cost_mode}")

    # Fuel-mix-based CC capture quantity:
    # 0.9 * [E * r * 0.112 + E * (1-r) * 0.0901], and if EC is chosen multiply by 0.8.
    capture_base_expr = {
        i: 0.9
        * model_data["energy_gj"][i]
        * (0.0901 * x_cc[i] + (0.112 - 0.0901) * br_xcc[i])
        for i in plants
    }
    capture_ec_reduction_expr = {
        i: 0.2
        * 0.9
        * model_data["energy_gj"][i]
        * (0.0901 * y_ec_cc[i] + (0.112 - 0.0901) * br_xec_cc[i])
        for i in plants
    }
    cc_capture_expr = {i: capture_base_expr[i] - capture_ec_reduction_expr[i] for i in plants}
    cc_capture_no_ec_expr = {
        i: 0.9
        * model_data["energy_gj"][i]
        * (0.0901 * (x_cc[i] - y_ec_cc[i]) + (0.112 - 0.0901) * (br_xcc[i] - br_xec_cc[i]))
        for i in plants
    }
    cc_capture_with_ec_expr = {
        i: 0.8
        * 0.9
        * model_data["energy_gj"][i]
        * (0.0901 * y_ec_cc[i] + (0.112 - 0.0901) * br_xec_cc[i])
        for i in plants
    }
    cc_cost_expr = gp.quicksum(
        model_data["cc_capex"][i] * x_cc[i]
        + model_data["cc_opex_per_t"][i] * cc_capture_expr[i]
        + model_data["cc_ele1_per_t"][i] * cc_capture_no_ec_expr[i]
        + model_data["cc_ele2_per_t"][i] * cc_capture_with_ec_expr[i]
        for i in plants
    )
    # Fallback for plants without decomposed CC data: keep legacy total-cost term.
    cc_legacy_cost_expr = gp.quicksum(
        model_data["cc_cost"][i] * x_cc[i]
        for i in plants
        if model_data["cc_capex"][i] == 0 and model_data["cc_opex_per_t"][i] == 0 and model_data["cc_ele1_per_t"][i] == 0 and model_data["cc_ele2_per_t"][i] == 0
    )

    if float(rmb_per_usd) <= 0:
        raise ValueError("rmb_per_usd must be positive.")

    # Biomass subsidy only applies to the biomass co-fired share of annual generation:
    # subsidy base = effective_blend_ratio * annual_generation_kWh.
    bc_subsidy_usd_per_kwh = float(bc_subsidy_rmb_per_kwh) / float(rmb_per_usd)
    cc_subsidy_usd_per_ton = float(cc_subsidy_rmb_per_ton) / float(rmb_per_usd)
    bc_subsidy_credit_expr = gp.quicksum(
        bc_subsidy_usd_per_kwh
        * model_data["generation_kwh"][i]
        * (blend_ratio[i] - 0.2 * br_xec[i])
        for i in plants
    )
    cc_subsidy_credit_expr = cc_subsidy_usd_per_ton * gp.quicksum(cc_capture_expr[i] for i in plants)

    total_cost_expr = (
        gp.quicksum(model_data["ec_cost"][i] * x_ec[i] for i in plants)
        + cc_cost_expr
        + cc_legacy_cost_expr
        + bc_cost_expr
        + gp.quicksum(model_data["transport_cost"][(j, i)] * b_flow[(j, i)] for (j, i) in arcs)
        + gp.quicksum(
            (model_data["co2_transport_cost"][(i, k)] + model_data["co2_storage_cost_per_ton"]) * co2_flow[(i, k)]
            for (i, k) in sink_arcs
        )
        - bc_subsidy_credit_expr
        - cc_subsidy_credit_expr
    )
    # Unified BC abatement definition: blend ratio × plant baseline emissions.
    # Since bc_use[i] is in GJ and energy_gj[i] is fixed, this remains linear.
    total_abatement_expr = (
        gp.quicksum(model_data["ec_abate"][i] * x_ec[i] for i in plants)
        + gp.quicksum(cc_capture_expr[i] for i in plants)
        + gp.quicksum(
            (model_data["base_emission"][i] / model_data["energy_gj"][i]) * bc_use[i]
            if model_data["energy_gj"][i] > 0
            else 0.0
            for i in plants
        )
    )

    if objective_mode == "cost":
        model.setObjective(total_cost_expr, GRB.MINIMIZE)
    elif objective_mode == "abatement":
        model.setObjective(total_abatement_expr, GRB.MINIMIZE)
    else:
        raise ValueError(f"Unsupported objective_mode: {objective_mode}")

    # Biomass supply cap by source j.
    for j in biomass:
        out_vars = [b_flow[(jj, i)] for (jj, i) in arcs if jj == j]
        if out_vars:
            model.addConstr(gp.quicksum(out_vars) <= model_data["supply"][j])

    # Plant-level biomass usage and blend cap.
    for i in plants:
        in_vars = [b_flow[(j, ii)] for (j, ii) in arcs if ii == i]
        if in_vars:
            model.addConstr(bc_use[i] == gp.quicksum(in_vars))
        else:
            model.addConstr(bc_use[i] == 0)
        model.addConstr(bc_use[i] <= model_data["bc_cap_gj"][i])

        # Link blend ratio to biomass usage at each plant.
        e_i = model_data["energy_gj"][i]
        # Linearization for blend_ratio * x_ec to model EC-driven 20% demand reduction.
        model.addConstr(br_xec[i] <= blend_ratio[i])
        model.addConstr(br_xec[i] <= x_ec[i])
        model.addConstr(br_xec[i] >= blend_ratio[i] - (1 - x_ec[i]))
        model.addConstr(br_xec[i] >= 0)

        # Binary product y_ec_cc = x_ec * x_cc for CC interaction terms.
        model.addConstr(y_ec_cc[i] <= x_ec[i])
        model.addConstr(y_ec_cc[i] <= x_cc[i])
        model.addConstr(y_ec_cc[i] >= x_ec[i] + x_cc[i] - 1)

        # Linearization for blend_ratio * x_cc and blend_ratio * x_ec * x_cc.
        model.addConstr(br_xcc[i] <= blend_ratio[i])
        model.addConstr(br_xcc[i] <= x_cc[i])
        model.addConstr(br_xcc[i] >= blend_ratio[i] - (1 - x_cc[i]))
        model.addConstr(br_xcc[i] >= 0)

        model.addConstr(br_xec_cc[i] <= blend_ratio[i])
        model.addConstr(br_xec_cc[i] <= y_ec_cc[i])
        model.addConstr(br_xec_cc[i] >= blend_ratio[i] - (1 - y_ec_cc[i]))
        model.addConstr(br_xec_cc[i] >= 0)

        if e_i > 0:
            # If EC is selected, effective demand for BC scales to 0.8*e_i.
            model.addConstr(bc_use[i] == e_i * blend_ratio[i] - 0.2 * e_i * br_xec[i])
            model.addConstr(blend_ratio[i] <= model_data["bc_cap_gj"][i] / e_i)
        else:
            model.addConstr(bc_use[i] == 0)
            model.addConstr(blend_ratio[i] == 0)

        # Tier split constraints; high_ratio equals blend_ratio only when y_high=1.
        ratio_cap = model_data["bc_cap_gj"][i] / e_i if e_i > 0 else 0.0
        if ratio_cap >= tier_split_ratio:
            model.addConstr(blend_ratio[i] <= tier_split_ratio + (ratio_cap - tier_split_ratio) * y_high[i])
        else:
            model.addConstr(blend_ratio[i] <= ratio_cap)
        model.addConstr(high_ratio[i] <= blend_ratio[i])
        model.addConstr(high_ratio[i] <= ratio_cap * y_high[i])
        model.addConstr(high_ratio[i] >= blend_ratio[i] - ratio_cap * (1 - y_high[i]))

    # Captured CO2 from each plant must be routed to sinks, bounded by CC decision.
    for i in plants:
        out_to_sinks = [co2_flow[(ii, k)] for (ii, k) in sink_arcs if ii == i]
        if out_to_sinks:
            # If EC is selected, effective CC capture volume scales to 0.8.
            model.addConstr(gp.quicksum(out_to_sinks) == cc_capture_expr[i])
        else:
            model.addConstr(cc_capture_expr[i] == 0)

    # Sink capacity constraints.
    for k in sinks:
        in_to_sink = [co2_flow[(i, kk)] for (i, kk) in sink_arcs if kk == k]
        if in_to_sink:
            model.addConstr(gp.quicksum(in_to_sink) <= model_data["sink_capacity_ton"][k])

    # Sector-wide target abatement.
    target_lb_constr = model.addConstr(total_abatement_expr >= float(target), name="target_lb")
    target_ub_constr = model.addConstr(total_abatement_expr <= GRB.INFINITY, name="target_ub")
    if abatement_upper is not None:
        target_ub_constr.RHS = float(abatement_upper)

    vars_pack = {
        "x_ec": x_ec,
        "x_cc": x_cc,
        "b_flow": b_flow,
        "bc_use": bc_use,
        "co2_flow": co2_flow,
        "blend_ratio": blend_ratio,
        "high_ratio": high_ratio,
        "y_high": y_high,
        "br_xec": br_xec,
        "y_ec_cc": y_ec_cc,
        "br_xcc": br_xcc,
        "br_xec_cc": br_xec_cc,
    }
    model.update()
    return model, vars_pack, total_cost_expr, total_abatement_expr, target_lb_constr, target_ub_constr


def compute_feasible_max_abatement(
    model_data: Dict[str, object],
    use_gurobi: bool,
    relax_integrality: bool,
    time_limit_sec: float,
    mip_gap: float,
    threads: int,
) -> float:
    model, _, _, total_abatement_expr, target_lb, target_ub = _build_problem(
        model_data=model_data,
        target=0.0,
        objective_mode="abatement",
        relax_integrality=relax_integrality,
        bc_cost_mode="sheet_linear",
        tier_split_ratio=TIER_SPLIT_RATIO,
        bc_subsidy_rmb_per_kwh=0.0,
        cc_subsidy_rmb_per_ton=0.0,
        rmb_per_usd=RMB_PER_USD,
    )
    target_lb.RHS = 0.0
    target_ub.RHS = GRB.INFINITY
    model.setObjective(total_abatement_expr, GRB.MAXIMIZE)
    status = _solve_model(model, use_gurobi, time_limit_sec, mip_gap, threads)
    if status != "Optimal":
        raise RuntimeError(f"Failed to compute max abatement ({'LP' if relax_integrality else 'MILP'}), status={status}")
    return float(total_abatement_expr.getValue())


def _build_solver_contexts(
    model_data: Dict[str, object],
    target_match_mode: str,
    relax_integrality: bool,
    bc_cost_mode: str,
    tier_split_ratio: float,
    bc_subsidy_rmb_per_kwh: float,
    cc_subsidy_rmb_per_ton: float,
    rmb_per_usd: float,
) -> Dict[str, Dict[str, object]]:
    contexts: Dict[str, Dict[str, object]] = {}
    if target_match_mode == "min_overshoot":
        m1, vp1, tc1, ta1, lb1, ub1 = _build_problem(
            model_data=model_data,
            target=0.0,
            objective_mode="abatement",
            relax_integrality=relax_integrality,
            bc_cost_mode=bc_cost_mode,
            tier_split_ratio=tier_split_ratio,
            bc_subsidy_rmb_per_kwh=bc_subsidy_rmb_per_kwh,
            cc_subsidy_rmb_per_ton=cc_subsidy_rmb_per_ton,
            rmb_per_usd=rmb_per_usd,
        )
        contexts["stage1"] = {
            "model": m1,
            "vars_pack": vp1,
            "total_cost": tc1,
            "total_abate": ta1,
            "target_lb": lb1,
            "target_ub": ub1,
        }
        m2, vp2, tc2, ta2, lb2, ub2 = _build_problem(
            model_data=model_data,
            target=0.0,
            objective_mode="cost",
            relax_integrality=relax_integrality,
            bc_cost_mode=bc_cost_mode,
            tier_split_ratio=tier_split_ratio,
            bc_subsidy_rmb_per_kwh=bc_subsidy_rmb_per_kwh,
            cc_subsidy_rmb_per_ton=cc_subsidy_rmb_per_ton,
            rmb_per_usd=rmb_per_usd,
        )
        contexts["stage2"] = {
            "model": m2,
            "vars_pack": vp2,
            "total_cost": tc2,
            "total_abate": ta2,
            "target_lb": lb2,
            "target_ub": ub2,
        }
    elif target_match_mode == "cost_only":
        m, vp, tc, ta, lb, ub = _build_problem(
            model_data=model_data,
            target=0.0,
            objective_mode="cost",
            relax_integrality=relax_integrality,
            bc_cost_mode=bc_cost_mode,
            tier_split_ratio=tier_split_ratio,
            bc_subsidy_rmb_per_kwh=bc_subsidy_rmb_per_kwh,
            cc_subsidy_rmb_per_ton=cc_subsidy_rmb_per_ton,
            rmb_per_usd=rmb_per_usd,
        )
        contexts["single"] = {
            "model": m,
            "vars_pack": vp,
            "total_cost": tc,
            "total_abate": ta,
            "target_lb": lb,
            "target_ub": ub,
        }
    else:
        raise ValueError(f"Unsupported target_match_mode: {target_match_mode}")
    return contexts


def solve_for_target(
    model_data: Dict[str, object],
    target: float,
    use_gurobi: bool,
    target_match_mode: str,
    relax_integrality: bool,
    bc_cost_mode: str,
    tier_split_ratio: float,
    time_limit_sec: float,
    mip_gap: float,
    threads: int,
    contexts: Dict[str, Dict[str, object]],
) -> Tuple[str, float, float, Dict[str, object]]:
    if target_match_mode == "min_overshoot":
        s1 = contexts["stage1"]
        s1["target_lb"].RHS = float(target)
        s1["target_ub"].RHS = GRB.INFINITY
        status = _solve_model(s1["model"], use_gurobi, time_limit_sec, mip_gap, threads)
        if status != "Optimal":
            return status, np.nan, np.nan, {}
        min_abate = float(s1["total_abate"].getValue())

        s2 = contexts["stage2"]
        s2["target_lb"].RHS = float(target)
        s2["target_ub"].RHS = float(min_abate + 1e-6)
        status = _solve_model(s2["model"], use_gurobi, time_limit_sec, mip_gap, threads)
        vars_pack = s2["vars_pack"]
        total_cost = s2["total_cost"]
        total_abate = s2["total_abate"]
    elif target_match_mode == "cost_only":
        s = contexts["single"]
        s["target_lb"].RHS = float(target)
        s["target_ub"].RHS = GRB.INFINITY
        status = _solve_model(s["model"], use_gurobi, time_limit_sec, mip_gap, threads)
        vars_pack = s["vars_pack"]
        total_cost = s["total_cost"]
        total_abate = s["total_abate"]
    else:
        raise ValueError(f"Unsupported target_match_mode: {target_match_mode}")

    if status != "Optimal":
        return status, np.nan, np.nan, {}

    achieved = float(total_abate.getValue())
    cost = float(total_cost.getValue())

    x_ec = vars_pack["x_ec"]
    x_cc = vars_pack["x_cc"]
    b_flow = vars_pack["b_flow"]
    bc_use = vars_pack["bc_use"]
    co2_flow = vars_pack["co2_flow"]

    summary = {
        "ec_adoption_sum": float(sum(_value_or_zero(v) for v in x_ec.values())),
        "cc_adoption_sum": float(sum(_value_or_zero(v) for v in x_cc.values())),
        "total_biomass_use_gj": float(sum(_value_or_zero(v) for v in bc_use.values())),
        "weighted_transport_distance_km": 0.0,
        "total_co2_stored_ton": float(sum(_value_or_zero(v) for v in co2_flow.values())),
        "weighted_co2_distance_km": 0.0,
        "vars_pack": vars_pack,
    }
    flow_qty = 0.0
    flow_dist = 0.0
    for (j, i), var in b_flow.items():
        q = _value_or_zero(var)
        if q <= 0:
            continue
        d = float(model_data["distance"][(j, i)])
        flow_qty += q
        flow_dist += q * d
    if flow_qty > 0:
        summary["weighted_transport_distance_km"] = flow_dist / flow_qty

    co2_qty = 0.0
    co2_dist = 0.0
    for (i, k), var in co2_flow.items():
        q = _value_or_zero(var)
        if q <= 0:
            continue
        d = float(model_data["sink_distance"][(i, k)])
        co2_qty += q
        co2_dist += q * d
    if co2_qty > 0:
        summary["weighted_co2_distance_km"] = co2_dist / co2_qty

    return status, achieved, cost, summary


def build_targets(
    n_targets: int,
    max_abatement: float,
    min_target: float = 0.0,
    max_target: float = None,
    extra_targets: List[float] = None,
) -> np.ndarray:
    # Keep a small margin from the theoretical max to reduce numeric issues in stage-2 bounding.
    safe_upper = max_abatement * (1.0 - 1e-5)
    if max_target is None:
        max_target = safe_upper
    max_target = min(max_target, safe_upper)
    lower = max(min_target, 0.0)
    if n_targets < 2:
        base_targets = np.array([lower])
    else:
        base_targets = np.linspace(lower, max_target, n_targets)

    extras = [] if extra_targets is None else [float(x) for x in extra_targets]
    extras = [x for x in extras if lower <= x <= safe_upper]
    all_targets = np.concatenate([base_targets, np.array(extras, dtype=float)]) if extras else base_targets
    all_targets = np.unique(np.round(all_targets, decimals=6))
    return np.sort(all_targets)


def run_sweep(
    model_data: Dict[str, object],
    targets: np.ndarray,
    use_gurobi: bool,
    target_match_mode: str,
    target_mode_switch_abatement: Optional[float],
    relax_integrality: bool,
    bc_cost_mode: str,
    tier_split_ratio: float,
    bc_subsidy_rmb_per_kwh: float,
    cc_subsidy_rmb_per_ton: float,
    rmb_per_usd: float,
    time_limit_sec: float,
    mip_gap: float,
    threads: int,
    detail_writer: pd.ExcelWriter = None,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    rows = []
    mode = "LP" if relax_integrality else "MILP"

    switch_threshold = None
    if target_mode_switch_abatement is not None:
        switch_threshold = float(target_mode_switch_abatement)
        if switch_threshold < 0:
            raise ValueError("target_mode_switch_abatement must be non-negative.")

    if switch_threshold is None:
        contexts_by_mode = {
            target_match_mode: _build_solver_contexts(
                model_data=model_data,
                target_match_mode=target_match_mode,
                relax_integrality=relax_integrality,
                bc_cost_mode=bc_cost_mode,
                tier_split_ratio=tier_split_ratio,
                bc_subsidy_rmb_per_kwh=bc_subsidy_rmb_per_kwh,
                cc_subsidy_rmb_per_ton=cc_subsidy_rmb_per_ton,
                rmb_per_usd=rmb_per_usd,
            )
        }
    else:
        contexts_by_mode = {
            "min_overshoot": _build_solver_contexts(
                model_data=model_data,
                target_match_mode="min_overshoot",
                relax_integrality=relax_integrality,
                bc_cost_mode=bc_cost_mode,
                tier_split_ratio=tier_split_ratio,
                bc_subsidy_rmb_per_kwh=bc_subsidy_rmb_per_kwh,
                cc_subsidy_rmb_per_ton=cc_subsidy_rmb_per_ton,
                rmb_per_usd=rmb_per_usd,
            ),
            "cost_only": _build_solver_contexts(
                model_data=model_data,
                target_match_mode="cost_only",
                relax_integrality=relax_integrality,
                bc_cost_mode=bc_cost_mode,
                tier_split_ratio=tier_split_ratio,
                bc_subsidy_rmb_per_kwh=bc_subsidy_rmb_per_kwh,
                cc_subsidy_rmb_per_ton=cc_subsidy_rmb_per_ton,
                rmb_per_usd=rmb_per_usd,
            ),
        }

    for idx, target in enumerate(targets, start=1):
        current_match_mode = target_match_mode
        if switch_threshold is not None:
            current_match_mode = "min_overshoot" if float(target) <= switch_threshold else "cost_only"

        status, achieved, cost, summary = solve_for_target(
            model_data=model_data,
            target=float(target),
            use_gurobi=use_gurobi,
            target_match_mode=current_match_mode,
            relax_integrality=relax_integrality,
            bc_cost_mode=bc_cost_mode,
            tier_split_ratio=tier_split_ratio,
            time_limit_sec=time_limit_sec,
            mip_gap=mip_gap,
            threads=threads,
            contexts=contexts_by_mode[current_match_mode],
        )
        rows.append(
            {
                "target_id": idx,
                "target_abatement": float(target),
                "achieved_abatement": achieved,
                "total_cost": cost,
                "status": status,
                "relaxation": mode,
                "match_mode_used": current_match_mode,
                "ec_adoption_sum": summary.get("ec_adoption_sum", np.nan),
                "cc_adoption_sum": summary.get("cc_adoption_sum", np.nan),
                "total_biomass_use_gj": summary.get("total_biomass_use_gj", np.nan),
                "weighted_transport_distance_km": summary.get("weighted_transport_distance_km", np.nan),
                "total_co2_stored_ton": summary.get("total_co2_stored_ton", np.nan),
                "weighted_co2_distance_km": summary.get("weighted_co2_distance_km", np.nan),
                "bc_subsidy_rmb_per_kwh": float(bc_subsidy_rmb_per_kwh),
                "cc_subsidy_rmb_per_ton": float(cc_subsidy_rmb_per_ton),
                "rmb_per_usd": float(rmb_per_usd),
            }
        )
        print(
            f"[{mode} {idx}/{len(targets)}] mode={current_match_mode} target={target:,.2f} status={status} "
            f"achieved={achieved:,.2f} cost={cost:,.2f}"
        )

        if status == "Optimal" and "vars_pack" in summary:
            tables = _extract_solution_tables(
                model_data=model_data,
                vars_pack=summary["vars_pack"],
                target_id=idx,
                target=float(target),
                achieved=float(achieved),
                status=status,
                relaxation=mode,
                bc_cost_mode=bc_cost_mode,
            )
            if detail_writer is not None:
                _write_detail_tables_for_target(detail_writer, idx, mode, tables)

    df = pd.DataFrame(rows).sort_values("target_abatement").reset_index(drop=True)
    df["marginal_cost_diff"] = df["total_cost"].diff() / df["target_abatement"].diff().replace(0, np.nan)
    df["avg_cost_from_origin"] = df["total_cost"] / df["achieved_abatement"].replace(0, np.nan)
    return df, []


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fleet-wide joint MILP for EC/BC/CC with two subsidy scenarios: "
            "(1) biomass co-firing electricity subsidy and (2) CC capture subsidy, plus "
            "spatial biomass allocation B_{j,i}, "
            "biomass supply constraints, plant blending cap, and parametric MAC sweep."
        )
    )
    parser.add_argument("--input", default=None, help="Workbook path. Default: data/1885_coal_power_plants_original.xlsx")
    parser.add_argument("--distance-csv", default=None, help="Biomass-to-plant distance CSV. Default: data/Dji_200.csv")
    parser.add_argument("--co2-distance-csv", default=None, help="Plant-to-sink distance CSV. Default: data/Dki_4000.csv")
    parser.add_argument("--n-targets", type=int, default=N_TARGETS, help="Number of sweep points for target abatement.")
    parser.add_argument("--extra-targets", type=float, nargs="*", default=EXTRA_TARGETS, help="Additional specific abatement targets to include alongside the regular grid.")
    parser.add_argument("--min-target", type=float, default=0.0, help="Minimum target abatement.")
    parser.add_argument("--max-target", type=float, default=None, help="Maximum target abatement; default is model feasible max.")
    parser.add_argument(
        "--fixed-max-abatement",
        type=float,
        default=FIXED_MAX_ABATEMENT,
        help="Use an exogenous max abatement value and skip the internal max-abatement solve.",
    )
    parser.add_argument("--blend-cap-ratio", type=float, default=BLEND_CAP_RATIO, help="Plant biomass blending cap ratio of energy demand.")
    parser.add_argument("--transport-cost-per-gj-km", type=float, default=TRANSPORT_COST_PER_GJ_KM, help="Transport cost coefficient per GJ-km.")
    parser.add_argument("--max-links-per-plant", type=int, default=MAX_LINKS_PER_PLANT, help="Keep only shortest K biomass links per plant (speed control).")
    parser.add_argument("--biomass-feedstock-cost-per-gj", type=float, default=BIOMASS_FEEDSTOCK_COST_PER_GJ, help="Biomass feedstock cost (USD per GJ), used in tiered_static BC mode.")
    parser.add_argument("--bc-cost-mode", choices=["sheet_linear", "tiered_static"], default=BC_COST_MODE, help="BC cost formulation mode.")
    parser.add_argument("--tier-split-ratio", type=float, default=TIER_SPLIT_RATIO, help="Blend-ratio threshold separating low/high BC investment tiers.")
    parser.add_argument("--co2-transport-cost-per-ton-km", type=float, default=CO2_TRANSPORT_COST_PER_TON_KM, help="CO2 transport cost coefficient (USD per ton-km).")
    parser.add_argument("--co2-storage-cost-per-ton", type=float, default=CO2_STORAGE_COST_PER_TON, help="CO2 storage cost (USD per ton).")
    parser.add_argument("--max-sink-links-per-plant", type=int, default=MAX_SINK_LINKS_PER_PLANT, help="Keep only shortest K sink links per plant (speed control).")
    parser.add_argument("--time-limit-sec", type=float, default=SOLVER_TIME_LIMIT_SEC, help="MILP solver time limit per solve (seconds).")
    parser.add_argument("--mip-gap", type=float, default=SOLVER_MIP_GAP, help="MILP relative optimality gap tolerance.")
    parser.add_argument("--solver-threads", type=int, default=SOLVER_THREADS, help="Threads for Gurobi (0 lets solver decide).")
    parser.add_argument("--rmb-per-usd", type=float, default=RMB_PER_USD, help="Exchange rate for converting subsidies from RMB to USD.")
    parser.add_argument("--bc-subsidy-rmb-per-kwh", type=float, default=BC_SUBSIDY_RMB_PER_KWH, help="Biomass co-firing subsidy (RMB per kWh biomass electricity).")
    parser.add_argument("--cc-subsidy-rmb-per-ton", type=float, default=CC_SUBSIDY_RMB_PER_TON, help="CC subsidy (RMB per ton CO2 captured).")
    parser.add_argument("--target-match-mode", choices=["min_overshoot", "cost_only"], default=TARGET_MATCH_MODE)
    parser.add_argument(
        "--target-mode-switch-abatement",
        type=float,
        default=TARGET_MODE_SWITCH_ABATEMENT,
        help="If set, targets <= this value use min_overshoot, while targets above this value use cost_only.",
    )
    parser.add_argument(
        "--diagnostic-lp-relax",
        "--also-run-lp-relax",
        dest="run_lp_diagnostic",
        action="store_true",
        help="Run LP-relaxation as an optional diagnostic benchmark and export a second curve file.",
    )
    parser.add_argument(
        "--no-diagnostic-lp-relax",
        dest="run_lp_diagnostic",
        action="store_false",
        help="Disable LP-relaxation diagnostic sweep.",
    )
    parser.set_defaults(run_lp_diagnostic=RUN_LP_DIAGNOSTIC)
    parser.add_argument("--no-gurobi", action="store_true", help="Use CBC instead of Gurobi.")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX, help="Output prefix in data folder.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, DATA_DIR_NAME)
    input_path = args.input or os.path.join(data_dir, INPUT_WORKBOOK_NAME)
    distance_path = args.distance_csv or os.path.join(data_dir, INPUT_BIOMASS_DISTANCE_NAME)
    co2_distance_path = args.co2_distance_csv or os.path.join(data_dir, INPUT_CO2_DISTANCE_NAME)
    use_gurobi = not args.no_gurobi

    plants, ec, bc, cc, biomass, biomass_dist, sinks, co2_dist = load_inputs(input_path, distance_path, co2_distance_path)
    model_data = build_model_data(
        plants=plants,
        ec=ec,
        bc=bc,
        cc=cc,
        biomass=biomass,
        biomass_dist=biomass_dist,
        sinks=sinks,
        co2_dist=co2_dist,
        blend_cap_ratio=args.blend_cap_ratio,
        max_links_per_plant=args.max_links_per_plant,
        max_sink_links_per_plant=args.max_sink_links_per_plant,
        transport_cost_per_gj_km=args.transport_cost_per_gj_km,
        co2_transport_cost_per_ton_km=args.co2_transport_cost_per_ton_km,
        co2_storage_cost_per_ton=args.co2_storage_cost_per_ton,
        biomass_feedstock_cost_per_gj=args.biomass_feedstock_cost_per_gj,
    )

    if args.fixed_max_abatement is not None:
        if args.fixed_max_abatement <= 0:
            raise ValueError("--fixed-max-abatement must be positive.")
        max_abatement = float(args.fixed_max_abatement)
        print(f"Using fixed max abatement (skip max-abatement solve): {max_abatement:,.2f}")
    else:
        max_abatement = compute_feasible_max_abatement(
            model_data,
            use_gurobi,
            relax_integrality=False,
            time_limit_sec=args.time_limit_sec,
            mip_gap=args.mip_gap,
            threads=args.solver_threads,
        )
    targets = build_targets(args.n_targets, max_abatement, args.min_target, args.max_target, args.extra_targets)
    if args.fixed_max_abatement is None:
        print(f"MILP feasible max abatement: {max_abatement:,.2f}")
    print(
        f"Plants: {len(model_data['plants'])}, Biomass points: {len(model_data['biomass'])}, "
        f"Biomass arcs: {len(model_data['arcs'])}, Sinks: {len(model_data['sinks'])}, Sink arcs: {len(model_data['sink_arcs'])}"
    )

    scenario_defs = [
        {
            "tag": "bc_subsidy",
            "desc": "Biomass electricity subsidy only",
            "bc_subsidy_rmb_per_kwh": float(args.bc_subsidy_rmb_per_kwh),
            "cc_subsidy_rmb_per_ton": 0.0,
        },
        {
            "tag": "cc_subsidy",
            "desc": "CC capture subsidy only",
            "bc_subsidy_rmb_per_kwh": 0.0,
            "cc_subsidy_rmb_per_ton": float(args.cc_subsidy_rmb_per_ton),
        },
    ]

    for scenario in scenario_defs:
        tag = scenario["tag"]
        out_curve = os.path.join(data_dir, f"{args.output_prefix}_{tag}{OUTPUT_CURVE_SUFFIX}")
        out_xlsx = os.path.join(data_dir, f"{args.output_prefix}_{tag}{OUTPUT_RESULTS_SUFFIX}")
        out_lp_curve = os.path.join(data_dir, f"{args.output_prefix}_{tag}{OUTPUT_LP_CURVE_SUFFIX}")
        lp_curve = None

        print(
            f"\n=== Scenario: {scenario['desc']} | "
            f"BC subsidy={scenario['bc_subsidy_rmb_per_kwh']:.4f} RMB/kWh, "
            f"CC subsidy={scenario['cc_subsidy_rmb_per_ton']:.4f} RMB/tCO2 ==="
        )

        with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as writer:
            milp_curve, _ = run_sweep(
                model_data=model_data,
                targets=targets,
                use_gurobi=use_gurobi,
                target_match_mode=args.target_match_mode,
                target_mode_switch_abatement=args.target_mode_switch_abatement,
                relax_integrality=False,
                bc_cost_mode=args.bc_cost_mode,
                tier_split_ratio=args.tier_split_ratio,
                bc_subsidy_rmb_per_kwh=scenario["bc_subsidy_rmb_per_kwh"],
                cc_subsidy_rmb_per_ton=scenario["cc_subsidy_rmb_per_ton"],
                rmb_per_usd=args.rmb_per_usd,
                time_limit_sec=args.time_limit_sec,
                mip_gap=args.mip_gap,
                threads=args.solver_threads,
                detail_writer=writer,
            )
            milp_curve.to_excel(writer, sheet_name="cm", index=False)
            milp_curve.to_csv(out_curve, index=False)

            if args.run_lp_diagnostic:
                if args.fixed_max_abatement is not None:
                    lp_max = float(args.fixed_max_abatement)
                    print(f"Using fixed max abatement for LP diagnostic: {lp_max:,.2f}")
                else:
                    lp_max = compute_feasible_max_abatement(
                        model_data,
                        use_gurobi,
                        relax_integrality=True,
                        time_limit_sec=args.time_limit_sec,
                        mip_gap=args.mip_gap,
                        threads=args.solver_threads,
                    )
                lp_targets = build_targets(args.n_targets, lp_max, args.min_target, args.max_target, args.extra_targets)
                if args.fixed_max_abatement is None:
                    print(f"LP feasible max abatement: {lp_max:,.2f}")
                lp_curve, _ = run_sweep(
                    model_data=model_data,
                    targets=lp_targets,
                    use_gurobi=use_gurobi,
                    target_match_mode=args.target_match_mode,
                    target_mode_switch_abatement=args.target_mode_switch_abatement,
                    relax_integrality=True,
                    bc_cost_mode=args.bc_cost_mode,
                    tier_split_ratio=args.tier_split_ratio,
                    bc_subsidy_rmb_per_kwh=scenario["bc_subsidy_rmb_per_kwh"],
                    cc_subsidy_rmb_per_ton=scenario["cc_subsidy_rmb_per_ton"],
                    rmb_per_usd=args.rmb_per_usd,
                    time_limit_sec=args.time_limit_sec,
                    mip_gap=args.mip_gap,
                    threads=args.solver_threads,
                    detail_writer=writer,
                )
                lp_curve.to_excel(writer, sheet_name="cl", index=False)
                lp_curve.to_csv(out_lp_curve, index=False)

        print(f"Scenario done: {scenario['desc']}")
        print(f"  MILP curve output: {out_curve}")
        if lp_curve is not None:
            print(f"  LP-relax curve output: {out_lp_curve}")
        print(f"  Excel output: {out_xlsx}")

    print("\nDone.")
    print(f"Input workbook: {input_path}")
    print(f"Input distance: {distance_path}")
    print(f"Input CO2 distance: {co2_distance_path}")


if __name__ == "__main__":
    main()
