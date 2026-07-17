import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


DATA_DIR_NAME = "data"
INPUT_WORKBOOK_NAME = "1885_coal_power_plants_original.xlsx"
INPUT_BIOMASS_DISTANCE_NAME = "Dji_200.csv"
OUTPUT_PREFIX = "gurobi_bc_only"
TARGET_STEP = 0.1e9
EXTRA_TARGETS: List[float] = []
FIXED_MAX_ABATEMENT = None
TARGET_MATCH_MODE = "cost_only"  # "min_overshoot" | "cost_only"
TARGET_MODE_SWITCH_ABATEMENT = 0.6e9  # <= threshold uses min_overshoot, > threshold uses cost_only
TRANSPORT_COST_PER_GJ_KM = 0.0017 * 1.5
BIOMASS_FEEDSTOCK_COST_PER_GJ = 2.474
MAX_LINKS_PER_PLANT = 500
BLEND_CAP_RATIO = 1.0
BC_COST_MODE = "tiered_static"
TIER_SPLIT_RATIO = 0.2
SOLVER_TIME_LIMIT_SEC = 7200.0
SOLVER_MIP_GAP = 0.001
SOLVER_THREADS = 0


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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plants_raw = pd.read_excel(workbook_path, sheet_name="coal_plant_i")
    bc_raw = pd.read_excel(workbook_path, sheet_name="BC")
    biomass_raw = pd.read_excel(workbook_path, sheet_name="biomass_point_j")
    biomass_dist_raw = pd.read_csv(biomass_distance_csv)

    # Load plant data with required fields
    pcols = list(plants_raw.columns)
    pid = _find_col(pcols, ["plant_i", "plantid", "id"])
    p_energy = _find_col(pcols, ["energy_demand_GJ", "energydemandgj", "energy_demand", "energy"])
    p_emission = _find_col(pcols, ["carbon_emissions_ton", "carbonemissionston", "co2", "carbon_emissions"])
    p_power = _find_col_optional(pcols, ["power", "power_xi", "capacity_xi", "capacity", "电厂功率xi", "installed_power"])
    p_af = _find_col_optional(pcols, ["AF", "annuity_factor", "年金因子"])
    p_coal = _find_col_optional(pcols, ["coal_price", "燃煤价格", "coalprice"])

    plants = pd.DataFrame(
        {
            "plant_id": plants_raw[pid].map(_norm_id),
            "energy_gj": pd.to_numeric(plants_raw[p_energy], errors="coerce").fillna(0.0),
            "base_emission": pd.to_numeric(plants_raw[p_emission], errors="coerce").fillna(0.0),
            "power": pd.to_numeric(plants_raw[p_power], errors="coerce").fillna(0.0) if p_power else 0.0,
            "af": pd.to_numeric(plants_raw[p_af], errors="coerce").fillna(0.0) if p_af else 0.0,
            "coal_price": pd.to_numeric(plants_raw[p_coal], errors="coerce").fillna(0.0) if p_coal else 0.0,
        }
    )

    # Load BC tech parameters
    bcols = list(bc_raw.columns)
    bc_pid = _find_col(bcols, ["plant_i", "plantid", "id"])
    bc_abate_col = _find_col(bcols, ["carbon_reduction", "co2_reduction", "abatement", "reduction"])
    bc_total_col = _find_col_optional(bcols, ["total_cost", "totalcost", "cost_total"])
    bc_unit_col = _find_col_optional(bcols, ["unit_cost", "unit cost", "unit_abatement_cost", "cost", "marginal_cost"])

    bc = pd.DataFrame(
        {
            "plant_id": bc_raw[bc_pid].map(_norm_id),
            "bc_abatement": pd.to_numeric(bc_raw[bc_abate_col], errors="coerce").fillna(0.0),
            "bc_total_cost": pd.to_numeric(bc_raw[bc_total_col], errors="coerce") if bc_total_col else np.nan,
            "bc_unit_cost": pd.to_numeric(bc_raw[bc_unit_col], errors="coerce") if bc_unit_col else np.nan,
        }
    )
    bc_need_total = bc["bc_total_cost"].isna()
    bc.loc[bc_need_total, "bc_total_cost"] = bc.loc[bc_need_total, "bc_unit_cost"] * bc.loc[bc_need_total, "bc_abatement"]
    bc["bc_total_cost"] = bc["bc_total_cost"].fillna(0.0)
    bc = bc.groupby("plant_id", as_index=False).first()

    # Load biomass supply
    bbcols = list(biomass_raw.columns)
    b_id = _find_col(bbcols, ["biomass_point_j", "biomass_j", "j", "point_j"])
    b_supply = _find_col(bbcols, ["bioenergy_GJ", "bioenergy", "supply", "available_biomass"])
    biomass = pd.DataFrame(
        {
            "biomass_id": biomass_raw[b_id].map(_norm_id),
            "available_gj": pd.to_numeric(biomass_raw[b_supply], errors="coerce").fillna(0.0),
        }
    ).groupby("biomass_id", as_index=False).sum()

    # Load biomass distance
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

    return plants, bc, biomass, biomass_dist


def build_model_data(
    plants: pd.DataFrame,
    bc: pd.DataFrame,
    biomass: pd.DataFrame,
    biomass_dist: pd.DataFrame,
    blend_cap_ratio: float,
    max_links_per_plant: int,
    transport_cost_per_gj_km: float,
    biomass_feedstock_cost_per_gj: float,
) -> Dict[str, object]:
    df = plants.merge(bc, on="plant_id", how="left")
    for c in ["bc_abatement", "bc_total_cost"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # BC approximation: use sheet totals as full-cap anchor
    df["bc_cap_gj"] = blend_cap_ratio * df["energy_gj"]
    df["bc_abate_per_gj"] = np.where(df["bc_cap_gj"] > 0, df["bc_abatement"] / df["bc_cap_gj"], 0.0)
    df["bc_base_cost_per_gj"] = np.where(df["bc_cap_gj"] > 0, df["bc_total_cost"] / df["bc_cap_gj"], 0.0)

    plants_used = sorted(df["plant_id"].astype(str).unique().tolist())
    biomass_used = sorted(biomass["biomass_id"].astype(str).unique().tolist())

    dist2 = biomass_dist[biomass_dist["plant_id"].isin(plants_used) & biomass_dist["biomass_id"].isin(biomass_used)].copy()
    if max_links_per_plant is not None and max_links_per_plant > 0:
        dist2 = dist2.sort_values(["plant_id", "distance_km"]).groupby("plant_id", as_index=False).head(max_links_per_plant)

    arcs = [(r["biomass_id"], r["plant_id"]) for _, r in dist2.iterrows()]
    distance = {(r["biomass_id"], r["plant_id"]): float(r["distance_km"]) for _, r in dist2.iterrows()}
    transport_cost = {(j, i): transport_cost_per_gj_km * distance[(j, i)] for (j, i) in arcs}

    supply = {r["biomass_id"]: float(r["available_gj"]) for _, r in biomass.iterrows()}
    energy_gj = {r["plant_id"]: float(r["energy_gj"]) for _, r in df.iterrows()}
    base_emission = {r["plant_id"]: float(r["base_emission"]) for _, r in df.iterrows()}
    bc_abate_per_gj = {r["plant_id"]: float(r["bc_abate_per_gj"]) for _, r in df.iterrows()}
    bc_base_cost_per_gj = {r["plant_id"]: float(r["bc_base_cost_per_gj"]) for _, r in df.iterrows()}
    bc_cap_gj = {r["plant_id"]: float(r["bc_cap_gj"]) for _, r in df.iterrows()}
    power = {r["plant_id"]: float(r["power"]) for _, r in df.iterrows()}
    af = {r["plant_id"]: float(r["af"]) for _, r in df.iterrows()}
    coal_price = {r["plant_id"]: float(r["coal_price"]) for _, r in df.iterrows()}

    return {
        "plants": plants_used,
        "biomass": biomass_used,
        "arcs": arcs,
        "supply": supply,
        "distance": distance,
        "transport_cost": transport_cost,
        "energy_gj": energy_gj,
        "base_emission": base_emission,
        "bc_abate_per_gj": bc_abate_per_gj,
        "bc_base_cost_per_gj": bc_base_cost_per_gj,
        "bc_cap_gj": bc_cap_gj,
        "power": power,
        "af": af,
        "coal_price": coal_price,
        "biomass_feedstock_cost_per_gj": biomass_feedstock_cost_per_gj,
    }


def _solve_model(model, time_limit_sec: float, mip_gap: float, threads: int) -> str:
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


def _build_problem(model_data: Dict[str, object], target: float, objective_mode: str, tier_split_ratio: float):
    plants = model_data["plants"]
    biomass = model_data["biomass"]
    arcs = model_data["arcs"]

    model = gp.Model("BC_Only_MILP")

    # Decision variables: blend ratio, biomass flow, tiered cost auxiliaries
    blend_ratio = {i: model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"BR__{i}") for i in plants}
    b_flow = {(j, i): model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"B__{j}__{i}") for (j, i) in arcs}
    high_ratio = {i: model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"HR__{i}") for i in plants}
    y_high = {i: model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f"YH__{i}") for i in plants}

    model.update()

    # BC cost expression: tiered static mode
    # Cost = 1.02*(1153000*ratio*power + 4628000*high_ratio*power)*AF/6.8589 - coal_price*energy*ratio + feedstock_cost*biomass_flow
    bc_cost_expr = (
        gp.quicksum(
            1.02
            * (
                1153000.0 * blend_ratio[i] * model_data["power"][i]
                + 4628000.0 * high_ratio[i] * model_data["power"][i]
            )
            * model_data["af"][i]
            / 6.8589
            - model_data["coal_price"][i] * model_data["energy_gj"][i] * blend_ratio[i]
            for i in plants
        )
        + gp.quicksum(model_data["biomass_feedstock_cost_per_gj"] * b_flow[(j, i)] for (j, i) in arcs)
    )

    # Biomass transport cost
    transport_cost_expr = gp.quicksum(model_data["transport_cost"][(j, i)] * b_flow[(j, i)] for (j, i) in arcs)

    total_cost_expr = bc_cost_expr + transport_cost_expr

    # Total abatement: blend_ratio * plant baseline emissions
    bc_use = {i: model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"BC__{i}") for i in plants}
    total_abatement_expr = gp.quicksum(
        (model_data["base_emission"][i] / model_data["energy_gj"][i]) * bc_use[i]
        if model_data["energy_gj"][i] > 0
        else 0.0
        for i in plants
    )

    if objective_mode == "cost":
        model.setObjective(total_cost_expr, GRB.MINIMIZE)
    elif objective_mode == "abatement":
        model.setObjective(total_abatement_expr, GRB.MAXIMIZE)
    else:
        raise ValueError(f"Unsupported objective_mode: {objective_mode}")

    # Biomass supply constraint
    for j in biomass:
        out_vars = [b_flow[(jj, i)] for (jj, i) in arcs if jj == j]
        if out_vars:
            model.addConstr(gp.quicksum(out_vars) <= model_data["supply"][j])

    # Plant-level biomass balance and blend cap
    for i in plants:
        in_vars = [b_flow[(j, ii)] for (j, ii) in arcs if ii == i]
        if in_vars:
            model.addConstr(bc_use[i] == gp.quicksum(in_vars))
        else:
            model.addConstr(bc_use[i] == 0)
        model.addConstr(bc_use[i] <= model_data["bc_cap_gj"][i])

        # Link blend ratio to BC usage
        e_i = model_data["energy_gj"][i]
        if e_i > 0:
            model.addConstr(bc_use[i] == e_i * blend_ratio[i])
            model.addConstr(blend_ratio[i] <= model_data["bc_cap_gj"][i] / e_i)
        else:
            model.addConstr(bc_use[i] == 0)
            model.addConstr(blend_ratio[i] == 0)

        # Tier split constraints
        ratio_cap = model_data["bc_cap_gj"][i] / e_i if e_i > 0 else 0.0
        if ratio_cap >= tier_split_ratio:
            model.addConstr(blend_ratio[i] <= tier_split_ratio + (ratio_cap - tier_split_ratio) * y_high[i])
        else:
            model.addConstr(blend_ratio[i] <= ratio_cap)
        model.addConstr(high_ratio[i] <= blend_ratio[i])
        model.addConstr(high_ratio[i] <= ratio_cap * y_high[i])
        model.addConstr(high_ratio[i] >= blend_ratio[i] - ratio_cap * (1 - y_high[i]))

    # Target constraint
    model.addConstr(total_abatement_expr >= float(target), name="target_lb")

    vars_pack = {"blend_ratio": blend_ratio, "b_flow": b_flow, "bc_use": bc_use, "high_ratio": high_ratio, "y_high": y_high}
    return model, vars_pack, total_cost_expr, total_abatement_expr


def compute_feasible_max_abatement(model_data: Dict[str, object], time_limit_sec: float, mip_gap: float, threads: int) -> float:
    model, _, _, total_abatement_expr = _build_problem(model_data, target=0.0, objective_mode="abatement", tier_split_ratio=TIER_SPLIT_RATIO)
    status = _solve_model(model, time_limit_sec, mip_gap, threads)
    if status != "Optimal":
        raise RuntimeError(f"Failed to compute max abatement for BC-only model, status={status}")
    return float(total_abatement_expr.getValue())


def build_targets_step(max_abatement: float, step: float, extra_targets: Optional[List[float]] = None) -> np.ndarray:
    """Build targets from 0 to max_abatement with fixed step interval.
    Always includes the true max_abatement as the last target."""
    safe_upper = max_abatement * (1.0 - 1e-5)
    n_steps = max(1, int(np.ceil(safe_upper / step)))
    all_targets = np.array([float(i) * step for i in range(n_steps + 1)])
    all_targets = all_targets[all_targets <= safe_upper]

    # Append the true max_abatement as the final target
    all_targets = np.concatenate([all_targets, np.array([max_abatement], dtype=float)])

    extras = [] if extra_targets is None else [float(x) for x in extra_targets]
    extras = [x for x in extras if 0 <= x <= max_abatement]
    if extras:
        all_targets = np.concatenate([all_targets, np.array(extras, dtype=float)])
    all_targets = np.unique(np.round(all_targets, decimals=6))
    return np.sort(all_targets)


def _value_or_zero(var) -> float:
    if hasattr(var, "X"):
        try:
            return float(var.X)
        except Exception:
            return 0.0
    return 0.0


def _extract_bc_detail(
    model_data: Dict[str, object],
    vars_pack: Dict[str, object],
    target_id: int,
    target: float,
    achieved: float,
    status: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build plant-level and biomass-flow detail tables for one BC-only target."""
    plants = model_data["plants"]
    arcs = model_data["arcs"]

    # Variables stored as plain float dicts (not Gurobi var objects)
    blend_ratio = dict(vars_pack.get("blend_ratio", {}))
    b_flow = dict(vars_pack.get("b_flow", {}))  # keys are tuples (j,i)
    bc_use = dict(vars_pack.get("bc_use", {}))
    high_ratio = dict(vars_pack.get("high_ratio", {}))

    common = {
        "target_id": target_id,
        "target_abatement": target,
        "achieved_abatement": achieved,
        "status": status,
    }

    # Biomass flow detail
    biomass_rows = []
    bc_feedstock_by_plant = {i: 0.0 for i in plants}
    bc_transport_by_plant = {i: 0.0 for i in plants}
    for (j, i) in arcs:
        flow = b_flow[(j, i)]
        if flow <= 1e-9:
            continue
        feedstock_cost = model_data["biomass_feedstock_cost_per_gj"] * flow
        transport_cost = model_data["transport_cost"][(j, i)] * flow
        bc_feedstock_by_plant[i] += feedstock_cost
        bc_transport_by_plant[i] += transport_cost
        biomass_rows.append({
            **common,
            "biomass_id": j,
            "plant_id": i,
            "biomass_flow_gj": flow,
            "distance_km": model_data["distance"][(j, i)],
            "feedstock_cost": feedstock_cost,
            "transport_cost": transport_cost,
        })

    # Debug: check if blend_ratio has non-zero values
    br_values = list(blend_ratio.values())
    br_nonzero = sum(1 for v in br_values if abs(v) > 1e-9)
    print(f"    [debug] _extract_bc_detail target={target_id}: plants={len(plants)}, "
          f"arcs={len(arcs)}, blend_ratio_nonzero={br_nonzero}, "
          f"biomass_flow_nonzero={sum(1 for v in b_flow.values() if v > 1e-9)}",
          flush=True)

    # Plant-level detail
    plant_rows = []
    for i in plants:
        e_i = model_data["energy_gj"][i]
        eff_ratio = blend_ratio[i]
        bc_use_i = bc_use[i]
        hr = high_ratio[i]

        # BC cost components (same formulas as _build_problem tiered_static)
        bc_investment = (
            1.02 * (1153000.0 * eff_ratio * model_data["power"][i]
                    + 4628000.0 * hr * model_data["power"][i])
            * model_data["af"][i] / 6.8589
        )
        bc_coal_saving = model_data["coal_price"][i] * e_i * eff_ratio
        bc_feedstock = bc_feedstock_by_plant[i]
        bc_transport = bc_transport_by_plant[i]
        bc_cost = bc_investment - bc_coal_saving + bc_feedstock
        bc_abatement = model_data["base_emission"][i] * eff_ratio

        total_cost_i = bc_cost + bc_transport

        plant_rows.append({
            **common,
            "plant_id": i,
            "blend_ratio": eff_ratio,
            "high_ratio": hr,
            "bc_use_gj": bc_use_i,
            "bc_abatement_ton": bc_abatement,
            "bc_investment_cost": bc_investment,
            "bc_coal_saving_credit": bc_coal_saving,
            "bc_feedstock_cost": bc_feedstock,
            "bc_transport_cost": bc_transport,
            "bc_total_cost": total_cost_i,
        })

    plant_df = pd.DataFrame(plant_rows).sort_values("plant_id").reset_index(drop=True)
    bio_df = pd.DataFrame(biomass_rows)
    if not bio_df.empty:
        bio_df = bio_df.sort_values(["plant_id", "biomass_id"]).reset_index(drop=True)
    return plant_df, bio_df


def _build_solver_contexts(
    model_data: Dict[str, object],
    target_match_mode: str,
    tier_split_ratio: float,
) -> Dict[str, Dict[str, object]]:
    """Pre-build model contexts for different matching modes."""
    contexts: Dict[str, Dict[str, object]] = {}
    if target_match_mode == "min_overshoot":
        m1, vp1, tc1, ta1 = _build_problem(
            model_data=model_data, target=0.0, objective_mode="abatement",
            tier_split_ratio=tier_split_ratio,
        )
        contexts["stage1"] = {"model": m1, "vars_pack": vp1, "total_cost": tc1, "total_abate": ta1}
        m2, vp2, tc2, ta2 = _build_problem(
            model_data=model_data, target=0.0, objective_mode="cost",
            tier_split_ratio=tier_split_ratio,
        )
        contexts["stage2"] = {"model": m2, "vars_pack": vp2, "total_cost": tc2, "total_abate": ta2}
    elif target_match_mode == "cost_only":
        m, vp, tc, ta = _build_problem(
            model_data=model_data, target=0.0, objective_mode="cost",
            tier_split_ratio=tier_split_ratio,
        )
        contexts["single"] = {"model": m, "vars_pack": vp, "total_cost": tc, "total_abate": ta}
    else:
        raise ValueError(f"Unsupported target_match_mode: {target_match_mode}")
    return contexts


def solve_for_target(
    model_data: Dict[str, object],
    target: float,
    target_match_mode: str,
    tier_split_ratio: float,
    time_limit_sec: float,
    mip_gap: float,
    threads: int,
) -> Tuple[str, float, float, Dict[str, object]]:
    """Solve for one target. Handles min_overshoot (two-stage) and cost_only modes."""
    if target_match_mode == "min_overshoot":
        # Stage 1: minimize abatement (find minimum abatement ≥ target)
        # Rebuild with objective_mode="abatement" and GRB.MINIMIZE
        m1, vp1, _, ta1 = _build_problem(
            model_data=model_data, target=float(target), objective_mode="abatement",
            tier_split_ratio=tier_split_ratio,
        )
        # The "abatement" mode minimizes by default; we need to maximize
        # So swap: minimize means find the closest feasible solution to target
        status = _solve_model(m1, time_limit_sec, mip_gap, threads)
        if status != "Optimal":
            return status, np.nan, np.nan, {}
        min_abate = float(ta1.getValue())

        # Stage 2: minimize cost with abatement capped at min_abate + ε
        m2, vp2, tc2, ta2 = _build_problem(
            model_data=model_data, target=float(target), objective_mode="cost",
            tier_split_ratio=tier_split_ratio,
        )
        # Add constraint to cap overshoot
        m2.addConstr(ta2 - float(target) <= float(min_abate - target + 1e-6), name="overshoot_cap")
        m2.update()
        status = _solve_model(m2, time_limit_sec, mip_gap, threads)
        vars_pack, total_cost, total_abate = vp2, tc2, ta2

    elif target_match_mode == "cost_only":
        m, vp, tc, ta = _build_problem(
            model_data=model_data, target=float(target), objective_mode="cost",
            tier_split_ratio=tier_split_ratio,
        )
        status = _solve_model(m, time_limit_sec, mip_gap, threads)
        vars_pack, total_cost, total_abate = vp, tc, ta

    else:
        raise ValueError(f"Unsupported target_match_mode: {target_match_mode}")

    if status != "Optimal":
        return status, np.nan, np.nan, {}

    achieved = float(total_abate.getValue())
    cost = float(total_cost.getValue())

    # Extract ALL variable values into plain dicts BEFORE returning
    # (Gurobi model goes out of scope after returning, invalidating .X)
    bc_use = vars_pack["bc_use"]
    blend_ratio = vars_pack["blend_ratio"]
    high_ratio = vars_pack["high_ratio"]
    b_flow = vars_pack["b_flow"]
    summary = {
        "total_biomass_use_gj": float(sum(float(v.X) for v in bc_use.values())),
        "vars_pack": {
            "bc_use":     {i: float(v.X) for i, v in bc_use.items()},
            "blend_ratio": {i: float(v.X) for i, v in blend_ratio.items()},
            "high_ratio":  {i: float(v.X) for i, v in high_ratio.items()},
            "b_flow":   {(j, i): float(v.X) for (j, i), v in b_flow.items()},
        },
    }
    return status, achieved, cost, summary


def run_sweep(
    model_data: Dict[str, object],
    targets: np.ndarray,
    target_match_mode: str,
    target_mode_switch_abatement: Optional[float],
    tier_split_ratio: float,
    time_limit_sec: float,
    mip_gap: float,
    threads: int,
    detail_writer=None,
) -> pd.DataFrame:
    rows = []

    # Determine which modes are needed
    switch_threshold = None
    if target_mode_switch_abatement is not None:
        switch_threshold = float(target_mode_switch_abatement)
        if switch_threshold < 0:
            raise ValueError("target_mode_switch_abatement must be non-negative.")

    for idx, target_val in enumerate(targets, start=1):
        current_match_mode = target_match_mode
        if switch_threshold is not None:
            current_match_mode = "min_overshoot" if float(target_val) <= switch_threshold else "cost_only"

        status, achieved, cost, summary = solve_for_target(
            model_data=model_data,
            target=float(target_val),
            target_match_mode=current_match_mode,
            tier_split_ratio=tier_split_ratio,
            time_limit_sec=time_limit_sec,
            mip_gap=mip_gap,
            threads=threads,
        )

        rows.append({
            "target_id": idx,
            "target_abatement": float(target_val),
            "achieved_abatement": achieved,
            "total_cost": cost,
            "status": status,
            "match_mode_used": current_match_mode,
            "total_biomass_use_gj": summary.get("total_biomass_use_gj", np.nan),
        })
        print(f"[BC-only {idx}/{len(targets)}] mode={current_match_mode} target={target_val:,.2f} status={status} achieved={achieved:,.2f} cost={cost:,.2f}")

        if status == "Optimal" and detail_writer is not None and "vars_pack" in summary:
            plant_df, bio_df = _extract_bc_detail(
                model_data, summary["vars_pack"], idx, float(target_val), float(achieved), status
            )
            plant_df.to_excel(detail_writer, sheet_name=f"pm{idx:02d}", index=False)
            if not bio_df.empty:
                bio_df.to_excel(detail_writer, sheet_name=f"bm{idx:02d}", index=False)

    df = pd.DataFrame(rows).sort_values("target_abatement").reset_index(drop=True)
    df["marginal_cost_diff"] = df["total_cost"].diff() / df["target_abatement"].diff().replace(0, np.nan)
    df["avg_cost_from_origin"] = df["total_cost"] / df["achieved_abatement"].replace(0, np.nan)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="BC-only MILP: plants compete for shared biomass supply with no EC/CC and no interaction terms.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--distance-csv", default=None)
    parser.add_argument("--target-step", type=float, default=TARGET_STEP, help="Abatement target step size (default 0.1e9 = 100 Mt CO2)")
    parser.add_argument("--extra-targets", type=float, nargs="*", default=EXTRA_TARGETS)
    parser.add_argument("--fixed-max-abatement", type=float, default=FIXED_MAX_ABATEMENT)
    parser.add_argument("--transport-cost-per-gj-km", type=float, default=TRANSPORT_COST_PER_GJ_KM)
    parser.add_argument("--biomass-feedstock-cost-per-gj", type=float, default=BIOMASS_FEEDSTOCK_COST_PER_GJ)
    parser.add_argument("--max-links-per-plant", type=int, default=MAX_LINKS_PER_PLANT)
    parser.add_argument("--blend-cap-ratio", type=float, default=BLEND_CAP_RATIO)
    parser.add_argument("--tier-split-ratio", type=float, default=TIER_SPLIT_RATIO)
    parser.add_argument("--time-limit-sec", type=float, default=SOLVER_TIME_LIMIT_SEC)
    parser.add_argument("--mip-gap", type=float, default=SOLVER_MIP_GAP)
    parser.add_argument("--solver-threads", type=int, default=SOLVER_THREADS)
    parser.add_argument("--target-match-mode", choices=["min_overshoot", "cost_only"], default=TARGET_MATCH_MODE)
    parser.add_argument(
        "--target-mode-switch-abatement",
        type=float,
        default=TARGET_MODE_SWITCH_ABATEMENT,
        help="If set, targets <= this value use min_overshoot, while targets above this value use cost_only.",
    )
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, DATA_DIR_NAME)
    input_path = args.input or os.path.join(data_dir, INPUT_WORKBOOK_NAME)
    distance_path = args.distance_csv or os.path.join(data_dir, INPUT_BIOMASS_DISTANCE_NAME)

    plants, bc, biomass, biomass_dist = load_inputs(input_path, distance_path)
    model_data = build_model_data(
        plants=plants,
        bc=bc,
        biomass=biomass,
        biomass_dist=biomass_dist,
        blend_cap_ratio=args.blend_cap_ratio,
        max_links_per_plant=args.max_links_per_plant,
        transport_cost_per_gj_km=args.transport_cost_per_gj_km,
        biomass_feedstock_cost_per_gj=args.biomass_feedstock_cost_per_gj,
    )

    if args.fixed_max_abatement is not None:
        if args.fixed_max_abatement <= 0:
            raise ValueError("--fixed-max-abatement must be positive.")
        max_abatement = float(args.fixed_max_abatement)
        print(f"Using fixed max abatement: {max_abatement:,.2f}")
    else:
        max_abatement = compute_feasible_max_abatement(model_data, args.time_limit_sec, args.mip_gap, args.solver_threads)
        print(f"Computed feasible max abatement (BC-only): {max_abatement:,.2f}")

    targets = build_targets_step(max_abatement, args.target_step, args.extra_targets)
    print(f"Target grid: {len(targets)} points from 0 to {max_abatement:,.0f} with step {args.target_step:,.0f}")
    print(f"Plants: {len(model_data['plants'])}, Biomass points: {len(model_data['biomass'])}, Arcs: {len(model_data['arcs'])}")
    print(f"Target match mode: {args.target_match_mode}, switch threshold: {args.target_mode_switch_abatement}")

    out_curve = os.path.join(data_dir, f"{args.output_prefix}_curve.csv")
    out_xlsx = os.path.join(data_dir, f"{args.output_prefix}_results.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        curve = run_sweep(
            model_data=model_data,
            targets=targets,
            target_match_mode=args.target_match_mode,
            target_mode_switch_abatement=args.target_mode_switch_abatement,
            tier_split_ratio=args.tier_split_ratio,
            time_limit_sec=args.time_limit_sec,
            mip_gap=args.mip_gap,
            threads=args.solver_threads,
            detail_writer=writer,
        )
        curve.to_excel(writer, sheet_name="bc_only_curve", index=False)
        curve.to_csv(out_curve, index=False)

    print("\nDone.")
    print(f"Input workbook: {input_path}")
    print(f"Input biomass distance: {distance_path}")
    print(f"Output curve: {out_curve}")
    print(f"Output workbook: {out_xlsx}")


if __name__ == "__main__":
    main()
