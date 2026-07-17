import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


DATA_DIR_NAME = "data"
INPUT_WORKBOOK_NAME = "1885_coal_power_plants_original.xlsx"
INPUT_CO2_DISTANCE_NAME = "Dki_4000.csv"
OUTPUT_PREFIX = "gurobi_cc_only"
TARGET_STEP = 0.1e9  # 100 Mt CO2 step between targets
EXTRA_TARGETS: List[float] = []
FIXED_MAX_ABATEMENT = None
TARGET_MATCH_MODE = "cost_only"
CO2_STORAGE_COST_PER_TON = 5.0
CO2_TRANSPORT_COST_PER_TON_KM = 0.1
MAX_SINK_LINKS_PER_PLANT = 10
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
    co2_distance_csv: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plants_raw = pd.read_excel(workbook_path, sheet_name="coal_plant_i")
    cc_raw = pd.read_excel(workbook_path, sheet_name="CC")
    sinks_raw = pd.read_excel(workbook_path, sheet_name="carbon_sink_k")
    co2_dist_raw = pd.read_csv(co2_distance_csv)

    pcols = list(plants_raw.columns)
    pid = _find_col(pcols, ["plant_i", "plantid", "id"])
    plants = pd.DataFrame({"plant_id": plants_raw[pid].map(_norm_id)})

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
    cc_need_total = cc["cc_total_cost"].isna()
    cc.loc[cc_need_total, "cc_total_cost"] = cc.loc[cc_need_total, "cc_unit_cost"] * cc.loc[cc_need_total, "cc_abatement"]
    cc["cc_total_cost"] = cc["cc_total_cost"].fillna(0.0)
    cc = cc.groupby("plant_id", as_index=False).first()

    scols = list(sinks_raw.columns)
    s_id = _find_col(scols, ["sink_k", "sink", "k"])
    s_cap = _find_col(scols, ["Estimated_storage_potential_Mt", "estimatedstoragepotentialmt", "capacity", "storage"])
    sinks = pd.DataFrame(
        {
            "sink_id": sinks_raw[s_id].map(_norm_id),
            "capacity_raw": pd.to_numeric(sinks_raw[s_cap], errors="coerce").fillna(0.0),
        }
    ).groupby("sink_id", as_index=False).sum()
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

    return plants, cc, sinks, co2_dist


def build_model_data(
    plants: pd.DataFrame,
    cc: pd.DataFrame,
    sinks: pd.DataFrame,
    co2_dist: pd.DataFrame,
    max_sink_links_per_plant: int,
    co2_transport_cost_per_ton_km: float,
    co2_storage_cost_per_ton: float,
) -> Dict[str, object]:
    df = plants.merge(cc, on="plant_id", how="left")
    for c in ["cc_abatement", "cc_total_cost", "cc_capex_per_year", "cc_opex_per_t", "cc_ele1_per_t", "cc_ele2_per_t"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    plant_ids = sorted(df["plant_id"].astype(str).unique().tolist())
    sink_ids = sorted(sinks["sink_id"].astype(str).unique().tolist())

    co2_dist2 = co2_dist[co2_dist["plant_id"].isin(plant_ids) & co2_dist["sink_id"].isin(sink_ids)].copy()
    if max_sink_links_per_plant is not None and max_sink_links_per_plant > 0:
        co2_dist2 = co2_dist2.sort_values(["plant_id", "distance_km"]).groupby("plant_id", as_index=False).head(max_sink_links_per_plant)

    sink_arcs = [(r["plant_id"], r["sink_id"]) for _, r in co2_dist2.iterrows()]
    sink_distance = {(r["plant_id"], r["sink_id"]): float(r["distance_km"]) for _, r in co2_dist2.iterrows()}

    cc_abate = {r["plant_id"]: float(r["cc_abatement"]) for _, r in df.iterrows()}
    cc_cost = {r["plant_id"]: float(r["cc_total_cost"]) for _, r in df.iterrows()}
    cc_capex = {r["plant_id"]: float(r["cc_capex_per_year"]) for _, r in df.iterrows()}
    cc_opex_per_t = {r["plant_id"]: float(r["cc_opex_per_t"]) for _, r in df.iterrows()}
    cc_ele1_per_t = {r["plant_id"]: float(r["cc_ele1_per_t"]) for _, r in df.iterrows()}
    cc_ele2_per_t = {r["plant_id"]: float(r["cc_ele2_per_t"]) for _, r in df.iterrows()}

    sink_capacity_ton = {r["sink_id"]: float(r["capacity_ton"]) for _, r in sinks.iterrows()}
    co2_transport_cost = {(i, k): co2_transport_cost_per_ton_km * sink_distance[(i, k)] for (i, k) in sink_arcs}

    return {
        "plants": plant_ids,
        "sinks": sink_ids,
        "sink_arcs": sink_arcs,
        "sink_distance": sink_distance,
        "sink_capacity_ton": sink_capacity_ton,
        "co2_transport_cost": co2_transport_cost,
        "co2_storage_cost_per_ton": co2_storage_cost_per_ton,
        "cc_abate": cc_abate,
        "cc_cost": cc_cost,
        "cc_capex": cc_capex,
        "cc_opex_per_t": cc_opex_per_t,
        "cc_ele1_per_t": cc_ele1_per_t,
        "cc_ele2_per_t": cc_ele2_per_t,
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


def _build_problem(model_data: Dict[str, object], target: float, objective_mode: str):
    plants = model_data["plants"]
    sink_arcs = model_data["sink_arcs"]
    sinks = model_data["sinks"]

    model = gp.Model("CC_Only_MILP")

    x_cc = {i: model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f"x_cc__{i}") for i in plants}
    co2_flow = {(i, k): model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"CO2__{i}__{k}") for (i, k) in sink_arcs}

    cc_capture_expr = {i: model_data["cc_abate"][i] * x_cc[i] for i in plants}

    cc_cost_expr = gp.quicksum(
        (
            model_data["cc_capex"][i] * x_cc[i]
            + (model_data["cc_opex_per_t"][i] + model_data["cc_ele1_per_t"][i]) * cc_capture_expr[i]
        )
        if not (
            model_data["cc_capex"][i] == 0
            and model_data["cc_opex_per_t"][i] == 0
            and model_data["cc_ele1_per_t"][i] == 0
            and model_data["cc_ele2_per_t"][i] == 0
        )
        else model_data["cc_cost"][i] * x_cc[i]
        for i in plants
    )

    ts_cost_expr = gp.quicksum(
        (model_data["co2_transport_cost"][(i, k)] + model_data["co2_storage_cost_per_ton"]) * co2_flow[(i, k)]
        for (i, k) in sink_arcs
    )

    total_cost_expr = cc_cost_expr + ts_cost_expr
    total_abatement_expr = gp.quicksum(cc_capture_expr[i] for i in plants)

    if objective_mode == "cost":
        model.setObjective(total_cost_expr, GRB.MINIMIZE)
    elif objective_mode == "abatement":
        model.setObjective(total_abatement_expr, GRB.MAXIMIZE)
    else:
        raise ValueError(f"Unsupported objective_mode: {objective_mode}")

    for i in plants:
        out_to_sinks = [co2_flow[(ii, k)] for (ii, k) in sink_arcs if ii == i]
        if out_to_sinks:
            model.addConstr(gp.quicksum(out_to_sinks) == cc_capture_expr[i])
        else:
            model.addConstr(cc_capture_expr[i] == 0)

    for k in sinks:
        in_to_sink = [co2_flow[(i, kk)] for (i, kk) in sink_arcs if kk == k]
        if in_to_sink:
            model.addConstr(gp.quicksum(in_to_sink) <= model_data["sink_capacity_ton"][k])

    model.addConstr(total_abatement_expr >= float(target), name="target_lb")

    vars_pack = {"x_cc": x_cc, "co2_flow": co2_flow}
    return model, vars_pack, total_cost_expr, total_abatement_expr


def compute_feasible_max_abatement(model_data: Dict[str, object], time_limit_sec: float, mip_gap: float, threads: int) -> float:
    model, _, _, total_abatement_expr = _build_problem(model_data, target=0.0, objective_mode="abatement")
    status = _solve_model(model, time_limit_sec, mip_gap, threads)
    if status != "Optimal":
        raise RuntimeError(f"Failed to compute max abatement for CC-only model, status={status}")
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


def _extract_cc_detail(
    model_data: Dict[str, object],
    vars_pack: Dict[str, object],
    target_id: int,
    target: float,
    achieved: float,
    status: str,
) -> pd.DataFrame:
    """Build plant-level detail table for one CC-only target."""
    plants = model_data["plants"]
    sink_arcs = model_data["sink_arcs"]

    x_cc = {i: _value_or_zero(v) for i, v in vars_pack["x_cc"].items()}
    co2_flow = {(i, k): _value_or_zero(v) for (i, k), v in vars_pack["co2_flow"].items()}

    common = {
        "target_id": target_id,
        "target_abatement": target,
        "achieved_abatement": achieved,
        "status": status,
    }

    # Group CO2 flow by plant and sink for detail Co2 sheet
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
        co2_rows.append({
            **common,
            "plant_id": i,
            "sink_id": k,
            "co2_flow_ton": flow,
            "distance_km": model_data["sink_distance"][(i, k)],
            "transport_cost": transport_cost,
            "storage_cost": storage_cost,
            "total_cost": transport_cost + storage_cost,
        })

    # Plant-level detail
    plant_rows = []
    for i in plants:
        cc_abate = model_data["cc_abate"][i]
        cc_capture = cc_abate * x_cc[i]
        cc_is_decomposed = not (
            model_data["cc_capex"][i] == 0
            and model_data["cc_opex_per_t"][i] == 0
            and model_data["cc_ele1_per_t"][i] == 0
            and model_data["cc_ele2_per_t"][i] == 0
        )
        if cc_is_decomposed:
            cc_cost = (
                model_data["cc_capex"][i] * x_cc[i]
                + model_data["cc_opex_per_t"][i] * cc_capture
                + model_data["cc_ele1_per_t"][i] * cc_capture
            )
        else:
            cc_cost = model_data["cc_cost"][i] * x_cc[i]

        ts_cost = co2_transport_cost_by_plant[i] + co2_storage_cost_by_plant[i]

        plant_rows.append({
            **common,
            "plant_id": i,
            "x_cc": x_cc[i],
            "cc_abatement_potential_ton": cc_abate,
            "cc_capture_ton": cc_capture,
            "co2_stored_ton": co2_stored_by_plant[i],
            "cc_cost": cc_cost,
            "cc_capex": model_data["cc_capex"][i] * x_cc[i],
            "cc_opex": model_data["cc_opex_per_t"][i] * cc_capture,
            "cc_energy": model_data["cc_ele1_per_t"][i] * cc_capture,
            "co2_transport_cost": co2_transport_cost_by_plant[i],
            "co2_storage_cost": co2_storage_cost_by_plant[i],
            "ts_total_cost": ts_cost,
            "total_cost": cc_cost + ts_cost,
        })

    plant_df = pd.DataFrame(plant_rows).sort_values("plant_id").reset_index(drop=True)
    co2_df = pd.DataFrame(co2_rows)
    if not co2_df.empty:
        co2_df = co2_df.sort_values(["plant_id", "sink_id"]).reset_index(drop=True)
    return plant_df, co2_df


def run_sweep(model_data: Dict[str, object], targets: np.ndarray, time_limit_sec: float, mip_gap: float, threads: int, detail_writer=None) -> pd.DataFrame:
    rows = []
    for idx, target in enumerate(targets, start=1):
        model, vars_pack, total_cost_expr, total_abatement_expr = _build_problem(model_data, target=float(target), objective_mode="cost")
        status = _solve_model(model, time_limit_sec, mip_gap, threads)

        if status == "Optimal":
            achieved = float(total_abatement_expr.getValue())
            cost = float(total_cost_expr.getValue())
            x_cc = vars_pack["x_cc"]
            co2_flow = vars_pack["co2_flow"]
            cc_adoption_sum = float(sum(float(v.X) for v in x_cc.values()))
            total_co2_stored_ton = float(sum(float(v.X) for v in co2_flow.values()))
        else:
            achieved = np.nan
            cost = np.nan
            cc_adoption_sum = np.nan
            total_co2_stored_ton = np.nan

        rows.append(
            {
                "target_id": idx,
                "target_abatement": float(target),
                "achieved_abatement": achieved,
                "total_cost": cost,
                "status": status,
                "cc_adoption_sum": cc_adoption_sum,
                "total_co2_stored_ton": total_co2_stored_ton,
            }
        )
        print(f"[CC-only {idx}/{len(targets)}] target={target:,.2f} status={status} achieved={achieved:,.2f} cost={cost:,.2f}")

        # Write detail sheets if writer provided
        if status == "Optimal" and detail_writer is not None:
            plant_df, co2_df = _extract_cc_detail(
                model_data, vars_pack, idx, float(target), float(achieved), status
            )
            plant_df.to_excel(detail_writer, sheet_name=f"pm{idx:02d}", index=False)
            if not co2_df.empty:
                co2_df.to_excel(detail_writer, sheet_name=f"cm{idx:02d}", index=False)

    df = pd.DataFrame(rows).sort_values("target_abatement").reset_index(drop=True)
    df["marginal_cost_diff"] = df["total_cost"].diff() / df["target_abatement"].diff().replace(0, np.nan)
    df["avg_cost_from_origin"] = df["total_cost"] / df["achieved_abatement"].replace(0, np.nan)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="CC-only MILP: plants compete for shared CO2 sink capacity with no EC/BC and no interaction terms.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--co2-distance-csv", default=None)
    parser.add_argument("--target-step", type=float, default=TARGET_STEP, help="Abatement target step size (default 0.1e9 = 100 Mt CO2)")
    parser.add_argument("--extra-targets", type=float, nargs="*", default=EXTRA_TARGETS)
    parser.add_argument("--fixed-max-abatement", type=float, default=FIXED_MAX_ABATEMENT)
    parser.add_argument("--co2-transport-cost-per-ton-km", type=float, default=CO2_TRANSPORT_COST_PER_TON_KM)
    parser.add_argument("--co2-storage-cost-per-ton", type=float, default=CO2_STORAGE_COST_PER_TON)
    parser.add_argument("--max-sink-links-per-plant", type=int, default=MAX_SINK_LINKS_PER_PLANT)
    parser.add_argument("--time-limit-sec", type=float, default=SOLVER_TIME_LIMIT_SEC)
    parser.add_argument("--mip-gap", type=float, default=SOLVER_MIP_GAP)
    parser.add_argument("--solver-threads", type=int, default=SOLVER_THREADS)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, DATA_DIR_NAME)
    input_path = args.input or os.path.join(data_dir, INPUT_WORKBOOK_NAME)
    co2_distance_path = args.co2_distance_csv or os.path.join(data_dir, INPUT_CO2_DISTANCE_NAME)

    plants, cc, sinks, co2_dist = load_inputs(input_path, co2_distance_path)
    model_data = build_model_data(
        plants=plants,
        cc=cc,
        sinks=sinks,
        co2_dist=co2_dist,
        max_sink_links_per_plant=args.max_sink_links_per_plant,
        co2_transport_cost_per_ton_km=args.co2_transport_cost_per_ton_km,
        co2_storage_cost_per_ton=args.co2_storage_cost_per_ton,
    )

    if args.fixed_max_abatement is not None:
        if args.fixed_max_abatement <= 0:
            raise ValueError("--fixed-max-abatement must be positive.")
        max_abatement = float(args.fixed_max_abatement)
        print(f"Using fixed max abatement: {max_abatement:,.2f}")
    else:
        max_abatement = compute_feasible_max_abatement(model_data, args.time_limit_sec, args.mip_gap, args.solver_threads)
        print(f"Computed feasible max abatement (CC-only): {max_abatement:,.2f}")

    targets = build_targets_step(max_abatement, args.target_step, args.extra_targets)
    print(f"Target grid: {len(targets)} points from 0 to {max_abatement:,.0f} with step {args.target_step:,.0f}")
    print(
        f"Plants: {len(model_data['plants'])}, Sinks: {len(model_data['sinks'])}, "
        f"Sink arcs: {len(model_data['sink_arcs'])}"
    )

    out_curve = os.path.join(data_dir, f"{args.output_prefix}_curve.csv")
    out_xlsx = os.path.join(data_dir, f"{args.output_prefix}_results.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        curve = run_sweep(model_data, targets, args.time_limit_sec, args.mip_gap, args.solver_threads, detail_writer=writer)
        curve.to_excel(writer, sheet_name="cc_only_curve", index=False)
        curve.to_csv(out_curve, index=False)

    print("\nDone.")
    print(f"Input workbook: {input_path}")
    print(f"Input CO2 distance: {co2_distance_path}")
    print(f"Output curve: {out_curve}")
    print(f"Output workbook: {out_xlsx}")


if __name__ == "__main__":
    main()
