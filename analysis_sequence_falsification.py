import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gurobipy import GRB

import optimization_1_fleet_joint_milp_persistent as base


# ---- User config (edit before running) ----
DATA_DIR_NAME = "data"
DEFAULT_POLICIES = ["unconstrained", "ec_bc_cc", "ec_cc_bc", "bc_cc", "cc_bc"]
DEFAULT_REFERENCE_POLICY = "unconstrained"
DEFAULT_OUTPUT_PREFIX = "sequence_falsification"
# -------------------------------------------


def _policy_help() -> str:
    return (
        "Supported policies:\n"
        "  unconstrained : no cross-technology ordering constraints (joint optimum reference)\n"
        "  ec_bc_cc      : EC -> BC -> CC (CC requires BC, BC requires EC)\n"
        "  ec_cc_bc      : EC -> CC -> BC (BC requires CC, CC requires EC)\n"
        "  bc_cc         : BC -> CC, and EC disabled\n"
        "  cc_bc         : CC -> BC, and EC disabled"
    )


def _validate_policies(policies: List[str]) -> None:
    allowed = {"unconstrained", "ec_bc_cc", "ec_cc_bc", "bc_cc", "cc_bc"}
    unknown = sorted(set(policies) - allowed)
    if unknown:
        raise ValueError(f"Unknown policies: {unknown}.\n{_policy_help()}")


def _apply_sequence_constraints(
    model,
    vars_pack: Dict[str, object],
    model_data: Dict[str, object],
    policy: str,
) -> None:
    if policy == "unconstrained":
        return

    plants = model_data["plants"]
    x_ec = vars_pack["x_ec"]
    x_cc = vars_pack["x_cc"]
    bc_use = vars_pack["bc_use"]

    # Introduce a BC adoption binary (z_bc) so sequence rules can reference BC as a discrete stage.
    vtype = GRB.CONTINUOUS if next(iter(x_ec.values())).VType == GRB.CONTINUOUS else GRB.BINARY
    z_bc = {
        i: model.addVar(lb=0.0, ub=1.0, vtype=vtype, name=f"z_bc__{i}")
        for i in plants
    }
    model.update()

    for i in plants:
        cap = float(model_data["bc_cap_gj"][i])
        model.addConstr(bc_use[i] <= cap * z_bc[i], name=f"bc_link__{i}")

        if policy == "ec_bc_cc":
            model.addConstr(z_bc[i] <= x_ec[i], name=f"ord_ec_bc__{i}")
            model.addConstr(x_cc[i] <= z_bc[i], name=f"ord_bc_cc__{i}")

        elif policy == "ec_cc_bc":
            model.addConstr(x_cc[i] <= x_ec[i], name=f"ord_ec_cc__{i}")
            model.addConstr(z_bc[i] <= x_cc[i], name=f"ord_cc_bc__{i}")

        elif policy == "bc_cc":
            model.addConstr(x_ec[i] == 0.0, name=f"disable_ec__{i}")
            model.addConstr(x_cc[i] <= z_bc[i], name=f"ord_bc_cc__{i}")

        elif policy == "cc_bc":
            model.addConstr(x_ec[i] == 0.0, name=f"disable_ec__{i}")
            model.addConstr(z_bc[i] <= x_cc[i], name=f"ord_cc_bc__{i}")

        else:
            raise ValueError(f"Unsupported policy: {policy}")

    model.update()


def _build_solver_contexts_with_policy(
    model_data: Dict[str, object],
    target_match_mode: str,
    relax_integrality: bool,
    bc_cost_mode: str,
    tier_split_ratio: float,
    policy: str,
) -> Dict[str, Dict[str, object]]:
    contexts: Dict[str, Dict[str, object]] = {}

    def _build_one(objective_mode: str) -> Dict[str, object]:
        model, vars_pack, total_cost, total_abate, target_lb, target_ub = base._build_problem(
            model_data=model_data,
            target=0.0,
            objective_mode=objective_mode,
            relax_integrality=relax_integrality,
            bc_cost_mode=bc_cost_mode,
            tier_split_ratio=tier_split_ratio,
        )
        _apply_sequence_constraints(model, vars_pack, model_data, policy)
        return {
            "model": model,
            "vars_pack": vars_pack,
            "total_cost": total_cost,
            "total_abate": total_abate,
            "target_lb": target_lb,
            "target_ub": target_ub,
        }

    if target_match_mode == "min_overshoot":
        contexts["stage1"] = _build_one("abatement")
        contexts["stage2"] = _build_one("cost")
    elif target_match_mode in ("cost_only", "equality"):
        contexts["single"] = _build_one("cost")
    else:
        raise ValueError(f"Unsupported target_match_mode: {target_match_mode}")

    return contexts


def run_sweep_for_policy(
    model_data: Dict[str, object],
    targets: np.ndarray,
    use_gurobi: bool,
    target_match_mode: str,
    target_mode_switch_abatement: Optional[float],
    relax_integrality: bool,
    bc_cost_mode: str,
    tier_split_ratio: float,
    time_limit_sec: float,
    mip_gap: float,
    threads: int,
    policy: str,
) -> pd.DataFrame:
    rows = []
    mode = "LP" if relax_integrality else "MILP"

    switch_threshold = None
    if target_mode_switch_abatement is not None:
        switch_threshold = float(target_mode_switch_abatement)
        if switch_threshold < 0:
            raise ValueError("target_mode_switch_abatement must be non-negative.")

    if switch_threshold is None:
        contexts_by_mode = {
            target_match_mode: _build_solver_contexts_with_policy(
                model_data=model_data,
                target_match_mode=target_match_mode,
                relax_integrality=relax_integrality,
                bc_cost_mode=bc_cost_mode,
                tier_split_ratio=tier_split_ratio,
                policy=policy,
            )
        }
    else:
        contexts_by_mode = {
            "min_overshoot": _build_solver_contexts_with_policy(
                model_data=model_data,
                target_match_mode="min_overshoot",
                relax_integrality=relax_integrality,
                bc_cost_mode=bc_cost_mode,
                tier_split_ratio=tier_split_ratio,
                policy=policy,
            ),
            "cost_only": _build_solver_contexts_with_policy(
                model_data=model_data,
                target_match_mode="cost_only",
                relax_integrality=relax_integrality,
                bc_cost_mode=bc_cost_mode,
                tier_split_ratio=tier_split_ratio,
                policy=policy,
            ),
        }

    for idx, target in enumerate(targets, start=1):
        current_match_mode = target_match_mode
        if switch_threshold is not None:
            current_match_mode = "min_overshoot" if float(target) <= switch_threshold else "cost_only"

        status, achieved, cost, summary = base.solve_for_target(
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
                "policy": policy,
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
            }
        )

        print(
            f"[{policy} | {mode} {idx}/{len(targets)}] mode={current_match_mode} "
            f"target={target:,.2f} status={status} achieved={achieved:,.2f} cost={cost:,.2f}"
        )

    curve = pd.DataFrame(rows).sort_values("target_abatement").reset_index(drop=True)
    curve["marginal_cost_diff"] = curve["total_cost"].diff() / curve["target_abatement"].diff().replace(0, np.nan)
    curve["avg_cost_from_origin"] = curve["total_cost"] / curve["achieved_abatement"].replace(0, np.nan)
    return curve


def build_gap_tables(curves: pd.DataFrame, reference_policy: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if reference_policy not in set(curves["policy"].unique()):
        raise ValueError(f"reference_policy={reference_policy} is not in computed policies.")

    ref = curves[curves["policy"] == reference_policy][
        ["target_id", "target_abatement", "total_cost", "status"]
    ].rename(
        columns={
            "total_cost": "total_cost_ref",
            "status": "status_ref",
        }
    )

    joined = curves.merge(ref, on=["target_id", "target_abatement"], how="left")
    # Optimal, or time-limited with an accepted near-optimal incumbent (cost reported).
    ok = joined["status"].isin(["Optimal", "TimeLimit"]) & joined["total_cost"].notna()
    ok_ref = joined["status_ref"].isin(["Optimal", "TimeLimit"]) & joined["total_cost_ref"].notna()
    joined["both_optimal"] = ok & ok_ref

    joined["gap_abs_vs_ref"] = np.where(
        joined["both_optimal"],
        joined["total_cost"] - joined["total_cost_ref"],
        np.nan,
    )
    denom = joined["total_cost_ref"].abs()
    joined["gap_pct_vs_ref"] = np.where(
        joined["both_optimal"] & (denom > 1e-12),
        joined["gap_abs_vs_ref"] / denom,
        np.nan,
    )

    detail_cols = [
        "policy",
        "target_id",
        "target_abatement",
        "status",
        "status_ref",
        "both_optimal",
        "total_cost",
        "total_cost_ref",
        "gap_abs_vs_ref",
        "gap_pct_vs_ref",
    ]
    detail = joined[detail_cols].sort_values(["policy", "target_abatement"]).reset_index(drop=True)

    rows = []
    for policy, g in detail.groupby("policy"):
        valid = g[g["both_optimal"]].copy()
        gap_pct = valid["gap_pct_vs_ref"].dropna()
        gap_abs = valid["gap_abs_vs_ref"].dropna()
        rows.append(
            {
                "policy": policy,
                "reference_policy": reference_policy,
                "n_targets_total": int(len(g)),
                "n_targets_both_optimal": int(len(valid)),
                "median_gap_pct_vs_ref": float(np.nanmedian(gap_pct)) if len(gap_pct) else np.nan,
                "p90_gap_pct_vs_ref": float(np.nanpercentile(gap_pct, 90)) if len(gap_pct) else np.nan,
                "max_gap_pct_vs_ref": float(np.nanmax(gap_pct)) if len(gap_pct) else np.nan,
                "median_gap_abs_vs_ref": float(np.nanmedian(gap_abs)) if len(gap_abs) else np.nan,
                "p90_gap_abs_vs_ref": float(np.nanpercentile(gap_abs, 90)) if len(gap_abs) else np.nan,
                "max_gap_abs_vs_ref": float(np.nanmax(gap_abs)) if len(gap_abs) else np.nan,
            }
        )

    summary = pd.DataFrame(rows).sort_values("policy").reset_index(drop=True)
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Falsification test for preset technology ordering: compare sequence-constrained TAC curves "
            "against unconstrained joint optimum and report optimality-gap distribution (median/P90/max)."
        )
    )
    parser.add_argument("--input", default=None, help="Workbook path. Default: data/1885_coal_power_plants_original.xlsx")
    parser.add_argument("--distance-csv", default=None, help="Biomass-to-plant distance CSV. Default: data/Dji_200.csv")
    parser.add_argument("--co2-distance-csv", default=None, help="Plant-to-sink distance CSV. Default: data/Dki_4000.csv")

    parser.add_argument("--n-targets", type=int, default=base.N_TARGETS, help="Number of target abatement points.")
    parser.add_argument("--extra-targets", type=float, nargs="*", default=base.EXTRA_TARGETS, help="Extra specific targets.")
    parser.add_argument("--min-target", type=float, default=0.0, help="Minimum target abatement.")
    parser.add_argument("--max-target", type=float, default=None, help="Maximum target abatement.")
    parser.add_argument("--fixed-max-abatement", type=float, default=base.FIXED_MAX_ABATEMENT, help="Use exogenous feasible-max abatement.")
    parser.add_argument(
        "--targets-from-curve",
        default=None,
        help="Optional path to a curve CSV whose target_abatement column is used as the exact target grid.",
    )

    parser.add_argument("--blend-cap-ratio", type=float, default=base.BLEND_CAP_RATIO)
    parser.add_argument("--transport-cost-per-gj-km", type=float, default=base.TRANSPORT_COST_PER_GJ_KM)
    parser.add_argument("--max-links-per-plant", type=int, default=base.MAX_LINKS_PER_PLANT)
    parser.add_argument("--biomass-feedstock-cost-per-gj", type=float, default=base.BIOMASS_FEEDSTOCK_COST_PER_GJ)
    parser.add_argument("--bc-cost-mode", choices=["sheet_linear", "tiered_static"], default=base.BC_COST_MODE)
    parser.add_argument("--tier-split-ratio", type=float, default=base.TIER_SPLIT_RATIO)
    parser.add_argument("--co2-transport-cost-per-ton-km", type=float, default=base.CO2_TRANSPORT_COST_PER_TON_KM)
    parser.add_argument("--co2-storage-cost-per-ton", type=float, default=base.CO2_STORAGE_COST_PER_TON)
    parser.add_argument("--max-sink-links-per-plant", type=int, default=base.MAX_SINK_LINKS_PER_PLANT)

    parser.add_argument("--time-limit-sec", type=float, default=base.SOLVER_TIME_LIMIT_SEC)
    parser.add_argument("--mip-gap", type=float, default=base.SOLVER_MIP_GAP)
    parser.add_argument("--solver-threads", type=int, default=base.SOLVER_THREADS)
    parser.add_argument("--target-match-mode", choices=["equality", "min_overshoot", "cost_only"], default=base.TARGET_MATCH_MODE)
    parser.add_argument("--target-mode-switch-abatement", type=float, default=None)
    parser.add_argument("--target-start-index", type=int, default=1, help="1-based start index for slicing the target grid.")
    parser.add_argument("--target-end-index", type=int, default=None, help="1-based inclusive end index for slicing the target grid.")

    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES, help=_policy_help())
    parser.add_argument("--reference-policy", default=DEFAULT_REFERENCE_POLICY, help="Policy used as TAC optimum benchmark.")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help="Output prefix under data/.")
    parser.add_argument("--no-gurobi", action="store_true", help="Use CBC instead of Gurobi (not recommended for this script).")
    parser.add_argument("--ec-package", default=base.EC_PACKAGE, help="EC package tag (same as the fleet model).")
    parser.add_argument("--ec-package-file", default=None, help="EC package CSV. Default: data/ec_packages_r2.csv")
    args = parser.parse_args()

    _validate_policies(args.policies)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, DATA_DIR_NAME)

    input_path = args.input or os.path.join(data_dir, base.INPUT_WORKBOOK_NAME)
    distance_path = args.distance_csv or os.path.join(data_dir, base.INPUT_BIOMASS_DISTANCE_NAME)
    co2_distance_path = args.co2_distance_csv or os.path.join(data_dir, base.INPUT_CO2_DISTANCE_NAME)
    use_gurobi = not args.no_gurobi

    plants, ec, bc, cc, biomass, biomass_dist, sinks, co2_dist = base.load_inputs(
        input_path,
        distance_path,
        co2_distance_path,
    )
    if args.ec_package:
        ec = base.apply_ec_package(ec, args.ec_package_file or os.path.join(data_dir, base.EC_PACKAGE_FILE_NAME), args.ec_package)
    plants, ec, cc, biomass = base.apply_input_scenario(
        workbook_path=input_path, plants=plants, ec=ec, cc=cc, biomass=biomass, coal_factor=1.0, discount_case="base",
        cc_capex_factor=1.0, biomass_supply_factor=1.0, cc_capture_rate=base.BASE_CC_CAPTURE_RATE,
        scenario_year=None, retirement_age=30.0,
    )
    model_data = base.build_model_data(
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
    model_data = base.apply_model_scenario(
        model_data, cc_capture_rate=base.BASE_CC_CAPTURE_RATE, bc_subsidy_rmb_per_kwh=0.0, cc_subsidy_rmb_per_ton=0.0,
        disable_ec=False, disable_ec_interactions=False, disable_bc_cc_interaction=False,
        storage_injection_years=base.STORAGE_INJECTION_YEARS,
    )

    if args.targets_from_curve:
        target_curve_path = args.targets_from_curve
        if not os.path.isabs(target_curve_path):
            target_curve_path = os.path.join(data_dir, target_curve_path)
        if not os.path.exists(target_curve_path):
            raise FileNotFoundError(f"Target-curve CSV not found: {target_curve_path}")

        target_curve_df = pd.read_csv(target_curve_path)
        if "target_abatement" not in target_curve_df.columns:
            raise ValueError(f"target_abatement column is missing in {target_curve_path}")
        targets = pd.to_numeric(target_curve_df["target_abatement"], errors="coerce").dropna().astype(float).to_numpy()
        targets = np.unique(np.round(targets, 6))
        targets = np.sort(targets)
        if len(targets) == 0:
            raise ValueError(f"No valid target_abatement values found in {target_curve_path}")
        print(f"Using target grid from curve: {target_curve_path} (n={len(targets)})")
    else:
        if args.fixed_max_abatement is not None:
            if args.fixed_max_abatement <= 0:
                raise ValueError("--fixed-max-abatement must be positive.")
            max_abatement = float(args.fixed_max_abatement)
            print(f"Using fixed max abatement: {max_abatement:,.2f}")
        else:
            max_abatement = base.compute_feasible_max_abatement(
                model_data,
                use_gurobi,
                relax_integrality=False,
                time_limit_sec=args.time_limit_sec,
                mip_gap=args.mip_gap,
                threads=args.solver_threads,
            )
            print(f"Computed feasible max abatement: {max_abatement:,.2f}")

        targets = base.build_targets(
            n_targets=args.n_targets,
            max_abatement=max_abatement,
            min_target=args.min_target,
            max_target=args.max_target,
            extra_targets=args.extra_targets,
        )

    if args.target_start_index < 1:
        raise ValueError("--target-start-index must be >= 1.")
    start_idx = args.target_start_index - 1
    end_idx = len(targets) if args.target_end_index is None else min(int(args.target_end_index), len(targets))
    if end_idx < args.target_start_index:
        raise ValueError("--target-end-index must be >= --target-start-index.")
    targets = targets[start_idx:end_idx]
    if len(targets) == 0:
        raise ValueError("Selected target slice is empty.")
    print(f"Using target slice: start={args.target_start_index}, end={end_idx}, n={len(targets)}")

    curves_all = []
    for policy in args.policies:
        print("\n" + "=" * 90)
        # per-policy checkpoint: a completed policy is reloaded instead of re-solved after an interruption
        ckpt = os.path.join(data_dir, f"{args.output_prefix}_checkpoint_{policy}.csv")
        if os.path.exists(ckpt):
            done = pd.read_csv(ckpt)
            if len(done) == len(targets) and np.allclose(np.sort(done["target_abatement"].to_numpy(float)), targets):
                print(f"Policy {policy}: loaded completed checkpoint {ckpt}")
                curves_all.append(done)
                continue
        print(f"Running policy: {policy}")
        curve = run_sweep_for_policy(
            model_data=model_data,
            targets=targets,
            use_gurobi=use_gurobi,
            target_match_mode=args.target_match_mode,
            target_mode_switch_abatement=args.target_mode_switch_abatement,
            relax_integrality=False,
            bc_cost_mode=args.bc_cost_mode,
            tier_split_ratio=args.tier_split_ratio,
            time_limit_sec=args.time_limit_sec,
            mip_gap=args.mip_gap,
            threads=args.solver_threads,
            policy=policy,
        )
        curve.to_csv(ckpt, index=False)
        curves_all.append(curve)

    curves_df = pd.concat(curves_all, ignore_index=True)
    gap_detail, gap_summary = build_gap_tables(curves_df, reference_policy=args.reference_policy)

    out_curve = os.path.join(data_dir, f"{args.output_prefix}_all_policies_curve.csv")
    out_detail = os.path.join(data_dir, f"{args.output_prefix}_optimality_gap_detail.csv")
    out_summary = os.path.join(data_dir, f"{args.output_prefix}_optimality_gap_summary.csv")
    out_xlsx = os.path.join(data_dir, f"{args.output_prefix}_comparison.xlsx")

    curves_df.to_csv(out_curve, index=False)
    gap_detail.to_csv(out_detail, index=False)
    gap_summary.to_csv(out_summary, index=False)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        curves_df.to_excel(writer, sheet_name="all_policies_curve", index=False)
        gap_detail.to_excel(writer, sheet_name="optimality_gap_detail", index=False)
        gap_summary.to_excel(writer, sheet_name="optimality_gap_summary", index=False)

    print("\nDone.")
    print(f"Input workbook: {input_path}")
    print(f"Input distance: {distance_path}")
    print(f"Input CO2 distance: {co2_distance_path}")
    print(f"Policies: {args.policies}")
    print(f"Reference policy: {args.reference_policy}")
    print(f"Output curve: {out_curve}")
    print(f"Output gap detail: {out_detail}")
    print(f"Output gap summary: {out_summary}")
    print(f"Output workbook: {out_xlsx}")


if __name__ == "__main__":
    main()
