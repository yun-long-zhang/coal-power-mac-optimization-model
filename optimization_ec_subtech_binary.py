import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pulp import GUROBI_CMD, LpBinary, LpMaximize, LpMinimize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum, value


# ---- User config (edit before running) ----
DATA_DIR_NAME = "data"
INPUT_WORKBOOK_NAME = "1885_coal_power_plants_original.xlsx"
INPUT_BIOMASS_DISTANCE_NAME = "Dji_200.csv"
INPUT_CO2_DISTANCE_NAME = "Dki_4000.csv"
OUTPUT_CURVE_SUFFIX = "_curve.csv"
OUTPUT_RESULTS_SUFFIX = "_results.xlsx"
OUTPUT_LP_CURVE_SUFFIX = "_lp_curve.csv"
OUTPUT_SUMMARY_SUFFIX = "_summary.csv"
TARGET_RATIOS = [x / 1000.0 for x in range(150, 201, 25)]  # 0.05, 0.075, ..., 0.20
OUTPUT_PREFIX = "ec_noovershoot"
SOLVER_TIME_LIMIT_SEC = 300.0
SOLVER_MIP_GAP = 0.001
SOLVER_THREADS = 0  # 0 means let solver decide
OVERLAP_COEF = 0.0  # baseline: 0 means additive EC effects
OVERLAP_COEFS = [x / 100.0 for x in range(0, 76, 25)]  # 0%, 5%, ..., 40%
ABATEMENT_CONSTRAINT_MODE = "per_plant_tight"  # fleet_tight (default), per_plant, or per_plant_tight
OVER_TARGET_TOL_RATIO = 0.0001  # allow at most 1% overshoot around target by default
MIN_OVERSHOOT_PENALTY = 1000.0  # CNY per ton overshoot; encourages solutions close to target
OVERSHOOT_RELAX_PENALTY = 1e7  # large penalty for violating overshoot cap when unavoidable
EC_UNIT_COLS_SPEC = ""  # optional fixed unit-cost columns, e.g. "O:AH"; empty: detect tech_1 ... tech_20 by name
EC_ABATE_COLS_SPEC = ""  # optional fixed abatement columns, e.g. "AK:BD"; empty: detect tech_1.1 ... tech_20.1 by name
EC_UNIT_REV_COL_SPEC = ""  # optional single column, e.g. "AV" or "48"

# Process-based classification for 20 EC technologies.
# Classification basis comes from the provided technology names and process links.
TECH_NAME_BY_NUM = {
    1: "Integrated performance improvement of steam turbine",
    2: "Oxygen-enriched ignition stable combustion and oil-saving",
    3: "Efficient centrifugal spray device in cooling tower",
    4: "Contacting sealing of rotary air preheater",
    5: "High-parameter large-capacity brown coal powder boiler",
    6: "Steam-heating startup from neighboring unit",
    7: "Energy-saving seal of rotary air preheater",
    8: "Boiler combustion temperature monitor and optimization",
    9: "Condenser vacuum maintenance and energy-saving",
    10: "Spiral-strip condenser anti-scaling device",
    11: "Boiler coking early warning and intelligent optimization",
    12: "FGD flue-gas waste-heat recovery and fan optimization",
    13: "Steam turbine seal system retrofit",
    14: "Energy-saving and efficiency control of ESP",
    15: "High-efficiency combined evaporative condenser",
    16: "Flue-gas integrated optimization and advanced heat recovery",
    17: "ESP energy-saving with quasi-stable DC power",
    18: "Modernized steam turbine flow-passage retrofit",
    19: "Coal mill energy-saving retrofit",
    20: "Liquid-coupled feed pump VFD retrofit for large unit",
}

TECH_CATEGORY_BY_NUM = {
    2: "boiler_combustion_milling",
    5: "boiler_combustion_milling",
    6: "boiler_combustion_milling",
    8: "boiler_combustion_milling",
    11: "boiler_combustion_milling",
    19: "boiler_combustion_milling",
    1: "steam_turbine_body_flow",
    13: "steam_turbine_body_flow",
    18: "steam_turbine_body_flow",
    3: "cooling_condensing_system",
    9: "cooling_condensing_system",
    10: "cooling_condensing_system",
    15: "cooling_condensing_system",
    4: "air_preheater_sealing",
    7: "air_preheater_sealing",
    12: "flue_gas_heat_recovery",
    16: "flue_gas_heat_recovery",
    14: "electrical_aux_and_control",
    17: "electrical_aux_and_control",
    20: "electrical_aux_and_control",
}

DEFAULT_TECH_CATEGORY = "uncategorized"
# ------------------------------------------


def _canon(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _find_col(columns: List[str], candidates: List[str]) -> str:
    canon_to_raw = {_canon(c): c for c in columns}
    for cand in candidates:
        key = _canon(cand)
        if key in canon_to_raw:
            return canon_to_raw[key]
    raise KeyError(f"Could not find any of columns {candidates} in {columns}")


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


def _excel_col_to_idx(col: str) -> int:
    # 1-based Excel column to 0-based Python index (A->0, H->7, AA->26).
    col = col.strip().upper()
    val = 0
    for ch in col:
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Invalid Excel column: {col}")
        val = val * 26 + (ord(ch) - ord("A") + 1)
    return val - 1


def _parse_col_token(token: str) -> int:
    s = str(token).strip()
    if not s:
        raise ValueError("Empty column token.")
    if s.isdigit():
        idx_1based = int(s)
        if idx_1based <= 0:
            raise ValueError(f"Column index must be >=1, got: {s}")
        return idx_1based - 1
    return _excel_col_to_idx(s)


def _parse_col_range_spec(spec: str, expected_count: int = 20) -> List[int]:
    s = str(spec).strip()
    if not s:
        return []
    if ":" in s:
        left, right = s.split(":", 1)
        start = _parse_col_token(left)
        end = _parse_col_token(right)
        if end < start:
            raise ValueError(f"Invalid column range '{spec}': end before start.")
        idxs = list(range(start, end + 1))
    else:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        idxs = [_parse_col_token(p) for p in parts]

    if expected_count > 0 and len(idxs) != expected_count:
        raise ValueError(
            f"Column spec '{spec}' resolved to {len(idxs)} columns, expected {expected_count}."
        )
    return idxs


def _pick_cols_by_index(columns: List[str], idxs: List[int], label: str) -> List[str]:
    out = []
    n = len(columns)
    for idx in idxs:
        if idx < 0 or idx >= n:
            raise ValueError(
                f"{label} column index {idx + 1} out of range. Sheet has {n} columns."
            )
        out.append(columns[idx])
    return out


def _solve_model(model: LpProblem, use_gurobi: bool, time_limit_sec: float, mip_gap: float, threads: int) -> str:
    if use_gurobi:
        options = [
            ("TimeLimit", f"{float(time_limit_sec)}"),
            ("MIPGap", f"{float(mip_gap)}"),
        ]
        if int(threads) > 0:
            options.append(("Threads", f"{int(threads)}"))
        status_code = model.solve(GUROBI_CMD(options=options))
    else:
        status_code = model.solve(PULP_CBC_CMD(msg=False))
    return LpStatus[status_code]


def _parse_tech_num_from_header(name: str) -> int:
    s = str(name).strip().lower()
    if not s.startswith("tech_"):
        return -1
    tail = s[len("tech_"):]
    if tail.endswith(".1"):
        tail = tail[:-2]
    try:
        return int(tail)
    except Exception:
        return -1


def _parse_tech_num_from_id(tech_id: str) -> int:
    s = str(tech_id).strip().upper()
    if s.startswith("TECH_"):
        s = s[5:]
    elif s.startswith("EC_"):
        s = s[3:]
    try:
        return int(s)
    except Exception:
        return -1


def _tech_category(tech_id: str) -> str:
    return TECH_CATEGORY_BY_NUM.get(_parse_tech_num_from_id(tech_id), DEFAULT_TECH_CATEGORY)


def _category_counts(tech_ids: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in tech_ids:
        c = _tech_category(t)
        counts[c] = counts.get(c, 0) + 1
    return counts


def _pair_overlap_coef(tech_id_1: str, tech_id_2: str, overlap_coef: float, category_counts: Dict[str, int]) -> float:
    # Same-category pairs share delta / (n_c - 1), where n_c is the number of candidate
    # sub-technologies of that category at the plant. Each added measure k then loses at most
    # delta * a_k, so its net contribution stays >= (1 - delta) * a_k and abatement is
    # monotone in the selected set.
    c1 = _tech_category(tech_id_1)
    c2 = _tech_category(tech_id_2)
    if c1 != c2:
        return 0.0
    n_c = int(category_counts.get(c1, 0))
    if n_c < 2:
        return 0.0
    return float(overlap_coef) / float(n_c - 1)


def load_ec_subtech_inputs(
    workbook_path: str,
    unit_cols_spec: str = "",
    abate_cols_spec: str = "",
    unit_rev_col_spec: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    plants_raw = pd.read_excel(workbook_path, sheet_name="coal_plant_i")
    ec_raw = pd.read_excel(workbook_path, sheet_name="EC")

    # Baseline emissions from coal plant sheet.
    pcols = list(plants_raw.columns)
    pid = _find_col(pcols, ["plant_i", "plantid", "id"])
    p_emission = _find_col(pcols, ["carbon_emissions_ton", "carbonemissionston", "co2", "carbon_emissions"])
    plants = pd.DataFrame(
        {
            "plant_id": plants_raw[pid].map(_norm_id),
            "base_emission": pd.to_numeric(plants_raw[p_emission], errors="coerce").fillna(0.0),
        }
    )
    plants = plants.groupby("plant_id", as_index=False).sum()

    ecols = list(ec_raw.columns)
    ec_pid_col = _find_col(ecols, ["plant_i", "plantid", "id"])
    if str(unit_rev_col_spec).strip():
        unit_rev_idx = _parse_col_range_spec(unit_rev_col_spec, expected_count=1)[0]
        ec_unit_rev_col = _pick_cols_by_index(ecols, [unit_rev_idx], "EC unit revenue")[0]
    else:
        ec_unit_rev_col = _find_col(ecols, ["unit _revenue_from_coal_saving", "unit_revenue_from_coal_saving"])

    # Detect the 20 EC sub-tech columns from names:
    # - unit cost: tech_1 ... tech_20
    # - annual abatement: tech_1.1 ... tech_20.1
    # Keep workbook order for consistent tech indexing.
    def _is_unit_col(name: str) -> bool:
        s = str(name).strip().lower()
        return s.startswith("tech_") and not s.endswith(".1")

    def _is_abate_col(name: str) -> bool:
        s = str(name).strip().lower()
        return s.startswith("tech_") and s.endswith(".1")

    unit_cols = [c for c in ecols if _is_unit_col(c)]
    abate_cols = [c for c in ecols if _is_abate_col(c)]

    if str(unit_cols_spec).strip() or str(abate_cols_spec).strip():
        if not (str(unit_cols_spec).strip() and str(abate_cols_spec).strip()):
            raise ValueError("When using fixed EC column mapping, both --ec-unit-cols and --ec-abate-cols are required.")
        unit_idxs = _parse_col_range_spec(unit_cols_spec, expected_count=20)
        abate_idxs = _parse_col_range_spec(abate_cols_spec, expected_count=20)
        unit_cols = _pick_cols_by_index(ecols, unit_idxs, "EC unit-cost")
        abate_cols = _pick_cols_by_index(ecols, abate_idxs, "EC abatement")

    if len(unit_cols) != 20 or len(abate_cols) != 20:
        raise ValueError(
            "Cannot locate 20 EC sub-tech columns. "
            f"Detected unit={len(unit_cols)}, abatement={len(abate_cols)}."
        )

    unit_by_num = {}
    for c in unit_cols:
        n = _parse_tech_num_from_header(c)
        if n > 0:
            unit_by_num[n] = c

    abate_by_num = {}
    for c in abate_cols:
        n = _parse_tech_num_from_header(c)
        if n > 0:
            abate_by_num[n] = c

    tech_nums = sorted(set(unit_by_num.keys()) & set(abate_by_num.keys()))
    if len(tech_nums) != 20:
        raise ValueError(
            "Cannot form 20 matched tech pairs by number. "
            f"Matched={len(tech_nums)}, unit_keys={sorted(unit_by_num.keys())}, "
            f"abate_keys={sorted(abate_by_num.keys())}."
        )

    rows = []
    for n in tech_nums:
        tname = f"TECH_{n:02d}"
        ucol = unit_by_num[n]
        acol = abate_by_num[n]
        sub = pd.DataFrame(
            {
                "plant_id": ec_raw[ec_pid_col].map(_norm_id),
                "tech_id": tname,
                "tech_num": n,
                "tech_name": TECH_NAME_BY_NUM.get(n, f"Tech {n}"),
                "tech_category": TECH_CATEGORY_BY_NUM.get(n, DEFAULT_TECH_CATEGORY),
                "unit_cost": pd.to_numeric(ec_raw[ucol], errors="coerce").fillna(0.0),
                "abatement_ton": pd.to_numeric(ec_raw[acol], errors="coerce").fillna(0.0),
                "unit_revenue_from_coal_saving": pd.to_numeric(ec_raw[ec_unit_rev_col], errors="coerce").fillna(0.0),
            }
        )
        # Retrofit cost of sub-tech = unit abatement cost * annual abatement.
        sub["retrofit_cost"] = sub["unit_cost"] * sub["abatement_ton"]
        rows.append(sub)

    ec_subtech = pd.concat(rows, ignore_index=True)
    ec_subtech = ec_subtech.groupby(["plant_id", "tech_id"], as_index=False).first()

    # Keep plants that are in baseline table.
    ec_subtech = ec_subtech[ec_subtech["plant_id"].isin(set(plants["plant_id"]))].copy()

    return plants, ec_subtech


def build_problem(
    ec_subtech: pd.DataFrame,
    plants: pd.DataFrame,
    ratio: float,
    overlap_coef: float,
    constraint_mode: str,
    over_target_tol_ratio: float,
    min_overshoot_penalty: float,
    overshoot_relax_penalty: float,
    plant_ratio: Dict[str, float] = None,
) -> Tuple[LpProblem, Dict[Tuple[str, str], LpVariable], object, object]:
    model = LpProblem("EC_Subtech_Binary_MinCost", LpMinimize)

    plant_ids = sorted(plants["plant_id"].unique().tolist())
    plant_emission = plants.set_index("plant_id")["base_emission"].to_dict()
    tech_map: Dict[str, List[Tuple[str, float, float, float]]] = {}
    all_overlap_penalty_terms = []

    for _, r in ec_subtech.iterrows():
        p = str(r["plant_id"])
        t = str(r["tech_id"])
        u = float(r["unit_cost"])
        a = float(r["abatement_ton"])
        c = float(r["retrofit_cost"])
        # Keep only meaningful candidate technologies.
        if a <= 0:
            continue
        tech_map.setdefault(p, []).append((t, u, a, c))

    x: Dict[Tuple[str, str], LpVariable] = {}
    abate_expr_terms = []
    cost_expr_terms = []
    per_plant_over_terms = []
    per_plant_relax_terms = []

    for p in plant_ids:
        candidates = tech_map.get(p, [])
        if not candidates:
            continue

        plant_vars = []
        for t, _u, a, c in candidates:
            var = LpVariable(f"x__{p}__{t}", cat=LpBinary)
            x[(p, t)] = var
            plant_vars.append(var)
            abate_expr_terms.append(a * var)
            cost_expr_terms.append(c * var)

        # Overlap penalty for jointly selected EC technologies at the same plant.
        # Penalty_{t,u} = overlap_coef / (n_c - 1) * min(a_t, a_u) * y_{t,u}, where y_{t,u}=x_t*x_u.
        # Cross-category pairs are treated as no-overlap (coefficient 0).
        overlap_penalty_terms = []
        cat_counts = _category_counts([t for t, _u, _a, _c in candidates])
        if float(overlap_coef) > 0:
            for i in range(len(candidates)):
                t1, _u1, a1, _c1 = candidates[i]
                for j in range(i + 1, len(candidates)):
                    t2, _u2, a2, _c2 = candidates[j]
                    pair_coef = _pair_overlap_coef(t1, t2, overlap_coef, cat_counts)
                    if pair_coef <= 0:
                        continue
                    penalty_base = min(float(a1), float(a2))
                    if penalty_base <= 0:
                        continue
                    y = LpVariable(f"yov__{p}__{t1}__{t2}", cat=LpBinary)
                    model += y <= x[(p, t1)]
                    model += y <= x[(p, t2)]
                    model += y >= x[(p, t1)] + x[(p, t2)] - 1
                    penalty_term = float(pair_coef) * penalty_base * y
                    overlap_penalty_terms.append(penalty_term)
                    all_overlap_penalty_terms.append(penalty_term)

        if constraint_mode in {"per_plant", "per_plant_tight"}:
            # Per-plant abatement requirement.
            p_ratio = float(plant_ratio.get(p, ratio)) if plant_ratio is not None else float(ratio)
            required_abatement = float(plant_emission.get(p, 0.0)) * p_ratio
            if overlap_penalty_terms:
                plant_abatement_expr = lpSum(a * x[(p, t)] for t, _u, a, _c in candidates) - lpSum(overlap_penalty_terms)
            else:
                plant_abatement_expr = lpSum(a * x[(p, t)] for t, _u, a, _c in candidates)

            model += plant_abatement_expr >= required_abatement

            if constraint_mode == "per_plant_tight":
                # Bound per-plant overshoot and penalize both overshoot and cap violations.
                over_p = LpVariable(f"over_abatement_ton__{p}", lowBound=0)
                relax_p = LpVariable(f"overshoot_relax_ton__{p}", lowBound=0)
                model += over_p >= plant_abatement_expr - required_abatement
                model += plant_abatement_expr <= required_abatement * (1.0 + float(over_target_tol_ratio)) + relax_p
                per_plant_over_terms.append(over_p)
                per_plant_relax_terms.append(relax_p)

    if all_overlap_penalty_terms:
        total_abatement_expr = lpSum(abate_expr_terms) - lpSum(all_overlap_penalty_terms)
    else:
        total_abatement_expr = lpSum(abate_expr_terms)
    total_cost_expr = lpSum(cost_expr_terms)

    total_baseline = float(plants["base_emission"].sum())
    target_abatement = float(ratio) * total_baseline

    if constraint_mode == "fleet_tight":
        # Fleet-level target with bounded overshoot and min-overshoot style penalty.
        model += total_abatement_expr >= target_abatement

        over_abatement = LpVariable("over_abatement_ton", lowBound=0)
        model += over_abatement >= total_abatement_expr - target_abatement

        overshoot_relax = LpVariable("overshoot_relax_ton", lowBound=0)
        model += total_abatement_expr <= target_abatement * (1.0 + float(over_target_tol_ratio)) + overshoot_relax

        model += (
            total_cost_expr
            + float(min_overshoot_penalty) * over_abatement
            + float(overshoot_relax_penalty) * overshoot_relax
        )
    elif constraint_mode == "per_plant_tight":
        model += (
            total_cost_expr
            + float(min_overshoot_penalty) * lpSum(per_plant_over_terms)
            + float(overshoot_relax_penalty) * lpSum(per_plant_relax_terms)
        )
    else:
        model += total_cost_expr

    return model, x, total_abatement_expr, total_cost_expr


def compute_max_abatement_under_current_rules(ec_subtech: pd.DataFrame) -> float:
    """
    Maximum total abatement under current modeling rule:
    at most one EC sub-tech can be selected per plant.
    """
    if ec_subtech.empty:
        return 0.0
    per_plant_max = ec_subtech.groupby("plant_id", as_index=False)["abatement_ton"].max()
    return float(pd.to_numeric(per_plant_max["abatement_ton"], errors="coerce").fillna(0.0).sum())


def compute_per_plant_capability(ec_subtech: pd.DataFrame) -> pd.DataFrame:
    """
    Per-plant maximum abatement under multi-selection rule (sum over 20 techs).
    """
    if ec_subtech.empty:
        return pd.DataFrame(columns=["plant_id", "max_abatement_ton"])
    out = ec_subtech.groupby("plant_id", as_index=False)["abatement_ton"].sum()
    return out.rename(columns={"abatement_ton": "max_abatement_ton"})


def solve_scenario(
    ratio: float,
    plants: pd.DataFrame,
    ec_subtech: pd.DataFrame,
    overlap_coef: float,
    use_gurobi: bool,
    time_limit_sec: float,
    mip_gap: float,
    threads: int,
    constraint_mode: str,
    over_target_tol_ratio: float,
    min_overshoot_penalty: float,
    overshoot_relax_penalty: float,
    plant_ratio_cap: Dict[str, float] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    total_baseline = float(plants["base_emission"].sum())

    plant_required = plants[["plant_id", "base_emission"]].copy()
    per_plant_max = compute_per_plant_capability(ec_subtech)
    plant_required = plant_required.merge(per_plant_max, on="plant_id", how="left")
    plant_required["max_abatement_ton"] = pd.to_numeric(plant_required["max_abatement_ton"], errors="coerce").fillna(0.0)

    # Plant-specific target ratio. With a cap file, each plant targets
    # min(ratio, physical cap, technical max of the remaining EC measures).
    plant_required["target_ratio"] = float(ratio)
    if plant_ratio_cap is not None:
        cap = plant_required["plant_id"].map(plant_ratio_cap).astype(float).fillna(float(ratio))
        tech_max = np.where(
            plant_required["base_emission"] > 0,
            plant_required["max_abatement_ton"] / plant_required["base_emission"].replace(0, np.nan),
            0.0,
        )
        tech_max = np.nan_to_num(tech_max, nan=0.0) * (1.0 - 1e-9)
        plant_required["target_ratio"] = np.minimum(np.minimum(cap, float(ratio)), tech_max)
    plant_required["required_abatement_ton"] = plant_required["base_emission"] * plant_required["target_ratio"]
    plant_ratio = plant_required.set_index("plant_id")["target_ratio"].to_dict()
    target = float(plant_required["required_abatement_ton"].sum())
    plant_required["is_plant_feasible"] = (
        plant_required["required_abatement_ton"] <= plant_required["max_abatement_ton"] + 1e-6
    )

    infeasible_plants = int((~plant_required["is_plant_feasible"]).sum())
    max_feasible_abatement = float(plant_required["max_abatement_ton"].sum())
    max_feasible_ratio = max_feasible_abatement / total_baseline if total_baseline > 0 else np.nan

    tech_ids = sorted(ec_subtech["tech_id"].unique().tolist())
    decision_cols = [f"x_{t}" for t in tech_ids]

    summary = {
        "target_ratio": ratio,
        "target_abatement_ton": target,
        "overlap_coef": float(overlap_coef),
        "overlap_rule": "same_category_only",
        "constraint_mode": str(constraint_mode),
        "over_target_tol_ratio": float(over_target_tol_ratio),
        "status": "Not Solved",
        "achieved_abatement_ton": np.nan,
        "overshoot_ton": np.nan,
        "total_retrofit_cost": np.nan,
        "selected_plant_count": 0,
        "max_feasible_abatement_ton": max_feasible_abatement,
        "max_feasible_ratio": max_feasible_ratio,
        "infeasible_plant_count": infeasible_plants,
    }

    # Fast infeasibility check is valid for per-plant based modes.
    if constraint_mode in {"per_plant", "per_plant_tight"} and infeasible_plants > 0:
        summary["status"] = "Infeasible_PerPlant_Target_Above_EC_Max"
        detail = plant_required[["plant_id", "base_emission", "required_abatement_ton", "max_abatement_ton", "is_plant_feasible"]].copy()
        detail["selected_tech_id"] = ""
        detail["achieved_abatement_ton"] = np.nan
        detail["unit_abatement_cost"] = np.nan
        detail["unit_revenue_from_coal_saving"] = np.nan
        detail["unit_cost_after_coal_saving"] = np.nan
        for col in decision_cols:
            detail[col] = np.nan
        return summary, detail

    model, x, total_abatement_expr, total_cost_expr = build_problem(
        ec_subtech=ec_subtech,
        plants=plants,
        ratio=ratio,
        overlap_coef=overlap_coef,
        constraint_mode=constraint_mode,
        over_target_tol_ratio=over_target_tol_ratio,
        min_overshoot_penalty=min_overshoot_penalty,
        overshoot_relax_penalty=overshoot_relax_penalty,
        plant_ratio=plant_ratio,
    )
    status = _solve_model(model, use_gurobi, time_limit_sec, mip_gap, threads)
    summary["status"] = status

    if status != "Optimal":
        detail = plant_required[["plant_id", "base_emission", "required_abatement_ton", "max_abatement_ton", "is_plant_feasible"]].copy()
        detail["selected_tech_id"] = ""
        detail["achieved_abatement_ton"] = np.nan
        detail["unit_abatement_cost"] = np.nan
        detail["unit_revenue_from_coal_saving"] = np.nan
        detail["unit_cost_after_coal_saving"] = np.nan
        for col in decision_cols:
            detail[col] = np.nan
        return summary, detail

    achieved = float(value(total_abatement_expr))
    cost = float(value(total_cost_expr))

    abate_lookup = ec_subtech.set_index(["plant_id", "tech_id"])["abatement_ton"].to_dict()
    unit_cost_lookup = ec_subtech.set_index(["plant_id", "tech_id"])["unit_cost"].to_dict()
    unit_rev_lookup = ec_subtech.groupby("plant_id", as_index=False)["unit_revenue_from_coal_saving"].first().set_index("plant_id")[
        "unit_revenue_from_coal_saving"
    ].to_dict()
    decision_rows = []
    selected_plant_count = 0
    for _, row in plant_required.iterrows():
        p = str(row["plant_id"])
        d = {
            "plant_id": p,
            "base_emission": float(row["base_emission"]),
            "target_ratio": float(row["target_ratio"]),
            "required_abatement_ton": float(row["required_abatement_ton"]),
            "max_abatement_ton": float(row["max_abatement_ton"]),
            "is_plant_feasible": bool(row["is_plant_feasible"]),
        }
        selected_tech = ""
        achieved_p = 0.0
        overlap_penalty_p = 0.0
        total_retrofit_cost_p = 0.0
        selected_techs = []
        selected_tech_data = []  # (tech_id, gross_abatement_ton, unit_cost)
        for t in tech_ids:
            val = int(value(x[(p, t)]) > 0.5) if (p, t) in x else 0
            d[f"x_{t}"] = val
            if val == 1:
                selected_techs.append(t)
                a = float(abate_lookup.get((p, t), 0.0))
                u = float(unit_cost_lookup.get((p, t), 0.0))
                selected_tech_data.append((t, a, u))
                achieved_p += a
                total_retrofit_cost_p += a * u

        if float(overlap_coef) > 0 and len(selected_techs) >= 2:
            cat_counts = _category_counts([t for t in tech_ids if float(abate_lookup.get((p, t), 0.0)) > 0])
            for i in range(len(selected_techs)):
                for j in range(i + 1, len(selected_techs)):
                    t1 = selected_techs[i]
                    t2 = selected_techs[j]
                    pair_coef = _pair_overlap_coef(t1, t2, overlap_coef, cat_counts)
                    if pair_coef <= 0:
                        continue
                    a1 = float(abate_lookup.get((p, t1), 0.0))
                    a2 = float(abate_lookup.get((p, t2), 0.0))
                    overlap_penalty_p += float(pair_coef) * min(a1, a2)

        achieved_net_p = max(0.0, achieved_p - overlap_penalty_p)
        selected_tech = ",".join(selected_techs)

        if selected_techs:
            selected_plant_count += 1
        unit_abatement_cost_untrimmed_p = (total_retrofit_cost_p / achieved_net_p) if achieved_net_p > 1e-12 else np.nan

        # Trimmed unit-cost basis: remove overshoot cost from highest unit-cost selected tech slices,
        # then divide by required abatement (when overshoot exists).
        required_p = float(row["required_abatement_ton"])
        trimmed_excess_abatement_p = max(0.0, achieved_net_p - required_p)
        trimmed_total_retrofit_cost_p = float(total_retrofit_cost_p)

        if trimmed_excess_abatement_p > 1e-12 and achieved_net_p > 1e-12 and achieved_p > 1e-12 and selected_tech_data:
            net_scale = achieved_net_p / achieved_p
            tech_net_slices = []
            for _tid, gross_abate_ton, unit_cost in selected_tech_data:
                net_abate_ton = max(0.0, float(gross_abate_ton) * net_scale)
                if net_abate_ton > 0:
                    tech_net_slices.append((float(unit_cost), net_abate_ton))

            remaining_trim = trimmed_excess_abatement_p
            removed_cost = 0.0
            for unit_cost, net_abate_ton in sorted(tech_net_slices, key=lambda x: x[0], reverse=True):
                if remaining_trim <= 1e-12:
                    break
                cut = min(net_abate_ton, remaining_trim)
                removed_cost += cut * unit_cost
                remaining_trim -= cut

            trimmed_total_retrofit_cost_p = max(0.0, trimmed_total_retrofit_cost_p - removed_cost)

        if achieved_net_p > required_p + 1e-12 and required_p > 1e-12:
            trimmed_basis_abatement_p = required_p
        else:
            trimmed_basis_abatement_p = achieved_net_p

        unit_abatement_cost_p = (
            trimmed_total_retrofit_cost_p / trimmed_basis_abatement_p
            if trimmed_basis_abatement_p > 1e-12
            else np.nan
        )

        unit_rev_p = float(unit_rev_lookup.get(p, 0.0))
        d["selected_tech_id"] = selected_tech
        d["selected_tech_category"] = ",".join(
            sorted({TECH_CATEGORY_BY_NUM.get(_parse_tech_num_from_id(t), DEFAULT_TECH_CATEGORY) for t in selected_techs})
        )
        d["achieved_abatement_ton_gross"] = achieved_p
        d["overlap_penalty_ton"] = overlap_penalty_p
        d["achieved_abatement_ton"] = achieved_net_p
        d["trimmed_excess_abatement_ton"] = trimmed_excess_abatement_p
        d["trimmed_basis_abatement_ton"] = trimmed_basis_abatement_p
        d["total_retrofit_cost_untrimmed"] = total_retrofit_cost_p
        d["total_retrofit_cost_trimmed"] = trimmed_total_retrofit_cost_p
        d["unit_abatement_cost_untrimmed"] = unit_abatement_cost_untrimmed_p
        d["unit_abatement_cost"] = unit_abatement_cost_p
        d["unit_revenue_from_coal_saving"] = unit_rev_p
        d["unit_cost_after_coal_saving_untrimmed"] = (
            unit_abatement_cost_untrimmed_p - unit_rev_p if pd.notna(unit_abatement_cost_untrimmed_p) else np.nan
        )
        d["unit_cost_after_coal_saving"] = (
            unit_abatement_cost_p - unit_rev_p if pd.notna(unit_abatement_cost_p) else np.nan
        )
        decision_rows.append(d)

    chosen = pd.DataFrame(decision_rows)
    if not chosen.empty:
        # Sort by numeric plant_id, not string (to get 1, 2, 3, ... not 1, 10, 100, ...)
        chosen = chosen.sort_values("plant_id", key=lambda x: pd.to_numeric(x, errors='coerce')).reset_index(drop=True)

    summary["achieved_abatement_ton"] = achieved
    summary["overshoot_ton"] = achieved - target
    summary["total_retrofit_cost"] = cost
    summary["selected_plant_count"] = selected_plant_count
    return summary, chosen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Binary EC sub-technology selection model. "
            "Objective: minimize total retrofit cost. "
            "Constraint mode can be fleet-level tight target or per-plant target matching."
        )
    )
    parser.add_argument("--input", default=None, help="Workbook path. Default: data/1885_coal_power_plants_original.xlsx")
    parser.add_argument("--target-ratios", type=float, nargs="+", default=TARGET_RATIOS, help="Abatement target ratios, e.g. 0.1 0.2")
    parser.add_argument("--time-limit-sec", type=float, default=SOLVER_TIME_LIMIT_SEC, help="MILP solver time limit per scenario (seconds).")
    parser.add_argument("--mip-gap", type=float, default=SOLVER_MIP_GAP, help="MILP relative optimality gap tolerance.")
    parser.add_argument("--solver-threads", type=int, default=SOLVER_THREADS, help="Threads for Gurobi (0 lets solver decide).")
    parser.add_argument(
        "--overlap-coef",
        type=float,
        default=OVERLAP_COEF,
        help="Single overlap coefficient in [0,1]. Ignored when --overlap-coefs is provided.",
    )
    parser.add_argument(
        "--overlap-coefs",
        type=float,
        nargs="+",
        default=OVERLAP_COEFS,
        help="Series of same-category overlap coefficients in [0,1], e.g. 0 0.05 ... 0.4",
    )
    parser.add_argument("--no-gurobi", action="store_true", help="Use CBC instead of Gurobi.")
    parser.add_argument(
        "--constraint-mode",
        choices=["fleet_tight", "per_plant", "per_plant_tight"],
        default=ABATEMENT_CONSTRAINT_MODE,
        help="Abatement constraint mode. per_plant_tight enforces each plant target with tight overshoot control.",
    )
    parser.add_argument(
        "--over-target-tol-ratio",
        type=float,
        default=OVER_TARGET_TOL_RATIO,
        help="Allowed relative overshoot above target in fleet_tight mode (e.g., 0.01 for 1%).",
    )
    parser.add_argument(
        "--min-overshoot-penalty",
        type=float,
        default=MIN_OVERSHOOT_PENALTY,
        help="Penalty (CNY/ton) on overshoot amount in fleet_tight mode.",
    )
    parser.add_argument(
        "--overshoot-relax-penalty",
        type=float,
        default=OVERSHOOT_RELAX_PENALTY,
        help="Large penalty (CNY/ton) for violating overshoot cap in fleet_tight mode.",
    )
    parser.add_argument(
        "--ec-unit-cols",
        default=EC_UNIT_COLS_SPEC,
        help="Optional fixed EC unit-cost columns, e.g. 'H:AA' or '8:27'.",
    )
    parser.add_argument(
        "--ec-abate-cols",
        default=EC_ABATE_COLS_SPEC,
        help="Optional fixed EC abatement columns, e.g. 'AB:AU' or '28:47'.",
    )
    parser.add_argument(
        "--ec-unit-rev-col",
        default=EC_UNIT_REV_COL_SPEC,
        help="Optional fixed EC unit coal-saving revenue column, e.g. 'AV' or '48'.",
    )
    parser.add_argument(
        "--exclude-techs",
        type=int,
        nargs="*",
        default=[],
        help="EC sub-technology numbers to exclude, e.g. 5",
    )
    parser.add_argument(
        "--plant-ratio-cap-csv",
        default=None,
        help="Optional CSV with columns plant_id, ec_ratio_cap giving a plant-specific upper bound on the EC target ratio.",
    )
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX, help="Output prefix in data folder.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, DATA_DIR_NAME)
    output_dir = os.path.join(data_dir, "EC_Portfolio")
    os.makedirs(output_dir, exist_ok=True)
    input_path = args.input or os.path.join(data_dir, INPUT_WORKBOOK_NAME)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input workbook not found: {input_path}")

    ratios = [float(r) for r in args.target_ratios]
    if not ratios:
        raise ValueError("--target-ratios cannot be empty.")
    if any(r <= 0 or r > 1 for r in ratios):
        raise ValueError("Each target ratio must be in (0, 1].")

    overlap_coefs = [float(v) for v in args.overlap_coefs]
    if not overlap_coefs:
        overlap_coefs = [float(args.overlap_coef)]
    overlap_coefs = sorted(set(overlap_coefs))
    if any(v < 0 or v > 1 for v in overlap_coefs):
        raise ValueError("All overlap coefficients must be in [0, 1].")

    plants, ec_subtech = load_ec_subtech_inputs(
        input_path,
        unit_cols_spec=args.ec_unit_cols,
        abate_cols_spec=args.ec_abate_cols,
        unit_rev_col_spec=args.ec_unit_rev_col,
    )
    if args.exclude_techs:
        ec_subtech = ec_subtech[~ec_subtech["tech_num"].isin(set(args.exclude_techs))].copy()

    plant_ratio_cap = None
    if args.plant_ratio_cap_csv:
        cap_df = pd.read_csv(args.plant_ratio_cap_csv)
        plant_ratio_cap = dict(zip(cap_df["plant_id"].map(_norm_id), pd.to_numeric(cap_df["ec_ratio_cap"], errors="coerce")))
    use_gurobi = not args.no_gurobi

    summaries = []
    detail_tables: Dict[str, pd.DataFrame] = {}

    total_cases = len(overlap_coefs) * len(ratios)
    case_idx = 0
    for overlap_coef in overlap_coefs:
        print(f"\n=== Overlap coefficient = {overlap_coef:.0%} ({overlap_coef:.2f}) ===")
        for ratio in ratios:
            case_idx += 1
            summary, chosen = solve_scenario(
                ratio=ratio,
                plants=plants,
                ec_subtech=ec_subtech,
                overlap_coef=overlap_coef,
                use_gurobi=use_gurobi,
                time_limit_sec=args.time_limit_sec,
                mip_gap=args.mip_gap,
                threads=args.solver_threads,
                constraint_mode=args.constraint_mode,
                over_target_tol_ratio=args.over_target_tol_ratio,
                min_overshoot_penalty=args.min_overshoot_penalty,
                overshoot_relax_penalty=args.overshoot_relax_penalty,
                plant_ratio_cap=plant_ratio_cap,
            )
            summaries.append(summary)

            otag = f"o{int(round(overlap_coef * 100)):02d}"
            rtag = f"r{int(round(ratio * 100)):02d}"
            detail_tables[f"{otag}_{rtag}"] = chosen

            print(
                f"[{case_idx}/{total_cases}] overlap={overlap_coef:.0%} ratio={ratio:.2%} status={summary['status']} "
                f"target={summary['target_abatement_ton']:,.2f} "
                f"achieved={summary['achieved_abatement_ton'] if pd.notna(summary['achieved_abatement_ton']) else np.nan:,.2f} "
                f"overshoot={summary['overshoot_ton'] if pd.notna(summary['overshoot_ton']) else np.nan:,.2f} "
                f"cost={summary['total_retrofit_cost'] if pd.notna(summary['total_retrofit_cost']) else np.nan:,.2f} "
                f"infeasible_plants={summary['infeasible_plant_count']}"
            )

    summary_df = pd.DataFrame(summaries).sort_values("target_ratio").reset_index(drop=True)

    out_csv = os.path.join(output_dir, f"{args.output_prefix}{OUTPUT_SUMMARY_SUFFIX}")
    out_xlsx = os.path.join(output_dir, f"{args.output_prefix}{OUTPUT_RESULTS_SUFFIX}")
    summary_df.to_csv(out_csv, index=False)

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        # Sort detail_tables by numeric order: ratio first, then overlap
        def tag_sort_key(tag: str) -> Tuple[int, int]:
            # tag format: "o00_r05" → (5, 0) for sorting by ratio then overlap
            parts = tag.split('_')
            o_val = int(parts[0][1:])  # Remove 'o' prefix and convert to int
            r_val = int(parts[1][1:])  # Remove 'r' prefix and convert to int
            return (r_val, o_val)
        
        sorted_tags = sorted(detail_tables.keys(), key=tag_sort_key)
        for tag in sorted_tags:
            df = detail_tables[tag]
            sheet = f"sel_{tag}"
            if len(sheet) > 31:
                sheet = sheet[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)

    print("\nDone.")
    print(f"Input workbook: {input_path}")
    print(f"Overlap coefficients: {overlap_coefs}")
    if str(args.ec_unit_cols).strip() or str(args.ec_abate_cols).strip() or str(args.ec_unit_rev_col).strip():
        print(
            f"EC fixed columns: unit={args.ec_unit_cols or 'auto'}, "
            f"abatement={args.ec_abate_cols or 'auto'}, unit_revenue={args.ec_unit_rev_col or 'auto'}"
        )
    print("Overlap rule: same-category pairs penalized, cross-category pairs unpenalized")
    print(f"Constraint mode: {args.constraint_mode}")
    if args.constraint_mode in {"fleet_tight", "per_plant_tight"}:
        print(f"Over-target tolerance ratio: {args.over_target_tol_ratio}")
        print(f"Min-overshoot penalty: {args.min_overshoot_penalty}")
        print(f"Overshoot-relax penalty: {args.overshoot_relax_penalty}")
    print(f"Scenario count: {total_cases}")
    print(f"Output summary csv: {out_csv}")
    print(f"Output result xlsx: {out_xlsx}")


if __name__ == "__main__":
    main()
