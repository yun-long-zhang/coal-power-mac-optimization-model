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
OUTPUT_PREFIX = "gurobi"
TARGET_MATCH_MODE = "equality"  # "equality" | "min_overshoot" | "cost_only"
TARGET_MODE_SWITCH_ABATEMENT = None  # legacy: <= threshold uses min_overshoot, > threshold uses cost_only
EQUALITY_REL_TOL = 1e-6  # equality mode: target <= abatement <= target * (1 + tol)
BLEND_CAP_RATIO = 1.0
ROAD_TORTUOSITY = 1.5  # road distance / straight-line distance (Dji are straight-line distances)
TRANSPORT_COST_PER_GJ_KM = 0.0017 * ROAD_TORTUOSITY
BIOMASS_FEEDSTOCK_COST_PER_GJ = 294.3 / 17.3 / 6.8985  # collection, storage, processing: 294.3 RMB/t, LHV 17.3 GJ/t
BIOMASS_LHV_GJ_PER_T = 17.3
MAX_LINKS_PER_PLANT = 500
CO2_STORAGE_COST_PER_TON = 5.0
CO2_TRANSPORT_COST_PER_TON_KM = 0.1
MAX_SINK_LINKS_PER_PLANT = 25  # nearest storage sites per plant; 10 truncates deep-abatement potential once capacity is annualized
STORAGE_INJECTION_YEARS = 20.0  # annual injection capacity = cumulative storage potential / injection period
BC_COST_MODE = "tiered_static"  # "sheet_linear" | "tiered_static"
TIER_SPLIT_RATIO = 0.2
FIXED_MAX_ABATEMENT = None  # e.g. 4.7e9; None means compute internally
EC_PACKAGE_FILE_NAME = "ec_packages_r2.csv"  # built by build_ec_packages.py
EC_PACKAGE = "no5_r10_o00"  # baseline EC package: 10% without technology No. 5; "" keeps the legacy EC sheet columns
RMB_PER_USD = 6.8985  # 2019 average exchange rate (Table S2)
BC_FIXED_OM_SHARE = 0.02  # BC fixed O&M as a share of total investment per year
EF_COAL_T_PER_GJ = 0.0901
EF_BIOMASS_T_PER_GJ = 0.112
BASE_CC_CAPTURE_RATE = 0.90
CC_FIXED_OM_SHARE = 0.04
# Capture energy requirement per tonne CO2 (煤电CCS改造成本计算.xlsx): regeneration heat and compression/auxiliary power.
CC_STEAM_GJ_PER_T = 3.5
CC_POWER_MWH_PER_T = 0.23
CC_STEAM_BOILER_EFF = 0.90  # "boiler" basis: heat supplied by extra coal at boiler efficiency
CC_STEAM_TO_POWER = 0.25  # "extraction" basis: steam extracted from the turbine, lost power = 25% of the heat
# "extraction" basis ("avoided" accounting) uses current-generation solvent values instead of 3.5 GJ/t and 0.23 MWh/t:
#   regeneration heat 2.4 GJ/t: CHN Energy Taizhou 0.5 Mt/a (2023, < 2.4 GJ/t operating); Huaneng Longdong/Zhengning
#     1.5 Mt/a (2025, <= 2.3 GJ/t design); IEAGHG 2019-09 Cansolv 2.2-2.8 GJ/t;
#   capture + compression power 0.10 MWh/t: mean of Taizhou (< 90 kWh/t) and NETL Rev4a B12B.90 (~0.11 MWh/t).
# The workbook cc_ext_* columns hold the 3.5/0.23 values; the energy-dependent terms are rescaled in code.
CC_EXT_STEAM_GJ_PER_T = 2.4
CC_EXT_POWER_MWH_PER_T = 0.10
CC_CAPEX_SCALE_EXPONENT = 0.67  # capture-unit scaling with treated CO2 (as in the CC cost workbook)  # CC fixed O&M as a share of total investment per year (Table S2, c)
# Abatement accounting boundary (R2-1), set with the environment variable ABATEMENT_ACCOUNTING:
#   "captured": first-round baseline; capture heat priced as boiler coal, the CO2 of the capture energy and of
#               the logistics is not counted (CO2 captured = abatement).
#   "avoided":  operational energy of the abatement chain counted on the same boundary as its cost. Assumption:
#               the boiler has spare capacity, so regeneration steam is extracted from the turbine and the plant
#               fires extra coal to keep its net output (extra coal = (lost power + auxiliary power) / efficiency);
#               the flue gas of the extra coal passes the same capture unit (captured at the same rate, unit sized
#               up by F0^0.67). The uncaptured CO2 of biomass haulage/processing and of CO2 transport is deducted.
ABATEMENT_ACCOUNTING = os.environ.get("ABATEMENT_ACCOUNTING", "avoided")  # "avoided" is the revised baseline
if ABATEMENT_ACCOUNTING not in ("captured", "avoided"):
    raise ValueError("ABATEMENT_ACCOUNTING must be 'captured' or 'avoided'.")
# Logistics emission factors used in the "avoided" accounting:
LOGISTICS_EF_AVOIDED = {
    "bio_transport_kg_per_t_km": 0.04,  # truck haulage of biomass, kg CO2 per t-km of road distance (Lu et al. 2019, PNAS 116, 8206)
    "bio_processing_kg_per_gj": 10.0,  # biomass supply chain, mean of 5-15 kg CO2e/GJ for residues (Creutzig et al. 2015, GCB Bioenergy 7, 916)
    "co2_transport_kg_per_t_km": 0.0,  # pipeline pumping, neglected (small against the captured CO2)
}
_AVOIDED = ABATEMENT_ACCOUNTING == "avoided"
DEFAULT_CC_ENERGY_BASIS = "extraction" if _AVOIDED else "boiler"
DEFAULT_CC_PENALTY_EMISSIONS = _AVOIDED
DEFAULT_LOGISTICS_EF = LOGISTICS_EF_AVOIDED if _AVOIDED else {k: 0.0 for k in LOGISTICS_EF_AVOIDED}
DISCOUNT_RATE_BY_CASE = {"base": 0.08, "af5": 0.05, "af12": 0.12}  # BC and CC; the EC packages are annualized at 8% and kept fixed in the discount cases
BASE_YEAR = 2020
PLANT_LIFETIME_YEARS = 40.0
MIN_PAYBACK_YEARS = 5.0
TECH_SERVICE_LIFE_YEARS = 20.0
# Capture-rate scenarios (Supplementary Note 5.2): multipliers on CC capex, fixed O&M,
# capture electricity (ele1) and other variable O&M (opex).
CC_CAPTURE_RATE_COST_MULTIPLIERS = {
    0.90: (1.0, 1.0, 1.0, 1.0),
    0.95: (1.005, 1.004, 1.04, 1.134),
    0.997: (1.014, 1.013, 1.10, 1.199),
}
# Time-slice scenarios: CC cost learning factor by target year.
CC_COST_FACTOR_BY_YEAR = {2030: 0.90, 2035: 0.85, 2040: 0.80, 2045: 0.75}
RUN_LP_DIAGNOSTIC = False  # True to run LP-relax benchmark sweep
SOLVER_TIME_LIMIT_SEC = 7200.0
SOLVER_MIP_GAP = 0.001
SOLVER_MIP_GAP_ABS = 5.0e6  # USD/yr; also stop when the absolute gap is below this (near-zero total cost targets)
TIME_LIMIT_ACCEPT_GAP = 0.005  # at the time limit, keep the incumbent if its MIP gap is at most 0.5%
SOLVER_THREADS = 0  # 0 means let solver decide
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
    p_power = _find_col_optional(
        pcols,
        ["power", "power_xi", "capacity_xi", "capacity", "电厂功率xi", "installed_power"],
    )
    p_af = _find_col_optional(pcols, ["AF", "annuity_factor", "年金因子"])
    p_hours = _find_col_optional(pcols, ["running_hours", "runninghours", "hours", "年利用小时"])
    p_coal = _find_col_optional(pcols, ["coal_price", "燃煤价格", "coalprice"])

    plants = pd.DataFrame(
        {
            "plant_id": plants_raw[pid].map(_norm_id),
            "energy_gj": pd.to_numeric(plants_raw[p_energy], errors="coerce").fillna(0.0),
            "base_emission": pd.to_numeric(plants_raw[p_emission], errors="coerce").fillna(0.0),
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
    cc_af8_col = _find_col(ccols, ["annuity_factor_8%"])
    cc_life_col = _find_col(ccols, ["life_years"])

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
            "cc_payback_years": pd.to_numeric(cc_raw[cc_life_col], errors="coerce"),
        }
    )
    # Overnight capture investment recovered from the precomputed annualized capex (8%, payback years).
    cc["cc_investment"] = cc["cc_capex_per_year"] / pd.to_numeric(cc_raw[cc_af8_col], errors="coerce")
    cc["cc_fixed_om_per_year"] = CC_FIXED_OM_SHARE * cc["cc_investment"]
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
    s_cap = _find_col(scols, ["Estimated_storage_potential_t", "Estimated_storage_potential_Mt", "capacity", "storage"])
    sinks = pd.DataFrame(
        {
            "sink_id": sinks_raw[s_id].map(_norm_id),
            "capacity_raw": pd.to_numeric(sinks_raw[s_cap], errors="coerce").fillna(0.0),
        }
    ).groupby("sink_id", as_index=False).sum()
    # Storage capacity in the workbook is in t CO2; values in Mt (all below 1e5) are converted to t.
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


def _capture_coefs(model_data: Dict[str, object]) -> Tuple[float, float, float]:
    # Capture rate, coal emission factor and the extra biogenic factor of biomass (zero when the
    # BC-CC interaction is switched off, i.e. capture is computed as if the fuel were coal only).
    ef_bio_delta = 0.0 if model_data.get("disable_bc_cc_interaction", False) else EF_BIOMASS_T_PER_GJ - EF_COAL_T_PER_GJ
    return float(model_data["cc_capture_rate"]), EF_COAL_T_PER_GJ, ef_bio_delta


def _crf(rate: float, years) -> np.ndarray:
    years = np.maximum(np.asarray(years, dtype=float), 1.0)
    return rate / (1.0 - (1.0 + rate) ** (-years))


def _payback_years(remaining_life) -> np.ndarray:
    # Eq. (S3): investment sharing time = min(remaining plant life, technology service life),
    # with a minimum of MIN_PAYBACK_YEARS.
    return np.clip(np.asarray(remaining_life, dtype=float), MIN_PAYBACK_YEARS, TECH_SERVICE_LIFE_YEARS)


def apply_ec_package(ec: pd.DataFrame, package_csv: str, package: str) -> pd.DataFrame:
    # Replace EC abatement and annual cost with a plant-level EC package (cost-optimal sub-technology
    # portfolio). Gross cost and avoided coal cost are kept separately for the scenario adjustments.
    pk = pd.read_csv(package_csv, dtype={"plant_id": str})
    pk = pk[pk["package"] == package]
    if pk.empty:
        raise ValueError(f"EC package '{package}' not found in {package_csv}.")
    pk = pk.assign(plant_id=pk["plant_id"].map(_norm_id))
    out = ec[["plant_id"]].merge(
        pk[["plant_id", "ec_abatement_ton", "ec_gross_cost", "ec_coal_saving"]], on="plant_id", how="left"
    )
    if out["ec_abatement_ton"].isna().any():
        raise ValueError(f"EC package '{package}' is missing {int(out['ec_abatement_ton'].isna().sum())} plants.")
    out["ec_total_cost"] = out["ec_gross_cost"] - out["ec_coal_saving"]
    return out.rename(columns={"ec_abatement_ton": "ec_abatement"})


def apply_input_scenario(
    workbook_path: str,
    plants: pd.DataFrame,
    ec: pd.DataFrame,
    cc: pd.DataFrame,
    biomass: pd.DataFrame,
    coal_factor: float,
    discount_case: str,
    cc_capex_factor: float,
    biomass_supply_factor: float,
    cc_capture_rate: float,
    scenario_year: Optional[int],
    retirement_age: float,
    cc_energy_basis: str = DEFAULT_CC_ENERGY_BASIS,
    cc_ext_steam_gj_per_t: float = CC_EXT_STEAM_GJ_PER_T,
    cc_ext_power_mwh_per_t: float = CC_EXT_POWER_MWH_PER_T,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Annuity factors and scenario changes applied to input tables before the model data are built."""
    plants = plants.copy()
    ec = ec.copy()
    cc = cc.copy()
    biomass = biomass.copy()
    plants_raw = pd.read_excel(workbook_path, sheet_name="coal_plant_i")
    plants_raw["plant_id"] = plants_raw[_find_col(list(plants_raw.columns), ["plant_i", "plantid", "id"])].map(_norm_id)
    plants_raw = plants_raw.groupby("plant_id", as_index=False).first().set_index("plant_id")
    cc_raw = pd.read_excel(workbook_path, sheet_name="CC")
    cc_raw["plant_id"] = cc_raw[_find_col(list(cc_raw.columns), ["plant_i", "plantid", "id"])].map(_norm_id)
    cc_raw = cc_raw.groupby("plant_id", as_index=False).first().set_index("plant_id")

    def _plant_col(frame: pd.DataFrame, name: str) -> pd.Series:
        return pd.to_numeric(frame["plant_id"].map(plants_raw[_find_col(list(plants_raw.columns), [name])]), errors="coerce")

    # Payback years: remaining life in 2020 (40-year plant life), or in the scenario year for time slices.
    if scenario_year is not None:
        age = _plant_col(plants, f"{scenario_year}_year")
        if age.isna().any():
            raise ValueError(f"Missing plant ages for {scenario_year}.")
        plants = plants[age <= float(retirement_age)].copy()
        remaining = {pid: float(retirement_age) - a for pid, a in zip(plants["plant_id"], age[age <= float(retirement_age)])}
    else:
        online = _plant_col(plants, "online_year")
        remaining = {pid: PLANT_LIFETIME_YEARS - (BASE_YEAR - y) for pid, y in zip(plants["plant_id"], online)}
    online_all = pd.to_numeric(plants_raw[_find_col(list(plants_raw.columns), ["online_year"])], errors="coerce")
    st_2020 = pd.Series(_payback_years(PLANT_LIFETIME_YEARS - (BASE_YEAR - online_all)), index=online_all.index)
    st = pd.Series(_payback_years(list(remaining.values())), index=list(remaining.keys()))
    rate = DISCOUNT_RATE_BY_CASE[discount_case]

    # BC: annuity factor at the BC discount rate.
    plants["af"] = _crf(rate, plants["plant_id"].map(st))

    # CC: annualized capex = overnight investment x CRF; fixed O&M = share of investment.
    key = round(float(cc_capture_rate), 4)
    if key not in CC_CAPTURE_RATE_COST_MULTIPLIERS:
        raise ValueError(f"--cc-capture-rate must be one of {sorted(CC_CAPTURE_RATE_COST_MULTIPLIERS)}.")
    capex_mult, fom_mult, ele1_mult, opex_mult = CC_CAPTURE_RATE_COST_MULTIPLIERS[key]
    # Capture energy: priced as extra coal at the plant. "boiler": heat from extra coal at boiler efficiency
    # (steam cost in opex); "extraction": steam extracted from the turbine, the lost power (25% of the heat)
    # is added to the electricity requirement and priced like it at the plant's fuel cost of generation.
    eff = _plant_col(cc, "efficiency")
    if cc_energy_basis == "extraction":
        cc["cc_opex_per_t"] = pd.to_numeric(cc["plant_id"].map(cc_raw[_find_col(list(cc_raw.columns), ["cc_ext_opex_dollar_per_t"])]), errors="coerce")
        # electricity1 is CC_POWER_MWH_PER_T priced at the fuel cost of generation; rescale to the extraction penalty
        penalty_mwh = float(cc_ext_power_mwh_per_t) + float(cc_ext_steam_gj_per_t) * CC_STEAM_TO_POWER / 3.6
        cc["cc_ele1_per_t"] = cc["cc_ele1_per_t"] * penalty_mwh / CC_POWER_MWH_PER_T
        # all of the extraction-basis penalty is power, so it follows the capture-rate electricity multiplier
        cc["cc_penalty_coal_gj_per_t"] = penalty_mwh * ele1_mult * 3.6 / eff
    elif cc_energy_basis == "boiler":
        cc["cc_penalty_coal_gj_per_t"] = CC_STEAM_GJ_PER_T / CC_STEAM_BOILER_EFF + CC_POWER_MWH_PER_T * 3.6 / eff
    else:
        raise ValueError("cc_energy_basis must be 'boiler' or 'extraction'.")
    cc["cc_penalty_coal_gj_per_t"] = cc["cc_penalty_coal_gj_per_t"].fillna(0.0)
    learning = CC_COST_FACTOR_BY_YEAR.get(int(scenario_year), 1.0) if scenario_year is not None else 1.0
    cc_st = cc["plant_id"].map(st)
    if scenario_year is None:
        cc_st = cc["cc_payback_years"]  # precomputed payback years of the CC sheet (identical rule)
    investment = cc["cc_investment"] * float(cc_capex_factor) * learning
    cc["cc_capex_per_year"] = investment * capex_mult * _crf(rate, cc_st.fillna(TECH_SERVICE_LIFE_YEARS))
    cc["cc_fixed_om_per_year"] = CC_FIXED_OM_SHARE * investment * fom_mult
    cc["cc_ele1_per_t"] = cc["cc_ele1_per_t"] * ele1_mult
    cc["cc_opex_per_t"] = cc["cc_opex_per_t"] * opex_mult

    # EC: in time slices, rescale the annualized investment to the remaining plant life.
    if scenario_year is not None and "ec_gross_cost" in ec.columns:
        scale = _crf(0.08, ec["plant_id"].map(st)) / _crf(0.08, ec["plant_id"].map(st_2020))
        ec["ec_gross_cost"] = ec["ec_gross_cost"] * np.nan_to_num(scale, nan=1.0)

    # Coal price: BC and EC coal-saving credits, steam cost in CC opex (precomputed columns), and the
    # capture electricity cost, which is priced at plant generation cost (fuel share of LCOE scales).
    if abs(coal_factor - 1.0) > 1e-9:
        plants["coal_price"] = plants["coal_price"] * float(coal_factor)
        if cc_energy_basis == "boiler":
            # steam cost in opex: precomputed columns at +-25% coal price
            cc_opex_col = {0.75: "opex_dollar_per_t_coal_m_25%", 1.25: "opex_dollar_per_t_coal_p_25%"}
            ckey = round(float(coal_factor), 4)
            if ckey not in cc_opex_col:
                raise ValueError("--coal-factor supports 0.75, 1.0 and 1.25 (CC opex columns are precomputed).")
            opex_coal = pd.to_numeric(cc["plant_id"].map(cc_raw[_find_col(list(cc_raw.columns), [cc_opex_col[ckey]])]), errors="coerce")
            cc["cc_opex_per_t"] = opex_coal.fillna(cc["cc_opex_per_t"] / opex_mult) * opex_mult
        # capture electricity is priced at the fuel cost of generation, so it scales with the coal price
        cc["cc_ele1_per_t"] = cc["cc_ele1_per_t"] * float(coal_factor)
    if "ec_gross_cost" in ec.columns:
        ec["ec_total_cost"] = ec["ec_gross_cost"] - float(coal_factor) * ec["ec_coal_saving"]

    if abs(biomass_supply_factor - 1.0) > 1e-9:
        biomass["available_gj"] = biomass["available_gj"] * float(biomass_supply_factor)

    if scenario_year is not None:
        print(f"Scenario year {scenario_year}: {len(plants)} active plants (age <= {retirement_age}), CC investment x{learning:.2f}")
    return plants, ec, cc, biomass


def apply_model_scenario(
    model_data: Dict[str, object],
    cc_capture_rate: float,
    bc_subsidy_rmb_per_kwh: float,
    cc_subsidy_rmb_per_ton: float,
    disable_ec: bool,
    disable_ec_interactions: bool,
    disable_bc_cc_interaction: bool,
    storage_injection_years: Optional[float] = None,
    cc_penalty_emissions: bool = DEFAULT_CC_PENALTY_EMISSIONS,
    bio_transport_ef_kg_per_t_km: float = DEFAULT_LOGISTICS_EF["bio_transport_kg_per_t_km"],
    bio_processing_ef_kg_per_gj: float = DEFAULT_LOGISTICS_EF["bio_processing_kg_per_gj"],
    co2_transport_ef_kg_per_t_km: float = DEFAULT_LOGISTICS_EF["co2_transport_kg_per_t_km"],
    bc_capex_factor: float = 1.0,
) -> Dict[str, object]:
    """Scenario changes applied to the assembled model data."""
    model_data = dict(model_data)
    model_data["bc_capex_factor"] = float(bc_capex_factor)  # co-firing investment multiplier (DB and GB)
    if storage_injection_years:
        # Cumulative storage potential spread over the injection period gives the annual injection capacity.
        model_data["sink_capacity_ton"] = {
            k: v / float(storage_injection_years) for k, v in model_data["sink_capacity_ton"].items()
        }
    model_data["cc_capture_rate"] = float(cc_capture_rate)
    model_data["bc_subsidy_usd_per_kwh"] = float(bc_subsidy_rmb_per_kwh) / RMB_PER_USD
    model_data["cc_subsidy_usd_per_ton"] = float(cc_subsidy_rmb_per_ton) / RMB_PER_USD
    model_data["disable_ec"] = bool(disable_ec)
    model_data["disable_bc_cc_interaction"] = bool(disable_bc_cc_interaction)
    if disable_ec_interactions:
        # EC keeps its own abatement and cost but no longer changes biomass demand or capture volume.
        model_data["ec_ratio"] = {i: 0.0 for i in model_data["ec_ratio"]}
    # Capture electricity per tonne at plants with EC scales with the reduced fuel input (1 - s_i).
    model_data["cc_ele2_per_t"] = {
        i: (1.0 - model_data["ec_ratio"].get(i, 0.0)) * model_data["cc_ele1_per_t"][i] for i in model_data["cc_ele1_per_t"]
    }
    model_data["cc_penalty_emissions"] = bool(cc_penalty_emissions)
    if cc_penalty_emissions:
        # The capture energy is supplied by extra coal fired in the plant's boiler (assumed to have spare capacity;
        # priced at the plant's fuel cost) to keep the net output; its flue gas passes the same capture unit. With p_i GJ of extra coal per tonne captured, the captured CO2 per tonne of
        # base capture is F_i = 1 / (1 - rate * ef_coal * p_i); with EC, p_i falls to (1 - s_i) p_i.
        rate = float(model_data["cc_capture_rate"])
        f0, fec = {}, {}
        for i, p0 in model_data["cc_penalty_coal_gj_per_t"].items():
            s_i = model_data["ec_ratio"].get(i, 0.0)
            f0[i] = 1.0 / (1.0 - rate * EF_COAL_T_PER_GJ * p0)
            fec[i] = 1.0 / (1.0 - rate * EF_COAL_T_PER_GJ * p0 * (1.0 - s_i))
        model_data["cc_f0"], model_data["cc_fec"] = f0, fec
        # larger capture unit for the larger treated CO2 flow
        model_data["cc_capex"] = {i: v * f0[i] ** CC_CAPEX_SCALE_EXPONENT for i, v in model_data["cc_capex"].items()}
        model_data["cc_fixed_om"] = {i: v * f0[i] ** CC_CAPEX_SCALE_EXPONENT for i, v in model_data["cc_fixed_om"].items()}
    # Logistics emissions (not captured) are deducted from the abatement: biomass haulage per t-km (converted
    # with the biomass LHV, on road distance), biomass supply chain per GJ, and CO2 transport per t-km.
    # straight-line distance x road tortuosity, as in the transport cost
    model_data["bio_transport_ef_t_per_gj_km"] = float(bio_transport_ef_kg_per_t_km) / 1000.0 / BIOMASS_LHV_GJ_PER_T * ROAD_TORTUOSITY
    model_data["bio_processing_ef_t_per_gj"] = float(bio_processing_ef_kg_per_gj) / 1000.0
    model_data["co2_transport_ef_t_per_t_km"] = float(co2_transport_ef_kg_per_t_km) / 1000.0
    return model_data


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
        "cc_fixed_om_per_year",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "cc_penalty_coal_gj_per_t" not in df.columns:
        df["cc_penalty_coal_gj_per_t"] = 0.0
    df["cc_penalty_coal_gj_per_t"] = pd.to_numeric(df["cc_penalty_coal_gj_per_t"], errors="coerce").fillna(0.0)

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

    arcs = list(zip(dist2["biomass_id"], dist2["plant_id"]))
    distance = {(j, i): float(d) for j, i, d in zip(dist2["biomass_id"], dist2["plant_id"], dist2["distance_km"])}

    sink_ids = sorted(sinks["sink_id"].astype(str).unique().tolist())
    co2_dist2 = co2_dist[co2_dist["plant_id"].isin(plants_used) & co2_dist["sink_id"].isin(sink_ids)].copy()
    if max_sink_links_per_plant is not None and max_sink_links_per_plant > 0:
        co2_dist2 = (
            co2_dist2.sort_values(["plant_id", "distance_km"]).groupby("plant_id", as_index=False).head(max_sink_links_per_plant)
        )
    sink_arcs = list(zip(co2_dist2["plant_id"], co2_dist2["sink_id"]))
    sink_distance = {(i, k): float(d) for i, k, d in zip(co2_dist2["plant_id"], co2_dist2["sink_id"], co2_dist2["distance_km"])}

    supply = {r["biomass_id"]: float(r["available_gj"]) for _, r in biomass.iterrows()}
    energy_gj = {r["plant_id"]: float(r["energy_gj"]) for _, r in df.iterrows()}
    base_emission = {r["plant_id"]: float(r["base_emission"]) for _, r in df.iterrows()}
    ec_abate = {r["plant_id"]: float(r["ec_abatement"]) for _, r in df.iterrows()}
    # Plant-specific EC saving ratio s_i: share of baseline fuel (and coal CO2) avoided by the EC package.
    ec_ratio = {
        i: (min(max(ec_abate[i] / base_emission[i], 0.0), 0.999) if base_emission[i] > 0 else 0.0)
        for i in ec_abate
    }
    ec_cost = {r["plant_id"]: float(r["ec_total_cost"]) for _, r in df.iterrows()}
    cc_abate = {r["plant_id"]: float(r["cc_abatement"]) for _, r in df.iterrows()}
    cc_cost = {r["plant_id"]: float(r["cc_total_cost"]) for _, r in df.iterrows()}
    cc_capex = {r["plant_id"]: float(r["cc_capex_per_year"]) for _, r in df.iterrows()}
    cc_opex_per_t = {r["plant_id"]: float(r["cc_opex_per_t"]) for _, r in df.iterrows()}
    cc_ele1_per_t = {r["plant_id"]: float(r["cc_ele1_per_t"]) for _, r in df.iterrows()}
    cc_ele2_per_t = {r["plant_id"]: float(r["cc_ele2_per_t"]) for _, r in df.iterrows()}
    cc_fixed_om = {r["plant_id"]: float(r["cc_fixed_om_per_year"]) for _, r in df.iterrows()}
    cc_penalty_coal = {r["plant_id"]: float(r["cc_penalty_coal_gj_per_t"]) for _, r in df.iterrows()}
    bc_abate_per_gj = {r["plant_id"]: float(r["bc_abate_per_gj"]) for _, r in df.iterrows()}
    bc_base_cost_per_gj = {r["plant_id"]: float(r["bc_base_cost_per_gj"]) for _, r in df.iterrows()}
    bc_cap_gj = {r["plant_id"]: float(r["bc_cap_gj"]) for _, r in df.iterrows()}
    power = {r["plant_id"]: float(r["power"]) for _, r in df.iterrows()}
    running_hours = {r["plant_id"]: float(r["running_hours"]) for _, r in df.iterrows()}
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
        "ec_ratio": ec_ratio,
        "generation_kwh": {i: max(0.0, power[i] * running_hours[i] * 1000.0) for i in plants_used},
        "cc_capture_rate": BASE_CC_CAPTURE_RATE,
        "bc_subsidy_usd_per_kwh": 0.0,
        "cc_subsidy_usd_per_ton": 0.0,
        "disable_ec": False,
        "disable_bc_cc_interaction": False,
        "ec_cost": ec_cost,
        "cc_abate": cc_abate,
        "cc_cost": cc_cost,
        "cc_capex": cc_capex,
        "cc_opex_per_t": cc_opex_per_t,
        "cc_ele1_per_t": cc_ele1_per_t,
        "cc_ele2_per_t": cc_ele2_per_t,
        "cc_fixed_om": cc_fixed_om,
        "cc_penalty_coal_gj_per_t": cc_penalty_coal,
        "cc_f0": {i: 1.0 for i in plants_used},
        "cc_fec": {i: 1.0 for i in plants_used},
        "cc_penalty_emissions": False,
        "bc_abate_per_gj": bc_abate_per_gj,
        "bc_base_cost_per_gj": bc_base_cost_per_gj,
        "bc_cap_gj": bc_cap_gj,
        "power": power,
        "af": af,
        "coal_price": coal_price,
        "sinks": sink_ids,
        "sink_arcs": sink_arcs,
        "sink_distance": sink_distance,
        "sink_capacity_ton": sink_capacity_ton,
        "co2_transport_cost": co2_transport_cost,
        "co2_storage_cost_per_ton": co2_storage_cost_per_ton,
        "biomass_feedstock_cost_per_gj": biomass_feedstock_cost_per_gj,
        # logistics emissions (t CO2 per unit flow), switched on in apply_model_scenario
        "bio_transport_ef_t_per_gj_km": 0.0,
        "bio_processing_ef_t_per_gj": 0.0,
        "co2_transport_ef_t_per_t_km": 0.0,
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
    model.Params.MIPGapAbs = float(SOLVER_MIP_GAP_ABS)
    if int(threads) > 0:
        model.Params.Threads = int(threads)
    # Partial MIP start: keep the previous target's binary decisions and let Gurobi complete the
    # continuous variables (the full previous solution violates the new abatement target).
    if model.IsMIP and model.SolCount > 0:
        all_vars = model.getVars()
        bins = [v for v in all_vars if v.VType == GRB.BINARY]
        conts = [v for v in all_vars if v.VType != GRB.BINARY]
        bin_vals = model.getAttr("X", bins)
        model.setAttr("Start", bins, bin_vals)
        model.setAttr("Start", conts, [GRB.UNDEFINED] * len(conts))
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
    logistics_emission_by_plant = {i: 0.0 for i in plants}
    for (j, i) in arcs:
        flow = b_flow[(j, i)]
        if flow <= 1e-9:
            continue
        transport_cost = model_data["transport_cost"][(j, i)] * flow
        feedstock_cost = model_data["biomass_feedstock_cost_per_gj"] * flow
        biomass_transport_cost_by_plant[i] += transport_cost
        biomass_feedstock_cost_by_plant[i] += feedstock_cost
        logistics_emission_by_plant[i] += (
            model_data.get("bio_transport_ef_t_per_gj_km", 0.0) * model_data["distance"][(j, i)]
            + model_data.get("bio_processing_ef_t_per_gj", 0.0)
        ) * flow
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
        logistics_emission_by_plant[i] += model_data.get("co2_transport_ef_t_per_t_km", 0.0) * model_data["sink_distance"][(i, k)] * flow
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
        s_i = model_data["ec_ratio"][i]
        rate, ef_coal, ef_bio_delta = _capture_coefs(model_data)
        cc_capture = rate * model_data["energy_gj"][i] * (
            ef_coal * x_cc[i] + ef_bio_delta * br_xcc[i]
        ) - s_i * rate * model_data["energy_gj"][i] * (
            ef_coal * y_ec_cc[i] + ef_bio_delta * br_xec_cc[i]
        )
        cc_capture_no_ec = rate * model_data["energy_gj"][i] * (
            ef_coal * (x_cc[i] - y_ec_cc[i]) + ef_bio_delta * (br_xcc[i] - br_xec_cc[i])
        )
        cc_capture_with_ec = (1.0 - s_i) * rate * model_data["energy_gj"][i] * (
            ef_coal * y_ec_cc[i] + ef_bio_delta * br_xec_cc[i]
        )
        cc_capture_base = cc_capture
        f0_i, fec_i = model_data["cc_f0"][i], model_data["cc_fec"][i]
        if model_data.get("cc_penalty_emissions", False):
            cc_capture = f0_i * cc_capture_no_ec + fec_i * cc_capture_with_ec
            cc_net_abatement = cc_capture - (cc_capture - cc_capture_base) / rate
        else:
            f0_i = fec_i = 1.0
            cc_net_abatement = cc_capture

        ec_cost_component = model_data["ec_cost"][i] * x_ec[i]
        cc_is_decomposed = not (
            model_data["cc_capex"][i] == 0
            and model_data["cc_opex_per_t"][i] == 0
            and model_data["cc_ele1_per_t"][i] == 0
            and model_data["cc_ele2_per_t"][i] == 0
        )
        cc_cost_component = (
            (model_data["cc_capex"][i] + model_data["cc_fixed_om"][i]) * x_cc[i]
            + model_data["cc_opex_per_t"][i] * cc_capture
            + model_data["cc_ele1_per_t"][i] * f0_i * cc_capture_no_ec
            + model_data["cc_ele2_per_t"][i] * fec_i * cc_capture_with_ec
        )
        cc_legacy_cost_component = model_data["cc_cost"][i] * x_cc[i] if not cc_is_decomposed else 0.0

        # BC cost decomposition for reporting (always computed with the formula requested by user).
        # Annualized investment plus fixed O&M as a share of total investment.
        bc_investment_cost = (
            (
                1153000.0 * blend_ratio[i] * model_data["power"][i]
                + 4628000.0 * high_ratio[i] * model_data["power"][i]
            )
            * model_data.get("bc_capex_factor", 1.0)
            * (model_data["af"][i] + BC_FIXED_OM_SHARE)
            / RMB_PER_USD
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
                "ec_saving_ratio": s_i,
                "cc_capture_rate": rate,
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
                "cc_capture_base_ton": cc_capture_base,
                "cc_net_abatement_ton": cc_net_abatement,
                "cc_penalty_capture_factor": f0_i,
                # extra coal fired for the capture energy (GJ/yr); with EC the requirement per tonne falls to (1 - s_i)
                "cc_penalty_coal_gj": model_data["cc_penalty_coal_gj_per_t"][i] * (f0_i * cc_capture_no_ec + (1.0 - s_i) * fec_i * cc_capture_with_ec),
                "logistics_emission_ton": logistics_emission_by_plant[i],
                "total_abatement_ton": ec_abatement + bc_abatement_formula + cc_net_abatement - logistics_emission_by_plant[i],
                "total_abatement_ton_model": ec_abatement + bc_abatement_model + cc_net_abatement - logistics_emission_by_plant[i],
                "co2_stored_ton": co2_stored_by_plant[i],
                "bc_subsidy_credit": model_data["bc_subsidy_usd_per_kwh"] * model_data["generation_kwh"][i] * blend_ratio[i],
                "cc_fixed_cost": (model_data["cc_capex"][i] + model_data["cc_fixed_om"][i]) * x_cc[i],
                "cc_fixed_cost_if_adopted": model_data["cc_capex"][i] + model_data["cc_fixed_om"][i],
                "cc_opex_per_t": model_data["cc_opex_per_t"][i],
                "cc_ele1_per_t": model_data["cc_ele1_per_t"][i],
                "cc_ele2_per_t": model_data["cc_ele2_per_t"][i],
                "bc_af": model_data["af"][i],
                "cc_subsidy_credit": model_data["cc_subsidy_usd_per_ton"] * cc_capture,
                "ec_cost": ec_cost_component,
                "cc_cost": cc_cost_component + cc_legacy_cost_component,
                "bc_cost": bc_internal_cost,
                "bc_cost_formula": bc_internal_cost_formula,
                "bc_cost_model_linear": bc_internal_cost_model,
                "bc_investment_cost": bc_investment_cost,
                "bc_feedstock_cost": bc_feedstock_cost,
                "generation_mwh": model_data["generation_kwh"][i] / 1000.0,
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

    # Scenario switch: EC off (BC+CC only). The McCormick bounds then force the EC products to zero.
    # (Switching off EC interactions is handled in apply_scenario by setting s_i = 0 in the
    # interaction terms, so EC can still be combined with BC and CC.)
    if model_data.get("disable_ec", False):
        for i in plants:
            model.addConstr(x_ec[i] == 0, name=f"NO_EC_X__{i}")
    # Single-technology runs for the standalone benchmark (EC is also switched off).
    if model_data.get("only_tech") == "bc":
        for i in plants:
            model.addConstr(x_cc[i] == 0, name=f"ONLY_BC__{i}")
    elif model_data.get("only_tech") == "cc":
        for i in plants:
            model.addConstr(blend_ratio[i] == 0, name=f"ONLY_CC__{i}")

    if bc_cost_mode == "sheet_linear":
        bc_cost_expr = gp.quicksum(model_data["bc_base_cost_per_gj"][i] * bc_use[i] for i in plants)
    elif bc_cost_mode == "tiered_static":
        # Reference form:
        # (1153000*ratio*power + 4628000*high_ratio*power)*(AF + om_share)/RMB_PER_USD - coal_price*energy*ratio
        # + feedstock_cost*biomass_flow, with fixed O&M as a share of total investment.
        # Investment follows the nominal co-firing ratio; biomass use and avoided coal follow the EC-adjusted
        # effective ratio (ratio_eff = ratio - s_i*ratio*x_ec).
        bc_cost_expr = (
            gp.quicksum(
                (
                    # co-firing investment per kW of plant capacity at the nominal co-firing ratio (DB and GB)
                    1153000.0 * blend_ratio[i] * model_data["power"][i]
                    + 4628000.0 * high_ratio[i] * model_data["power"][i]
                )
                * model_data.get("bc_capex_factor", 1.0)
                * (model_data["af"][i] + BC_FIXED_OM_SHARE)
                / RMB_PER_USD
                - model_data["coal_price"][i] * model_data["energy_gj"][i] * (blend_ratio[i] - model_data["ec_ratio"][i] * br_xec[i])
                for i in plants
            )
            + gp.quicksum(model_data["biomass_feedstock_cost_per_gj"] * b_flow[(j, i)] for (j, i) in arcs)
        )
    else:
        raise ValueError(f"Unsupported bc_cost_mode: {bc_cost_mode}")

    # Fuel-mix-based CC capture quantity:
    # rate * [E * r * ef_bio + E * (1-r) * ef_coal], and if EC is chosen multiply by (1 - s_i).
    rate, ef_coal, ef_bio_delta = _capture_coefs(model_data)
    capture_base_expr = {
        i: rate
        * model_data["energy_gj"][i]
        * (ef_coal * x_cc[i] + ef_bio_delta * br_xcc[i])
        for i in plants
    }
    capture_ec_reduction_expr = {
        i: model_data["ec_ratio"][i]
        * rate
        * model_data["energy_gj"][i]
        * (ef_coal * y_ec_cc[i] + ef_bio_delta * br_xec_cc[i])
        for i in plants
    }
    cc_capture_base_expr = {i: capture_base_expr[i] - capture_ec_reduction_expr[i] for i in plants}
    cc_capture_no_ec_expr = {
        i: rate
        * model_data["energy_gj"][i]
        * (ef_coal * (x_cc[i] - y_ec_cc[i]) + ef_bio_delta * (br_xcc[i] - br_xec_cc[i]))
        for i in plants
    }
    cc_capture_with_ec_expr = {
        i: (1.0 - model_data["ec_ratio"][i])
        * rate
        * model_data["energy_gj"][i]
        * (ef_coal * y_ec_cc[i] + ef_bio_delta * br_xec_cc[i])
        for i in plants
    }
    # Energy-penalty emissions (optional): captured CO2 including the flue gas of the extra coal burned for the
    # capture energy (F = 1 without penalty), and the net abatement of capture after the uncaptured share.
    f0, fec = model_data["cc_f0"], model_data["cc_fec"]
    cc_capture_expr = {i: f0[i] * cc_capture_no_ec_expr[i] + fec[i] * cc_capture_with_ec_expr[i] for i in plants}
    if model_data.get("cc_penalty_emissions", False):
        cc_net_abatement_expr = {
            i: cc_capture_expr[i] - (cc_capture_expr[i] - cc_capture_base_expr[i]) / rate for i in plants
        }
    else:
        cc_capture_expr = cc_capture_base_expr
        cc_net_abatement_expr = cc_capture_base_expr
    cc_cost_expr = gp.quicksum(
        (model_data["cc_capex"][i] + model_data["cc_fixed_om"][i]) * x_cc[i]
        + model_data["cc_opex_per_t"][i] * cc_capture_expr[i]
        + model_data["cc_ele1_per_t"][i] * f0[i] * cc_capture_no_ec_expr[i]
        + model_data["cc_ele2_per_t"][i] * fec[i] * cc_capture_with_ec_expr[i]
        for i in plants
    )
    # Fallback for plants without decomposed CC data: keep legacy total-cost term.
    cc_legacy_cost_expr = gp.quicksum(
        model_data["cc_cost"][i] * x_cc[i]
        for i in plants
        if model_data["cc_capex"][i] == 0 and model_data["cc_opex_per_t"][i] == 0 and model_data["cc_ele1_per_t"][i] == 0 and model_data["cc_ele2_per_t"][i] == 0
    )

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
    )
    # Optional policy subsidies: per kWh of biomass-based electricity and per ton of CO2 captured and stored.
    if model_data["bc_subsidy_usd_per_kwh"] > 0:
        total_cost_expr -= gp.quicksum(
            model_data["bc_subsidy_usd_per_kwh"]
            * model_data["generation_kwh"][i]
            * blend_ratio[i]  # EC does not change generation: biomass electricity = theta x generation
            for i in plants
        )
    if model_data["cc_subsidy_usd_per_ton"] > 0:
        total_cost_expr -= model_data["cc_subsidy_usd_per_ton"] * gp.quicksum(cc_capture_expr[i] for i in plants)
    # Unified BC abatement definition: blend ratio × plant baseline emissions.
    # Since bc_use[i] is in GJ and energy_gj[i] is fixed, this remains linear.
    total_abatement_expr = (
        gp.quicksum(model_data["ec_abate"][i] * x_ec[i] for i in plants)
        + gp.quicksum(cc_net_abatement_expr[i] for i in plants)
        + gp.quicksum(
            (model_data["base_emission"][i] / model_data["energy_gj"][i]) * bc_use[i]
            if model_data["energy_gj"][i] > 0
            else 0.0
            for i in plants
        )
    )
    ef_bt = model_data.get("bio_transport_ef_t_per_gj_km", 0.0)
    ef_bp = model_data.get("bio_processing_ef_t_per_gj", 0.0)
    ef_ct = model_data.get("co2_transport_ef_t_per_t_km", 0.0)
    if ef_bt > 0 or ef_bp > 0:
        total_abatement_expr -= gp.quicksum(
            (ef_bt * model_data["distance"][(j, i)] + ef_bp) * b_flow[(j, i)] for (j, i) in arcs
        )
    if ef_ct > 0:
        total_abatement_expr -= gp.quicksum(
            ef_ct * model_data["sink_distance"][(i, k)] * co2_flow[(i, k)] for (i, k) in sink_arcs
        )

    if objective_mode == "cost":
        model.setObjective(total_cost_expr, GRB.MINIMIZE)
    elif objective_mode == "abatement":
        model.setObjective(total_abatement_expr, GRB.MINIMIZE)
    else:
        raise ValueError(f"Unsupported objective_mode: {objective_mode}")

    # Group arcs once by source/plant/sink instead of scanning all arcs per node.
    arcs_by_source: Dict[str, List[object]] = {}
    arcs_by_plant: Dict[str, List[object]] = {}
    for (j, i) in arcs:
        arcs_by_source.setdefault(j, []).append(b_flow[(j, i)])
        arcs_by_plant.setdefault(i, []).append(b_flow[(j, i)])
    sink_arcs_by_plant: Dict[str, List[object]] = {}
    sink_arcs_by_sink: Dict[str, List[object]] = {}
    for (i, k) in sink_arcs:
        sink_arcs_by_plant.setdefault(i, []).append(co2_flow[(i, k)])
        sink_arcs_by_sink.setdefault(k, []).append(co2_flow[(i, k)])

    # Biomass supply cap by source j.
    for j in biomass:
        out_vars = arcs_by_source.get(j, [])
        if out_vars:
            model.addConstr(gp.quicksum(out_vars) <= model_data["supply"][j])

    # Plant-level biomass usage and blend cap.
    for i in plants:
        in_vars = arcs_by_plant.get(i, [])
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
            # If EC is selected, effective demand for BC scales to (1 - s_i)*e_i.
            model.addConstr(bc_use[i] == e_i * blend_ratio[i] - model_data["ec_ratio"][i] * e_i * br_xec[i])
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
        out_to_sinks = sink_arcs_by_plant.get(i, [])
        if out_to_sinks:
            # If EC is selected, effective CC capture volume scales to (1 - s_i).
            model.addConstr(gp.quicksum(out_to_sinks) == cc_capture_expr[i])
        else:
            model.addConstr(cc_capture_expr[i] == 0)

    # Sink capacity constraints.
    for k in sinks:
        in_to_sink = sink_arcs_by_sink.get(k, [])
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
    )
    target_lb.RHS = 0.0
    target_ub.RHS = GRB.INFINITY
    model.setObjective(total_abatement_expr, GRB.MAXIMIZE)
    status = _solve_model(model, use_gurobi, time_limit_sec, mip_gap, threads)
    if status == "TimeLimit" and model.SolCount > 0:
        # Use the best feasible abatement found (a conservative maximum).
        print(f"Max-abatement solve hit the time limit; using incumbent (gap {model.MIPGap:.4%}).")
    elif status != "Optimal":
        raise RuntimeError(f"Failed to compute max abatement ({'LP' if relax_integrality else 'MILP'}), status={status}")
    return float(total_abatement_expr.getValue())


def _build_solver_contexts(
    model_data: Dict[str, object],
    target_match_mode: str,
    relax_integrality: bool,
    bc_cost_mode: str,
    tier_split_ratio: float,
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
        )
        contexts["stage2"] = {
            "model": m2,
            "vars_pack": vp2,
            "total_cost": tc2,
            "total_abate": ta2,
            "target_lb": lb2,
            "target_ub": ub2,
        }
    elif target_match_mode in ("cost_only", "equality"):
        m, vp, tc, ta, lb, ub = _build_problem(
            model_data=model_data,
            target=0.0,
            objective_mode="cost",
            relax_integrality=relax_integrality,
            bc_cost_mode=bc_cost_mode,
            tier_split_ratio=tier_split_ratio,
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
    elif target_match_mode in ("cost_only", "equality"):
        s = contexts["single"]
        s["target_lb"].RHS = float(target)
        # Equality mode fixes abatement at the target (Eq. 43); cost_only leaves it as a lower bound.
        s["target_ub"].RHS = (
            float(target) * (1.0 + EQUALITY_REL_TOL) + 1.0 if target_match_mode == "equality" else GRB.INFINITY
        )
        status = _solve_model(s["model"], use_gurobi, time_limit_sec, mip_gap, threads)
        vars_pack = s["vars_pack"]
        total_cost = s["total_cost"]
        total_abate = s["total_abate"]
        solved_model = s["model"]
    else:
        raise ValueError(f"Unsupported target_match_mode: {target_match_mode}")

    if target_match_mode == "min_overshoot":
        solved_model = s2["model"]
    # Keep the incumbent when the time limit is reached with a near-optimal solution.
    final_gap = float(solved_model.MIPGap) if solved_model.SolCount > 0 and solved_model.IsMIP else 0.0
    if status == "TimeLimit" and solved_model.SolCount > 0 and final_gap <= TIME_LIMIT_ACCEPT_GAP:
        print(f"  Time limit reached; keeping incumbent with MIP gap {final_gap:.4%}")
    elif status != "Optimal":
        return status, np.nan, np.nan, {}

    achieved = float(total_abate.getValue())
    cost = float(total_cost.getValue())

    x_ec = vars_pack["x_ec"]
    x_cc = vars_pack["x_cc"]
    b_flow = vars_pack["b_flow"]
    bc_use = vars_pack["bc_use"]
    co2_flow = vars_pack["co2_flow"]

    summary = {
        "mip_gap": final_gap,
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
    target_step: float = None,
) -> np.ndarray:
    # Keep a 0.1% margin below the maximum feasible abatement: an equality target at the exact
    # frontier leaves almost no feasible region and can fail to find any solution.
    safe_upper = max_abatement * (1.0 - 1e-3)
    if max_target is None:
        max_target = safe_upper
    max_target = min(max_target, safe_upper)
    lower = max(min_target, 0.0)
    if target_step is not None and target_step > 0:
        base_targets = lower + np.arange(int(np.floor((max_target - lower) / target_step)) + 1) * target_step
        if base_targets[-1] < max_target - 1e-9:
            base_targets = np.append(base_targets, max_target)
    elif n_targets < 2:
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
            ),
            "cost_only": _build_solver_contexts(
                model_data=model_data,
                target_match_mode="cost_only",
                relax_integrality=relax_integrality,
                bc_cost_mode=bc_cost_mode,
                tier_split_ratio=tier_split_ratio,
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
                "mip_gap": summary.get("mip_gap", np.nan),
                "ec_adoption_sum": summary.get("ec_adoption_sum", np.nan),
                "cc_adoption_sum": summary.get("cc_adoption_sum", np.nan),
                "total_biomass_use_gj": summary.get("total_biomass_use_gj", np.nan),
                "weighted_transport_distance_km": summary.get("weighted_transport_distance_km", np.nan),
                "total_co2_stored_ton": summary.get("total_co2_stored_ton", np.nan),
                "weighted_co2_distance_km": summary.get("weighted_co2_distance_km", np.nan),
            }
        )
        print(
            f"[{mode} {idx}/{len(targets)}] mode={current_match_mode} target={target:,.2f} status={status} "
            f"achieved={achieved:,.2f} cost={cost:,.2f}"
        )

        if status in ("Optimal", "TimeLimit") and "vars_pack" in summary:
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
            "Fleet-wide joint MILP for EC/BC/CC with spatial biomass allocation B_{j,i}, "
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
    parser.add_argument("--target-match-mode", choices=["equality", "min_overshoot", "cost_only"], default=TARGET_MATCH_MODE)
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
    parser.add_argument(
        "--ec-package",
        default=EC_PACKAGE,
        help="EC package tag in the EC package file (e.g. r20_o00, r15_o25, no5_r10_o00); empty keeps legacy EC sheet columns.",
    )
    parser.add_argument("--ec-package-file", default=None, help="EC package CSV. Default: data/ec_packages_r2.csv")
    parser.add_argument("--storage-injection-years", type=float, default=STORAGE_INJECTION_YEARS, help="Annualize storage potential over N years.")
    parser.add_argument("--target-step", type=float, default=None, help="Optional fixed target step (t CO2), overrides --n-targets.")
    parser.add_argument("--coal-factor", type=float, default=1.0, help="Coal price multiplier: 0.75, 1.0 or 1.25.")
    parser.add_argument("--discount-case", choices=["base", "af5", "af12"], default="base", help="Discount rate case for BC/CC annuity factors.")
    parser.add_argument("--cc-capex-factor", type=float, default=1.0, help="CC capex multiplier, e.g. 0.7 or 1.3.")
    parser.add_argument("--bc-capex-factor", type=float, default=1.0, help="Biomass co-firing investment multiplier (DB and GB), e.g. 0.7 or 1.3.")
    parser.add_argument("--biomass-supply-factor", type=float, default=1.0, help="Biomass availability multiplier, e.g. 0.25-0.75.")
    parser.add_argument("--cc-capture-rate", type=float, default=BASE_CC_CAPTURE_RATE, help="CO2 capture rate: 0.90, 0.95 or 0.997.")
    parser.add_argument("--bc-subsidy-rmb-per-kwh", type=float, default=0.0, help="Biomass electricity subsidy (RMB/kWh).")
    parser.add_argument("--cc-subsidy-rmb-per-ton", type=float, default=0.0, help="Subsidy for captured and stored CO2 (RMB/t).")
    parser.add_argument("--scenario-year", type=int, default=None, help="Time slice (2030/2035/2040/2045): retire old plants, apply CC learning.")
    parser.add_argument("--retirement-age", type=float, default=30.0, help="Plant retirement age used with --scenario-year.")
    parser.add_argument("--disable-ec", action="store_true", help="BC+CC only (EC not available).")
    parser.add_argument("--only-tech", choices=["bc", "cc"], default=None, help="Single-technology run (standalone benchmark); implies --disable-ec.")
    parser.add_argument("--disable-ec-interactions", action="store_true", help="EC no longer changes biomass demand or CO2 capture volume.")
    parser.add_argument("--disable-bc-cc-interaction", action="store_true", help="Capture volume independent of the fuel mix (biomass at the coal emission factor).")
    parser.add_argument("--cc-energy-basis", choices=["boiler", "extraction"], default=DEFAULT_CC_ENERGY_BASIS, help="Capture heat priced as extra coal in a boiler or as turbine-extraction power loss. Default from ABATEMENT_ACCOUNTING.")
    parser.add_argument("--cc-ext-steam-gj-per-t", type=float, default=CC_EXT_STEAM_GJ_PER_T, help="Regeneration heat (GJ/t CO2) on the extraction basis.")
    parser.add_argument("--cc-ext-power-mwh-per-t", type=float, default=CC_EXT_POWER_MWH_PER_T, help="Capture + compression power (MWh/t CO2) on the extraction basis.")
    parser.add_argument("--cc-penalty-emissions", action=argparse.BooleanOptionalAction, default=DEFAULT_CC_PENALTY_EMISSIONS, help="Count the CO2 of the extra coal fired (boiler with spare capacity) for the capture energy; its flue gas is captured at the same rate.")
    parser.add_argument("--bio-transport-ef-kg-per-t-km", type=float, default=DEFAULT_LOGISTICS_EF["bio_transport_kg_per_t_km"], help="CO2 of biomass haulage (kg per t-km of biomass, incl. empty return); deducted from abatement.")
    parser.add_argument("--bio-processing-ef-kg-per-gj", type=float, default=DEFAULT_LOGISTICS_EF["bio_processing_kg_per_gj"], help="CO2 of biomass collection/processing (kg per GJ delivered); deducted from abatement.")
    parser.add_argument("--co2-transport-ef-kg-per-t-km", type=float, default=DEFAULT_LOGISTICS_EF["co2_transport_kg_per_t_km"], help="CO2 emitted by CO2 transport (kg per t-km); deducted from abatement.")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX, help="Output prefix in data folder.")
    args = parser.parse_args()
    print(
        f"Abatement accounting: {ABATEMENT_ACCOUNTING} (cc energy basis={args.cc_energy_basis}"
        + (f" {args.cc_ext_steam_gj_per_t} GJ/t + {args.cc_ext_power_mwh_per_t} MWh/t" if args.cc_energy_basis == "extraction" else "")
        + f", penalty CO2={args.cc_penalty_emissions}, "
        f"logistics EF: bio {args.bio_transport_ef_kg_per_t_km} kg/t-km + {args.bio_processing_ef_kg_per_gj} kg/GJ, "
        f"CO2 {args.co2_transport_ef_kg_per_t_km} kg/t-km)"
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, DATA_DIR_NAME)
    input_path = args.input or os.path.join(data_dir, INPUT_WORKBOOK_NAME)
    distance_path = args.distance_csv or os.path.join(data_dir, INPUT_BIOMASS_DISTANCE_NAME)
    co2_distance_path = args.co2_distance_csv or os.path.join(data_dir, INPUT_CO2_DISTANCE_NAME)
    use_gurobi = not args.no_gurobi

    plants, ec, bc, cc, biomass, biomass_dist, sinks, co2_dist = load_inputs(input_path, distance_path, co2_distance_path)
    if args.ec_package:
        package_path = args.ec_package_file or os.path.join(data_dir, EC_PACKAGE_FILE_NAME)
        ec = apply_ec_package(ec, package_path, args.ec_package)
        print(f"EC package: {args.ec_package} ({package_path})")
    plants, ec, cc, biomass = apply_input_scenario(
        workbook_path=input_path,
        plants=plants,
        ec=ec,
        cc=cc,
        biomass=biomass,
        coal_factor=args.coal_factor,
        discount_case=args.discount_case,
        cc_capex_factor=args.cc_capex_factor,
        biomass_supply_factor=args.biomass_supply_factor,
        cc_capture_rate=args.cc_capture_rate,
        scenario_year=args.scenario_year,
        retirement_age=args.retirement_age,
        cc_energy_basis=args.cc_energy_basis,
        cc_ext_steam_gj_per_t=args.cc_ext_steam_gj_per_t,
        cc_ext_power_mwh_per_t=args.cc_ext_power_mwh_per_t,
    )
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
    model_data = apply_model_scenario(
        model_data,
        cc_capture_rate=args.cc_capture_rate,
        bc_subsidy_rmb_per_kwh=args.bc_subsidy_rmb_per_kwh,
        cc_subsidy_rmb_per_ton=args.cc_subsidy_rmb_per_ton,
        disable_ec=args.disable_ec or args.only_tech is not None,
        disable_ec_interactions=args.disable_ec_interactions,
        disable_bc_cc_interaction=args.disable_bc_cc_interaction,
        storage_injection_years=args.storage_injection_years,
        cc_penalty_emissions=args.cc_penalty_emissions,
        bio_transport_ef_kg_per_t_km=args.bio_transport_ef_kg_per_t_km,
        bio_processing_ef_kg_per_gj=args.bio_processing_ef_kg_per_gj,
        co2_transport_ef_kg_per_t_km=args.co2_transport_ef_kg_per_t_km,
        bc_capex_factor=args.bc_capex_factor,
    )
    model_data["only_tech"] = args.only_tech

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
    targets = build_targets(args.n_targets, max_abatement, args.min_target, args.max_target, args.extra_targets, args.target_step)
    if args.fixed_max_abatement is None:
        print(f"MILP feasible max abatement: {max_abatement:,.2f}")
    print(
        f"Plants: {len(model_data['plants'])}, Biomass points: {len(model_data['biomass'])}, "
        f"Biomass arcs: {len(model_data['arcs'])}, Sinks: {len(model_data['sinks'])}, Sink arcs: {len(model_data['sink_arcs'])}"
    )

    out_curve = os.path.join(data_dir, f"{args.output_prefix}{OUTPUT_CURVE_SUFFIX}")
    out_xlsx = os.path.join(data_dir, f"{args.output_prefix}{OUTPUT_RESULTS_SUFFIX}")
    out_lp_curve = os.path.join(data_dir, f"{args.output_prefix}{OUTPUT_LP_CURVE_SUFFIX}")
    lp_curve = None
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
                time_limit_sec=args.time_limit_sec,
                mip_gap=args.mip_gap,
                threads=args.solver_threads,
                detail_writer=writer,
            )
            lp_curve.to_excel(writer, sheet_name="cl", index=False)
            lp_curve.to_csv(out_lp_curve, index=False)

    print("\nDone.")
    print(f"Input workbook: {input_path}")
    print(f"Input distance: {distance_path}")
    print(f"Input CO2 distance: {co2_distance_path}")
    print(f"MILP curve output: {out_curve}")
    if lp_curve is not None:
        print(f"LP-relax curve output: {out_lp_curve}")
    print(f"Excel output: {out_xlsx}")


if __name__ == "__main__":
    main()
