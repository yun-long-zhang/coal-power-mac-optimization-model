"""
Plant-level standalone and pairwise techno-economic results (Fig. 1 and Fig. 2), computed with the same
cost formulas and parameters as the fleet model but for each plant independently (no competition for
biomass hubs or storage sites).

Standalone measures (per plant): EC packages (10%, 20%), direct co-firing DB and gasified co-firing GB at
co-firing ratios of 10% and 20%, and carbon capture CC at 90%.
  - BC: biomass input theta*P is supplied from the nearest hubs (within the Dji_200 arcs) up to each hub's
    resource; cost = investment x (CRF + fixed O&M share) + feedstock + transport - avoided coal cost.
  - CC: captured CO2 is sent to the nearest storage sites up to their annual injection capacity; cost =
    annualized capex + fixed O&M + variable O&M + capture electricity + transport + storage.
Plant parameters come from the fleet model's own data pipeline (baseline scenario), so the accounting is the
model's: under the avoided accounting CC treats F times the base flue gas (F = 1 / (1 - rate * ef_coal * p),
p = extra coal per tonne captured, (1 - s) p with EC), the unit and its fixed O&M are scaled by F0^0.67, and
the abatement is net of the uncaptured CO2 of the extra coal; BC abatement is net of the biomass supply-chain
and haulage emissions of the arcs it uses.
Pairwise combinations: BC or CC on top of an EC package (fuel input reduced by s_i; BC keeps its nominal
co-firing ratio, so its biomass input falls by s_i and is re-sourced from the nearest hubs; CC captures the
reduced flue-gas CO2 and uses the EC electricity coefficient), and CC on top of DB co-firing (biogenic CO2
added to the capture stream).
"""

import os

import numpy as np
import pandas as pd

import optimization_1_fleet_joint_milp_persistent as base


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "standalone_pairwise_r2.csv")
EC_PACKAGE_FILE = os.path.join(DATA_DIR, base.EC_PACKAGE_FILE_NAME)
BC_INV_DB_RMB_PER_MW = 1153000.0
BC_INV_GB_RMB_PER_MW = 1153000.0 + 4628000.0
THETAS = (0.10, 0.20)


def _model_data() -> dict:
    """Baseline model data, assembled exactly as in the fleet model."""
    wb = os.path.join(DATA_DIR, base.INPUT_WORKBOOK_NAME)
    plants, ec, bc, cc, biomass, biomass_dist, sinks, co2_dist = base.load_inputs(
        wb, os.path.join(DATA_DIR, base.INPUT_BIOMASS_DISTANCE_NAME), os.path.join(DATA_DIR, base.INPUT_CO2_DISTANCE_NAME)
    )
    ec = base.apply_ec_package(ec, EC_PACKAGE_FILE, base.EC_PACKAGE)
    plants, ec, cc, biomass = base.apply_input_scenario(
        workbook_path=wb, plants=plants, ec=ec, cc=cc, biomass=biomass, coal_factor=1.0, discount_case="base",
        cc_capex_factor=1.0, biomass_supply_factor=1.0, cc_capture_rate=base.BASE_CC_CAPTURE_RATE,
        scenario_year=None, retirement_age=30.0,
    )
    md = base.build_model_data(
        plants=plants, ec=ec, bc=bc, cc=cc, biomass=biomass, biomass_dist=biomass_dist, sinks=sinks, co2_dist=co2_dist,
        blend_cap_ratio=base.BLEND_CAP_RATIO, max_links_per_plant=base.MAX_LINKS_PER_PLANT,
        max_sink_links_per_plant=base.MAX_SINK_LINKS_PER_PLANT, transport_cost_per_gj_km=base.TRANSPORT_COST_PER_GJ_KM,
        co2_transport_cost_per_ton_km=base.CO2_TRANSPORT_COST_PER_TON_KM, co2_storage_cost_per_ton=base.CO2_STORAGE_COST_PER_TON,
        biomass_feedstock_cost_per_gj=base.BIOMASS_FEEDSTOCK_COST_PER_GJ,
    )
    return base.apply_model_scenario(
        md, cc_capture_rate=base.BASE_CC_CAPTURE_RATE, bc_subsidy_rmb_per_kwh=0.0, cc_subsidy_rmb_per_ton=0.0,
        disable_ec=False, disable_ec_interactions=False, disable_bc_cc_interaction=False,
        storage_injection_years=base.STORAGE_INJECTION_YEARS,
    )


def _plants(md: dict) -> pd.DataFrame:
    ids = list(md["plants"])
    m = lambda key: pd.Series([float(md[key][i]) for i in ids], index=ids)
    out = pd.DataFrame(
        {
            "P_gj": m("energy_gj"),
            "E0": m("base_emission"),
            "X_mw": m("power"),
            "G_mwh": m("generation_kwh") / 1000.0,
            "coal_price": m("coal_price"),
            "af_bc": m("af"),
            # unit size and fixed O&M already include the F0^0.67 scale-up under the avoided accounting
            "cc_fixed_per_year": m("cc_capex") + m("cc_fixed_om"),
            "cc_opex_per_t": m("cc_opex_per_t"),
            "cc_ele1_per_t": m("cc_ele1_per_t"),
            "cc_penalty_gj_per_t": m("cc_penalty_coal_gj_per_t"),
        }
    )
    out.index.name = "plant_id"
    return out


def _ec_packages(plant_ids: pd.Series) -> pd.DataFrame:
    pk = pd.read_csv(EC_PACKAGE_FILE, dtype={"plant_id": str})
    p20 = pk[pk["package"] == "r20_o00"].set_index("plant_id")
    p10 = pk[pk["package"] == "no5_r10_o00"].set_index("plant_id")  # baseline: 10% without technology No. 5
    out = pd.DataFrame({"plant_id": plant_ids})
    out["ec20_abate"] = out["plant_id"].map(p20["ec_abatement_ton"])
    out["ec20_cost"] = out["plant_id"].map(p20["ec_net_cost"])
    out["ec10_abate"] = out["plant_id"].map(p10["ec_abatement_ton"])
    out["ec10_cost"] = out["plant_id"].map(p10["ec_net_cost"])
    return out


def _nearest_fill(arcs: pd.DataFrame, need: pd.Series, cap_col: str, unit_cost_col: str, unit_ef_col: str = None):
    """For each plant, fill `need` from its arcs in order of unit cost (nearest first); return the cost
    (NaN if the reachable resource is insufficient) and, with unit_ef_col, the logistics emissions."""
    a = arcs.merge(need.rename("need"), left_on="plant_id", right_index=True)
    a = a.sort_values(["plant_id", unit_cost_col])
    a["cum_before"] = a.groupby("plant_id")[cap_col].cumsum() - a[cap_col]
    a["take"] = np.clip(a["need"] - a["cum_before"], 0.0, a[cap_col])
    cost = (a["take"] * a[unit_cost_col]).groupby(a["plant_id"]).sum()
    supplied = a.groupby("plant_id")["take"].sum()
    short = supplied < need.reindex(supplied.index) * (1 - 1e-9)
    cost[short] = np.nan
    if unit_ef_col is None:
        return cost.reindex(need.index)
    emis = (a["take"] * a[unit_ef_col]).groupby(a["plant_id"]).sum()
    emis[short] = np.nan
    return cost.reindex(need.index), emis.reindex(need.index)


def main() -> None:
    md = _model_data()
    pl = _plants(md)
    ec = _ec_packages(pd.Series(pl.index)).set_index("plant_id")
    s20 = (ec["ec20_abate"] / pl["E0"]).clip(0, 0.999)
    s10 = (ec["ec10_abate"] / pl["E0"]).clip(0, 0.999)

    # Biomass arcs (hub resource) and storage arcs (annual injection capacity)
    wb = os.path.join(DATA_DIR, base.INPUT_WORKBOOK_NAME)
    hubs = pd.read_excel(wb, sheet_name="biomass_point_j")
    hubs = pd.DataFrame({"biomass_id": hubs["biomass_point_j"].map(base._norm_id), "supply_gj": hubs["bioenergy_GJ"].astype(float)})
    dji = pd.read_csv(os.path.join(DATA_DIR, base.INPUT_BIOMASS_DISTANCE_NAME))
    bio_arcs = pd.DataFrame({"biomass_id": dji["biomass_j"].map(base._norm_id), "plant_id": dji["plant_i"].map(base._norm_id), "d": dji["Dji"].astype(float)})
    bio_arcs = bio_arcs.merge(hubs, on="biomass_id")
    bio_arcs["unit_cost"] = base.BIOMASS_FEEDSTOCK_COST_PER_GJ + base.TRANSPORT_COST_PER_GJ_KM * bio_arcs["d"]
    bio_arcs["unit_ef"] = md["bio_processing_ef_t_per_gj"] + md["bio_transport_ef_t_per_gj_km"] * bio_arcs["d"]
    rate = float(md["cc_capture_rate"])
    penalty = bool(md.get("cc_penalty_emissions", False))
    sinks = pd.read_excel(wb, sheet_name="carbon_sink_k")
    sinks = pd.DataFrame({"sink_id": sinks["sink_k"].map(base._norm_id), "cap_t": sinks["Estimated_storage_potential_t"].astype(float) / base.STORAGE_INJECTION_YEARS})
    dki = pd.read_csv(os.path.join(DATA_DIR, base.INPUT_CO2_DISTANCE_NAME))
    co2_arcs = pd.DataFrame({"sink_id": dki["sink_k"].map(base._norm_id), "plant_id": dki["plant_i"].map(base._norm_id), "d": dki["Dki"].astype(float)})
    co2_arcs = co2_arcs.merge(sinks, on="sink_id")
    co2_arcs["unit_cost"] = base.CO2_TRANSPORT_COST_PER_TON_KM * co2_arcs["d"] + base.CO2_STORAGE_COST_PER_TON

    res = pd.DataFrame(index=pl.index)
    res["G_mwh"] = pl["G_mwh"]
    res["s_ec10"], res["s_ec20"] = s10, s20
    for lvl, s in (("10", s10), ("20", s20)):
        res[f"EC{lvl}_cost"] = ec[f"ec{lvl}_cost"]
        res[f"EC{lvl}_uac"] = ec[f"ec{lvl}_cost"] / ec[f"ec{lvl}_abate"]

    reachable = bio_arcs.groupby("plant_id")["supply_gj"].sum().reindex(pl.index).fillna(0.0)

    def bc(theta: float, inv_rmb_per_mw: float, s: pd.Series) -> pd.DataFrame:
        # biomass input = displaced coal energy, limited to the biomass reachable within 200 km
        bio = np.minimum(theta * (1.0 - s) * pl["P_gj"], reachable)
        supply, logistics = _nearest_fill(bio_arcs, bio, "supply_gj", "unit_cost", "unit_ef")
        # investment per kW of capacity at the nominal co-firing ratio (unchanged by EC)
        invest = inv_rmb_per_mw * theta * pl["X_mw"] * (pl["af_bc"] + base.BC_FIXED_OM_SHARE) / base.RMB_PER_USD
        gross = invest + supply  # annual cost of BC (investment, O&M, feedstock, transport)
        net = gross - pl["coal_price"] * bio
        share = bio / ((1.0 - s) * pl["P_gj"])  # co-firing ratio actually reached (theta unless biomass is short)
        return pd.DataFrame({"gross": gross, "net": net, "abate": bio * base.EF_COAL_T_PER_GJ - logistics,
                             "share": share})

    def cc(theta: float, s: pd.Series, ele_scale: pd.Series) -> pd.DataFrame:
        q_base = rate * (1.0 - s) * pl["P_gj"] * (
            base.EF_COAL_T_PER_GJ * (1 - theta) + base.EF_BIOMASS_T_PER_GJ * theta
        )
        if penalty:
            # flue gas of the extra coal for the capture energy is captured too; with EC the penalty is (1 - s) p
            f = 1.0 / (1.0 - rate * base.EF_COAL_T_PER_GJ * pl["cc_penalty_gj_per_t"] * (1.0 - s))
        else:
            f = 1.0
        q = f * q_base
        ts = _nearest_fill(co2_arcs, q, "cap_t", "unit_cost")
        cost = pl["cc_fixed_per_year"] + (pl["cc_opex_per_t"] + ele_scale * pl["cc_ele1_per_t"]) * q + ts
        abate = q - (q - q_base) / rate if penalty else q
        return pd.DataFrame({"cost": cost, "abate": abate, "captured": q})

    zero = pd.Series(0.0, index=pl.index)
    one = pd.Series(1.0, index=pl.index)
    for theta in THETAS:
        t = f"{int(theta * 100)}"
        for name, inv in (("DB", BC_INV_DB_RMB_PER_MW), ("GB", BC_INV_GB_RMB_PER_MW)):
            b0 = bc(theta, inv, zero)
            res[f"{name}{t}_net"], res[f"{name}{t}_gross"], res[f"{name}{t}_abate"] = b0["net"], b0["gross"], b0["abate"]
            res[f"{name}{t}_uac"] = b0["net"] / b0["abate"]
            res[f"{name}{t}_biopower_cost_per_mwh"] = b0["gross"] / (b0["share"] * pl["G_mwh"])
            for lvl, s in (("10", s10), ("20", s20)):
                b1 = bc(theta, inv, s)
                key = f"{name}{t}_EC{lvl}"
                res[f"{key}_uac"] = b1["net"] / b1["abate"]
                res[f"{key}_dgen_combo"] = (ec[f"ec{lvl}_cost"] + b1["net"] - b0["net"]) / pl["G_mwh"]
                res[f"{key}_biopower_cost_per_mwh"] = b1["gross"] / (b1["share"] * pl["G_mwh"])
        c_bc = cc(theta, zero, one)  # CC on a DB co-fired plant
        res[f"CC_DB{t}_uac"] = c_bc["cost"] / c_bc["abate"]

    c0 = cc(0.0, zero, one)
    res["CC90_cost"], res["CC90_abate"], res["CC90_captured"] = c0["cost"], c0["abate"], c0["captured"]
    res["CC90_uac"] = c0["cost"] / c0["abate"]
    for lvl, s in (("10", s10), ("20", s20)):
        c1 = cc(0.0, s, 1.0 - s)
        res[f"CC90_EC{lvl}_uac"] = c1["cost"] / c1["abate"]
        res[f"CC90_EC{lvl}_dgen_combo"] = (ec[f"ec{lvl}_cost"] + c1["cost"] - c0["cost"]) / pl["G_mwh"]
        res[f"EC{lvl}_dgen_alone"] = ec[f"ec{lvl}_cost"] / pl["G_mwh"]

    rrh = pd.read_excel(wb, sheet_name="RRH")
    res["RRH_uac"] = pd.Series(rrh["unit _cost"].astype(float).values, index=rrh["plant_i"].map(base._norm_id))
    res.reset_index().to_csv(OUTPUT_CSV, index=False)

    def q(col: str) -> str:
        v = res[col].dropna()
        return f"n={len(v)}, mean {v.mean():7.2f}, median {v.median():7.2f}, p10 {v.quantile(.1):7.2f}, p90 {v.quantile(.9):7.2f}"

    print("Standalone UAC (USD/t):")
    for c in ["EC10_uac", "EC20_uac", "DB10_uac", "DB20_uac", "GB10_uac", "GB20_uac", "CC90_uac", "RRH_uac"]:
        print(f"  {c:12s} {q(c)}")
    print("Share of plants with UAC below RRH:")
    for c in ["EC10_uac", "EC20_uac", "DB10_uac", "DB20_uac", "GB10_uac", "GB20_uac", "CC90_uac"]:
        ok = res[[c, "RRH_uac"]].dropna()
        print(f"  {c:12s} {(ok[c] < ok['RRH_uac']).mean() * 100:5.1f}%")
    print("Pairwise (EC20 on top):")
    for x in ["DB10", "DB20", "GB10", "GB20"]:
        d = (res[f"{x}_EC20_uac"] - res[f"{x}_uac"]).dropna()
        b = (res[f"{x}_biopower_cost_per_mwh"] - res[f"{x}_EC20_biopower_cost_per_mwh"]).dropna()
        print(f"  {x}: dUAC {d.mean():+.2f} USD/t; biopower cost saving {b.mean():.2f} USD/MWh")
    d = (res["CC90_EC20_uac"] - res["CC90_uac"]).dropna()
    print(f"  CC90: dUAC {d.mean():+.2f} USD/t")
    for t in ("10", "20"):
        d = (res["CC90_uac"] - res[f"CC_DB{t}_uac"]).dropna()
        print(f"  BC{t}% lowers CC UAC by {d.mean():.2f} USD/t (median {d.median():.2f})")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
