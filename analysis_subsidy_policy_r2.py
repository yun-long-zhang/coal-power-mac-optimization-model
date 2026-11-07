"""
Policy support from the fleet optimization (Reviewer 2, comment 7), avoided-accounting result set.

Replaces the earlier estimate (pairwise unit savings x optimized volumes) by quantities the fleet MILP
produces directly, under uniform per-unit instruments:

  (1) Uniform carbon price (or a uniform per-tonne abatement payment). For each curve the least-cost
      response is A*(p) = argmin_A T(A) - p A on the lower convex hull of (A, T). The price needed to reach
      abatement A is set by the marginal plant (hull slope), and the transfer of a uniform payment is p x A.
      Comparing base with the no-interaction runs shows how the interactions change the required price and
      the transfer.
  (2) Uniform technology subsidies (biomass electricity RMB/kWh, captured CO2 RMB/t). At each target the
      subsidy spending is the sum of the credits paid to all adopting plants; the resource cost is the
      objective plus the spending. Without a target (private cost minimized) the subsidy alone draws in
      deployment A*(0).

Outputs data/subsidy_policy_r2.xlsx (sheets: carbon_price, carbon_price_reference, subsidy_targets,
subsidy_free_response) in R2_SET_DIR (default results_r2_ec10_avoided).
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SET_DIR = os.environ.get("R2_SET_DIR", os.path.join(HERE, "data", "results_r2_ec10_avoided"))
FIG_DIR = os.environ.get("R2_FIG_DIR", os.path.join(SET_DIR, "figures"))
OUT_XLSX = os.path.join(FIG_DIR, "subsidy_policy_r2.xlsx")
RMB_PER_USD = 7.0

INTERACTION_RUNS = ["base", "no_interaction", "no_ec_interaction", "no_bccc_interaction"]
SUBSIDY_RUNS = {
    "subsidy_bc_0p15": "BC 0.15 RMB/kWh",
    "subsidy_bc_0p25": "BC 0.25 RMB/kWh",
    "subsidy_cc_35": "CC 35 RMB/t",
    "subsidy_cc_105": "CC 105 RMB/t",
}
REPORT_GT = [1.0, 2.0, 3.0, 3.5, 4.0]
REFERENCE_PRICES = {"China ETS": 9.0, "UK ETS": 66.0, "EU ETS": 83.0}


def _curve(name: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(SET_DIR, f"gurobi_r2_{name}_curve.csv"))
    d = d[d["status"].astype(str).isin(["Optimal", "TimeLimit"])]
    d = d[["target_id", "achieved_abatement", "total_cost"]].sort_values("achieved_abatement")
    # the origin (no retrofit) is always available
    d = pd.concat([pd.DataFrame({"target_id": [0], "achieved_abatement": [0.0], "total_cost": [0.0]}), d])
    return d.drop_duplicates("achieved_abatement").reset_index(drop=True)


def _hull(d: pd.DataFrame) -> pd.DataFrame:
    """Lower convex hull of (A, T): the points a uniform price can select."""
    pts = d[["achieved_abatement", "total_cost"]].to_numpy(float)
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(tuple(p))
    h = pd.DataFrame(hull, columns=["A", "T"])
    # keep the part a non-negative price can reach (from the cost minimum upwards)
    h = h.iloc[int(h["T"].idxmin()):].reset_index(drop=True)
    return h


def _price_to_reach(h: pd.DataFrame, a: float) -> float:
    """Lowest uniform price at which the least-cost response reaches at least `a` (slope of the hull segment)."""
    if a <= h["A"].iloc[0]:
        return 0.0
    if a > h["A"].iloc[-1] + 1e-6:
        return np.nan
    k = int(np.searchsorted(h["A"].to_numpy(), a - 1e-6))
    return float((h["T"].iloc[k] - h["T"].iloc[k - 1]) / (h["A"].iloc[k] - h["A"].iloc[k - 1]))


def _response(h: pd.DataFrame, p: float):
    """A*(p) and T at A*(p) (largest abatement among ties)."""
    v = h["T"] - p * h["A"]
    k = int(np.flatnonzero(np.isclose(v, v.min(), rtol=0, atol=1e3))[-1])
    return float(h["A"].iloc[k]), float(h["T"].iloc[k])


def carbon_price_tables():
    hulls = {r: _hull(_curve(r)) for r in INTERACTION_RUNS}
    rows = []
    for g in REPORT_GT:
        a = g * 1e9
        row = {"abatement_gt": g}
        for r in INTERACTION_RUNS:
            p = _price_to_reach(hulls[r], a)
            row[f"{r}_price_usd_t"] = p
            row[f"{r}_transfer_bn"] = p * a / 1e9 if np.isfinite(p) else np.nan
        rows.append(row)
    price = pd.DataFrame(rows)
    ref = []
    for lab, p in REFERENCE_PRICES.items():
        row = {"reference": lab, "price_usd_t": p}
        for r in INTERACTION_RUNS:
            a, t = _response(hulls[r], p)
            row[f"{r}_abatement_gt"] = a / 1e9
            row[f"{r}_transfer_bn"] = p * a / 1e9
            row[f"{r}_resource_cost_bn"] = t / 1e9
        ref.append(row)
    return price, pd.DataFrame(ref)


def _plant_sums(name: str) -> pd.DataFrame:
    x = pd.ExcelFile(os.path.join(SET_DIR, f"gurobi_r2_{name}_results.xlsx"))
    rows = []
    for s in x.sheet_names:
        if not s.startswith("pm"):
            continue
        p = x.parse(s)
        if p.empty:
            continue
        g = lambda c: float(pd.to_numeric(p[c], errors="coerce").fillna(0).sum()) if c in p.columns else 0.0
        rows.append({
            "target_id": int(p["target_id"].iloc[0]),
            "bc_subsidy_bn": g("bc_subsidy_credit") / 1e9,
            "cc_subsidy_bn": g("cc_subsidy_credit") / 1e9,
            "biomass_electricity_twh": float((pd.to_numeric(p["generation_mwh"], errors="coerce").fillna(0)
                                              * pd.to_numeric(p["effective_blend_ratio"], errors="coerce").fillna(0)).sum()) / 1e6,
            "co2_captured_gt": g("cc_capture_ton") / 1e9,
            "plant_total_cost_bn": g("total_cost") / 1e9,
        })
    return pd.DataFrame(rows)


def subsidy_tables():
    base = _curve("base")
    base_sums = _plant_sums("base").set_index("target_id")
    rows, free = [], []
    for run, lab in SUBSIDY_RUNS.items():
        c = _curve(run)
        sums = _plant_sums(run).set_index("target_id")
        m = c.merge(sums, left_on="target_id", right_index=True, how="inner")
        m["spending_bn"] = m["bc_subsidy_bn"] + m["cc_subsidy_bn"]
        m["private_cost_bn"] = m["total_cost"] / 1e9
        m["resource_cost_bn"] = m["private_cost_bn"] + m["spending_bn"]
        for g in REPORT_GT:
            a = g * 1e9
            if a > m["achieved_abatement"].max():
                continue
            ip = lambda col, d=m: float(np.interp(a, d["achieved_abatement"], d[col]))
            bt = float(np.interp(a, base["achieved_abatement"], base["total_cost"])) / 1e9
            rows.append({
                "scenario": run, "instrument": lab, "abatement_gt": g,
                "spending_bn": ip("spending_bn"), "private_cost_bn": ip("private_cost_bn"),
                "resource_cost_bn": ip("resource_cost_bn"), "base_resource_cost_bn": bt,
                "resource_cost_change_bn": ip("resource_cost_bn") - bt,
                "private_cost_reduction_bn": bt - ip("private_cost_bn"),
                "biomass_electricity_twh": ip("biomass_electricity_twh"),
                "base_biomass_electricity_twh": float(np.interp(a, base["achieved_abatement"].iloc[1:],
                                                                base_sums.loc[base["target_id"].iloc[1:], "biomass_electricity_twh"])),
                "co2_captured_gt": ip("co2_captured_gt"),
                "base_co2_captured_gt": float(np.interp(a, base["achieved_abatement"].iloc[1:],
                                                        base_sums.loc[base["target_id"].iloc[1:], "co2_captured_gt"])),
            })
        # no target: least private cost on the subsidy curve (what the subsidy alone draws in)
        k = int(m["private_cost_bn"].idxmin())
        k0 = int(base["total_cost"].idxmin())
        free.append({
            "scenario": run, "instrument": lab,
            "abatement_gt": m.loc[k, "achieved_abatement"] / 1e9, "spending_bn": m.loc[k, "spending_bn"],
            "private_cost_bn": m.loc[k, "private_cost_bn"], "resource_cost_bn": m.loc[k, "resource_cost_bn"],
            "biomass_electricity_twh": m.loc[k, "biomass_electricity_twh"], "co2_captured_gt": m.loc[k, "co2_captured_gt"],
            "base_abatement_gt": base.loc[k0, "achieved_abatement"] / 1e9, "base_cost_bn": base.loc[k0, "total_cost"] / 1e9,
        })
    return pd.DataFrame(rows), pd.DataFrame(free)


def main() -> None:
    price, ref = carbon_price_tables()
    sub, free = subsidy_tables()
    os.makedirs(os.path.dirname(OUT_XLSX), exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as w:
        price.to_excel(w, sheet_name="carbon_price", index=False)
        ref.to_excel(w, sheet_name="carbon_price_reference", index=False)
        sub.to_excel(w, sheet_name="subsidy_targets", index=False)
        free.to_excel(w, sheet_name="subsidy_free_response", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("Uniform price needed to reach A (USD/t) and transfer p*A (bn USD):")
    print(price.round(2).to_string(index=False))
    print("\nResponse at reference prices:")
    print(ref.round(2).to_string(index=False))
    print("\nUniform subsidies at fixed targets:")
    print(sub.round(2).to_string(index=False))
    print("\nUniform subsidies without a target (least private cost):")
    print(free.round(2).to_string(index=False))
    print(f"Saved: {OUT_XLSX}")


if __name__ == "__main__":
    main()
