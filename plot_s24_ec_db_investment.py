"""
Supplementary Fig. S24: unit abatement cost, initial investment, and simple payback period of energy
conservation (EC) and direct biomass co-firing (DB) at each plant, for the 10% and 20% reduction levels.

  UAC        from the standalone assessment (standalone_pairwise_r2.csv, net of avoided coal cost)
  CAPEX EC   initial investment of the plant's EC package. The model uses the annualized investment of each
             measure k, A_k = I_k x CRF_k with CRF_k = r / (1 - (1 + r)^-ST_k), r = 8% and
             ST_k = min(remaining plant life, service life of k) (Supplementary Eqs. S2-S4; remaining life
             = 40 yr - age in 2020). I_k = A_k / CRF_k; the package investment scales the package's annualized
             investment by the ratio sum(I_k) / sum(A_k) of its selected measures.
  CAPEX DB   1,153 RMB per kW of capacity and unit co-firing ratio x ratio x capacity (Table S2)
  Payback    initial investment / annual net cash saving
             EC: avoided coal cost of the package
             DB: avoided coal cost - biomass feedstock and transport (nearest hubs first) - fixed O&M (2% of investment)
             (only for plants with a positive annual saving)

Panels: (a) UAC of DB vs EC; (b) investment of DB vs EC and (c) payback period of DB vs EC, for plants at
which both options have a negative UAC. Outputs fig_s24_ec_db_investment.png/.xlsx in data/.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis_standalone_pairwise as sp
from plot_export_utils import save_figure_bundle

base = sp.base
DATA = sp.DATA_DIR
SET_DIR = os.environ.get("R2_SET_DIR", os.path.join(DATA, "results_r2_ec10_avoided"))
# plant-level standalone results written by analysis_standalone_pairwise.py (data/ during a figure run)
PAIRWISE = next((p for p in (os.path.join(DATA, "standalone_pairwise_r2.csv"),
                             os.path.join(os.environ.get("R2_FIG_DIR", os.path.join(SET_DIR, "figures")), "standalone_pairwise_r2.csv"))
                 if os.path.exists(p)), os.path.join(DATA, "standalone_pairwise_r2.csv"))
OUT_PNG = os.path.join(DATA, "fig_s24_ec_db_investment.png")
R_EC = 0.08
PLANT_LIFE = 40.0
BASE_YEAR = 2020
# service life of the EC measures (yr, Supplementary Table S3)
SERVICE_LIFE = {1: 30, 2: 30, 3: 10, 4: 15, 5: 30, 6: 30, 7: 15, 8: 30, 9: 20, 10: 5, 11: 30, 12: 15, 13: 9, 14: 30,
                15: 20, 16: 20, 17: 30, 18: 30, 19: 10, 20: 30}
PACKAGES = {"10": "no5_r10_o00", "20": "r20_o00"}
COLORS = {"10": "#9ecae1", "20": "#2171b5"}


def _ec_capex(plants: pd.DataFrame) -> pd.DataFrame:
    wb = os.path.join(DATA, base.INPUT_WORKBOOK_NAME)
    ec = pd.read_excel(wb, sheet_name="EC")
    cp = pd.read_excel(wb, sheet_name="coal_plant_i")[["plant_i", "online_year"]]
    ec = ec.merge(cp, on="plant_i")
    ec["plant_id"] = ec["plant_i"].map(base._norm_id)
    rl = (PLANT_LIFE - (BASE_YEAR - ec["online_year"])).clip(lower=1.0)
    ann, inv = {}, {}
    for k, sl in SERVICE_LIFE.items():
        a = pd.to_numeric(ec[f"tech_{k}"], errors="coerce").fillna(0) * pd.to_numeric(ec[f"tech_{k}.1"], errors="coerce").fillna(0)
        st = np.minimum(rl, sl)
        crf = R_EC / (1.0 - (1.0 + R_EC) ** (-st))
        ann[k] = pd.Series(a.values, index=ec["plant_id"])
        inv[k] = pd.Series((a / crf).values, index=ec["plant_id"])
    pk = pd.read_csv(sp.EC_PACKAGE_FILE, dtype={"plant_id": str})
    rows = []
    for lvl, name in PACKAGES.items():
        sub = pk[pk["package"] == name]
        for _, r in sub.iterrows():
            pid = base._norm_id(r["plant_id"])
            ks = [int(t.split("_")[1]) for t in str(r["selected_tech_id"]).split(",") if t.startswith("TECH_")]
            a = sum(ann[k].get(pid, 0.0) for k in ks)
            i = sum(inv[k].get(pid, 0.0) for k in ks)
            ratio = i / a if a > 0 else np.nan
            rows.append({"plant_id": pid, "level": lvl, "ec_capex": float(r["ec_gross_cost"]) * ratio,
                         "ec_saving": float(r["ec_coal_saving"]), "ec_annualized_inv": float(r["ec_gross_cost"])})
    return pd.DataFrame(rows)


def _db(plants: pd.DataFrame, md: dict) -> pd.DataFrame:
    wb = os.path.join(DATA, base.INPUT_WORKBOOK_NAME)
    hubs = pd.read_excel(wb, sheet_name="biomass_point_j")
    hubs = pd.DataFrame({"biomass_id": hubs["biomass_point_j"].map(base._norm_id), "supply_gj": hubs["bioenergy_GJ"].astype(float)})
    dji = pd.read_csv(os.path.join(DATA, base.INPUT_BIOMASS_DISTANCE_NAME))
    arcs = pd.DataFrame({"biomass_id": dji["biomass_j"].map(base._norm_id), "plant_id": dji["plant_i"].map(base._norm_id), "d": dji["Dji"].astype(float)})
    arcs = arcs.merge(hubs, on="biomass_id")
    arcs["unit_cost"] = base.BIOMASS_FEEDSTOCK_COST_PER_GJ + base.TRANSPORT_COST_PER_GJ_KM * arcs["d"]
    rows = []
    for theta in sp.THETAS:
        lvl = f"{int(theta * 100)}"
        bio = theta * plants["P_gj"]
        supply = sp._nearest_fill(arcs, bio, "supply_gj", "unit_cost")
        capex = sp.BC_INV_DB_RMB_PER_MW * theta * plants["X_mw"] / base.RMB_PER_USD
        saving = plants["coal_price"] * bio - supply - base.BC_FIXED_OM_SHARE * capex
        rows.append(pd.DataFrame({"plant_id": plants.index, "level": lvl, "db_capex": capex.values, "db_saving": saving.values}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    md = sp._model_data()
    plants = sp._plants(md)
    df = _ec_capex(plants).merge(_db(plants, md), on=["plant_id", "level"])
    pw = pd.read_csv(PAIRWISE, dtype={"plant_id": str})
    pw["plant_id"] = pw["plant_id"].map(base._norm_id)
    uac = pd.concat([pd.DataFrame({"plant_id": pw["plant_id"], "level": l, "ec_uac": pw[f"EC{l}_uac"], "db_uac": pw[f"DB{l}_uac"]}) for l in PACKAGES])
    df = df.merge(uac, on=["plant_id", "level"])
    cap = pd.Series(plants["X_mw"].values, index=plants.index)
    df["capacity_mw"] = df["plant_id"].map(cap)
    df["ec_pp"] = np.where(df["ec_saving"] > 0, df["ec_capex"] / df["ec_saving"], np.nan)
    df["db_pp"] = np.where(df["db_saving"] > 0, df["db_capex"] / df["db_saving"], np.nan)
    df["both_negative"] = (df["ec_uac"] < 0) & (df["db_uac"] < 0)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    ax = axes[0]
    for l in PACKAGES:
        s = df[df["level"] == l]
        ax.scatter(s["ec_uac"], s["db_uac"], s=10, facecolors="none", edgecolors=COLORS[l], linewidths=0.6, label=f"{l}% reduction")
    lim = [min(df["ec_uac"].min(), df["db_uac"].min()) - 2, max(df["ec_uac"].max(), df["db_uac"].max()) + 2]
    ax.plot(lim, lim, color="black", lw=1.2)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.fill_between([lim[0], 0], lim[0], 0, color="#fde0dd", alpha=0.5, zorder=0, label="both negative")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.text(lim[1] * 0.55, lim[1] * 0.55 - 6, "x = y", fontsize=10)
    ax.set_xlabel("UAC of EC (USD t CO$_2^{-1}$)")
    ax.set_ylabel("UAC of DB (USD t CO$_2^{-1}$)")
    ax.set_title("(a) Unit abatement cost", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    pos = df[df["both_negative"]]
    for ax, (xc, yc, scale, lab, unit, ttl) in zip(axes[1:], [
        ("ec_capex", "db_capex", 1e6, "Initial investment", "million USD", "(b) Initial investment"),
        ("ec_pp", "db_pp", 1.0, "Simple payback period", "years", "(c) Simple payback period"),
    ]):
        for l in PACKAGES:
            s = pos[pos["level"] == l]
            ax.scatter(s[xc] / scale, s[yc] / scale, s=10, facecolors="none", edgecolors=COLORS[l], linewidths=0.6, label=f"{l}% reduction")
        hi = np.nanmax([pos[xc].max(), pos[yc].max()]) / scale * 1.05
        ax.plot([0, hi], [0, hi], color="black", lw=1.2)
        ax.text(hi * 0.8, hi * 0.72, "x = y", fontsize=10)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel(f"{lab} of EC ({unit})")
        ax.set_ylabel(f"{lab} of DB ({unit})")
        ax.set_title(ttl, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for a in axes:
        a.grid(alpha=0.2, ls="--")
    fig.tight_layout()

    rows = []
    for l in PACKAGES:
        s = df[df["level"] == l]
        p = s[s["both_negative"]]
        w = s["capacity_mw"]
        rows.append({
            "level": l, "plants": len(s),
            "share_plants_ec_uac_below_db": float((s["ec_uac"] < s["db_uac"]).mean()),
            "share_capacity_db_uac_below_ec": float(w[s["db_uac"] < s["ec_uac"]].sum() / w.sum()),
            "share_capacity_negative_either": float(w[(s["ec_uac"] < 0) | (s["db_uac"] < 0)].sum() / w.sum()),
            "plants_both_negative": len(p),
            "share_both_negative_db_capex_below_ec": float((p["db_capex"] < p["ec_capex"]).mean()),
            "share_both_negative_ec_pp_below_db": float((p["ec_pp"] < p["db_pp"]).mean()),
            "median_ec_capex_musd": p["ec_capex"].median() / 1e6, "median_db_capex_musd": p["db_capex"].median() / 1e6,
            "median_ec_pp_yr": p["ec_pp"].median(), "median_db_pp_yr": p["db_pp"].median(),
        })
    summary = pd.DataFrame(rows)
    save_figure_bundle(fig, OUT_PNG, dpi=300, bbox_inches="tight", data_frames={"summary": summary, "plants": df})
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(summary.round(3).T.to_string())
    # check: summed annualized cost of the selected measures vs the package's annualized investment
    print("Saved:", OUT_PNG)


if __name__ == "__main__":
    main()
