"""Supplementary Fig. S27: the comparison of Fig. 4a repeated with the EC package of at least 20% CO2 reduction.

The joint model with the 20% package (gurobi_r2_ec_r20_o00), the joint model without EC (gurobi_r2_no_ec), and the
standalone technology ranking built with the 20% package (same BC-only and CC-only results as for Fig. 4a) are
plotted in the style of Fig. 4a. The cost gap between the joint model and the standalone ranking is listed at
equal abatement.

  python3 plot_ec20_comparison.py
Inputs are read from R2_SET_DIR (default data/); outputs fig_ec20_comparison.{png,pdf,xlsx} go to R2_FIG_DIR
(default data/).
"""

import os
import shutil
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_2_paper_composite_figures as p2
import plot_standalone_mac as psm
from plot_export_utils import save_figure_bundle

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SET_DIR = os.environ.get("R2_SET_DIR") or DATA
FIG_DIR = os.environ.get("R2_FIG_DIR") or DATA
EC_PACKAGE = "r20_o00"
BASE_CURVE = "gurobi_r2_ec_r20_o00_curve.csv"
NO_EC_CURVE = "gurobi_r2_no_ec_curve.csv"
REPORT_GT = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]


def _cost_at(curve: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    d = curve.dropna(subset=["total_cost"]).sort_values("achieved_abatement")
    xs = np.r_[0.0, d["achieved_abatement"].to_numpy()]
    ys = np.r_[0.0, d["total_cost"].to_numpy()]
    y = np.interp(x, xs, ys, right=np.nan)
    y[x > xs.max()] = np.nan
    return y


def main() -> None:
    psm.EC_PACKAGE = EC_PACKAGE
    units = pd.concat([psm.load_ec_data(), psm.load_bc_data(), psm.load_cc_data()], ignore_index=True)
    standalone, _ = psm.build_mac_curve(units, step=psm.TARGET_STEP)
    raw, _ = psm.build_mac_curve(units, step=psm.TARGET_STEP, raw=True)
    base = pd.read_csv(os.path.join(SET_DIR, BASE_CURVE))

    with tempfile.TemporaryDirectory() as tmp:  # file names expected by the Fig. 4a panel
        shutil.copy(os.path.join(SET_DIR, BASE_CURVE), os.path.join(tmp, "gurobi_r2_base_curve.csv"))
        shutil.copy(os.path.join(SET_DIR, NO_EC_CURVE), os.path.join(tmp, NO_EC_CURVE))
        raw.to_csv(os.path.join(tmp, "standalone_mac_raw_units.csv"), index=False)
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        p2._plot_mac_comparison_panel(ax, tmp)
    labels = {"Baseline (joint model)": "Joint model, EC 20%",
              "Standalone technology ranking": "Standalone technology ranking, EC 20%"}
    for t in ax.get_legend().get_texts():
        t.set_text(labels.get(t.get_text(), t.get_text()))
    ax.set_title("MAC curves with the 20% EC package", fontsize=13)

    x = np.array(REPORT_GT) * 1e9
    joint, alone = _cost_at(base, x), _cost_at(standalone, x)
    gap = pd.DataFrame({
        "abatement_Gt": REPORT_GT,
        "joint_cost_bn": joint / 1e9,
        "standalone_cost_bn": alone / 1e9,
        "gap_bn": (joint - alone) / 1e9,
        "gap_pct_of_joint": 100 * (joint - alone) / np.abs(joint),
    })
    max_joint = base["achieved_abatement"].max()
    max_alone = standalone["achieved_abatement"].max()
    summary = pd.DataFrame({
        "item": ["max abatement joint (Gt)", "max abatement standalone (Gt)", "standalone overestimate (%)"],
        "value": [max_joint / 1e9, max_alone / 1e9, 100 * (max_alone / max_joint - 1)],
    })
    os.makedirs(FIG_DIR, exist_ok=True)
    save_figure_bundle(fig, os.path.join(FIG_DIR, "fig_ec20_comparison.png"), dpi=320,
                       data_frames={"cost_gap": gap, "max_abatement": summary, "standalone_ec20": standalone})
    plt.close(fig)
    print(gap.round(2).to_string(index=False))
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
