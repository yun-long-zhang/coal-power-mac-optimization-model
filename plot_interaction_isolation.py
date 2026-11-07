"""Isolating the interaction effect (reply to R2-5).

Compares, at equal abatement, the total annual cost of
  baseline (joint model)                     gurobi_r2_base
  no interactions (same model, s_i = 0 and no BC-CC coupling)  gurobi_r2_no_interaction
  no EC interactions only                    gurobi_r2_no_ec_interaction
  no BC-CC interaction only                  gurobi_r2_no_bccc_interaction (optional)
  standalone technology ranking              figures/standalone_mac_curve.csv
and splits baseline - standalone into
  interaction effect      = baseline - no interactions
  construction/constraints = no interactions - standalone.

  python3 plot_interaction_isolation.py --set ec10
Outputs go to data/results_r2_<set>/figures/.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_export_utils import save_figure_bundle

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REPORT_GT = [1.0, 2.0, 3.0, 3.5, 4.0, 4.5]


def _curve(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path).dropna(subset=["total_cost"])
    d = d[["achieved_abatement", "total_cost"]].sort_values("achieved_abatement")
    return pd.concat([pd.DataFrame({"achieved_abatement": [0.0], "total_cost": [0.0]}), d], ignore_index=True)


def _at(d: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    y = np.interp(x, d["achieved_abatement"], d["total_cost"], right=np.nan)
    y[x > d["achieved_abatement"].max()] = np.nan
    return y / 1e9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["ec10", "ec20", "ec10_avoided"], default=os.environ.get("R2_SET", "ec10"))
    args = parser.parse_args()
    set_dir = os.environ.get("R2_SET_DIR") or os.path.join(DATA, f"results_r2_{args.set}")
    fig_dir = os.environ.get("R2_FIG_DIR") or os.path.join(set_dir, "figures")
    ec_label = "EC 20%" if args.set == "ec20" else "EC 10%"

    runs = {
        "base": _curve(os.path.join(set_dir, "gurobi_r2_base_curve.csv")),
        "no_interaction": _curve(os.path.join(set_dir, "gurobi_r2_no_interaction_curve.csv")),
        "no_ec_interaction": _curve(os.path.join(set_dir, "gurobi_r2_no_ec_interaction_curve.csv")),
        "no_bccc_interaction": _curve(os.path.join(set_dir, "gurobi_r2_no_bccc_interaction_curve.csv")),
        # during make_r2_figures.py the fresh standalone curve is still in data/; afterwards it is in figures/
        "standalone": _curve(next(p for p in (os.path.join(DATA, "standalone_mac_curve.csv"),
                                              os.path.join(fig_dir, "standalone_mac_curve.csv")) if os.path.exists(p))),
    }
    have_bccc = runs["no_bccc_interaction"] is not None

    x_max = runs["base"]["achieved_abatement"].max()
    x = np.linspace(0.0, x_max, 400)
    c = {k: _at(v, x) for k, v in runs.items() if v is not None}
    interaction = c["base"] - c["no_interaction"]
    construction = c["no_interaction"] - c["standalone"]
    ec_channel = c["base"] - c["no_ec_interaction"]
    bccc_channel = c["base"] - c["no_bccc_interaction"] if have_bccc else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = axes[0]
    styles = [
        ("base", f"Baseline (joint model, {ec_label})", "#1F1F1F", "-", 2.4),
        ("no_interaction", "No interactions (joint model)", "#E6862A", "--", 1.8),
        ("no_ec_interaction", "No EC interactions (joint model)", "#3BA03B", "--", 1.4),
        ("no_bccc_interaction", "No BC–CC interaction (joint model)", "#7B5BB5", ":", 1.6),
        ("standalone", "Standalone technology ranking", "#D62728", "-.", 1.6),
    ]
    for key, lab, col, ls, lw in styles:
        if key not in runs or runs[key] is None:
            continue
        d = runs[key]
        ax.plot(d["achieved_abatement"] / 1e9, d["total_cost"] / 1e9, color=col, ls=ls, lw=lw, label=lab)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("Abatement (Gt CO$_2$ yr$^{-1}$)")
    ax.set_ylabel("Total annual abatement cost (billion USD yr$^{-1}$)")
    ax.set_title("(a) Total cost at equal abatement", loc="left")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    xg = x / 1e9
    ax.plot(xg, c["base"] - c["standalone"], color="#1F1F1F", lw=2.2, label="Baseline − standalone (total gap)")
    ax.plot(xg, interaction, color="#E6862A", lw=1.8, ls="--", label="Interaction effect (baseline − no interactions)")
    ax.plot(xg, construction, color="#7F7F7F", lw=1.4, ls="-.", label="Curve construction (no interactions − standalone)")
    ax.plot(xg, ec_channel, color="#3BA03B", lw=1.2, ls=":", label="EC channel (baseline − no EC interactions)")
    if have_bccc:
        ax.plot(xg, bccc_channel, color="#7B5BB5", lw=1.2, ls=":", label="BC–CC channel (baseline − no BC–CC interaction)")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("Abatement (Gt CO$_2$ yr$^{-1}$)")
    ax.set_ylabel("Cost difference at equal abatement (billion USD yr$^{-1}$)")
    ax.set_title("(b) Decomposition of the gap to the standalone benchmark", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()

    rows = []
    for g in REPORT_GT:
        xi = np.array([g * 1e9])
        v = {k: float(_at(d, xi)[0]) for k, d in runs.items() if d is not None}
        row = {"abatement_gt": g, **{f"cost_{k}_bn": v.get(k, np.nan) for k in runs}}
        row["interaction_effect_bn"] = v["base"] - v["no_interaction"]
        row["construction_effect_bn"] = v["no_interaction"] - v["standalone"]
        row["total_gap_bn"] = v["base"] - v["standalone"]
        row["ec_channel_bn"] = v["base"] - v["no_ec_interaction"]
        row["bccc_channel_bn"] = v["base"] - v["no_bccc_interaction"] if have_bccc else np.nan
        row["interaction_effect_pct_of_baseline"] = 100 * row["interaction_effect_bn"] / v["base"] if v["base"] > 0 else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    maxima = pd.DataFrame(
        [{"run": k, "max_abatement_gt": d["achieved_abatement"].max() / 1e9} for k, d in runs.items() if d is not None]
    )

    out = os.path.join(fig_dir, "fig_interaction_isolation.png")
    save_figure_bundle(fig, out, dpi=300, bbox_inches="tight", data_frames={"at_targets": table, "max_abatement": maxima})
    plt.close(fig)
    pd.set_option("display.width", 200)
    print(table.round(2).to_string(index=False))
    print(maxima.round(3).to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
