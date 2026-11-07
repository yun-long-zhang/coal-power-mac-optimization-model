"""Step 4: all analyses, figures, and source-data tables of the paper from the scenario results.

  python3 make_figures.py                       # results in data/ (written by run_all_scenarios.py)
  python3 make_figures.py --results-dir PATH    # results kept in another folder (linked into data/ for the run)

Each analysis and plotting script is run in turn; their outputs (PNG/PDF figures and the XLSX/CSV data behind
each figure) are collected in figures/. A log is written to figures/make_figures.log.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

STEPS = [
    ["analysis_carbon_price_response.py", "--output-prefix", "carbon_price_response_main"],
    ["analysis_uac_interaction_decomposition.py"],
    ["analysis_interaction_cost_gap.py"],
    ["analysis_standalone_pairwise.py"],
    ["analysis_subsidy_policy_r2.py"],
    ["plot_standalone_mac.py"],
    ["plot_interaction_isolation.py"],
    ["plot_ec20_comparison.py"],
    ["plot_accounting_sensitivity.py"],
    ["plot_2_paper_composite_figures.py"],
    ["plot_1_fleet_joint_macc.py", "--input-prefix", "gurobi_r2_base", "--plot-state-evolution"],
    ["plot_lcoe.py"],
    ["plot_macc_retirement_scenarios.py"],
    ["plot_mac_subsidy_scenarios.py"],
    ["plot_sequence_falsification.py"],
    ["plot_carbon_price_response.py"],
    ["plot_decomposition_bc_cc_combined.py"],
    ["plot_interaction_cost_gap_provinces.py"],
    ["plot_fig1_fig2_standalone_pairwise.py"],
    ["plot_ec_packages_s9_s10.py"],
    ["plot_s24_ec_db_investment.py"],
]

OUTPUT_PATTERNS = [
    "paper_composite_1_sensitivity_nodes.*", "paper_composite_2_uncertainty_abc.*", "fig_scenario_comparison.*",
    "fig_lcoe_distribution.*", "gurobi_r2_lcoe_*", "fig_macc_retirement_scenarios.*", "fig_mac_subsidy_scenarios.*",
    "fig_sequence_falsification.*", "fig_carbon_price_response.*", "carbon_price_response_main_*",
    "fig_decomp_4panel.*", "uac_interaction_decomp_*", "interaction_cost_gap_r2.xlsx",
    "fig_interaction_cost_gap_provinces.*", "standalone_mac_*", "fig_standalone_mac*", "fleet_joint_*",
    "fig_accounting_sensitivity.*", "fig_ec20_comparison.*", "fig1_standalone_uac_r2.*", "fig2_pairwise_interactions_r2.*", "standalone_pairwise_r2.csv",
    "fig_s9_ec_category_contribution.*", "fig_s10_ec_package_uac.*", "fig_s24_ec_db_investment.*",
]


def _is_result_file(name: str) -> bool:
    return name.startswith("sequence_falsification_") or (
        name.startswith("gurobi_r2_") and (name.endswith("_curve.csv") or name.endswith("_results.xlsx")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=DATA, help="Folder with the gurobi_r2_* scenario results (default: data/).")
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "figures"), help="Output folder (default: figures/).")
    args = ap.parse_args()
    results = os.path.abspath(args.results_dir)
    fig_dir = os.path.abspath(args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    linked = []
    if results != DATA:  # make external results visible to the scripts, which read data/
        for path in glob.glob(os.path.join(results, "*")):
            name = os.path.basename(path)
            target = os.path.join(DATA, name)
            if _is_result_file(name) and not os.path.exists(target):
                os.symlink(path, target)
                linked.append(target)

    env = dict(os.environ, R2_SET="ec10_avoided", R2_SET_DIR=DATA, R2_FIG_DIR=fig_dir, MPLBACKEND="Agg")
    log_path = os.path.join(fig_dir, "make_figures.log")
    failed = []
    try:
        with open(log_path, "w") as log:
            for step in STEPS:
                log.write(f"\n===== {' '.join(step)}\n")
                log.flush()
                rc = subprocess.run([sys.executable, *step], cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                print(f"{'ok    ' if rc == 0 else 'FAILED'} {step[0]}")
                if rc != 0:
                    failed.append(step[0])
    finally:
        for target in linked:
            os.remove(target)
    moved = 0
    for pattern in OUTPUT_PATTERNS:
        for path in glob.glob(os.path.join(DATA, pattern)):
            if not os.path.isdir(path):
                shutil.move(path, os.path.join(fig_dir, os.path.basename(path)))
                moved += 1
    for d in ["fleet_joint_scenario_sorted_macc"]:  # per-scenario node figures written as a folder
        src = os.path.join(DATA, d)
        if os.path.isdir(src):
            dst = os.path.join(fig_dir, d)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.move(src, dst)
    print(f"{moved} output files -> {fig_dir}")
    if failed:
        print(f"Failed steps (see {log_path}): {failed}")


if __name__ == "__main__":
    main()
