#!/bin/bash
# Full reproduction: EC packages -> fleet scenarios -> technology-ordering test -> figures.
#   bash run_all.sh                  (2 scenarios at a time, 4 Gurobi threads each)
#   JOBS=3 THREADS=4 bash run_all.sh
# Restartable: finished scenarios (curve file present) are skipped.
set -e
cd "$(dirname "$0")"
JOBS=${JOBS:-2}
THREADS=${THREADS:-4}
export ABATEMENT_ACCOUNTING=avoided
mkdir -p logs

# Step 1: plant-level EC packages (Methods Eqs. 1-6; Supplementary Note 4) -> data/ec_packages_r2.csv
python3 -u optimization_ec_subtech_binary.py --constraint-mode per_plant --target-ratios 0.15 0.175 0.2 \
  --overlap-coefs 0 0.25 0.5 --output-prefix ec_pkg_r2_all > logs/step1_ec_all.log 2>&1
python3 -u optimization_ec_subtech_binary.py --constraint-mode per_plant --target-ratios 0.1 --overlap-coefs 0 \
  --exclude-techs 5 --output-prefix ec_pkg_r2_no5 > logs/step1_ec_no5.log 2>&1
python3 -u optimization_ec_subtech_binary.py --constraint-mode per_plant --target-ratios 0.1 --overlap-coefs 0 \
  --output-prefix ec_r2_fixed_pp > logs/step1_ec_with5.log 2>&1
python3 -u build_ec_packages.py > logs/step1_build_packages.log 2>&1

# Step 2: fleet-wide scenarios (Supplementary Table S1) -> data/gurobi_r2_<scenario>_curve.csv / _results.xlsx
python3 -u run_all_scenarios.py --accounting avoided --jobs "$JOBS" --threads "$THREADS" > logs/step2_scenarios.log 2>&1
python3 -u run_all_scenarios.py --accounting avoided --jobs "$JOBS" --threads "$THREADS" --rerun-incomplete > logs/step2_fill.log 2>&1
python3 check_results.py > logs/CHECK_RESULTS.txt 2>&1

# Step 3: technology-ordering test (Supplementary Fig. S18)
python3 -u analysis_sequence_falsification.py --targets-from-curve gurobi_r2_base_curve.csv \
  --solver-threads $((JOBS * THREADS)) > logs/step3_sequence.log 2>&1

# Step 4: analyses and figures -> figures/
python3 make_figures.py > logs/step4_figures.log 2>&1
echo "done $(date)"
