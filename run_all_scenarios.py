import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import pandas as pd


# ---- User config (edit before running) ----
MODEL_SCRIPT = "optimization_1_fleet_joint_milp_persistent.py"
DATA_DIR_NAME = "data"
LOG_DIR_NAME = "logs_r2"
OUTPUT_PREFIX = "gurobi_r2"
DEFAULT_JOBS = 3
DEFAULT_THREADS_PER_JOB = 4
# -------------------------------------------

# Scenario name -> extra arguments for the fleet model. Output files: data/{OUTPUT_PREFIX}_{name}_curve.csv / _results.xlsx
SCENARIOS: Dict[str, List[str]] = {
    "base": [],
    # Energy conservation packages (Fig. 4d); the baseline uses the 10% package without technology No. 5
    "ec_r20_o00": ["--ec-package", "r20_o00"],
    "ec_r175_o00": ["--ec-package", "r175_o00"],
    "ec_r15_o00": ["--ec-package", "r15_o00"],
    "ec_r15_o25": ["--ec-package", "r15_o25"],
    "ec_r15_o50": ["--ec-package", "r15_o50"],
    "ec_r10_with5": ["--ec-package", "r10_o00"],
    # Model structure (Fig. 4a, interaction effect)
    "no_ec": ["--disable-ec"],
    "no_ec_interaction": ["--disable-ec-interactions"],
    "no_interaction": ["--disable-ec-interactions", "--disable-bc-cc-interaction"],
    "no_bccc_interaction": ["--disable-bc-cc-interaction"],
    # Capture energy (R2-1): turbine-extraction pricing, and with the CO2 of the extra coal counted
    # (run with the cost-workbook energy values 3.5 GJ/t and 0.23 MWh/t)
    "cc_ext_cost": ["--cc-energy-basis", "extraction", "--cc-ext-steam-gj-per-t", "3.5", "--cc-ext-power-mwh-per-t", "0.23"],
    "cc_ext_penalty": ["--cc-energy-basis", "extraction", "--cc-penalty-emissions", "--cc-ext-steam-gj-per-t", "3.5", "--cc-ext-power-mwh-per-t", "0.23"],
    # Economic parameters (Fig. 4b)
    "coal_m25": ["--coal-factor", "0.75"],
    "coal_p25": ["--coal-factor", "1.25"],
    "af5": ["--discount-case", "af5"],
    "af12": ["--discount-case", "af12"],
    "capex_0p70": ["--cc-capex-factor", "0.7"],
    "capex_1p30": ["--cc-capex-factor", "1.3"],
    # biomass co-firing investment, direct and gasified co-firing together (R2-4)
    "bc_capex_0p70": ["--bc-capex-factor", "0.7"],
    "bc_capex_1p30": ["--bc-capex-factor", "1.3"],
    # Resources and engineering (Fig. 4c)
    "bio_supply_25": ["--biomass-supply-factor", "0.25"],
    "bio_supply_50": ["--biomass-supply-factor", "0.5"],
    "bio_supply_75": ["--biomass-supply-factor", "0.75"],
    "cc_rate_95": ["--cc-capture-rate", "0.95", "--n-targets", "52"],
    "cc_rate_99p7": ["--cc-capture-rate", "0.997", "--n-targets", "52"],
    # Subsidies (Fig. S16)
    "subsidy_bc_0p15": ["--bc-subsidy-rmb-per-kwh", "0.15"],
    "subsidy_bc_0p25": ["--bc-subsidy-rmb-per-kwh", "0.25"],
    "subsidy_cc_35": ["--cc-subsidy-rmb-per-ton", "35"],
    "subsidy_cc_105": ["--cc-subsidy-rmb-per-ton", "105"],
}
# Time slices with plant retirement (Fig. S17)
for _age in (30, 40):
    for _year in (2030, 2035, 2040, 2045):
        SCENARIOS[f"t{_age}_{_year}"] = [
            "--scenario-year", str(_year), "--retirement-age", str(_age), "--target-step", "1e8",
        ]


# Second result set: 20% EC baseline (package r20_o00). Scenarios that only depend on the EC package
# (EC sensitivities, no_ec) are shared with the 10% set and are not rerun; the base curve is ec_r20_o00.
EC20_SHARED = {"base", "no_ec", "ec_r20_o00", "ec_r175_o00", "ec_r15_o00", "ec_r15_o25", "ec_r15_o50", "ec_r10_with5"}
SCENARIOS_EC20: Dict[str, List[str]] = {
    k: v + ["--ec-package", "r20_o00"] for k, v in SCENARIOS.items() if k not in EC20_SHARED
}

# "avoided" accounting (ABATEMENT_ACCOUNTING=avoided): the capture-energy variants are the baseline itself;
# the first-round "captured" accounting is kept as one sensitivity run.
CAPTURED_BASIS_ARGS = [
    "--cc-energy-basis", "boiler", "--no-cc-penalty-emissions", "--bio-transport-ef-kg-per-t-km", "0",
    "--bio-processing-ef-kg-per-gj", "0", "--co2-transport-ef-kg-per-t-km", "0",
]


def _avoided(scen: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out = {k: v for k, v in scen.items() if k not in ("cc_ext_cost", "cc_ext_penalty")}
    out["captured_basis"] = CAPTURED_BASIS_ARGS
    # Capture energy of a conventional MEA unit (cost workbook) instead of 2.4 GJ/t and 0.10 MWh/t
    out["cc_energy_high"] = ["--cc-ext-steam-gj-per-t", "3.5", "--cc-ext-power-mwh-per-t", "0.23"]
    # Biomass supply-chain emissions at the ends of the 5-15 kg CO2e/GJ range (Creutzig et al. 2015) instead of 10
    out["bio_ef_5"] = ["--bio-processing-ef-kg-per-gj", "5"]
    out["bio_ef_15"] = ["--bio-processing-ef-kg-per-gj", "15"]
    # Single-technology runs for the standalone benchmark (only the maximum-abatement target is used)
    out["bc_only"] = ["--only-tech", "bc", "--n-targets", "3"]
    out["cc_only"] = ["--only-tech", "cc", "--n-targets", "3"]
    return out


def _paths(script_dir: str, name: str) -> Dict[str, str]:
    prefix = f"{OUTPUT_PREFIX}_{name}"
    return {
        "prefix": prefix,
        "curve": os.path.join(script_dir, DATA_DIR_NAME, f"{prefix}_curve.csv"),
        "log": os.path.join(script_dir, LOG_DIR_NAME, f"{prefix}.log"),
    }


def run_one(script_dir: str, name: str, extra: List[str], threads: int) -> Dict[str, object]:
    p = _paths(script_dir, name)
    cmd = [
        sys.executable, os.path.join(script_dir, MODEL_SCRIPT),
        "--output-prefix", p["prefix"], "--solver-threads", str(threads), *extra,
    ]
    t0 = time.time()
    with open(p["log"], "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        rc = subprocess.run(cmd, cwd=script_dir, stdout=log, stderr=subprocess.STDOUT).returncode
    minutes = (time.time() - t0) / 60.0
    ok = rc == 0 and os.path.exists(p["curve"])
    return {"scenario": name, "status": "ok" if ok else f"failed (rc={rc})", "minutes": round(minutes, 1), "log": p["log"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fleet-model scenarios in parallel; finished scenarios are skipped.")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="Number of scenarios run at the same time.")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS_PER_JOB, help="Gurobi threads per scenario.")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these scenario names.")
    parser.add_argument("--rerun", action="store_true", help="Rerun scenarios even if their curve file exists.")
    parser.add_argument("--rerun-incomplete", action="store_true", help="Also rerun scenarios whose curve file has unsolved targets.")
    parser.add_argument("--dry-run", action="store_true", help="Print the queue without running.")
    parser.add_argument("--set", choices=["ec10", "ec20"], default="ec10", help="Result set (EC baseline 10% or 20%).")
    parser.add_argument("--accounting", choices=["captured", "avoided"], default=os.environ.get("ABATEMENT_ACCOUNTING", "avoided"),
                        help="Abatement accounting boundary passed to the model (see optimization_1_fleet_joint_milp_persistent.py).")
    args = parser.parse_args()
    global OUTPUT_PREFIX, SCENARIOS
    if args.set == "ec20":
        OUTPUT_PREFIX, SCENARIOS = "gurobi_r2e20", SCENARIOS_EC20
    os.environ["ABATEMENT_ACCOUNTING"] = args.accounting  # inherited by the model subprocesses
    if args.accounting == "avoided":
        SCENARIOS = _avoided(SCENARIOS)
    print(f"Abatement accounting: {args.accounting}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, LOG_DIR_NAME), exist_ok=True)

    names = list(SCENARIOS) if not args.only else args.only
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}. Available: {list(SCENARIOS)}")
    def _done(n: str) -> bool:
        curve = _paths(script_dir, n)["curve"]
        if not os.path.exists(curve):
            return False
        if args.rerun_incomplete and pd.read_csv(curve)["total_cost"].isna().any():
            return False
        return True

    queue = [n for n in names if args.rerun or not _done(n)]
    skipped = [n for n in names if n not in queue]

    print(f"Queue ({len(queue)}): {queue}")
    if skipped:
        print(f"Skipped, output exists ({len(skipped)}): {skipped}")
    if args.dry_run or not queue:
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, script_dir, n, SCENARIOS[n], args.threads): n for n in queue}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            print(f"[{time.strftime('%H:%M')}] {r['scenario']}: {r['status']} in {r['minutes']} min", flush=True)

    summary = pd.DataFrame(results).sort_values("scenario")
    summary_path = os.path.join(script_dir, LOG_DIR_NAME, f"run_summary_{time.strftime('%Y%m%d_%H%M')}.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
