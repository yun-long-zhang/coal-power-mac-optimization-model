"""Check the environment before starting the runs (takes under a minute).

  python3 check_env.py
"""
import importlib
import os
import sys

ok = True
print(f"Python {sys.version.split()[0]}")
if sys.version_info < (3, 9):
    print("  ERROR: Python 3.9 or newer is required")
    ok = False
for mod in ["numpy", "pandas", "matplotlib", "gurobipy", "pulp", "xlsxwriter", "openpyxl"]:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:10s} {getattr(m, '__version__', '')}")
    except ImportError:
        print(f"  ERROR: {mod} is not installed (pip install {mod})")
        ok = False

here = os.path.dirname(os.path.abspath(__file__))
for f in ["1885_coal_power_plants_original.xlsx", "Dji_200.csv", "Dki_4000.csv", "plant_province_r2.csv"]:
    p = os.path.join(here, "data", f)
    print(f"  data/{f}: {'ok' if os.path.exists(p) else 'MISSING'}")
    ok &= os.path.exists(p)

if ok:
    import gurobipy as gp
    try:
        m = gp.Model()
        x = m.addVars(3000, ub=1.0)  # larger than the size-limited pip licence allows
        m.setObjective(x.sum(), gp.GRB.MAXIMIZE)
        m.Params.OutputFlag = 0
        m.optimize()
        print(f"  Gurobi {gp.gurobi.version()} licence ok")
    except gp.GurobiError as e:
        print(f"  ERROR: Gurobi licence problem: {e}")
        ok = False

if ok:
    os.environ["ABATEMENT_ACCOUNTING"] = "avoided"
    sys.path.insert(0, here)
    import optimization_1_fleet_joint_milp_persistent as base

    d = os.path.join(here, "data")
    plants, ec, bc, cc, biomass, biomass_dist, sinks, co2_dist = base.load_inputs(
        os.path.join(d, base.INPUT_WORKBOOK_NAME), os.path.join(d, base.INPUT_BIOMASS_DISTANCE_NAME),
        os.path.join(d, base.INPUT_CO2_DISTANCE_NAME))
    n_emis = float(plants["base_emission"].sum())
    print(f"  plants {len(plants)}, fleet emissions {n_emis / 1e9:.3f} Gt (expected 3.563)")
    import pandas as pd

    cc_cols = pd.read_excel(os.path.join(d, base.INPUT_WORKBOOK_NAME), sheet_name="CC", nrows=1).columns
    if abs(n_emis / 1e9 - 3.563) > 0.01 or "cc_ext_opex_dollar_per_t" not in cc_cols:
        print("  ERROR: workbook values not as expected")
        ok = False
    print(f"  accounting: {base.ABATEMENT_ACCOUNTING}, logistics EF: {base.DEFAULT_LOGISTICS_EF}")

print("\nREADY: run  bash run_all.sh" if ok else "\nNOT READY: fix the errors above")
