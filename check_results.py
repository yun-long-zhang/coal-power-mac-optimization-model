"""Summary of the finished runs: status of every target and the main numbers.

  python3 check_results.py
"""
import glob
import os

import pandas as pd

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
rows = []
for p in sorted(glob.glob(os.path.join(here, "gurobi_r2_*_curve.csv"))):
    d = pd.read_csv(p)
    rows.append({
        "scenario": os.path.basename(p)[len("gurobi_r2_"):-len("_curve.csv")],
        "targets": len(d),
        "optimal": int((d["status"] == "Optimal").sum()),
        "unsolved": int(d["total_cost"].isna().sum()),
        "max_abatement_Gt": round(d["achieved_abatement"].max() / 1e9, 3),
        "max_cost_bn": round(d["total_cost"].max() / 1e9, 1),
    })
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False))
s18 = os.path.join(here, "sequence_falsification_all_policies_curve.csv")
print(f"\nS18 sequence check: {'done' if os.path.exists(s18) else 'not done'}")
