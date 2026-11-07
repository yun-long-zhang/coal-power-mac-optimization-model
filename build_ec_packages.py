import argparse
import os

import pandas as pd


# ---- User config (edit before running) ----
DATA_DIR_NAME = "data"
EC_PORTFOLIO_DIR_NAME = "EC_Portfolio"
OUTPUT_NAME = "ec_packages_r2.csv"
# package tag -> (EC portfolio results workbook, sheet)
PACKAGES = {
    "r20_o00": ("ec_pkg_r2_all_results.xlsx", "sel_o00_r20"),
    "r175_o00": ("ec_pkg_r2_all_results.xlsx", "sel_o00_r18"),
    "r15_o00": ("ec_pkg_r2_all_results.xlsx", "sel_o00_r15"),
    "r175_o25": ("ec_pkg_r2_all_results.xlsx", "sel_o25_r18"),
    "r175_o50": ("ec_pkg_r2_all_results.xlsx", "sel_o50_r18"),
    "r15_o25": ("ec_pkg_r2_all_results.xlsx", "sel_o25_r15"),
    "r15_o50": ("ec_pkg_r2_all_results.xlsx", "sel_o50_r15"),
    "no5_r10_o00": ("ec_pkg_r2_no5_results.xlsx", "sel_o00_r10"),  # baseline: 10% without technology No. 5
    "r10_o00": ("ec_r2_fixed_pp_results.xlsx", "sel_o00_r10"),  # 10% with all 20 measures (sensitivity)
}
# -------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect plant-level EC packages (net abatement, net annual cost, saving ratio) "
        "from EC sub-technology portfolio results into one input file for the fleet model."
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, DATA_DIR_NAME)
    portfolio_dir = os.path.join(data_dir, EC_PORTFOLIO_DIR_NAME)

    frames = []
    for tag, (fname, sheet) in PACKAGES.items():
        d = pd.read_excel(os.path.join(portfolio_dir, fname), sheet_name=sheet)
        if d["achieved_abatement_ton"].isna().any():
            raise RuntimeError(f"Package {tag} has unsolved plants ({fname}:{sheet}).")
        abate = d["achieved_abatement_ton"]
        # Net annual cost = annualized retrofit cost - avoided coal cost, on the achieved (net) abatement.
        net_cost = abate * d["unit_cost_after_coal_saving_untrimmed"]
        frames.append(
            pd.DataFrame(
                {
                    "plant_id": d["plant_id"].astype(str),
                    "package": tag,
                    "ec_abatement_ton": abate,
                    "ec_net_cost": net_cost,
                    "ec_gross_cost": abate * d["unit_abatement_cost_untrimmed"],
                    "ec_coal_saving": abate * d["unit_revenue_from_coal_saving"],
                    "ec_saving_ratio": abate / d["base_emission"],
                    "selected_tech_id": d["selected_tech_id"].fillna(""),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out_path = args.output or os.path.join(data_dir, OUTPUT_NAME)
    out.to_csv(out_path, index=False)

    summary = out.groupby("package").agg(
        abatement_mt=("ec_abatement_ton", lambda s: s.sum() / 1e6),
        net_cost_musd=("ec_net_cost", lambda s: s.sum() / 1e6),
        ratio_min=("ec_saving_ratio", "min"),
        ratio_max=("ec_saving_ratio", "max"),
    )
    summary["net_unit_cost"] = summary["net_cost_musd"] / summary["abatement_mt"]
    print(summary.round(3).to_string())
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
