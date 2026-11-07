# Interaction-aware marginal abatement cost model for China's coal-fired power fleet

Code and data for *"Technology interactions reshape the economics of China's coal power decarbonization"*.
The repository reproduces all results, figures, and source-data tables of the main text and the Supplementary
Information from the input data.

## Contents

```
data/                 input data
  1885_coal_power_plants_original.xlsx   plants, EC measures, BC and CC parameters, storage sites,
                                         biomass collection sites, RRH data (Supplementary Tables S2-S8;
                                         columns described below)
  Dji_200.csv                            distances between biomass collection sites and plants (km)
  Dki_4000.csv                           distances between CO2 storage sites and plants (km)
  plant_province_r2.csv                  province of each plant (Supplementary Fig. S23)
source_data/          data behind every figure of the paper (one workbook per figure, see below)

optimization_ec_subtech_binary.py        Step 1  plant-level EC portfolio MILP (Methods Eqs. 1-6)
build_ec_packages.py                     Step 1  collects the EC packages used by the fleet model
optimization_1_fleet_joint_milp_persistent.py   Step 2  unified fleet-wide MILP (Methods Eqs. 7-44)
run_all_scenarios.py                     Step 2  runs all scenarios of Supplementary Table S1
analysis_sequence_falsification.py       Step 3  technology-ordering test (Supplementary Fig. S18)
make_figures.py                          Step 4  all analyses and figures (calls the analysis_*.py and plot_*.py scripts)
run_all.sh                               Steps 1-4 in sequence
build_distances.py                       distance tables Dji_200.csv and Dki_4000.csv from the coordinates
check_env.py, check_results.py           environment check; summary of finished runs
plot_export_utils.py, r2_result_columns.py   shared helpers
```

## Requirements

- Python 3.9 or newer (tested with 3.12.9) and the packages in `requirements.txt`:
  `pip install -r requirements.txt`
- Gurobi Optimizer (tested with 12.0.1) with a full licence. The size-limited licence that comes with `pip install gurobipy` cannot solve the
  fleet model; academic licences are available free of charge from Gurobi.

Check the setup with `python3 check_env.py`; the last line should read `READY`.

## Reproducing the results

```
bash run_all.sh                    # all steps; JOBS=3 THREADS=4 bash run_all.sh to change the parallelism
```

or step by step:

| Step | Command | Output | Run time* |
|---|---|---|---|
| 1 EC packages | the three `optimization_ec_subtech_binary.py` calls in `run_all.sh`, then `python3 build_ec_packages.py` | `data/ec_packages_r2.csv` | about 1 min |
| 2 Fleet scenarios | `ABATEMENT_ACCOUNTING=avoided python3 run_all_scenarios.py --accounting avoided --jobs 2 --threads 4` | `data/gurobi_r2_<scenario>_curve.csv`, `_results.xlsx` | 2.5–4.5 h per scenario; 42 scenarios |
| 3 Ordering test | `python3 analysis_sequence_falsification.py --targets-from-curve gurobi_r2_base_curve.csv --solver-threads 8` | `data/sequence_falsification_*` | about 16 h (8 threads) |
| 4 Figures | `python3 make_figures.py` | `figures/` | about 10 min |

\* Apple M4 Pro, 4 Gurobi threads per scenario.

Step 2 can be restarted after an interruption: scenarios whose curve file exists are skipped, and
`--rerun-incomplete` re-solves scenarios with unsolved targets. Step 3 saves each finished policy and resumes
from there. A single scenario can be run with `python3 run_all_scenarios.py --accounting avoided --only base`;
`python3 run_all_scenarios.py --accounting avoided --dry-run` lists all scenarios, and the model options of
each scenario are defined in `SCENARIOS` in `run_all_scenarios.py` (`python3
optimization_1_fleet_joint_milp_persistent.py --help` lists all options).

Each scenario solves 48 equally spaced abatement targets (equality constraint) with a relative MIP gap of 0.1%
(absolute gap 5 × 10^6 USD). Results are reproducible up to this tolerance; a re-run with a different number
of threads can return a different solution within the gap.

## Figures and source data

`make_figures.py` writes every figure together with the data behind it (an `.xlsx` workbook with one sheet per
panel). The workbooks of the published figures are in `source_data/`:

| Figure | Source data |
|---|---|
| Fig. 1 | `Fig1.xlsx` |
| Fig. 2 | `Fig2.xlsx` |
| Fig. 3 | `Fig3.xlsx` |
| Fig. 4 | `Fig4.xlsx` |
| Supplementary Figs. S9, S10 | `FigS9.xlsx`, `FigS10.xlsx` |
| Supplementary Figs. S13–S26 | `FigS13.xlsx` … `FigS26.xlsx` |
| Supplementary Note 6, Table S9 | `Note6_policy_instruments.xlsx` |

## Input workbook

| Sheet | Rows | Columns |
|---|---|---|
| `coal_plant_i` | 1,885 plants | `plant_i`; `longitude`, `latitude` (WGS-84); `capacity_Xi` (MW); `running_hours` (h yr^-1); `efficiency`; `energy_demand_GJ` (coal input, GJ yr^-1); `carbon_emissions_ton` (t CO2 yr^-1); `coal_price` (USD GJ^-1); `annuity_factor` (read but recomputed in the model from the discount rate and remaining life); `online_year`; `2030_year` … `2045_year` (plant age in that year, retirement scenarios); `LCOE (RMB/kWh)` (Supplementary Fig. S13) |
| `RRH` | 1,885 | `plant_i`; `unit _cost`: UAC of reducing running hours (USD t^-1) |
| `EC` | 1,885 | `plant_i`; `tech_k`: unit abatement cost of EC measure k (USD t^-1); `tech_k.1`: its annual abatement (t CO2 yr^-1); `unit _revenue_from_coal_saving` (USD t^-1); `carbon_reduction`, `unit _cost`: abatement and net UAC of a precomputed EC package (reference only; the fleet model uses the packages of Step 1) |
| `BC` | 1,885 | `plant_i`; `carbon_reduction`, `unit _cost`: reference values at the maximum co-firing ratio (only used to bound the maximum-abatement solve; the BC cost is computed in the model) |
| `CC` | 1,885 | `plant_i`; `capex_dollar_per_year` (annualized at 8 %, `annuity_factor_8%`, over `life_years`); `opex_dollar_per_t` (boiler-steam basis, `captured_basis` scenario only); `cc_ext_opex_dollar_per_t` (opex without steam cost, used with turbine extraction); `electricity1_dollar_per_t` (capture power at 0.23 MWh t^-1, rescaled in the model); `electricity2_dollar_per_t`, `carbon_reduction`, `unit _cost` (reference only) |
| `carbon_sink_k` | 429 storage sites | `sink_k`; `longitude`, `latitude`; `Estimated_storage_potential_t` (cumulative storage potential, t CO2) |
| `biomass_point_j` | 18,840 collection sites | `biomass_point_j`; `longitude`, `latitude`; `bioenergy_GJ` (available residues, GJ yr^-1) |

## Notes on the input data

- Biomass residues were mapped at 1 km resolution in ArcGIS and aggregated to 20 × 20 km collection cells
  (cells with less than 100 GJ yr^-1 excluded; 18,840 sites; Supplementary Note 3). The GIS processing is not
  part of this code; the resulting sites and distances are provided in the workbook and `Dji_200.csv`.
- Distances are geodesic distances between the coordinates (`python3 build_distances.py` rebuilds both tables
  exactly in about 1 min); the road tortuosity factor of 1.5 is applied in the model.
- Monetary values are in 2019 USD (6.8985 RMB/USD).

## Licence and citation

[Authors: licence, citation, and Zenodo DOI of this version.]
