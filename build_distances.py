"""
Distance tables of the fleet model from the coordinates in the input workbook (geodesic distance, WGS-84, km):

  data/Dji_200.csv    biomass collection site j - plant i, pairs within 200 km
  data/Dki_4000.csv   CO2 storage site k - plant i, pairs within 4000 km

The road tortuosity factor (1.5) is applied in the model, not here. Both tables are provided in data/ and are
reproduced exactly by this script (about 1 min; a pre-filter with the great-circle distance skips pairs that are
clearly outside the radius). --max-sites N limits the run to the first N sites of each sheet for a quick check.
"""

import argparse
import os

import numpy as np
import pandas as pd
from geopy.distance import geodesic

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WORKBOOK = os.path.join(DATA, "1885_coal_power_plants_original.xlsx")
EARTH_RADIUS_KM = 6371.0
PREFILTER_MARGIN = 1.01  # great-circle and geodesic distances differ by less than 0.6%


def _great_circle_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def distances(sites: pd.DataFrame, site_col: str, plants: pd.DataFrame, radius_km: float, out_cols) -> pd.DataFrame:
    rows = []
    plat, plon, pid = plants["latitude"].to_numpy(), plants["longitude"].to_numpy(), plants["plant_i"].to_numpy()
    for _, s in sites.iterrows():
        near = np.flatnonzero(_great_circle_km(s["latitude"], s["longitude"], plat, plon) <= radius_km * PREFILTER_MARGIN)
        for k in near:
            d = float(geodesic((s["latitude"], s["longitude"]), (plat[k], plon[k])).km)
            if d <= radius_km:
                rows.append((s[site_col], pid[k], d))
    return pd.DataFrame(rows, columns=out_cols)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-sites", type=int, default=None, help="Use only the first N sites of each sheet (quick check).")
    ap.add_argument("--out-dir", default=DATA, help="Output folder (default: data/).")
    args = ap.parse_args()

    plants = pd.read_excel(WORKBOOK, sheet_name="coal_plant_i")
    biomass = pd.read_excel(WORKBOOK, sheet_name="biomass_point_j").iloc[: args.max_sites]
    sinks = pd.read_excel(WORKBOOK, sheet_name="carbon_sink_k").iloc[: args.max_sites]
    os.makedirs(args.out_dir, exist_ok=True)

    dji = distances(biomass, "biomass_point_j", plants, 200.0, ["biomass_j", "plant_i", "Dji"])
    dji.to_csv(os.path.join(args.out_dir, "Dji_200.csv"), index=False)
    dki = distances(sinks, "sink_k", plants, 4000.0, ["sink_k", "plant_i", "Dki"])
    dki.to_csv(os.path.join(args.out_dir, "Dki_4000.csv"), index=False)
    print(f"Dji_200.csv: {len(dji):,} pairs; Dki_4000.csv: {len(dki):,} pairs -> {args.out_dir}")


if __name__ == "__main__":
    main()
