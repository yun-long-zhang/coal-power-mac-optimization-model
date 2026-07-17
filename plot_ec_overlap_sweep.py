import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm

from plot_export_utils import save_figure_bundle

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---- User config (edit before running) ----
DATA_DIR_NAME = "data"
EC_PORTFOLIO_DIR_NAME = "EC_Portfolio"
SUMMARY_CSV_NAME = "ec_noovershoot_summary.csv"
INPUT_WORKBOOK_NAME = "1885_coal_power_plants_original.xlsx"
OUTPUT_PREFIX = "ec_noovershoot"
OUTPUT_METRIC_HEATMAPS_PNG_SUFFIX = "_metric_heatmaps.png"
OUTPUT_METRIC_HEATMAP_DATA_XLSX_SUFFIX = "_metric_heatmap_data.xlsx"
OUTPUT_UNTRIMMED_METRIC_HEATMAPS_PNG_SUFFIX = "_untrimmed_metric_heatmaps.png"
OUTPUT_UNTRIMMED_METRIC_HEATMAP_DATA_XLSX_SUFFIX = "_untrimmed_metric_heatmap_data.xlsx"
OUTPUT_GROSS_METRIC_HEATMAPS_PNG_SUFFIX = "_gross_metric_heatmaps.png"
OUTPUT_GROSS_METRIC_HEATMAP_DATA_XLSX_SUFFIX = "_gross_metric_heatmap_data.xlsx"
OUTPUT_CATEGORY_STACK_PNG_SUFFIX = "_category_stack.png"
OUTPUT_CATEGORY_SHARE_STACK_PNG_SUFFIX = "_category_share_stack.png"
# -------------------------------------------

TECH_CATEGORY_BY_NUM = {
    2: "boiler_combustion_milling",
    5: "boiler_combustion_milling",
    6: "boiler_combustion_milling",
    8: "boiler_combustion_milling",
    11: "boiler_combustion_milling",
    19: "boiler_combustion_milling",
    1: "steam_turbine_body_flow",
    13: "steam_turbine_body_flow",
    18: "steam_turbine_body_flow",
    3: "cooling_condensing_system",
    9: "cooling_condensing_system",
    10: "cooling_condensing_system",
    15: "cooling_condensing_system",
    4: "air_preheater_sealing",
    7: "air_preheater_sealing",
    12: "flue_gas_heat_recovery",
    16: "flue_gas_heat_recovery",
    14: "electrical_aux_and_control",
    17: "electrical_aux_and_control",
    20: "electrical_aux_and_control",
}

CATEGORY_ORDER = [
    "boiler_combustion_milling",
    "steam_turbine_body_flow",
    "cooling_condensing_system",
    "air_preheater_sealing",
    "flue_gas_heat_recovery",
    "electrical_aux_and_control",
]

CATEGORY_LABELS = {
    "boiler_combustion_milling": "Boiler & milling",
    "steam_turbine_body_flow": "Steam turbine",
    "cooling_condensing_system": "Cooling/condensing",
    "air_preheater_sealing": "Air preheater sealing",
    "flue_gas_heat_recovery": "Flue-gas heat recovery",
    "electrical_aux_and_control": "Auxiliary electric/control",
}

CATEGORY_COLORS = {
    "boiler_combustion_milling": "#4E79A7",
    "steam_turbine_body_flow": "#F28E2B",
    "cooling_condensing_system": "#59A14F",
    "air_preheater_sealing": "#E15759",
    "flue_gas_heat_recovery": "#76B7B2",
    "electrical_aux_and_control": "#B07AA1",
}

HEATMAP_HIGHLIGHTS = {
    (15.0, 50.0),
    (17.5, 25.0),
    (20.0, 0.0),
}


def _resolve_input_path(path: str, script_dir: str) -> str:
    if os.path.isabs(path):
        return path
    candidate = os.path.join(script_dir, path)
    if os.path.exists(path):
        return path
    return candidate


def _resolve_output_dir(path: str, script_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(script_dir, path)


def _norm_id(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if not s:
        return ""
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def _canon(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _find_col(columns: list[str], candidates: list[str]) -> str:
    canon_to_raw = {_canon(c): c for c in columns}
    for cand in candidates:
        key = _canon(cand)
        if key in canon_to_raw:
            return canon_to_raw[key]
    raise KeyError(f"Could not find any of columns {candidates} in {columns}")


def _parse_tech_num_from_header(name: str) -> int:
    s = str(name).strip().lower()
    if not s.startswith("tech_"):
        return -1
    tail = s[len("tech_"):]
    if tail.endswith(".1"):
        tail = tail[:-2]
    try:
        return int(tail)
    except Exception:
        return -1


def _load_tech_lookup(workbook_path: str) -> pd.DataFrame:
    if not os.path.exists(workbook_path):
        raise FileNotFoundError(
            f"Input workbook not found: {workbook_path}. Pass --input-workbook explicitly or check the path."
        )

    ec_raw = pd.read_excel(workbook_path, sheet_name="EC")
    ecols = list(ec_raw.columns)
    ec_pid_col = _find_col(ecols, ["plant_i", "plantid", "id"])

    def _is_unit_col(name: str) -> bool:
        s = str(name).strip().lower()
        return s.startswith("tech_") and not s.endswith(".1")

    def _is_abate_col(name: str) -> bool:
        s = str(name).strip().lower()
        return s.startswith("tech_") and s.endswith(".1")

    unit_cols = [c for c in ecols if _is_unit_col(c)]
    abate_cols = [c for c in ecols if _is_abate_col(c)]

    unit_by_num = {n: c for c in unit_cols if (n := _parse_tech_num_from_header(c)) > 0}
    abate_by_num = {n: c for c in abate_cols if (n := _parse_tech_num_from_header(c)) > 0}
    tech_nums = sorted(set(unit_by_num) & set(abate_by_num))
    if len(tech_nums) != 20:
        raise ValueError(f"Cannot build 20-tech lookup from workbook: matched={len(tech_nums)}")

    rows = []
    for n in tech_nums:
        cat = TECH_CATEGORY_BY_NUM.get(n, "uncategorized")
        sub = pd.DataFrame(
            {
                "plant_id": ec_raw[ec_pid_col].map(_norm_id),
                "tech_num": n,
                "tech_id": f"TECH_{n:02d}",
                "tech_category": cat,
                "unit_cost": pd.to_numeric(ec_raw[unit_by_num[n]], errors="coerce").fillna(0.0),
                "abatement_ton": pd.to_numeric(ec_raw[abate_by_num[n]], errors="coerce").fillna(0.0),
            }
        )
        rows.append(sub)

    lookup = pd.concat(rows, ignore_index=True)
    return lookup


def _load_summary(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Summary CSV not found: {path}. "
            "Pass --summary explicitly or run from a directory where the relative path is valid."
        )
    df = pd.read_csv(path)
    required = {
        "target_ratio",
        "target_abatement_ton",
        "overlap_coef",
        "status",
        "total_retrofit_cost",
        "achieved_abatement_ton",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in summary CSV: {sorted(missing)}")

    out = df.copy()
    out["target_ratio"] = pd.to_numeric(out["target_ratio"], errors="coerce")
    out["target_abatement_ton"] = pd.to_numeric(out["target_abatement_ton"], errors="coerce")
    out["overlap_coef"] = pd.to_numeric(out["overlap_coef"], errors="coerce")
    out["total_retrofit_cost"] = pd.to_numeric(out["total_retrofit_cost"], errors="coerce")
    out["achieved_abatement_ton"] = pd.to_numeric(out["achieved_abatement_ton"], errors="coerce")
    out = out.dropna(subset=["target_ratio", "overlap_coef", "status"])

    out["target_pct"] = (out["target_ratio"] * 100).round(3)
    out["overlap_pct"] = (out["overlap_coef"] * 100).round(3)
    return out


def _sheet_tag(overlap_coef: float, target_ratio: float) -> str:
    # Keep tag rule consistent with optimization_ec_subtech_binary.py
    otag = f"o{int(round(float(overlap_coef) * 100)):02d}"
    rtag = f"r{int(round(float(target_ratio) * 100)):02d}"
    return f"sel_{otag}_{rtag}"


def _collect_category_contributions(summary_df: pd.DataFrame, results_xlsx: str, input_workbook: str) -> pd.DataFrame:
    lookup = _load_tech_lookup(input_workbook)
    lookup_index = lookup.set_index(["plant_id", "tech_num"])
    xls = pd.ExcelFile(results_xlsx)
    available = set(xls.sheet_names)

    records = []
    tech_cols = [f"x_TECH_{i:02d}" for i in range(1, 21)]

    for _, row in summary_df.iterrows():
        if str(row["status"]).strip() != "Optimal":
            continue

        sheet = _sheet_tag(row["overlap_coef"], row["target_ratio"])
        if sheet not in available:
            continue

        detail = pd.read_excel(results_xlsx, sheet_name=sheet)
        if detail.empty:
            continue

        for col in ["plant_id", "achieved_abatement_ton_gross", "achieved_abatement_ton"]:
            if col in detail.columns:
                detail[col] = pd.to_numeric(detail[col], errors="coerce")

        category_gross = {cat: 0.0 for cat in CATEGORY_ORDER}
        gross_total = 0.0
        net_total = 0.0

        for _, prow in detail.iterrows():
            plant_id = _norm_id(prow["plant_id"])
            plant_gross = float(prow.get("achieved_abatement_ton_gross", np.nan))
            plant_net = float(prow.get("achieved_abatement_ton", np.nan))
            if not np.isfinite(plant_gross) or plant_gross <= 0:
                continue

            selected_tech_nums = []
            for col in tech_cols:
                if col in prow.index and pd.notna(prow[col]) and int(prow[col]) == 1:
                    selected_tech_nums.append(int(col.split("TECH_")[-1]))

            if not selected_tech_nums:
                continue

            row_gross = 0.0
            row_category_gross = {cat: 0.0 for cat in CATEGORY_ORDER}
            for tech_num in selected_tech_nums:
                key = (plant_id, tech_num)
                if key not in lookup_index.index:
                    continue
                abate = float(lookup_index.loc[key, "abatement_ton"])
                cat = str(lookup_index.loc[key, "tech_category"])
                if cat not in row_category_gross:
                    continue
                row_category_gross[cat] += abate
                row_gross += abate

            if row_gross <= 0:
                continue

            # Preserve the category mix implied by selected technologies, but scale to the net achieved reduction.
            scale = plant_net / row_gross if np.isfinite(plant_net) and plant_net > 0 else 1.0
            for cat, gross in row_category_gross.items():
                category_gross[cat] += gross * scale

            gross_total += row_gross * scale
            if np.isfinite(plant_net) and plant_net > 0:
                net_total += plant_net

        target_ratio = float(row["target_ratio"])
        overlap_coef = float(row["overlap_coef"])
        for cat in CATEGORY_ORDER:
            records.append(
                {
                    "target_ratio": target_ratio,
                    "overlap_coef": overlap_coef,
                    "category": cat,
                    "contribution_ton": category_gross[cat],
                    "gross_total_ton": gross_total,
                    "net_total_ton": net_total,
                    "sheet": sheet,
                }
            )

    return pd.DataFrame(records)


def _compute_trimmed_target_unit_cost(
    detail: pd.DataFrame,
    lookup_index: pd.DataFrame,
    target_abatement_ton: float,
) -> tuple[float, float]:
    tech_cols = [f"x_TECH_{i:02d}" for i in range(1, 21)]
    slices: list[tuple[float, float]] = []
    total_ab = 0.0
    total_cost = 0.0

    for _, prow in detail.iterrows():
        plant_id = _norm_id(prow.get("plant_id", ""))
        plant_net = float(prow.get("achieved_abatement_ton", np.nan))
        plant_gross = float(prow.get("achieved_abatement_ton_gross", np.nan))
        unit_rev = float(prow.get("unit_revenue_from_coal_saving", 0.0))

        if not np.isfinite(plant_net) or plant_net <= 0:
            continue

        selected_tech_nums = []
        for col in tech_cols:
            if col in prow.index and pd.notna(prow[col]) and int(prow[col]) == 1:
                selected_tech_nums.append(int(col.split("TECH_")[-1]))

        if not selected_tech_nums:
            continue

        row_gross = 0.0
        tech_entries = []
        for tech_num in selected_tech_nums:
            key = (plant_id, tech_num)
            if key not in lookup_index.index:
                continue
            abate = float(lookup_index.loc[key, "abatement_ton"])
            ucost = float(lookup_index.loc[key, "unit_cost"])
            if not np.isfinite(abate) or abate <= 0:
                continue
            tech_entries.append((abate, ucost))
            row_gross += abate

        if row_gross <= 0:
            continue

        if np.isfinite(plant_gross) and plant_gross > 0:
            scale = max(0.0, plant_net / plant_gross)
        else:
            scale = max(0.0, plant_net / row_gross)

        for abate, ucost in tech_entries:
            net_ab = abate * scale
            if net_ab <= 0:
                continue
            unit_net = ucost - unit_rev
            slices.append((unit_net, net_ab))
            total_ab += net_ab
            total_cost += net_ab * unit_net

    if total_ab <= 0 or not np.isfinite(target_abatement_ton) or target_abatement_ton <= 0:
        return np.nan, np.nan

    overshoot = max(0.0, total_ab - target_abatement_ton)
    if overshoot > 1e-9 and slices:
        removed_cost = 0.0
        remaining = overshoot
        for unit_net, net_ab in sorted(slices, key=lambda x: x[0], reverse=True):
            if remaining <= 1e-9:
                break
            cut = min(net_ab, remaining)
            removed_cost += cut * unit_net
            remaining -= cut
        total_cost -= removed_cost

    return total_cost, total_cost / target_abatement_ton


def _attach_net_unit_cost_from_results(summary_df: pd.DataFrame, results_xlsx: str, input_workbook: str) -> pd.DataFrame:
    out = summary_df.copy()
    out["unit_cost_after_coal_saving_achieved"] = np.nan
    out["unit_cost_after_coal_saving_target"] = np.nan
    out["total_net_cost_after_coal_saving"] = np.nan
    out["unit_cost_after_coal_saving_target_trimmed"] = np.nan
    out["total_net_cost_after_coal_saving_trimmed"] = np.nan
    out["unit_gross_cost_achieved"] = np.nan
    out["unit_gross_cost_target"] = np.nan
    out["total_gross_cost"] = np.nan
    out["unit_gross_cost_target_trimmed"] = np.nan
    out["total_gross_cost_trimmed"] = np.nan

    if not os.path.exists(results_xlsx):
        return out

    lookup = _load_tech_lookup(input_workbook)
    lookup_index = lookup.set_index(["plant_id", "tech_num"])

    xls = pd.ExcelFile(results_xlsx)
    available = set(xls.sheet_names)

    for i, row in out.iterrows():
        if str(row["status"]).strip() != "Optimal":
            continue
        sheet = _sheet_tag(row["overlap_coef"], row["target_ratio"])
        if sheet not in available:
            continue

        d = pd.read_excel(results_xlsx, sheet_name=sheet)
        if d.empty:
            continue

        for col in [
            "achieved_abatement_ton_gross",
            "achieved_abatement_ton",
            "trimmed_basis_abatement_ton",
            "unit_cost_after_coal_saving",
            "unit_cost_after_coal_saving_untrimmed",
            "unit_abatement_cost",
            "unit_abatement_cost_untrimmed",
            "unit_revenue_from_coal_saving",
            "total_retrofit_cost_untrimmed",
            "total_retrofit_cost_trimmed",
        ]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")

        if "achieved_abatement_ton" not in d.columns:
            continue

        # Untrimmed scenario-level net cost for achieved/target bases.
        if "unit_cost_after_coal_saving_untrimmed" in d.columns:
            per_ton_net_untrimmed = d["unit_cost_after_coal_saving_untrimmed"]
        elif "unit_cost_after_coal_saving" in d.columns:
            # Backward compatibility with old result files where this was untrimmed.
            per_ton_net_untrimmed = d["unit_cost_after_coal_saving"]
        elif "unit_abatement_cost_untrimmed" in d.columns and "unit_revenue_from_coal_saving" in d.columns:
            per_ton_net_untrimmed = d["unit_abatement_cost_untrimmed"] - d["unit_revenue_from_coal_saving"]
        elif "unit_abatement_cost" in d.columns and "unit_revenue_from_coal_saving" in d.columns:
            per_ton_net_untrimmed = d["unit_abatement_cost"] - d["unit_revenue_from_coal_saving"]
        else:
            continue

        ab = d["achieved_abatement_ton"].fillna(0.0)
        net_total = float((ab * per_ton_net_untrimmed.fillna(0.0)).sum())
        if "unit_abatement_cost_untrimmed" in d.columns:
            per_ton_gross_untrimmed = d["unit_abatement_cost_untrimmed"].fillna(0.0)
            gross_total = float((ab * per_ton_gross_untrimmed).sum())
        elif "total_retrofit_cost_untrimmed" in d.columns:
            gross_total = float(d["total_retrofit_cost_untrimmed"].fillna(0.0).sum())
        else:
            gross_total = float(row.get("total_retrofit_cost", np.nan))
        total_ab = float(ab.sum())
        if total_ab <= 0:
            continue

        out.at[i, "total_net_cost_after_coal_saving"] = net_total
        out.at[i, "unit_cost_after_coal_saving_achieved"] = net_total / total_ab
        out.at[i, "total_gross_cost"] = gross_total
        out.at[i, "unit_gross_cost_achieved"] = gross_total / total_ab
        target_ab = float(row.get("target_abatement_ton", np.nan))
        if np.isfinite(target_ab) and target_ab > 0:
            out.at[i, "unit_cost_after_coal_saving_target"] = net_total / target_ab
            out.at[i, "unit_gross_cost_target"] = gross_total / target_ab

            # Prefer directly reading trimmed columns produced by optimization model output.
            if "unit_cost_after_coal_saving" in d.columns and "trimmed_basis_abatement_ton" in d.columns:
                trimmed_basis = d["trimmed_basis_abatement_ton"].fillna(0.0)
                trimmed_total = float((trimmed_basis * d["unit_cost_after_coal_saving"].fillna(0.0)).sum())
                trimmed_unit = trimmed_total / target_ab if target_ab > 0 else np.nan
                if "unit_abatement_cost" in d.columns:
                    trimmed_gross_total = float((trimmed_basis * d["unit_abatement_cost"].fillna(0.0)).sum())
                elif "total_retrofit_cost_trimmed" in d.columns:
                    trimmed_gross_total = float(d["total_retrofit_cost_trimmed"].fillna(0.0).sum())
                else:
                    trimmed_gross_total = np.nan
            else:
                trimmed_total, trimmed_unit = _compute_trimmed_target_unit_cost(d, lookup_index, target_ab)
                trimmed_gross_total = np.nan

            out.at[i, "total_net_cost_after_coal_saving_trimmed"] = trimmed_total
            out.at[i, "unit_cost_after_coal_saving_target_trimmed"] = trimmed_unit
            out.at[i, "total_gross_cost_trimmed"] = trimmed_gross_total
            out.at[i, "unit_gross_cost_target_trimmed"] = (
                trimmed_gross_total / target_ab if np.isfinite(trimmed_gross_total) and target_ab > 0 else np.nan
            )

    return out


def _fmt_pct(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}%"
    return f"{v:.2f}%"


def _build_heatmap_source_table(
    df: pd.DataFrame,
    unit_cost_basis: str,
    show_net_benefit: bool = False,
    metric_mode: str = "net",
) -> pd.DataFrame:
    work = df.copy()
    work["is_optimal"] = work["status"].astype(str).str.strip().eq("Optimal")

    if unit_cost_basis not in {"target_trimmed", "target", "achieved"}:
        raise ValueError("unit_cost_basis must be 'target_trimmed', 'target' or 'achieved'.")

    if metric_mode not in {"net", "gross"}:
        raise ValueError("metric_mode must be 'net' or 'gross'.")

    if metric_mode == "gross":
        if unit_cost_basis == "target_trimmed" and "unit_gross_cost_target_trimmed" in work.columns:
            work["unit_cost_cny_per_ton"] = work["unit_gross_cost_target_trimmed"]
        elif unit_cost_basis == "target" and "unit_gross_cost_target" in work.columns:
            work["unit_cost_cny_per_ton"] = work["unit_gross_cost_target"]
        elif unit_cost_basis == "achieved" and "unit_gross_cost_achieved" in work.columns:
            work["unit_cost_cny_per_ton"] = work["unit_gross_cost_achieved"]
        else:
            work["unit_cost_cny_per_ton"] = work["total_retrofit_cost"] / work["achieved_abatement_ton"]
    else:
        if unit_cost_basis == "target_trimmed" and "unit_cost_after_coal_saving_target_trimmed" in work.columns:
            work["unit_cost_cny_per_ton"] = work["unit_cost_after_coal_saving_target_trimmed"]
        elif unit_cost_basis == "target" and "unit_cost_after_coal_saving_target" in work.columns:
            work["unit_cost_cny_per_ton"] = work["unit_cost_after_coal_saving_target"]
        elif unit_cost_basis == "achieved" and "unit_cost_after_coal_saving_achieved" in work.columns:
            work["unit_cost_cny_per_ton"] = work["unit_cost_after_coal_saving_achieved"]
        else:
            work["unit_cost_cny_per_ton"] = work["total_retrofit_cost"] / work["achieved_abatement_ton"]

    work.loc[~np.isfinite(work["unit_cost_cny_per_ton"]), "unit_cost_cny_per_ton"] = np.nan
    if metric_mode == "gross":
        work["unit_metric_cny_per_ton"] = work["unit_cost_cny_per_ton"]
        work["unit_metric_label"] = "gross_cost"
    elif show_net_benefit:
        # Net benefit is the sign-flipped net unit cost.
        work["unit_metric_cny_per_ton"] = -work["unit_cost_cny_per_ton"]
        work["unit_metric_label"] = "net_benefit"
    else:
        work["unit_metric_cny_per_ton"] = work["unit_cost_cny_per_ton"]
        work["unit_metric_label"] = "net_cost"
    return work


def _build_monotonic_violation_table(heatmap_source: pd.DataFrame, show_net_benefit: bool = False) -> pd.DataFrame:
    rows = []
    for overlap_pct in sorted(heatmap_source["overlap_pct"].dropna().unique()):
        panel = heatmap_source[
            (heatmap_source["overlap_pct"] == overlap_pct)
            & (heatmap_source["is_optimal"])
            & (pd.notna(heatmap_source["unit_metric_cny_per_ton"]))
        ].sort_values("target_pct")

        prev = None
        for _, r in panel.iterrows():
            cur = {
                "overlap_pct": float(r["overlap_pct"]),
                "target_pct": float(r["target_pct"]),
                "unit_metric_cny_per_ton": float(r["unit_metric_cny_per_ton"]),
                "achieved_abatement_ton": float(r.get("achieved_abatement_ton", np.nan)),
                "target_abatement_ton": float(r.get("target_abatement_ton", np.nan)),
                "overshoot_ton": float(r.get("overshoot_ton", np.nan)),
            }
            is_violation = False
            if prev is not None:
                if show_net_benefit:
                    # For benefit view, monotonic expectation is non-decreasing with target.
                    is_violation = cur["unit_metric_cny_per_ton"] < prev["unit_metric_cny_per_ton"] - 1e-9
                else:
                    # For cost view, monotonic expectation is non-decreasing with target.
                    is_violation = cur["unit_metric_cny_per_ton"] < prev["unit_metric_cny_per_ton"] - 1e-9
            if is_violation:
                rows.append(
                    {
                        "overlap_pct": cur["overlap_pct"],
                        "target_pct_prev": prev["target_pct"],
                        "target_pct_curr": cur["target_pct"],
                        "unit_metric_prev": prev["unit_metric_cny_per_ton"],
                        "unit_metric_curr": cur["unit_metric_cny_per_ton"],
                        "drop_abs": cur["unit_metric_cny_per_ton"] - prev["unit_metric_cny_per_ton"],
                        "drop_pct_of_prev": (cur["unit_metric_cny_per_ton"] - prev["unit_metric_cny_per_ton"]) / prev["unit_metric_cny_per_ton"] * 100.0,
                        "overshoot_prev_ton": prev["overshoot_ton"],
                        "overshoot_curr_ton": cur["overshoot_ton"],
                    }
                )
            prev = cur

    return pd.DataFrame(rows)


def export_heatmap_source_excel(
    df: pd.DataFrame,
    out_xlsx: str,
    unit_cost_basis: str,
    show_net_benefit: bool = False,
    metric_mode: str = "net",
) -> pd.DataFrame:
    source = _build_heatmap_source_table(
        df,
        unit_cost_basis,
        show_net_benefit=show_net_benefit,
        metric_mode=metric_mode,
    )
    source = source.sort_values(["overlap_pct", "target_pct"]).reset_index(drop=True)

    matrix = source.pivot_table(
        index="overlap_pct",
        columns="target_pct",
        values="unit_metric_cny_per_ton",
        aggfunc="first",
    ).sort_index().sort_index(axis=1)

    violations = _build_monotonic_violation_table(source, show_net_benefit=show_net_benefit)

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        source.to_excel(writer, sheet_name="heatmap_source", index=False)
        matrix.to_excel(writer, sheet_name="unit_cost_matrix")
        violations.to_excel(writer, sheet_name="monotonicity_violations", index=False)

    return violations


def plot_metric_heatmaps(
    df: pd.DataFrame,
    out_png: str,
    unit_cost_basis: str = "target",
    unit_cost_levels: int = 21,
    show_net_benefit: bool = False,
    metric_mode: str = "net",
) -> None:
    work = _build_heatmap_source_table(
        df,
        unit_cost_basis,
        show_net_benefit=show_net_benefit,
        metric_mode=metric_mode,
    )

    overlap_vals = sorted(work["overlap_pct"].unique())
    target_vals = sorted(work["target_pct"].unique())

    unit_cost_mat = np.full((len(overlap_vals), len(target_vals)), np.nan)
    unsolved_mat = np.zeros((len(overlap_vals), len(target_vals)), dtype=bool)

    overlap_idx = {v: i for i, v in enumerate(overlap_vals)}
    target_idx = {v: j for j, v in enumerate(target_vals)}

    for _, row in work.iterrows():
        i = overlap_idx[float(row["overlap_pct"])]
        j = target_idx[float(row["target_pct"])]
        if bool(row["is_optimal"]):
            unit_cost_mat[i, j] = row["unit_metric_cny_per_ton"]
        else:
            unsolved_mat[i, j] = True

    fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=220)

    finite = unit_cost_mat[np.isfinite(unit_cost_mat)]
    if finite.size == 0:
        raise ValueError("No Optimal scenarios with finite unit cost found.")

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    boundaries = np.linspace(vmin, vmax, int(max(2, unit_cost_levels)) + 1)
    cmap = plt.get_cmap("YlOrRd", int(max(2, unit_cost_levels))).copy()
    cmap.set_bad(color="#eeeeee")
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    masked = np.ma.masked_invalid(unit_cost_mat)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto", origin="upper")

    ax.set_xticks(np.arange(len(target_vals)))
    ax.set_xticklabels([_fmt_pct(v) for v in target_vals], fontsize=10)
    ax.set_yticks(np.arange(len(overlap_vals)))
    ax.set_yticklabels([_fmt_pct(v) for v in overlap_vals], fontsize=10)
    ax.set_xlabel("Target abatement ratio", fontsize=11)
    ax.set_ylabel("Overlap coefficient", fontsize=11)
    if metric_mode == "gross":
        ax.set_title(
            "Average Gross Unit Abatement Cost Heatmap"
            if unit_cost_basis == "target"
            else (
                "Average Gross Unit Abatement Cost Heatmap (Target-Trimmed Basis)"
                if unit_cost_basis == "target_trimmed"
                else "Average Gross Unit Abatement Cost Heatmap (Achieved Basis)"
            ),
            fontsize=12,
            pad=10,
        )
    elif show_net_benefit:
        ax.set_title(
            "Net Unit Abatement Benefit Heatmap"
            if unit_cost_basis == "target"
            else (
                "Net Unit Abatement Benefit Heatmap (Target-Trimmed Basis)"
                if unit_cost_basis == "target_trimmed"
                else "Net Unit Abatement Benefit Heatmap (Achieved Basis)"
            ),
            fontsize=12,
            pad=10,
        )
    else:
        ax.set_title(
            "Net Unit Abatement Cost Heatmap (Target-Trimmed Basis)"
            if unit_cost_basis == "target_trimmed"
            else (
                "Net Unit Abatement Cost Heatmap"
                if unit_cost_basis == "target"
                else "Net Unit Abatement Cost Heatmap (Achieved Basis)"
            ),
            fontsize=12,
            pad=10,
        )

    for i in range(len(overlap_vals)):
        for j in range(len(target_vals)):
            if unsolved_mat[i, j]:
                ax.text(j, i, "No solution", ha="center", va="center", fontsize=9, color="#111111", fontweight="bold")
            elif np.isfinite(unit_cost_mat[i, j]):
                # Use contrasting text color so labels remain readable across bins.
                rel = 0.5 if vmax <= vmin else (unit_cost_mat[i, j] - vmin) / (vmax - vmin)
                txt_color = "white" if rel > 0.55 else "#1a1a1a"
                ax.text(
                    j,
                    i,
                    f"{unit_cost_mat[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=txt_color,
                    fontweight="bold",
                )

    for target_pct, overlap_pct in HEATMAP_HIGHLIGHTS:
        if target_pct in target_idx and overlap_pct in overlap_idx:
            j = target_idx[target_pct]
            i = overlap_idx[overlap_pct]
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.8,
                    zorder=6,
                )
            )

    tick_count = min(7, len(boundaries))
    tick_positions = np.linspace(vmin, vmax, tick_count)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04, ticks=tick_positions)
    cbar.ax.set_yticklabels([f"{v:.2f}" for v in tick_positions])
    if metric_mode == "gross":
        cbar.set_label("Average gross unit abatement cost ($/tCO2)", fontsize=10)
    elif show_net_benefit:
        cbar.set_label("Net unit abatement benefit after coal-saving revenue ($/tCO2)", fontsize=10)
    else:
        cbar.set_label("Net unit abatement cost after coal-saving revenue ($/tCO2)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    save_figure_bundle(
        fig,
        out_png,
        bbox_inches="tight",
        data_frames={"summary": df},
    )
    plt.close(fig)


def plot_category_stack(summary_df: pd.DataFrame, results_xlsx: str, input_workbook: str, out_png: str) -> None:
    work = _collect_category_contributions(summary_df, results_xlsx, input_workbook)
    if work.empty:
        raise ValueError("No category contribution data available for plotting.")

    overlap_vals = sorted(work["overlap_coef"].unique())
    target_vals = sorted(work["target_ratio"].unique())
    n_panels = len(overlap_vals)
    ncols = 3 if n_panels > 3 else n_panels
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14.2, 10.2), dpi=220, sharey=True, constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for ax, overlap in zip(axes, overlap_vals):
        panel = work[work["overlap_coef"] == overlap].sort_values("target_ratio")
        x = np.arange(len(target_vals))
        bottom = np.zeros(len(target_vals))

        for cat in CATEGORY_ORDER:
            vals = []
            for target in target_vals:
                m = panel[(panel["target_ratio"] == target) & (panel["category"] == cat)]
                vals.append(float(m["contribution_ton"].iloc[0] / 1e6) if not m.empty else 0.0)
            vals = np.array(vals)
            ax.bar(
                x,
                vals,
                bottom=bottom,
                width=0.72,
                color=CATEGORY_COLORS[cat],
                edgecolor="white",
                linewidth=0.4,
                label=CATEGORY_LABELS[cat],
            )
            bottom += vals

        ax.set_title(f"Overlap {int(round(overlap * 100))}%", fontsize=11, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.2f}%" if abs(t - round(t)) > 1e-9 else f"{int(round(t))}%" for t in target_vals], fontsize=9)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)

    for ax in axes[n_panels:]:
        ax.axis("off")

    left_ax = axes[0]
    left_ax.set_ylabel("Allocated abatement (MtCO$_2$)", fontsize=11)
    for ax in axes[:n_panels]:
        ax.set_xlabel("Target abatement ratio", fontsize=10)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[cat], ec="white", lw=0.4, label=CATEGORY_LABELS[cat])
        for cat in CATEGORY_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Energy Conservation Technology Contribution to Abatement", fontsize=12)

    save_figure_bundle(
        fig,
        out_png,
        bbox_inches="tight",
        data_frames={"category_contribution": work},
    )
    plt.close(fig)


def plot_category_share_stack(summary_df: pd.DataFrame, results_xlsx: str, input_workbook: str, out_png: str) -> None:
    work = _collect_category_contributions(summary_df, results_xlsx, input_workbook)
    if work.empty:
        raise ValueError("No category contribution data available for plotting.")

    overlap_vals = sorted(work["overlap_coef"].unique())
    target_vals = sorted(work["target_ratio"].unique())
    n_panels = len(overlap_vals)
    ncols = 3 if n_panels > 3 else n_panels
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14.2, 10.2), dpi=220, sharey=True, constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for ax, overlap in zip(axes, overlap_vals):
        panel = work[work["overlap_coef"] == overlap].sort_values("target_ratio")
        x = np.arange(len(target_vals))
        bottom = np.zeros(len(target_vals))

        for cat in CATEGORY_ORDER:
            vals = []
            for target in target_vals:
                m = panel[(panel["target_ratio"] == target) & (panel["category"] == cat)]
                if m.empty:
                    vals.append(0.0)
                    continue
                total = float(m["gross_total_ton"].iloc[0])
                contrib = float(m["contribution_ton"].iloc[0])
                vals.append(contrib / total * 100.0 if total > 0 else 0.0)
            vals = np.array(vals)
            ax.bar(
                x,
                vals,
                bottom=bottom,
                width=0.72,
                color=CATEGORY_COLORS[cat],
                edgecolor="white",
                linewidth=0.4,
                label=CATEGORY_LABELS[cat],
            )
            bottom += vals

        ax.set_title(f"Overlap {int(round(overlap * 100))}%", fontsize=11, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.2f}%" if abs(t - round(t)) > 1e-9 else f"{int(round(t))}%" for t in target_vals], fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)

    for ax in axes[n_panels:]:
        ax.axis("off")

    left_ax = axes[0]
    left_ax.set_ylabel("Category share (%)", fontsize=11)
    for ax in axes[:n_panels]:
        ax.set_xlabel("Target abatement ratio", fontsize=10)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[cat], ec="white", lw=0.4, label=CATEGORY_LABELS[cat])
        for cat in CATEGORY_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Energy Conservation Technology Share of Abatement", fontsize=12)

    save_figure_bundle(
        fig,
        out_png,
        bbox_inches="tight",
        data_frames={"category_contribution": work},
    )
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot metric heatmaps (total abatement and unit abatement cost) for EC overlap sweep summary.")
    parser.add_argument(
        "--summary",
        default=os.path.join(DATA_DIR_NAME, EC_PORTFOLIO_DIR_NAME, SUMMARY_CSV_NAME),
        help="Input summary CSV path.")
    parser.add_argument(
        "--results-xlsx",
        default=None,
        help="Scenario detail workbook path. If provided, net unit cost after coal-saving revenue is used.")
    parser.add_argument(
        "--input-workbook",
        default=os.path.join(DATA_DIR_NAME, INPUT_WORKBOOK_NAME),
        help="Original workbook path used to reconstruct category-level abatement contributions.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(DATA_DIR_NAME, EC_PORTFOLIO_DIR_NAME),
        help="Directory to save output figures.")
    parser.add_argument(
        "--prefix",
        default=OUTPUT_PREFIX,
        help="Output filename prefix.")
    parser.add_argument(
        "--unit-cost-basis",
        choices=["target_trimmed", "target", "achieved"],
        default="target_trimmed",
        help="Unit-cost basis for heatmap. target_trimmed removes overshoot from highest-cost selected tech slices then divides by target.")
    parser.add_argument(
        "--unit-cost-levels",
        type=int,
        default=21,
        help="Number of discrete color bins for the unit-cost heatmap.")
    parser.add_argument(
        "--show-net-benefit",
        action="store_true",
        help="Plot and diagnose net benefit (-net cost) instead of net cost.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    summary_path = _resolve_input_path(args.summary, script_dir)
    output_dir = _resolve_output_dir(args.output_dir, script_dir)
    os.makedirs(output_dir, exist_ok=True)
    summary = _load_summary(summary_path)

    results_xlsx = args.results_xlsx
    if not results_xlsx:
        if summary_path.endswith("_summary.csv"):
            results_xlsx = summary_path.replace("_summary.csv", "_results.xlsx")
        else:
            results_xlsx = ""
    elif results_xlsx:
        results_xlsx = _resolve_input_path(results_xlsx, script_dir)

    input_workbook = _resolve_input_path(args.input_workbook, script_dir)

    if results_xlsx:
        summary = _attach_net_unit_cost_from_results(summary, results_xlsx, input_workbook)

    out_heat = os.path.join(output_dir, f"{args.prefix}{OUTPUT_METRIC_HEATMAPS_PNG_SUFFIX}")
    out_heat_data = os.path.join(output_dir, f"{args.prefix}{OUTPUT_METRIC_HEATMAP_DATA_XLSX_SUFFIX}")
    out_heat_untrimmed = os.path.join(output_dir, f"{args.prefix}{OUTPUT_UNTRIMMED_METRIC_HEATMAPS_PNG_SUFFIX}")
    out_heat_untrimmed_data = os.path.join(output_dir, f"{args.prefix}{OUTPUT_UNTRIMMED_METRIC_HEATMAP_DATA_XLSX_SUFFIX}")
    out_heat_gross = os.path.join(output_dir, f"{args.prefix}{OUTPUT_GROSS_METRIC_HEATMAPS_PNG_SUFFIX}")
    out_heat_gross_data = os.path.join(output_dir, f"{args.prefix}{OUTPUT_GROSS_METRIC_HEATMAP_DATA_XLSX_SUFFIX}")
    out_stack = os.path.join(output_dir, f"{args.prefix}{OUTPUT_CATEGORY_STACK_PNG_SUFFIX}")
    out_share = os.path.join(output_dir, f"{args.prefix}{OUTPUT_CATEGORY_SHARE_STACK_PNG_SUFFIX}")

    violations = export_heatmap_source_excel(
        summary,
        out_heat_data,
        unit_cost_basis=args.unit_cost_basis,
        show_net_benefit=args.show_net_benefit,
        metric_mode="net",
    )

    export_heatmap_source_excel(
        summary,
        out_heat_gross_data,
        unit_cost_basis=args.unit_cost_basis,
        show_net_benefit=False,
        metric_mode="gross",
    )
    export_heatmap_source_excel(
        summary,
        out_heat_untrimmed_data,
        unit_cost_basis="target",
        show_net_benefit=args.show_net_benefit,
        metric_mode="net",
    )

    plot_metric_heatmaps(
        summary,
        out_heat,
        unit_cost_basis=args.unit_cost_basis,
        unit_cost_levels=args.unit_cost_levels,
        show_net_benefit=args.show_net_benefit,
        metric_mode="net",
    )
    plot_metric_heatmaps(
        summary,
        out_heat_gross,
        unit_cost_basis=args.unit_cost_basis,
        unit_cost_levels=args.unit_cost_levels,
        show_net_benefit=False,
        metric_mode="gross",
    )
    plot_metric_heatmaps(
        summary,
        out_heat_untrimmed,
        unit_cost_basis="target",
        unit_cost_levels=args.unit_cost_levels,
        show_net_benefit=args.show_net_benefit,
        metric_mode="net",
    )
    if results_xlsx:
        plot_category_stack(summary, results_xlsx, input_workbook, out_stack)
        plot_category_share_stack(summary, results_xlsx, input_workbook, out_share)

    print(f"Saved: {out_heat}")
    print(f"Saved: {out_heat_data}")
    print(f"Saved: {out_heat_untrimmed}")
    print(f"Saved: {out_heat_untrimmed_data}")
    print(f"Saved: {out_heat_gross}")
    print(f"Saved: {out_heat_gross_data}")
    if results_xlsx:
        print(f"Saved: {out_stack}")
        print(f"Saved: {out_share}")
    print(f"Summary source: {summary_path}")
    if results_xlsx:
        print(f"Net unit cost source: {results_xlsx}")
        print(f"Input workbook: {input_workbook}")
    print(f"Unit cost basis: {args.unit_cost_basis}")
    print(f"Unit cost levels: {args.unit_cost_levels}")
    print(f"Show net benefit: {args.show_net_benefit}")
    if violations.empty:
        print("Monotonicity check: no decreasing steps along target ratio within each overlap row.")
    else:
        print(f"Monotonicity check: found {len(violations)} decreasing step(s); see sheet 'monotonicity_violations'.")


if __name__ == "__main__":
    main()