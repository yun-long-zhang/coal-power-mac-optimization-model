import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text

from plot_export_utils import save_figure_bundle


DATA_DIR_NAME = "data"
BASELINE_CURVE_NAME = "gurobi_original_curve.csv"
BASELINE_RESULTS_NAME = "gurobi_original_results.xlsx"
SENSITIVITY_ENVELOPE_NAME = "fleet_joint_macc_sensitivity_envelope.png"
SCENARIO_SORTED_DIR = "fleet_joint_scenario_sorted_macc"

COMPOSITE_1_OUT = "paper_composite_1_sensitivity_nodes.png"
COMPOSITE_2_OUT = "paper_composite_2_uncertainty_abc.png"

PLANT_COORD_XLSX_NAME = "1885_coal_power_plants_original.xlsx"

TECH_SPLIT_COLORS = {
    "EC": "#4DBBD5",
    "DB": "#3C5488",
    "GB": "#00A087",
    "CC": "#E64B35",
}


def _display_state_label(s: str) -> str:
    if len(s) == 6 and s[0] == "E" and s[2] == "C" and s[4] == "B":
        return f"{s[0:2]}-{s[2:4]}-{s[4:6]}"
    return s


def _display_ebc_state_label(s: str) -> str:
    if len(s) == 6 and s[0] == "E" and s[2] == "B" and s[4] == "C":
        return f"{s[0:2]}-{s[2:4]}-{s[4:6]}"
    return s


def _state_label_from_row_ebc(row: pd.Series, bc_threshold: float = 1e-8, binary_threshold: float = 0.5) -> str:
    ec = int(float(pd.to_numeric(row.get("x_ec", 0.0), errors="coerce") or 0.0) >= binary_threshold)
    cc = int(float(pd.to_numeric(row.get("x_cc", 0.0), errors="coerce") or 0.0) >= binary_threshold)
    br = float(pd.to_numeric(row.get("blend_ratio", 0.0), errors="coerce") or 0.0)
    bc = int(br > bc_threshold)
    return f"E{ec}B{bc}C{cc}"


def _state_label_from_row(row: pd.Series, bc_threshold: float = 1e-8, binary_threshold: float = 0.5) -> str:
    ec = int(float(pd.to_numeric(row.get("x_ec", 0.0), errors="coerce") or 0.0) >= binary_threshold)
    cc = int(float(pd.to_numeric(row.get("x_cc", 0.0), errors="coerce") or 0.0) >= binary_threshold)
    br = float(pd.to_numeric(row.get("blend_ratio", 0.0), errors="coerce") or 0.0)
    bc = int(br > bc_threshold)
    return f"E{ec}C{cc}B{bc}"


def _sheet_sort_key(name: str) -> Tuple[int, str]:
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        return int(digits), name
    return (10**9), name


def _load_state_share_series(result_xlsx: str) -> pd.DataFrame:
    xl = pd.ExcelFile(result_xlsx)
    plant_sheets = sorted([s for s in xl.sheet_names if s.startswith("pm")], key=_sheet_sort_key)

    rows = []
    for sname in plant_sheets:
        pdf = pd.read_excel(result_xlsx, sheet_name=sname)
        need = ["x_ec", "x_cc", "blend_ratio", "target_abatement"]
        if any(c not in pdf.columns for c in need) or pdf.empty:
            continue
        target = pd.to_numeric(pdf["target_abatement"], errors="coerce").iloc[0]
        if not np.isfinite(target):
            continue
        sub = pdf[["x_ec", "x_cc", "blend_ratio"]].copy()
        sub["state"] = sub.apply(_state_label_from_row, axis=1)
        sub["target_abatement"] = float(target)
        rows.append(sub[["target_abatement", "state"]])

    if not rows:
        return pd.DataFrame()

    state_df = pd.concat(rows, ignore_index=True)
    states = [
        "E0C0B0", "E1C0B0", "E0C1B0", "E1C1B0",
        "E0C0B1", "E1C0B1", "E0C1B1", "E1C1B1",
    ]
    counts = state_df.groupby(["target_abatement", "state"]).size().unstack(fill_value=0)
    for s in states:
        if s not in counts.columns:
            counts[s] = 0
    counts = counts[states].sort_index()
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return shares


def _load_tech_share_split_series(
    result_xlsx: str,
    bc_threshold: float = 1e-8,
    bc_split_ratio_threshold: float = 0.2,
) -> pd.DataFrame:
    xl = pd.ExcelFile(result_xlsx)
    plant_sheets = sorted([s for s in xl.sheet_names if s.startswith("pm")], key=_sheet_sort_key)

    rows = []
    for sname in plant_sheets:
        pdf = pd.read_excel(result_xlsx, sheet_name=sname)
        need = ["target_id", "target_abatement", "ec_abatement_ton", "bc_abatement_ton", "cc_capture_ton", "blend_ratio"]
        if any(c not in pdf.columns for c in need) or pdf.empty:
            continue

        target = pd.to_numeric(pdf["target_abatement"], errors="coerce").iloc[0]
        target_id = pd.to_numeric(pdf["target_id"], errors="coerce").iloc[0]
        if not np.isfinite(target):
            continue

        ec = pd.to_numeric(pdf["ec_abatement_ton"], errors="coerce").fillna(0.0)
        bc = pd.to_numeric(pdf["bc_abatement_ton"], errors="coerce").fillna(0.0)
        cc = pd.to_numeric(pdf["cc_capture_ton"], errors="coerce").fillna(0.0)
        br = pd.to_numeric(pdf["blend_ratio"], errors="coerce").fillna(0.0)

        db_mask = (br > bc_threshold) & (br <= bc_split_ratio_threshold)
        gb_mask = br > bc_split_ratio_threshold

        rows.append(
            {
                "target_id": int(target_id) if np.isfinite(target_id) else np.nan,
                "target_abatement": float(target),
                "EC": float(ec.sum()),
                "DB": float(bc[db_mask].sum()),
                "GB": float(bc[gb_mask].sum()),
                "CC": float(cc.sum()),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["target_abatement", "target_id"], na_position="last").reset_index(drop=True)
    total = out[["EC", "DB", "GB", "CC"]].sum(axis=1).replace(0.0, np.nan)
    out["share_EC"] = (out["EC"] / total).fillna(0.0)
    out["share_DB"] = (out["DB"] / total).fillna(0.0)
    out["share_GB"] = (out["GB"] / total).fillna(0.0)
    out["share_CC"] = (out["CC"] / total).fillna(0.0)
    return out


def _plot_tech_share_split_panel(ax: plt.Axes, data_dir: str, xlim: Tuple[float, float]) -> None:
    result_xlsx = os.path.join(data_dir, BASELINE_RESULTS_NAME)
    if not os.path.exists(result_xlsx):
        ax.text(0.5, 0.5, "Missing baseline results workbook", ha="center", va="center")
        return

    shares = _load_tech_share_split_series(result_xlsx)
    if shares.empty:
        ax.text(0.5, 0.5, "No technology share data", ha="center", va="center")
        return

    x = shares["target_abatement"].to_numpy(dtype=float)
    y_ec = shares["share_EC"].to_numpy(dtype=float) * 100.0
    y_db = shares["share_DB"].to_numpy(dtype=float) * 100.0
    y_gb = shares["share_GB"].to_numpy(dtype=float) * 100.0
    y_cc = shares["share_CC"].to_numpy(dtype=float) * 100.0

    labels = ["EC", "DB", "GB", "CC"]
    colors = [TECH_SPLIT_COLORS[k] for k in labels]
    ax.stackplot(x, y_ec, y_db, y_gb, y_cc, labels=labels, colors=colors, alpha=0.92)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(xlim)
    ax.set_ylabel("Abatement share (%)")
    ax.set_title("(b) Technology abatement share", fontfamily="Arial Unicode MS")
    ax.grid(alpha=0.2, linestyle="-")

    ax.legend(frameon=True, facecolor="white", edgecolor="#575757C9", ncol=1, loc="center right", bbox_to_anchor=(1.03, 0.5), fontsize=8)


def _load_curve(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"target_abatement", "total_cost"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    out = df.copy()
    out["target_abatement"] = pd.to_numeric(out["target_abatement"], errors="coerce")
    out["total_cost"] = pd.to_numeric(out["total_cost"], errors="coerce")
    if "marginal_cost_diff" in out.columns:
        out["marginal_cost_diff"] = pd.to_numeric(out["marginal_cost_diff"], errors="coerce")
    else:
        out["marginal_cost_diff"] = out["total_cost"].diff() / out["target_abatement"].diff().replace(0, np.nan)

    if "target_id" in out.columns:
        out["target_id"] = pd.to_numeric(out["target_id"], errors="coerce")

    out = out.dropna(subset=["target_abatement", "marginal_cost_diff"]).sort_values("target_abatement").reset_index(drop=True)
    return out


def _curve_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(df["target_abatement"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["marginal_cost_diff"], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    uniq_x, uniq_idx = np.unique(x, return_index=True)
    return uniq_x, y[uniq_idx]


def _interp_to_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if x.size < 2:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, x, y, left=np.nan, right=np.nan)


def _target_id_to_abatement(baseline_df: pd.DataFrame, target_ids: List[int]) -> Dict[int, float]:
    mapping: Dict[int, float] = {}

    if "target_id" in baseline_df.columns and baseline_df["target_id"].notna().any():
        tmp = baseline_df.dropna(subset=["target_id"]).copy()
        tmp["target_id"] = tmp["target_id"].astype(int)
        by_id = tmp.set_index("target_id")["target_abatement"].to_dict()
        for tid in target_ids:
            if tid in by_id:
                mapping[tid] = float(by_id[tid])

    if len(mapping) != len(target_ids):
        sorted_df = baseline_df.sort_values("target_abatement").reset_index(drop=True)
        for tid in target_ids:
            if tid in mapping:
                continue
            idx = tid - 1
            if 0 <= idx < len(sorted_df):
                mapping[tid] = float(sorted_df.loc[idx, "target_abatement"])

    return mapping


def _compute_bc_share_for_target(result_xlsx: str, target_id: int) -> float:
    if not os.path.exists(result_xlsx):
        return np.nan

    sheet = f"pm{target_id:02d}"
    try:
        pdf = pd.read_excel(result_xlsx, sheet_name=sheet)
    except Exception:
        return np.nan

    needed = {"bc_abatement_ton", "total_abatement_ton"}
    if not needed.issubset(set(pdf.columns)) or pdf.empty:
        return np.nan

    bc_total = pd.to_numeric(pdf["bc_abatement_ton"], errors="coerce").fillna(0.0).sum()
    ab_total = pd.to_numeric(pdf["total_abatement_ton"], errors="coerce").fillna(0.0).sum()
    if not np.isfinite(ab_total) or abs(ab_total) < 1e-12:
        return np.nan
    return float(bc_total / ab_total)


def _compute_share_for_target(result_xlsx: str, target_id: int, numerator_col: str, denominator_col: str = "total_abatement_ton") -> float:
    if not os.path.exists(result_xlsx):
        return np.nan

    sheet = f"pm{target_id:02d}"
    try:
        pdf = pd.read_excel(result_xlsx, sheet_name=sheet)
    except Exception:
        return np.nan

    if numerator_col not in pdf.columns or denominator_col not in pdf.columns or pdf.empty:
        return np.nan

    numerator = pd.to_numeric(pdf[numerator_col], errors="coerce").fillna(0.0).sum()
    denominator = pd.to_numeric(pdf[denominator_col], errors="coerce").fillna(0.0).sum()
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return np.nan
    return float(numerator / denominator)


def _load_plant_coordinates(data_dir: str) -> pd.DataFrame:
    coord_path = os.path.join(data_dir, PLANT_COORD_XLSX_NAME)
    if not os.path.exists(coord_path):
        return pd.DataFrame(columns=["plant_id", "longitude", "latitude"])

    pdf = pd.read_excel(coord_path)
    if pdf.empty:
        return pd.DataFrame(columns=["plant_id", "longitude", "latitude"])

    cols = {str(c).strip().lower(): c for c in pdf.columns}
    pid_col = cols.get("plant_i")
    lon_col = cols.get("longitude")
    lat_col = cols.get("latitude")
    if pid_col is None or lon_col is None or lat_col is None:
        return pd.DataFrame(columns=["plant_id", "longitude", "latitude"])

    out = pd.DataFrame(
        {
            "plant_id": pd.to_numeric(pdf[pid_col], errors="coerce"),
            "longitude": pd.to_numeric(pdf[lon_col], errors="coerce"),
            "latitude": pd.to_numeric(pdf[lat_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["plant_id", "longitude", "latitude"]).copy()
    out["plant_id"] = out["plant_id"].astype(int)
    out = out.drop_duplicates(subset=["plant_id"])
    return out


def _load_node_spatial_states(result_xlsx: str, target_id: int, coords: pd.DataFrame) -> pd.DataFrame:
    if coords.empty or not os.path.exists(result_xlsx):
        return pd.DataFrame(columns=["plant_id", "longitude", "latitude", "state"])

    sheet = f"pm{target_id:02d}"
    try:
        sdf = pd.read_excel(result_xlsx, sheet_name=sheet)
    except Exception:
        return pd.DataFrame(columns=["plant_id", "longitude", "latitude", "state"])

    need = {"plant_id", "x_ec", "x_cc", "blend_ratio"}
    if not need.issubset(set(sdf.columns)) or sdf.empty:
        return pd.DataFrame(columns=["plant_id", "longitude", "latitude", "state"])

    sub = sdf[["plant_id", "x_ec", "x_cc", "blend_ratio"]].copy()
    sub["plant_id"] = pd.to_numeric(sub["plant_id"], errors="coerce")
    sub = sub.dropna(subset=["plant_id"]).copy()
    sub["plant_id"] = sub["plant_id"].astype(int)
    sub["state"] = sub.apply(_state_label_from_row_ebc, axis=1)

    merged = sub.merge(coords, on="plant_id", how="inner")
    merged = merged.dropna(subset=["longitude", "latitude", "state"])
    return merged[["plant_id", "longitude", "latitude", "state"]]


def _plot_spatial_state_panel(
    ax: plt.Axes,
    node_df: pd.DataFrame,
    title: str,
    show_ylabel: bool = True,
    highlight_color: str = "#2F2F2F",
    previous_df: pd.DataFrame = None,
    hide_circle_legend: bool = False,
) -> None:
    states = [
        "E0B0C0", "E1B0C0", "E0B1C0", "E1B1C0",
        "E0B0C1", "E1B0C1", "E0B1C1", "E1B1C1",
    ]
    marker_map = {
        "E0B0C0": "x",
        "E1B0C0": "o",
        "E0B1C0": "s",
        "E1B1C0": "^",
        "E0B0C1": "D",
        "E1B0C1": "v",
        "E0B1C1": "<",
        "E1B1C1": ">",
    }

    ax.set_title(title, fontsize=12)
    ax.set_xlim(72.0, 136.0)
    ax.set_ylim(15.0, 55.0)
    ax.set_xlabel("Longitude (deg)")
    if show_ylabel:
        ax.set_ylabel("Latitude (deg)")
    else:
        ax.set_ylabel("")
    ax.grid(alpha=0.2, linestyle="-")

    if node_df.empty:
        ax.text(0.5, 0.5, "No spatial state data", ha="center", va="center", transform=ax.transAxes)
        return

    changed_ids = set()
    if previous_df is not None and not previous_df.empty:
        prev_states = previous_df[["plant_id", "state"]].rename(columns={"state": "prev_state"})
        cmp_df = node_df.merge(prev_states, on="plant_id", how="left")
        changed_ids = set(cmp_df.loc[cmp_df["prev_state"].notna() & (cmp_df["prev_state"] != cmp_df["state"]), "plant_id"].astype(int).tolist())

    legend_label_assigned = set()
    for s in states:
        sub = node_df[node_df["state"] == s]
        if sub.empty:
            continue
        mk = marker_map[s]
        changed = sub[sub["plant_id"].isin(changed_ids)]
        stable = sub[~sub["plant_id"].isin(changed_ids)]

        # Assign a legend label for this state exactly once — on the first
        # non-empty scatter call (stable preferred).  This ensures every
        # marker type that actually appears in the plot gets a legend entry.
        # Detect whether this state needs a legend label on this iteration.
        needs_label = s not in legend_label_assigned and not (hide_circle_legend and s == "E1B0C0")

        def _scatter_points(frame: pd.DataFrame, color: str, linewidth: float, force_label: bool = False) -> None:
            if frame.empty:
                return
            # Assign the legend label on the first non-empty branch, but always
            # use black (#2F2F2F) for the legend entry irrespective of the
            # actual scatter color (which may be highlight_color for changed points).
            assign_here = needs_label and force_label
            label_str = _display_ebc_state_label(s) if assign_here else None

            if mk == "x":
                ax.scatter(
                    frame["longitude"].to_numpy(dtype=float),
                    frame["latitude"].to_numpy(dtype=float),
                    marker=mk,
                    s=24,
                    c=color,
                    linewidths=linewidth,
                    alpha=0.95,
                    label=label_str,
                )
            else:
                ax.scatter(
                    frame["longitude"].to_numpy(dtype=float),
                    frame["latitude"].to_numpy(dtype=float),
                    marker=mk,
                    s=24,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=linewidth,
                    alpha=0.95,
                    label=label_str,
                )

        # Plot stable (black) first, with legend label if needed.
        _scatter_points(stable, "#2F2F2F", 0.9, force_label=True)
        if needs_label:
            legend_label_assigned.add(s)
        # Plot changed; do NOT assign a label here (legend already set by stable,
        # or will be set by a dummy proxy below if stable was empty).
        _scatter_points(changed, highlight_color, 1.2, force_label=False)

        # If stable was empty and this state still needs a label, create a
        # dummy invisible proxy legend handle in black.
        if stable.empty and needs_label:
            legend_label_assigned.add(s)
            ax.scatter([], [], marker=mk, s=24,
                       facecolors="none" if mk != "x" else "#2F2F2F",
                       edgecolors="#2F2F2F",
                       linewidths=0.9,
                       label=_display_ebc_state_label(s))

        _scatter_points(stable, "#2F2F2F", 0.9)
        # If stable was empty (all plants of this state are "changed"), assign
        # the legend label on the changed call instead.
        if stable.empty:
            _scatter_points(changed, highlight_color, 1.2)
        else:
            _scatter_points(changed, highlight_color, 1.2)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, ncol=1, frameon=False, fontsize=7, loc="lower right")


def _build_scenario_sorted_points(plant_df: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "plant_id",
        "ec_abatement_ton",
        "ec_cost",
        "bc_abatement_ton",
        "bc_cost",
        "cc_capture_ton",
        "cc_cost",
        "blend_ratio",
    ]
    missing = [c for c in needed if c not in plant_df.columns]
    if missing:
        raise ValueError(f"Plant scenario sheet missing columns: {sorted(missing)}")

    rows = []
    for _, r in plant_df.iterrows():
        plant_id = r["plant_id"]
        blend_ratio = float(pd.to_numeric(r.get("blend_ratio", 0.0), errors="coerce") or 0.0)
        tech_map = [
            ("EC", pd.to_numeric(r.get("ec_abatement_ton"), errors="coerce"), pd.to_numeric(r.get("ec_cost"), errors="coerce")),
            ("DB" if blend_ratio <= 0.2 else "GB", pd.to_numeric(r.get("bc_abatement_ton"), errors="coerce"), pd.to_numeric(r.get("bc_cost"), errors="coerce")),
            ("CC", pd.to_numeric(r.get("cc_capture_ton"), errors="coerce"), pd.to_numeric(r.get("cc_cost"), errors="coerce")),
        ]
        for tech, abate, cost in tech_map:
            if not np.isfinite(abate) or not np.isfinite(cost) or abate <= 0:
                continue
            rows.append(
                {
                    "plant_id": plant_id,
                    "tech": tech,
                    "abatement_ton": float(abate),
                    "cost": float(cost),
                    "unit_cost": float(cost / abate),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["plant_id", "tech", "abatement_ton", "cost", "unit_cost", "cum_abatement_ton"])

    out = pd.DataFrame(rows).sort_values(["unit_cost", "abatement_ton"], ascending=[True, False]).reset_index(drop=True)
    out["cum_abatement_ton"] = out["abatement_ton"].cumsum()
    return out


def _plot_sorted_unit_cost_panel(ax: plt.Axes, result_xlsx: str, target_id: int, title: str) -> None:
    if not os.path.exists(result_xlsx):
        ax.set_title(title, fontsize=12)
        ax.text(0.5, 0.5, "Missing baseline results workbook", ha="center", va="center", transform=ax.transAxes)
        return

    sheet = f"pm{target_id:02d}"
    try:
        pdf = pd.read_excel(result_xlsx, sheet_name=sheet)
    except Exception:
        ax.set_title(title, fontsize=12)
        ax.text(0.5, 0.5, f"Missing sheet {sheet}", ha="center", va="center", transform=ax.transAxes)
        return

    try:
        sorted_pts = _build_scenario_sorted_points(pdf)
    except Exception as exc:
        ax.set_title(title, fontsize=12)
        ax.text(0.5, 0.5, f"Cannot build sorted points\n{exc}", ha="center", va="center", transform=ax.transAxes)
        return

    ax.set_title(title, fontsize=12)
    if sorted_pts.empty:
        ax.text(0.5, 0.5, "No valid unit-cost points", ha="center", va="center", transform=ax.transAxes)
        return

    widths = sorted_pts["abatement_ton"].to_numpy(dtype=float)
    heights = sorted_pts["unit_cost"].to_numpy(dtype=float)
    x_right = np.cumsum(widths)
    x_left = x_right - widths

    tech_color = TECH_SPLIT_COLORS
    face_colors = sorted_pts["tech"].map(tech_color).fillna("#7F7F7F").to_numpy()

    ax.bar(
        x_left,
        heights,
        width=widths,
        align="edge",
        color=face_colors,
        edgecolor="none",
        linewidth=0.0,
        alpha=0.72,
        zorder=2,
    )

    x_step = np.concatenate(([0.0], x_right))
    y_step = np.concatenate(([heights[0]], heights))
    ax.step(x_step, y_step, where="post", color="#202020", linewidth=1.2, zorder=4)
    ax.fill_between(x_step, 0.0, y_step, step="post", color="#7F7F7F", alpha=0.05, zorder=1)

    ax.grid(alpha=0.2, linestyle="-")
    ax.axhline(0.0, color="black", linewidth=1.6, alpha=0.98, zorder=8)
    ax.set_xlabel("Cumulative abatement (t CO$_2$)", fontsize=12)
    ax.set_ylabel("Unit abatement cost (USD t CO$_2^{-1}$)", fontsize=12)
    ax.tick_params(labelsize=9)

    handles = [
        plt.Line2D([], [], color=TECH_SPLIT_COLORS["EC"], marker="o", linestyle="", markersize=4, label="EC"),
        plt.Line2D([], [], color=TECH_SPLIT_COLORS["DB"], marker="o", linestyle="", markersize=4, label="DB"),
        plt.Line2D([], [], color=TECH_SPLIT_COLORS["GB"], marker="o", linestyle="", markersize=4, label="GB"),
        plt.Line2D([], [], color=TECH_SPLIT_COLORS["CC"], marker="o", linestyle="", markersize=4, label="CC"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")


def _annotate_macc_point(
    ax: plt.Axes,
    x0: float,
    y0: float,
    label: str,
    color: str,
    text_offset: Tuple[float, float] = (8, -12),
) -> None:
    ax.scatter([x0], [y0], s=52, color=color, edgecolor="black", linewidth=0.7, zorder=11)
    ax.vlines(x0, 0.0, y0, colors=color, linestyles="-", linewidth=1.0, zorder=9)
    ax.annotate(
        label,
        xy=(x0, y0),
        xytext=text_offset,
        textcoords="offset points",
        fontsize=10,
        color=color,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.88},
    )


def _annotate_policy_thresholds(ax: plt.Axes, x_p12: float, x_p36: float, x_p47: float) -> None:
    ax.vlines(x_p12, -29.0, -20.0, colors="red", linewidth=3, linestyles="-", zorder=12)
    arrow_len = 0.34 * x_p12
    x_left = x_p12 - arrow_len
    ax.annotate(
        "",
        xy=(x_left, -24.5),
        xytext=(x_p12, -24.5),
        arrowprops={"arrowstyle": "->", "color": "red", "linewidth": 1.0},
    )
    ax.text(
        x_left - 0.0 * arrow_len,
        -40,
        "Profitable Technology Retrofits",
        ha="left",
        va="bottom",
        fontsize=9,
        color="red",
    )

    ax.vlines(x_p36, 60.0, 69.0, colors="red", linewidth=3, zorder=12)
    ax.text(x_p36 - 0.1 * x_p36, 72.0, "Carbon-Neutrality Threshold", ha="left", va="bottom", fontsize=9, color="red")
    ax.vlines(x_p47, 112.0, 121.0, colors="red", linewidth=3, zorder=12)
    ax.text(x_p47 - 0.019 * x_p47, ax.get_ylim()[1] * 0.86, "Maximum Abatement Potential", ha="right", va="center", fontsize=9, color="red")


def _ensure_node_images_exist(script_dir: str, data_dir: str, target_ids: List[int]) -> None:
    out_dir = os.path.join(data_dir, SCENARIO_SORTED_DIR)
    os.makedirs(out_dir, exist_ok=True)

    missing = []
    for tid in target_ids:
        p = os.path.join(out_dir, f"pm{tid:02d}_sorted_unit_cost_curve.png")
        if not os.path.exists(p):
            missing.append(tid)

    if not missing:
        return

    # Reuse existing plotting implementation to generate scenario sorted node curves.
    import plot_1_fleet_joint_macc as base_plot

    results_path = os.path.join(data_dir, BASELINE_RESULTS_NAME)
    if not os.path.exists(results_path):
        return

    try:
        base_plot._plot_all_scenario_sorted_macc(results_path, out_dir, mode="MILP")
    except Exception as exc:
        print(f"[warn] failed generating scenario sorted node figures: {exc}")


def _plot_top_sensitivity_with_markers(ax: plt.Axes, data_dir: str, node_map: Dict[int, float]) -> None:
    """Baseline MAC curve only — with carbon-price guide lines and policy annotations.
    (Sensitivity envelope curves have been moved to Figure 2, panel b.)"""
    baseline_path = os.path.join(data_dir, "gurobi_original_curve.csv")

    base_df = _load_curve(baseline_path)
    xb, yb = _curve_xy(base_df)
    grid = xb
    yb_grid = _interp_to_grid(xb, yb, grid)

    ax.plot(grid, yb_grid, color="#0B1F3A", linewidth=3.2, label="Baseline", zorder=6)

    finite_base = np.isfinite(grid) & np.isfinite(yb_grid)
    x_base = grid[finite_base]
    y_base = yb_grid[finite_base]
    x_left = 0.0  # fixed; aligned with build_composite_1 set_xlim(0, 4.8e9)

    def _x_at_base_level(y_level: float) -> float:
        if x_base.size < 2:
            return np.nan
        above = np.where(y_base >= y_level)[0]
        if above.size == 0:
            return np.nan
        i_hi = int(above[0])
        if i_hi == 0:
            return float(x_base[0])
        i_lo = i_hi - 1
        x_lo, x_hi = float(x_base[i_lo]), float(x_base[i_hi])
        y_lo, y_hi = float(y_base[i_lo]), float(y_base[i_hi])
        if not np.isfinite(y_lo) or not np.isfinite(y_hi) or abs(y_hi - y_lo) < 1e-12:
            return x_hi
        frac = (y_level - y_lo) / (y_hi - y_lo)
        frac = min(max(frac, 0.0), 1.0)
        return x_lo + frac * (x_hi - x_lo)

    guide_specs = [
        (9.0, "Carbon price = 9 (2025 average in China ETS)"),
        (66.0, "Carbon price = 66 (2025 average in UK ETS)"),
        (83.0, "Carbon price = 83 (2025 average in EU ETS)"),
    ]
    for y_level, text in guide_specs:
        x_end = _x_at_base_level(y_level)
        if not np.isfinite(x_end) or x_end <= 0:
            continue
        ax.hlines(y_level, x_left, x_end, colors="#7A7A7A", linestyles="--", linewidth=1.0, alpha=0.9, zorder=1)
        ax.text(
            x_left + (x_end - x_left) * 0.60,
            y_level + 1.8,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#000000",
            zorder=7,
        )

    x_scale = 1e9 if np.nanmax(grid) > 1e6 else 1.0
    p12_base_x = 1.17 * x_scale
    p12_x = p12_base_x - 0.03 * x_scale
    marker_specs = [
        ("p12", p12_x, "black"),
        ("p36", float(node_map.get(37, 3.6 * x_scale)), "#1F77B4"),
        ("p47", float(node_map.get(48, node_map.get(47, 0.0))), "#D62728"),
    ]
    for label, x0, color in marker_specs:
        if not np.isfinite(x0) or x0 <= 0:
            continue
        y0 = float(np.interp(x0, xb, yb)) if xb.size >= 2 else np.nan
        if not np.isfinite(y0):
            continue
        if label == "p12":
            text_offset = (-6, -16)
        elif label == "p47":
            text_offset = (0, 14)  # directly above the marker
        else:
            text_offset = (8, -12)
        _annotate_macc_point(ax, x0, y0, label, color, text_offset=text_offset)

    x_p12 = marker_specs[0][1]
    x_p36 = marker_specs[1][1]
    x_p47 = marker_specs[2][1]
    if all(np.isfinite(v) and v > 0 for v in [x_p12, x_p36, x_p47]):
        _annotate_policy_thresholds(ax, x_p12, x_p36, x_p47)

    ax.set_title("(a) Maginal abatement cost curve under sensitivity analysis", fontsize=13)
    ax.set_xlabel("Target abatement (Gt CO$_2$)", fontsize=12)
    ax.set_ylabel("Marginal abatement cost (USD t CO$_2^{-1}$)", fontsize=12)
    ax.set_xlim(0.0, 4.8e9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1e9:.1f}"))
    ax.set_xticks([0, 1e9, 2e9, 3e9, 4e9])
    ax.grid(alpha=0.2, linestyle="-")
    ax.axhline(0.0, color="black", linewidth=1.8, alpha=1.0, zorder=12)
    # Droplines at the last baseline point (x axis in tons, same as plot units)
    finite_base = np.isfinite(grid) & np.isfinite(yb_grid)
    if finite_base.any():
        _add_droplines(ax, grid[finite_base][-1], yb_grid[finite_base][-1], "#1F1F1F", lw=1.2)


def _draw_image_or_placeholder(ax: plt.Axes, img_path: str, title: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    if os.path.exists(img_path):
        arr = mpimg.imread(img_path)
        ax.imshow(arr)
        ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, f"Missing image\n{os.path.basename(img_path)}", ha="center", va="center", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)


from matplotlib.legend import Legend as MplLegend

def _increase_figure_fontsizes(fig: plt.Figure, delta: float = 4.0) -> None:
    if abs(delta) < 1e-12:
        return
    for txt in fig.findobj(match=lambda artist: isinstance(artist, Text)):
        parent = getattr(txt, 'axes', None)
        if isinstance(parent, MplLegend):
            continue
        size = txt.get_fontsize()
        if np.isfinite(size):
            txt.set_fontsize(float(size) + delta)


def build_composite_1(script_dir: str, data_dir: str, target_ids: List[int], out_path: str) -> str:
    baseline_df = _load_curve(os.path.join(data_dir, BASELINE_CURVE_NAME))
    node_map = _target_id_to_abatement(baseline_df, sorted(set(target_ids + [36, 47])))
    result_xlsx = os.path.join(data_dir, BASELINE_RESULTS_NAME)
    coords = _load_plant_coordinates(data_dir)

    fig = plt.figure(figsize=(15.2, 19.2))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.15, 1.15, 1.15, 1.15], hspace=0.22, wspace=0.12)

    ax_top = fig.add_subplot(gs[0, :])
    _plot_top_sensitivity_with_markers(ax_top, data_dir, node_map)

    # Add RRH unit-cost boxplot at x = 4.5e9 (panel a overlay)
    rrh_path = os.path.join(data_dir, PLANT_COORD_XLSX_NAME)
    if os.path.exists(rrh_path):
        rrh_df = pd.read_excel(rrh_path, sheet_name="RRH")
        rrh_uc = pd.to_numeric(rrh_df["unit _cost"], errors="coerce").dropna()
        if len(rrh_uc) > 0:
            box_data = [rrh_uc.to_numpy(dtype=float)]
            bp = ax_top.boxplot(
                box_data,
                positions=[4.0e9],
                widths=0.05e9,
                showfliers=False,
                whis=(0,100),
                patch_artist=True,
                zorder=20,
                manage_ticks=False,
            )
            # Style the box: fill white with dark edge, hide whiskers/caps/medians
            for line in bp["whiskers"]:
                line.set_visible(True)
                line.set_linewidth(3)
                line.set_color("#118E62")
            for cap in bp["caps"]:
                cap.set_visible(False)
            for line in bp["medians"]:
                line.set_visible(False)
            for line in bp["fliers"]:
                line.set_visible(False)
            for box in bp["boxes"]:
                box.set_facecolor("#118E62")
                box.set_edgecolor("#118E62")
                box.set_linewidth(1.5)
                box.set_alpha(0.92)
            # Add median marker manually
            median_val = float(np.median(rrh_uc))
            ax_top.scatter(
                [4.0e9], [median_val],
                marker="_", color="#FFFEFE", s=80, linewidth=2.0, zorder=22,
            )
            # Add a small label
            ax_top.text(
                4.05e9,
                median_val,
                "Median",
                ha="left",
                va="center",
                fontsize=7,
                color="#2F2F2F",
                zorder=25,
            )
            ax_top.text(
                3.7e9, median_val + 20.0,
                "unit cost range of RRH",
                ha="left", va="center", fontsize=8,
                color="#2F2F2F", fontweight="bold",
                zorder=25,
            )

    # Format x-axis ticks in scientific notation (×10⁹)
    ax_top.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x / 1e9:.1f}")
    )
    ax_top.set_xticks([0, 1e9, 2e9, 3e9, 4e9])

    # ── Dual y-axis: average carbon intensity of electricity (tCO2/MWh) ──
    # Load baseline total CO2 emissions from the curve / result
    # Total annual generation (coal fleet) is constant — unaffected by retrofit type.
    plants_raw = pd.read_excel(rrh_path, sheet_name="coal_plant_i")
    pcols = list(plants_raw.columns)
    def _pick_col(candidates):
        for c in pcols:
            if str(c).strip().lower() in candidates:
                return c
        return ""
    em_col  = _pick_col(["carbon_emissions_ton", "co2", "carbon_emissions"])
    pw_col  = _pick_col(["power", "capacity_xi", "capacity", "power_xi"])
    rh_col  = _pick_col(["running_hours", "hours", "runninghours"])

    total_base_emission = float(pd.to_numeric(plants_raw[em_col], errors="coerce").sum())
    total_gen_kwh = (
        pd.to_numeric(plants_raw[pw_col], errors="coerce").fillna(0.0)
        * pd.to_numeric(plants_raw[rh_col], errors="coerce").fillna(0.0)
        * 1000.0
    ).sum()

    baseline_path = os.path.join(data_dir, BASELINE_CURVE_NAME)
    base_df = _load_curve(baseline_path)
    x_abate = base_df["achieved_abatement"].to_numpy(dtype=float)
    # Residual emissions = baseline - achieved_abatement
    resid_emission = total_base_emission - x_abate
    # Carbon intensity: tCO2 / (kWh / 1e6) = tCO2/MWh
    carbon_intensity = resid_emission / (total_gen_kwh / 1e6)

    ax_top2 = ax_top.twinx()
    ax_top2.plot(
        x_abate, carbon_intensity,
        color="#D61AE4", linewidth=2.0, linestyle="-", alpha=0.85,
        zorder=5,
    )
    ax_top2.set_ylabel("Avg carbon intensity (tCO$_2$/MWh)", fontsize=11, color="#D61AE4")
    ax_top2.tick_params(axis="y", labelcolor="#D61AE4")
    ax_top2.set_ylim(-300, carbon_intensity[0] * 1.12)
    # Align right spine with plot boundary
    ax_top2.spines["right"].set_position(("outward", 0))
    ax_top.spines["right"].set_visible(False)

    xlim_top = ax_top.get_xlim()

    ax_mid = fig.add_subplot(gs[1, :], sharex=ax_top)
    _plot_tech_share_split_panel(ax_mid, data_dir, xlim_top)
    ax_mid.set_xlabel("Target abatement (Gt CO$_2$)", fontsize=12)

    panel_label_map = {12: 12, 37: 36, 48: 47}

    for i, tid in enumerate(target_ids):
        ax = fig.add_subplot(gs[2, i])
        pdisp = panel_label_map.get(tid, tid)
        _plot_sorted_unit_cost_panel(ax, result_xlsx, tid, f"(c{i+1}) Detailed portfolio in P{pdisp}")
        if i > 0:
            ax.set_ylabel("")

    spatial_frames = [_load_node_spatial_states(result_xlsx, tid, coords) for tid in target_ids]
    phase_titles = [
        "Target 1: \naiming at low-regret opportunities",
        "Target 2: \naiming to net-zero transition",
        "Target 3: \naiming for negative emissions",
    ]
    phase_descriptions = [
        "Layout 1: \nNational EC adoption \n& Southeast DB pilot",
        "Layout 2: \nGB expansion in \nCentral & North; \nCCS deployment in \nNorth & West",
        "Layout 3: \nCCS rollout in South & East",
    ]
    for i, tid in enumerate(target_ids):
        ax = fig.add_subplot(gs[3, i])
        prev_df = spatial_frames[i - 1] if i > 0 else None
        highlight_color = "#00A087" if i == 1 else "#D62728" if i == 2 else "#868181"
        pdisp = panel_label_map.get(tid, tid)
        _plot_spatial_state_panel(
            ax,
            spatial_frames[i],
            f"(d{i+1}) Fleet retrofit layout in P{pdisp}",
            show_ylabel=(i == 0),
            highlight_color=highlight_color,
            previous_df=prev_df,
            hide_circle_legend=(i == 2),
        )
        if i > 0:
            ax.set_ylabel("")
        # Phase title near the upper border
        ax.text(0.5, 0.97, phase_titles[i], transform=ax.transAxes,
                ha="center", va="top", fontsize=10, color="#2F2F2F",
                fontweight="bold", zorder=30)
        # Phase description near the left border
        ax.text(0.02, 0.45, phase_descriptions[i], transform=ax.transAxes,
                ha="left", va="center", fontsize=7, color="#2F2F2F",
                fontweight="bold", zorder=30)

    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.05, right=0.98, hspace=0.22, wspace=0.12)
    _increase_figure_fontsizes(fig, delta=4.0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    baseline_df = _load_curve(os.path.join(data_dir, BASELINE_CURVE_NAME))
    node_df = pd.DataFrame([
        {"target_id": tid, "target_abatement": node_map.get(tid, np.nan)}
        for tid in sorted(target_ids + [35, 47])
    ])
    save_figure_bundle(
        fig,
        out_path,
        dpi=320,
        bbox_inches="tight",
        data_frames={"mac_curve": baseline_df, "policy_nodes": node_df},
    )
    plt.close(fig)
    return out_path


def build_scenario_comparison(script_dir: str, data_dir: str, target_ids: List[int], out_path: str) -> str:
    """Standalone figure: cost + abatement bar-line comparison across scenarios."""
    baseline_curve = _load_curve(os.path.join(data_dir, BASELINE_CURVE_NAME))
    target_map = _target_id_to_abatement(baseline_curve, target_ids)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()

    scenarios = [
        ("Baseline", "gurobi_original_curve.csv", "#1F1F1F"),
        ("No-EC", "gurobi_no_ec_curve.csv", "#1F77B4"),
        ("Standalone", "standalone_mac_curve.csv", "#D62728"),
    ]

    x_selected = np.array([target_map[t] for t in target_ids if t in target_map], dtype=float)
    if x_selected.size == 0:
        ax.text(0.5, 0.5, "Selected targets not found", ha="center", va="center", transform=ax.transAxes)
        save_figure_bundle(
            fig,
            out_path,
            dpi=320,
            bbox_inches="tight",
            data_frames={"scenario_nodes": pd.DataFrame({"target_abatement": x_selected})},
        )
        plt.close(fig)
        return out_path

    n_groups = len(x_selected)
    n_bars = len(scenarios)
    x_pos = np.arange(n_groups)
    bar_w = min(0.6 / max(1, n_groups) * 0.5, 0.12)

    for idx, (label, fname, color) in enumerate(scenarios):
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            cdf = _load_curve(path)
            costs, abates = [], []
            for tid in target_ids:
                tgt = target_map.get(tid)
                if tgt is None:
                    costs.append(np.nan); abates.append(np.nan); continue
                row = cdf.iloc[(cdf["target_abatement"] - tgt).abs().idxmin()]
                costs.append(float(row["total_cost"]))
                abates.append(float(row["achieved_abatement"]))
            costs = np.array(costs); abates = np.array(abates)

            offset = (idx - (n_bars - 1) / 2) * bar_w
            ax2.bar(x_pos + offset, costs / 1e9, width=bar_w * 0.85,
                    facecolor="none", edgecolor=color, linewidth=2.0,
                    label=f"Cost - {label}", zorder=3)
            ax.scatter(x_pos + offset, abates / 1e9,
                       marker="o", s=40, color=color, edgecolor="white", linewidth=0.5,
                       label=f"Abatement - {label}", zorder=8)
        except Exception:
            continue

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{v/1e9:.2f}" for v in x_selected], fontsize=12)
    ax.set_xlabel("Abatement target (Gt CO₂)", fontsize=12)
    ax.set_ylabel("Abatement (Gt CO₂)", fontsize=12)
    ax.grid(alpha=0.2, linestyle="-")
    ax.set_title("Scenario comparison: total cost and abatement at nodes")
    ax2.set_ylabel("Total cost (Billion USD)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=True, fontsize=12, ncol=2, loc="upper left", bbox_to_anchor=(0, 0.85))

    fig.tight_layout()
    save_figure_bundle(fig, out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _set_3col_legend(ax: plt.Axes, legend_specs: list) -> None:
    """Draw three separate legends side-by-side to simulate a 3-column layout.
    
    legend_specs: list of tuples
      - ("Baseline", color, ls)  – only the first spec is baseline; placed in col 1
      - (["Item1", "Item2", ...], color, [ls1, ls2, ...]) – each is one column
    """
    def _mk_handle(color, ls):
        return plt.Line2D([], [], color=color, linestyle=ls, linewidth=2.0)

    # Build per-column handles and labels
    col_handles = []
    col_labels = []
    baseline_entry = None

    for spec in legend_specs:
        if isinstance(spec[0], str):
            label, color, ls = spec
            baseline_entry = (_mk_handle(color, ls), label)
        elif isinstance(spec[0], list):
            sub_labels, color, linestyles = spec
            hh, ll = [], []
            for i, lbl in enumerate(sub_labels):
                ls = linestyles[i % len(linestyles)]
                hh.append(_mk_handle(color, ls))
                ll.append(lbl)
            col_handles.append(hh)
            col_labels.append(ll)

    ncols = len(col_handles)
    if ncols == 0:
        return

    # Prepend baseline to the first column if any
    if baseline_entry:
        bh, bl = baseline_entry
        col_handles[0] = [bh] + col_handles[0]
        col_labels[0] = [bl] + col_labels[0]

    # Use three separate legends with ncol=1 each, positioned side by side
    # bbox_to_anchor positions in axes fraction
    positions = [(0.0, 0.8), (0.15, 0.72), (0.30, 0.72)]

    for ci in range(min(ncols, len(positions))):
        if any(l.strip() != "" for l in col_labels[ci] if l != " "):
            leg = ax.legend(
                col_handles[ci], col_labels[ci],
                frameon=False,
                prop={'size': 12},
                ncol=1,
                loc="upper left",
                bbox_to_anchor=(positions[ci][0], positions[ci][1]),
                bbox_transform=ax.transAxes,
                handlelength=1.2,
                handletextpad=0.6,
                labelspacing=0.35,
                borderaxespad=0.0,
                
            )
            print("Legend", ci)
            for t in leg.get_texts():
                print(
                    t.get_text(),
                    t.get_fontsize(),
                    t.get_fontproperties().get_size_in_points() )
            renderer = ax.figure.canvas.get_renderer()
            bbox = leg.get_window_extent(renderer)
            print(
                f"Legend {ci}: width={bbox.width:.1f}, height={bbox.height:.1f}"
            )
            print(len(ax.artists))
            print(ax.legend_)
            ax.add_artist(leg)
    ax.legend_ = None

def _plot_sensitivity_envelope_only(
    ax: plt.Axes,
    case_specs: List[Dict[str, str]],
    target_ids: List[int],
    target_map: Dict[int, float],
    panel_title: str,
    common_xlim: Tuple[float, float],
    common_ylim: Tuple[float, float],
) -> None:
    """Plot MAC curves with fill-between envelopes — no dual axis, no bar charts."""

    valid_specs = []
    for s in case_specs:
        if os.path.exists(s["curve_path"]):
            valid_specs.append(s)

    if not valid_specs:
        ax.text(0.5, 0.5, "No scenario curve files", ha="center", va="center")
        return

    x_selected = np.array([target_map[t] for t in target_ids if t in target_map], dtype=float)
    if x_selected.size == 0:
        ax.text(0.5, 0.5, "Selected targets not found", ha="center", va="center")
        return

    saved_last_points = []

    # Group specs by color category
    color_groups = {}
    for s in valid_specs:
        color_groups.setdefault(s["color"], []).append(s)

    for color, members in color_groups.items():
        arr = []
        for s in members:
            cdf = _load_curve(s["curve_path"])
            x, y = _curve_xy(cdf)
            if x.size == 0:
                continue
            # Interpolate onto a common grid
            grid = np.linspace(common_xlim[0], common_xlim[1], 200)
            ys = _interp_to_grid(x, y, grid)
            arr.append(ys)

            is_baseline = bool(s.get("is_baseline", False))
            line_ok = np.isfinite(ys)
            if line_ok.any():
                ax.plot(
                    grid[line_ok],
                    ys[line_ok],
                    color=color,
                    linewidth=3.2 if is_baseline else 1.6,
                    linestyle=s.get("linestyle", "-"),
                    alpha=1.0 if is_baseline else 0.88,
                    label=s["label"],
                    zorder=8 if is_baseline else 5,
                )
            if x.size >= 2:
                saved_last_points.append((float(x[-1]), float(y[-1])))

        if len(arr) >= 2:
            mat = np.vstack(arr)
            y_min = np.nanmin(mat, axis=0)
            y_max = np.nanmax(mat, axis=0)
            finite = np.isfinite(y_min) & np.isfinite(y_max)
            if finite.any():
                ax.fill_between(grid[finite], y_min[finite], y_max[finite],
                                color=color, alpha=0.12, zorder=2)

    ax.set_title(panel_title, fontsize=13)
    ax.set_ylabel("Marginal abatement cost (USD/tCO₂)", fontsize=12)
    ax.set_xlim(common_xlim)
    ax.set_ylim(common_ylim)
    for x_last, y_last in saved_last_points:
        _add_droplines(ax, x_last, y_last, "#1F1F1F", lw=1.2)
    ax.grid(alpha=0.2, linestyle="-")
    ax.set_xticks(x_selected)
    ax.set_xticklabels([f"{v/1e9:.2f}" for v in x_selected], fontsize=10)
    ax.axhline(0.0, color="black", linewidth=1.6, alpha=1.0, zorder=12)
    # Legend handled externally by _set_3col_legend


def _plot_multi_curve_only(
    ax: plt.Axes,
    spec_groups: List[Tuple[str, List[Dict], int]],
    target_ids: List[int],
    target_map: Dict[int, float],
    panel_title: str,
    common_xlim: Tuple[float, float],
    common_ylim: Tuple[float, float],
) -> None:
    """Plot multiple groups of MAC curves on a single panel — no dual axis, no bars."""
    all_saved = []
    baseline_plotted = False  # only one baseline line across all groups

    for group_label, specs, line_style_idx in spec_groups:
        valid = [s for s in specs if os.path.exists(s["curve_path"])]
        if not valid:
            continue

        # Use the first entry's color as the group theme color
        line_styles = ["-", "-.", ":", "--"]

        for s in valid:
            cdf = _load_curve(s["curve_path"])
            x, y = _curve_xy(cdf)
            if x.size == 0:
                continue

            is_baseline = bool(s.get("is_baseline", False))
            if is_baseline:
                if baseline_plotted:
                    continue  # skip duplicate baseline
                baseline_plotted = True
                label_str = "Baseline (100% biomass, 20% EC, 90% CC)"
                plot_color = "#1F1F1F"
                lw = 3.2
                ls = "-"
                zorder = 8
            else:
                label_str = s["label"]
                plot_color = s["color"]
                lw = 2.0
                ls = s.get("linestyle", line_styles[line_style_idx % len(line_styles)])
                zorder = 5

            ax.plot(
                x, y,
                color=plot_color,
                linewidth=lw,
                linestyle=ls,
                alpha=1.0 if is_baseline else 0.88,
                label=label_str,
                zorder=zorder,
            )
            if x.size >= 2:
                all_saved.append((float(x[-1]), float(y[-1])))

    ax.set_title(panel_title, fontsize=13)
    ax.set_ylabel("Marginal abatement cost (USD/tCO₂)", fontsize=12)
    ax.set_xlim(common_xlim)
    ax.set_ylim(common_ylim)
    for x_last, y_last in all_saved:
        _add_droplines(ax, x_last, y_last, "#1F1F1F", lw=1.2)
    ax.grid(alpha=0.2, linestyle="-")
    ax.set_xticks([target_map[t] for t in target_ids if t in target_map])
    ax.set_xticklabels([f"{target_map[t]/1e9:.2f}" for t in target_ids if t in target_map], fontsize=10)
    ax.axhline(0.0, color="black", linewidth=1.6, alpha=1.0, zorder=12)
    # Legend handled externally by _set_3col_legend


def _plot_uncertainty_dualaxis_panel(
    ax: plt.Axes,
    case_specs: List[Dict[str, str]],
    target_ids: List[int],
    target_map: Dict[int, float],
    panel_title: str,
    share_ylabel: str,
    common_xlim: Tuple[float, float],
    common_ylim: Tuple[float, float],
) -> None:
    ax2 = ax.twinx()

    valid_specs = []
    for s in case_specs:
        if os.path.exists(s["curve_path"]):
            valid_specs.append(s)

    if not valid_specs:
        ax.text(0.5, 0.5, "No scenario curve files", ha="center", va="center")
        return

    x_selected = np.array([target_map[t] for t in target_ids if t in target_map], dtype=float)
    if x_selected.size == 0:
        ax.text(0.5, 0.5, "Selected targets not found", ha="center", va="center")
        return

    x_span = max(float(np.nanmax(x_selected) - np.nanmin(x_selected)), 1.0)
    base_w = x_span / max(30.0, 4.0 * len(x_selected))

    # Wider spread for bar widths to make scenario differences more visible
    width_scale = np.linspace(1.0, 0.2, len(valid_specs))
    saved_last_points = []

    for idx, s in enumerate(valid_specs):
        cdf = _load_curve(s["curve_path"])
        x, y = _curve_xy(cdf)
        if x.size == 0:
            continue

        is_baseline = bool(s.get("is_baseline", False))
        ax.plot(
            x,
            y,
            color=s["color"],
            linewidth=3.2 if is_baseline else 2.0,
            linestyle=s.get("linestyle", "-"),
            alpha=1.0 if is_baseline else 0.88,
            marker="o" if is_baseline else None,
            markersize=3.6 if is_baseline else 0,
            label=s["label"],
            zorder=8 if is_baseline else 5,
        )

        share_col = s.get("share_col", "")
        shares = []
        for tid in target_ids:
            if tid not in target_map:
                shares.append(np.nan)
                continue
            share = _compute_share_for_target(s["results_path"], tid, share_col) if share_col else _compute_bc_share_for_target(s["results_path"], tid)
            shares.append(share * 100.0 if np.isfinite(share) else np.nan)
        shares = np.array(shares, dtype=float)

        bar_w = base_w * width_scale[idx]
        ok = np.isfinite(shares)
        if ok.any():
            ax2.bar(
                x_selected[ok],
                shares[ok],
                width=bar_w,
                facecolor="none",
                edgecolor=s["color"],
                linewidth=2.0,
                label=f"{share_ylabel} - {s['label']}",
                zorder=2,
            )

        if x.size >= 2:
            saved_last_points.append((float(x[-1]), float(y[-1])))

    ax.set_title(panel_title)
    ax.set_ylabel("Marginal abatement cost (USD/tCO₂)")
    ax.set_xlim(common_xlim)
    ax.set_ylim(common_ylim)
    # Droplines after xlim/ylim are set so they reach the actual axes
    for x_last, y_last in saved_last_points:
        _add_droplines(ax, x_last, y_last, "#1F1F1F", lw=1.2)
    ax.grid(alpha=0.2, linestyle="-")
    ax.set_xticks(x_selected)
    ax.set_xticklabels([f"{v/1e9:.2f}" for v in x_selected])

    ax2.set_ylabel(f"{share_ylabel} (%)")
    ax2.set_ylim(0.0, 100.0)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, ncol=2, fontsize=12, loc="upper left", bbox_to_anchor=(0, 0.85))


def _add_droplines(ax: plt.Axes, x_last: float, y_last: float, color: str,
                   ls: str = "--", lw: float = 1.2, alpha: float = 0.45) -> None:
    """Add drop lines from (x_last, y_last) to both axes."""
    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()
    ax.vlines(x_last, 0, y_last, colors=color, linestyles=ls, linewidth=lw, alpha=alpha, zorder=2)
    ax.hlines(y_last, x_left, x_last, colors=color, linestyles=ls, linewidth=lw, alpha=alpha, zorder=2)


def _plot_mac_comparison_panel(ax: plt.Axes, data_dir: str) -> None:
    """Panel (a): MAC comparison – baseline vs standalone vs no-EC.
    Standalone uses raw unit-level data (sorted, no target-grid compression)."""
    last_points = []

    # Baseline and No-EC: loaded via _load_curve (joint-model format)
    for label, fname, color, ls, lw in [
        ("Baseline (joint model)", "gurobi_original_curve.csv", "#1F1F1F", "-", 3.2),
        ("No-EC (joint model)", "gurobi_no_ec_curve.csv", "#1F77B4", ":", 2.4),
    ]:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            cdf = _load_curve(path)
            x, y = _curve_xy(cdf)
            ax.plot(x / 1e9, y, color=color, linewidth=lw, linestyle=ls,
                    alpha=0.92, label=label, zorder=5)
            if len(x) >= 2:
                last_points.append((float(x[-1]) / 1e9, float(y[-1])))
        except Exception:
            continue

    # Standalone: load raw unit CSV
    raw_path = os.path.join(data_dir, "standalone_mac_raw_units.csv")
    if os.path.exists(raw_path):
        try:
            raw_df = pd.read_csv(raw_path)
            raw_x = pd.to_numeric(raw_df["cum_abatement"], errors="coerce").to_numpy()
            raw_y = pd.to_numeric(raw_df["unit_cost"], errors="coerce").to_numpy()
            mask = np.isfinite(raw_x) & np.isfinite(raw_y)
            x_plot = raw_x[mask] / 1e9
            y_plot = raw_y[mask]
            ax.plot(x_plot, y_plot, color="#D62728", linewidth=2.4,
                    linestyle="--", alpha=0.92, label="Standalone (EC+BC+CC)", zorder=5)
            if len(x_plot) >= 2:
                last_points.append((float(x_plot[-1]), float(y_plot[-1])))
        except Exception:
            pass

    ax.axhline(0, color="black", linewidth=1.5, alpha=0.9, zorder=4)
    ax.set_xlabel("Abatement target (Gt CO₂)", fontsize=12)
    ax.set_ylabel("Marginal abatement cost (USD/tCO₂)", fontsize=12)
    ax.set_title("(a) MAC curve comparison: baseline, standalone, and no-EC", fontsize=13)
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False, fontsize=12, loc="upper left", bbox_to_anchor=(0, 0.85))
    ax.set_xlim(0.0, 5.2)
    ax.set_ylim(-60.0, 140.0)
    # Droplines after xlim is set so they reach the actual axes
    for x_last, y_last in last_points:
        _add_droplines(ax, x_last, y_last, "#1F1F1F", lw=1.2)


def _plot_scenario_comparison_panel(
    ax: plt.Axes, data_dir: str, target_ids: List[int], target_map: Dict[int, float],
) -> None:
    """Panel (b): bar+line dual-axis – total cost and total abatement for three scenarios."""
    scenarios = [
        ("Baseline", "gurobi_original_curve.csv", "#1F1F1F"),
        ("No-EC", "gurobi_no_ec_curve.csv", "#1F77B4"),
        ("Standalone", "standalone_mac_curve.csv", "#D62728"),
    ]
    x_selected = np.array([target_map[t] for t in target_ids if t in target_map], dtype=float)
    if x_selected.size == 0:
        ax.text(0.5, 0.5, "Selected targets not found", ha="center", va="center")
        return

    n_groups = len(x_selected)
    n_bars = len(scenarios)
    bar_width = min(0.6 * x_selected[0] / n_bars * 0.6, 0.06e9) if n_groups > 1 else 0.05e9

    ax2 = ax.twinx()
    x_pos = np.arange(n_groups)

    for idx, (label, fname, color) in enumerate(scenarios):
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            cdf = _load_curve(path)
            # Get cost and abatement at each selected target abatement
            costs = []
            abates = []
            for tid in target_ids:
                tgt = target_map.get(tid)
                if tgt is None:
                    costs.append(np.nan)
                    abates.append(np.nan)
                    continue
                row = cdf.iloc[(cdf["target_abatement"] - tgt).abs().idxmin()]
                costs.append(float(row["total_cost"]))
                abates.append(float(row["achieved_abatement"]))
            costs = np.array(costs)
            abates = np.array(abates)

            offset = (idx - (n_bars - 1) / 2) * bar_width
            ax2.bar(
                x_pos + offset, costs / 1e9, width=bar_width * 0.85,
                facecolor="none", edgecolor=color, linewidth=2.0,
                label=f"Cost - {label}", zorder=2,
            )
            ax.scatter(
                x_pos + offset, abates / 1e9,
                marker="o", s=32, color=color, edgecolor="white", linewidth=0.5,
                label=f"Abatement - {label}", zorder=8,
            )
        except Exception:
            continue

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{v/1e9:.2f}" for v in x_selected], fontsize=12)
    ax.set_xlabel("Abatement target (Gt CO₂)", fontsize=12)
    ax.set_ylabel("Abatement (Gt CO₂)", fontsize=12)
    ax.grid(alpha=0.2, linestyle="-")
    ax.set_title("(b)  Scenario comparison: total cost and abatement at selected nodes", loc="left", fontsize=13, fontweight="bold")

    ax2.set_ylabel("Total cost (Billion USD)", fontsize=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=True, fontsize=12, ncol=3, loc="upper left", bbox_to_anchor=(0, 0.85))


def build_composite_2(script_dir: str, data_dir: str, target_ids: List[int], out_path: str) -> str:
    baseline_curve = _load_curve(os.path.join(data_dir, BASELINE_CURVE_NAME))
    target_map = _target_id_to_abatement(baseline_curve, target_ids)
    fig, axes = plt.subplots(3, 1, figsize=(14.8, 14.0), sharex=False)

    # --- Sensitivity specs for panel (b) ---
    sensitivity_specs = [
        {
            "label": "Baseline",
            "curve_path": os.path.join(data_dir, "gurobi_original_curve.csv"),
            "results_path": os.path.join(data_dir, "gurobi_original_results.xlsx"),
            "color": "#0B1F3A",
            "linestyle": "-",
            "is_baseline": True,
        },
        {
            "label": "CAPEX -30%",
            "curve_path": os.path.join(data_dir, "gurobi_capex_0p70_curve.csv"),
            "results_path": os.path.join(data_dir, "gurobi_capex_0p70_results.xlsx"),
            "color": "#0DBFEC",
            "linestyle": "-",
        },
        {
            "label": "CAPEX +30%",
            "curve_path": os.path.join(data_dir, "gurobi_capex_1p30_curve.csv"),
            "results_path": os.path.join(data_dir, "gurobi_capex_1p30_results.xlsx"),
            "color": "#0DBFEC",
            "linestyle": "-.",
        },
        {
            "label": "Coal -25%",
            "curve_path": os.path.join(data_dir, "gurobi_coal_m25_curve.csv"),
            "results_path": os.path.join(data_dir, "gurobi_coal_m25_results.xlsx"),
            "color": "#C0392B",
            "linestyle": "-",
        },
        {
            "label": "Coal +25%",
            "curve_path": os.path.join(data_dir, "gurobi_coal_p25_curve.csv"),
            "results_path": os.path.join(data_dir, "gurobi_coal_p25_results.xlsx"),
            "color": "#C0392B",
            "linestyle": "-.",
        },
        {
            "label": "Discount AF 5%",
            "curve_path": os.path.join(data_dir, "gurobi_af5_curve.csv"),
            "results_path": os.path.join(data_dir, "gurobi_af5_results.xlsx"),
            "color": "#1E8449",
            "linestyle": "-",
        },
        {
            "label": "Discount AF 12%",
            "curve_path": os.path.join(data_dir, "gurobi_af12_curve.csv"),
            "results_path": os.path.join(data_dir, "gurobi_af12_results.xlsx"),
            "color": "#1E8449",
            "linestyle": "-.",
        },
    ]
    biomass_specs = [
        {
            "label": "Baseline (100%)",
            "curve_path": os.path.join(data_dir, "gurobi_original_curve.csv"),
            "results_path": os.path.join(data_dir, "gurobi_original_results.xlsx"),
            "color": "#1F1F1F",
            "linestyle": "-",
            "is_baseline": True,
            "share_col": "bc_abatement_ton_model",
        },
        {
            "label": "Supply 75%",
            "curve_path": os.path.join(data_dir, "biomass_supply_sensitivity_bio_supply_75_curve.csv"),
            "results_path": os.path.join(data_dir, "biomass_supply_sensitivity_bio_supply_75_results.xlsx"),
            "color": "#006D2C",
            "linestyle": "-",
            "share_col": "bc_abatement_ton_model",
        },
        {
            "label": "Supply 50%",
            "curve_path": os.path.join(data_dir, "biomass_supply_sensitivity_bio_supply_50_curve.csv"),
            "results_path": os.path.join(data_dir, "biomass_supply_sensitivity_bio_supply_50_results.xlsx"),
            "color": "#31A354",
            "linestyle": "-.",
            "share_col": "bc_abatement_ton_model",
        },
        {
            "label": "Supply 25%",
            "curve_path": os.path.join(data_dir, "biomass_supply_sensitivity_bio_supply_25_curve.csv"),
            "results_path": os.path.join(data_dir, "biomass_supply_sensitivity_bio_supply_25_results.xlsx"),
            "color": "#78C679",
            "linestyle": ":",
            "share_col": "bc_abatement_ton_model",
        },
    ]

    ecsave_specs = [
        {
            "label": "Baseline (EC 20%)",
            "curve_path": os.path.join(data_dir, "gurobi_original_curve.csv"),
            "results_path": os.path.join(data_dir, "gurobi_original_results.xlsx"),
            "color": "#D94801",
            "linestyle": "-",
            "is_baseline": True,
            "share_col": "ec_abatement_ton",
        },
        {
            "label": "EC 17.5%",
            "curve_path": os.path.join(data_dir, "gurobi_ecsave_17p5_curve.csv"),
            "results_path": os.path.join(data_dir, "gurobi_ecsave_17p5_results.xlsx"),
            "color": "#FDAE6B",
            "linestyle": "-.",
            "share_col": "ec_abatement_ton",
        },
        {
            "label": "EC 15%",
            "curve_path": os.path.join(data_dir, "gurobi_ecsave_15_curve.csv"),
            "results_path": os.path.join(data_dir, "gurobi_ecsave_15_results.xlsx"),
            "color": "#FDD0A2",
            "linestyle": ":",
            "share_col": "ec_abatement_ton",
        },
    ]

    ccrate_specs = [
        {
            "label": "Baseline (90%)",
            "curve_path": os.path.join(data_dir, "gurobi_original_curve.csv"),
            "results_path": os.path.join(data_dir, "gurobi_original_results.xlsx"),
            "color": "#54278F",
            "linestyle": "-",
            "is_baseline": True,
            "share_col": "cc_capture_ton",
        },
        {
            "label": "Capture 95%",
            "curve_path": os.path.join(data_dir, "gurobi_cc_rate_95_curve.csv"),
            "results_path": os.path.join(data_dir, "gurobi_cc_rate_95_results.xlsx"),
            "color": "#9E9AC8",
            "linestyle": "-.",
            "share_col": "cc_capture_ton",
        },
        {
            "label": "Capture 99.7%",
            "curve_path": os.path.join(data_dir, "gurobi_cc_rate_99p7_curve.csv"),
            "results_path": os.path.join(data_dir, "gurobi_cc_rate_99p7_results.xlsx"),
            "color": "#CBC9E2",
            "linestyle": ":",
            "share_col": "cc_capture_ton",
        },
    ]

    common_xlim = (0.0, 5.2e9)
    common_ylim = (-60.0, 140.0)

    _plot_mac_comparison_panel(axes[0], data_dir)

    # Panel (b): sensitivity envelope – coal, CAPEX, discount
    _plot_sensitivity_envelope_only(
        axes[1],
        sensitivity_specs,
        target_ids,
        target_map,
        "(b)  Parametric sensitivity of fleet MAC: coal price, CAPEX, and discount rate",
        common_xlim,
        common_ylim,
    )
    _set_3col_legend(axes[1], [
        ("Baseline", "#1F1F1F", "-"),
        (["CAPEX -30%", "CAPEX +30%"], "#0DBFEC", ["-", "-."]),
        (["Coal -25%", "Coal +25%"], "#C0392B", ["-", "-."]),
        (["Discount rate 5%", "Discount rate 12%"], "#1E8449", ["-", "-."]),
    ])

    # Panel (c): combined uncertainty — biomass supply, EC saving rate, CC capture rate
    _plot_multi_curve_only(
        axes[2],
        [
            ("Biomass supply", biomass_specs, 0),
            ("EC saving rate", ecsave_specs, 1),
            ("CC capture rate", ccrate_specs, 2),
        ],
        target_ids,
        target_map,
        "(c)  Multi-dimensional uncertainty: biomass supply, EC saving rate, and CC capture rate",
        common_xlim,
        common_ylim,
    )
    _set_3col_legend(axes[2], [
        ("Baseline (100% biomass, 20% EC, 90% Capture)", "#1F1F1F", "-"),
        (["Supply 75%", "Supply 50%", "Supply 25%"], "#006D2C", ["-", "-.", ":"]),
        (["EC 17.5%", "EC 15%"], "#D94801", ["-.", ":"]),
        (["Capture 95%", "Capture 99.7%"], "#54278F", ["-.", ":"]),
    ])

    axes[-1].set_xlabel("Abatement target (Gt CO$_2$)", fontsize=12)

    fig.tight_layout()
    _increase_figure_fontsizes(fig, delta=4.0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    targets_sorted = sorted(target_ids)
    rows = []
    for tid in targets_sorted:
        tgt = target_map.get(tid)
        if tgt is not None:
            rows.append({"target_id": tid, "target_abatement": tgt})
    save_figure_bundle(
        fig,
        out_path,
        dpi=320,
        bbox_inches="tight",
        data_frames={"policy_nodes": pd.DataFrame(rows)},
    )
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready composite figures for sensitivity and uncertainty MAC analysis.")
    parser.add_argument("--target-ids", nargs="+", type=int, default=[12, 36, 48], help="Selected target IDs, e.g., 12 36 48")
    parser.add_argument("--out1", default=None, help="Output path for composite figure 1")
    parser.add_argument("--out2", default=None, help="Output path for composite figure 2")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, DATA_DIR_NAME)

    out1 = args.out1 or os.path.join(data_dir, COMPOSITE_1_OUT)
    out2 = args.out2 or os.path.join(data_dir, COMPOSITE_2_OUT)
    out3 = os.path.join(data_dir, "fig_scenario_comparison.png")

    p1 = build_composite_1(script_dir=script_dir, data_dir=data_dir, target_ids=args.target_ids, out_path=out1)
    p2 = build_composite_2(script_dir=script_dir, data_dir=data_dir, target_ids=args.target_ids, out_path=out2)
    p3 = build_scenario_comparison(script_dir=script_dir, data_dir=data_dir, target_ids=[12, 24, 36, 48], out_path=out3)

    print("Done.")
    print(f"Composite 1: {p1}")
    print(f"Composite 2: {p2}")
    print(f"Scenario comparison: {p3}")


if __name__ == "__main__":
    main()
