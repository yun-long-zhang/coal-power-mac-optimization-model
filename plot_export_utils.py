import os
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _root_from_png_path(png_path: str) -> str:
    lower = png_path.lower()
    if lower.endswith(".png"):
        return png_path[:-4]
    return png_path


def _safe_sheet_name(name: str, used: set) -> str:
    base = "".join(ch if ch.isalnum() or ch in " _-" else "_" for ch in str(name)).strip() or "sheet"
    base = base[:31]
    cand = base
    idx = 2
    while cand in used:
        suffix = f"_{idx}"
        cand = (base[: 31 - len(suffix)] + suffix)
        idx += 1
    used.add(cand)
    return cand


def _extract_matplotlib_data(fig) -> Dict[str, pd.DataFrame]:
    sheets: Dict[str, pd.DataFrame] = {}

    for ax_idx, ax in enumerate(getattr(fig, "axes", []), start=1):
        rows_line = []
        for line_idx, line in enumerate(ax.get_lines(), start=1):
            x = np.asarray(line.get_xdata())
            y = np.asarray(line.get_ydata())
            n = min(len(x), len(y))
            if n == 0:
                continue
            label = line.get_label()
            for k in range(n):
                rows_line.append(
                    {
                        "axis_index": ax_idx,
                        "series_index": line_idx,
                        "series_label": label,
                        "x": x[k],
                        "y": y[k],
                    }
                )
        if rows_line:
            sheets[f"ax{ax_idx}_lines"] = pd.DataFrame(rows_line)

        rows_scatter = []
        for coll_idx, coll in enumerate(ax.collections, start=1):
            if not hasattr(coll, "get_offsets"):
                continue
            try:
                offsets = np.asarray(coll.get_offsets())
            except Exception:
                continue
            if offsets.size == 0:
                continue
            label = getattr(coll, "get_label", lambda: "")()
            arr = None
            if hasattr(coll, "get_array"):
                try:
                    arr = coll.get_array()
                    if arr is not None:
                        arr = np.asarray(arr)
                except Exception:
                    arr = None
            for k, xy in enumerate(offsets):
                rec = {
                    "axis_index": ax_idx,
                    "collection_index": coll_idx,
                    "series_label": label,
                    "x": float(xy[0]) if len(xy) > 0 else np.nan,
                    "y": float(xy[1]) if len(xy) > 1 else np.nan,
                }
                if arr is not None and k < len(arr):
                    rec["value"] = arr[k]
                rows_scatter.append(rec)
        if rows_scatter:
            sheets[f"ax{ax_idx}_scatter"] = pd.DataFrame(rows_scatter)

        rows_bar = []
        for patch_idx, patch in enumerate(ax.patches, start=1):
            if not hasattr(patch, "get_width") or not hasattr(patch, "get_height"):
                continue
            try:
                x_left = float(patch.get_x())
                width = float(patch.get_width())
                height = float(patch.get_height())
            except Exception:
                continue
            rows_bar.append(
                {
                    "axis_index": ax_idx,
                    "patch_index": patch_idx,
                    "x_left": x_left,
                    "x_center": x_left + width / 2.0,
                    "width": width,
                    "height": height,
                }
            )
        if rows_bar:
            sheets[f"ax{ax_idx}_bars"] = pd.DataFrame(rows_bar)

    return sheets


def _write_excel(path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    if not sheets:
        return
    _ensure_parent(path)
    used = set()
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            if df is None:
                continue
            if isinstance(df, pd.Series):
                df = df.to_frame()
            if not isinstance(df, pd.DataFrame):
                continue
            safe_name = _safe_sheet_name(name, used)
            df.to_excel(writer, sheet_name=safe_name, index=False)


def save_figure_bundle(
    fig,
    png_path: str,
    dpi: Optional[int] = 300,
    bbox_inches: str = "tight",
    data_frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, str]:
    _ensure_parent(png_path)
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)

    root = _root_from_png_path(png_path)
    pdf_path = root + ".pdf"
    xlsx_path = root + ".xlsx"

    fig.savefig(pdf_path, dpi=dpi, bbox_inches=bbox_inches)

    sheets: Dict[str, pd.DataFrame] = {}
    sheets.update(_extract_matplotlib_data(fig))
    if data_frames:
        sheets.update(data_frames)
    _write_excel(xlsx_path, sheets)

    return {"png": png_path, "pdf": pdf_path, "xlsx": xlsx_path}


def _extract_plotly_data(fig) -> Dict[str, pd.DataFrame]:
    sheets: Dict[str, pd.DataFrame] = {}
    data = getattr(fig, "data", [])
    for idx, trace in enumerate(data, start=1):
        d = trace.to_plotly_json() if hasattr(trace, "to_plotly_json") else dict(trace)
        rec = {
            "trace_name": d.get("name", f"trace_{idx}"),
            "trace_type": d.get("type", "unknown"),
        }

        x = d.get("x")
        y = d.get("y")
        if x is not None or y is not None:
            x_arr = np.asarray(x) if x is not None else np.array([])
            y_arr = np.asarray(y) if y is not None else np.array([])
            n = max(len(x_arr), len(y_arr))
            rows = []
            for k in range(n):
                rows.append(
                    {
                        **rec,
                        "index": k,
                        "x": x_arr[k] if k < len(x_arr) else np.nan,
                        "y": y_arr[k] if k < len(y_arr) else np.nan,
                    }
                )
            sheets[f"trace_{idx}"] = pd.DataFrame(rows)
        else:
            sheets[f"trace_{idx}"] = pd.DataFrame([rec])
    return sheets


def save_plotly_bundle(
    fig,
    png_path: str,
    width: int = 1500,
    height: int = 900,
    scale: int = 2,
    data_frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, str]:
    _ensure_parent(png_path)
    fig.write_image(png_path, width=width, height=height, scale=scale)

    root = _root_from_png_path(png_path)
    pdf_path = root + ".pdf"
    xlsx_path = root + ".xlsx"

    fig.write_image(pdf_path, width=width, height=height, scale=scale)

    sheets: Dict[str, pd.DataFrame] = {}
    sheets.update(_extract_plotly_data(fig))
    if data_frames:
        sheets.update(data_frames)
    _write_excel(xlsx_path, sheets)

    return {"png": png_path, "pdf": pdf_path, "xlsx": xlsx_path}
