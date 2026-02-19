
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        GDP YoY Component Pre-Processing
#
# Author:       ChatGPT 5.2, prompted and ammended by Jan Ole Westphal
#
# Description:  Takes the ifo forecast overview excel and extracts vintage tables for all compontents
#
# ==================================================================================================
# --------------------------------------------------------------------------------------------------




"""
Fix this by hand
"""

from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter


# ======================================================================================
# Paths (relative to your working directory)
# ======================================================================================
IN_DIR  = Path("0_0_Data/0_Forecast_Inputs/3_Yearly_Forecasts_raw")
OUT_DIR = Path("0_0_Data/0_Forecast_Inputs/3_YoY_Forecasts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_MIN, YEAR_MAX = 1969, 2025
TARGET_YEARS = list(range(YEAR_MIN, YEAR_MAX + 1))

# Correct chronological within-year order:
# F=Frühjahr (Mar), S=Sommer (Jun), H=Herbst (Sep), W=Winter (Dec)
SEASONS_ORDER = ["F", "S", "H", "W"]
LETTER_TO_MONTHDAY = {"F": (3, 15), "S": (6, 15), "H": (9, 15), "W": (12, 15)}


# ======================================================================================
# Component renaming
# ======================================================================================
COMPONENT_RENAME = {
    "CONG":   "PUBCON",
    "CONP":   "PRIVCON",
    "EX":     "EXPORT",
    "IM":     "IMPORT",
    "GFCFCO": "CONSTR",
    "GFCFME": "EQUIPMENT",
    "GFCFOP": "OPA",
}


# ======================================================================================
# Helpers
# ======================================================================================
def parse_filename(p: Path) -> tuple[str, str]:
    """
    Example: 'BBK multiple years forecasts for CONG.xlsx'
      -> leading='BBK', component='CONG'
    If leading == 'DB' -> 'DB_Research'
    """
    stem = p.stem.strip()
    parts = stem.split()
    if len(parts) < 2:
        raise ValueError(f"Cannot parse filename: {p.name}")

    leading = parts[0].strip()
    if leading.upper() == "DB":
        leading = "DB_Research"

    component = re.sub(r"[^A-Za-z0-9_]+", "", parts[-1]).strip()
    if not component:
        raise ValueError(f"Cannot parse component from filename: {p.name}")

    return leading, component


def safe_sheet_name(name: str) -> str:
    name = re.sub(r"[\[\]\:\*\?\/\\]", "_", str(name))
    return name[:31] if len(name) > 31 else name


def read_excel_flex(path: Path) -> pd.DataFrame:
    """Read first sheet; return empty df if unreadable or empty."""
    try:
        df = pd.read_excel(path, sheet_name=0)
        if df is None:
            return pd.DataFrame()
        df = df.dropna(how="all").dropna(axis=1, how="all")
        return df
    except Exception:
        return pd.DataFrame()


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in cols_norm:
            return cols_norm[key]
    return None


def ensure_long_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce a robust long schema.
    Required: Release Year, Year, Value.
    """
    if df.empty:
        return df

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    col_release_year = _first_existing_col(df, ["Release Year", "release_year", "ReleaseYear"])
    col_year         = _first_existing_col(df, ["Year", "year", "Target Year", "Forecast Year"])
    col_value        = _first_existing_col(df, ["Value", "value", "Forecast", "Forecast Value"])

    if col_release_year is None or col_year is None or col_value is None:
        return pd.DataFrame()

    col_release_month   = _first_existing_col(df, ["Release Month", "release_month", "Month"])
    col_release_quarter = _first_existing_col(df, ["Release Quarter", "release_quarter", "Quarter"])
    col_data            = _first_existing_col(df, ["Data", "data", "Provider", "Source"])

    out = pd.DataFrame({
        "Release Year": df[col_release_year],
        "Year": df[col_year],
        "Value": df[col_value],
    })
    if col_release_month is not None:
        out["Release Month"] = df[col_release_month]
    if col_release_quarter is not None:
        out["Release Quarter"] = df[col_release_quarter]
    if col_data is not None:
        out["Data"] = df[col_data]

    out = out.dropna(how="all")
    return out


def build_vintage_date_vectorised(df_long: pd.DataFrame) -> pd.Series:
    """
    Vintage date = 15th of Release Month (if present) else inferred from Release Quarter.
    """
    ry = pd.to_numeric(df_long.get("Release Year"), errors="coerce")

    rm = pd.to_numeric(df_long.get("Release Month"), errors="coerce") if "Release Month" in df_long.columns else pd.Series(np.nan, index=df_long.index)
    rq = pd.to_numeric(df_long.get("Release Quarter"), errors="coerce") if "Release Quarter" in df_long.columns else pd.Series(np.nan, index=df_long.index)

    rm_filled = rm.where(rm.notna(), rq.map({1: 2, 2: 5, 3: 8, 4: 11}))  # quarterly vintages: Feb/May/Aug/Nov

    vd = pd.to_datetime(
        dict(year=ry.astype("Int64"), month=rm_filled.astype("Int64"), day=15),
        errors="coerce",
    )
    return vd


def long_to_ifo_like_matrix(
    df_long_raw: pd.DataFrame,
    provider_name: str,
    target_years: list[int] = TARGET_YEARS,
    seasons: list[str] = SEASONS_ORDER,
) -> pd.DataFrame:
    """
    Output format:
      Row1: Prognose | F S H W repeated by year
      Row2: Jahr     | year repeated 4x
      Row3: Datenstand | 15 Mar / 15 Jun / 15 Sep / 15 Dec of that year
      Row4+: vintage dates (quarterly) | values (sparse allowed)
    Always produces full grid of columns for years 1969..2025 and 4 seasons per year.
    """
    # full column grid metadata
    col_meta = []
    for y in target_years:
        for letter in seasons:
            md = LETTER_TO_MONTHDAY[letter]
            datenstand = pd.Timestamp(datetime(int(y), md[0], md[1]))
            col_meta.append((letter, int(y), datenstand))

    # headers-only template if nothing usable
    def headers_only() -> pd.DataFrame:
        n_rows, n_cols = 3, 1 + len(col_meta)
        mat = pd.DataFrame(np.empty((n_rows, n_cols), dtype="object"))
        mat.loc[:, :] = None
        mat.iat[0, 0] = "Prognose"
        mat.iat[1, 0] = "Jahr"
        mat.iat[2, 0] = "Datenstand"
        for j, (letter, y, ds) in enumerate(col_meta, start=1):
            mat.iat[0, j] = letter
            mat.iat[1, j] = y
            mat.iat[2, j] = ds.to_pydatetime()
        return mat

    df_long = ensure_long_schema(df_long_raw)
    if df_long.empty:
        return headers_only()

    # provider filter if present (drops "Observed" blocks etc. naturally as well)
    if "Data" in df_long.columns:
        df_long = df_long[df_long["Data"].astype(str).str.strip().eq(str(provider_name))].copy()

    # coerce numeric and drop garbage rows
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
    df_long["Release Year"] = pd.to_numeric(df_long["Release Year"], errors="coerce")
    if "Release Month" in df_long.columns:
        df_long["Release Month"] = pd.to_numeric(df_long["Release Month"], errors="coerce")
    if "Release Quarter" in df_long.columns:
        df_long["Release Quarter"] = pd.to_numeric(df_long["Release Quarter"], errors="coerce")

    df_long = df_long.dropna(subset=["Release Year", "Year"])  # keep missing Value (sparse allowed)

    # keep target years range
    df_long = df_long[df_long["Year"].between(min(target_years), max(target_years), inclusive="both")].copy()
    if df_long.empty:
        return headers_only()

    # build vintage date
    df_long["vintage_date"] = build_vintage_date_vectorised(df_long)
    df_long = df_long.dropna(subset=["vintage_date", "Year"])
    if df_long.empty:
        return headers_only()

    df_long["target_year"] = df_long["Year"].astype(int)

    # pivot: index=vintage_date, columns=target_year
    pivot = (
        df_long.pivot_table(
            index="vintage_date",
            columns="target_year",
            values="Value",
            aggfunc="first",
        )
        .sort_index()
        .reindex(columns=target_years)
    )

    vintage_index = pivot.index.sort_values()
    if len(vintage_index) == 0:
        return headers_only()

    # matrix (object dtype, so headers + dates + floats coexist cleanly)
    n_rows = 3 + len(vintage_index)
    n_cols = 1 + len(col_meta)
    mat = pd.DataFrame(np.empty((n_rows, n_cols), dtype="object"))
    mat.loc[:, :] = None

    # headers
    mat.iat[0, 0] = "Prognose"
    mat.iat[1, 0] = "Jahr"
    mat.iat[2, 0] = "Datenstand"
    for j, (letter, y, ds) in enumerate(col_meta, start=1):
        mat.iat[0, j] = letter
        mat.iat[1, j] = y
        mat.iat[2, j] = ds.to_pydatetime()

    # vintage dates
    for i, vd in enumerate(vintage_index, start=3):
        mat.iat[i, 0] = pd.Timestamp(vd).to_pydatetime()

    # fill values with truncation: vintage must be <= datenstand for that column
    for j, (_, y, ds) in enumerate(col_meta, start=1):
        ser = pivot[y].reindex(vintage_index)
        ser = ser.where(ser.index <= ds)  # set later vintages to NaN
        mat.iloc[3:, j] = ser.to_numpy()

    return mat


# ======================================================================================
# Excel formatting (post-write): dates, alignment, freeze panes
# ======================================================================================
def format_workbook(path: Path) -> None:
    wb = openpyxl.load_workbook(path)
    header_font = Font(bold=True)
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")

    for ws in wb.worksheets:
        max_row = ws.max_row
        max_col = ws.max_column

        # freeze at B4 (keep headers + vintage-date column visible)
        ws.freeze_panes = "B4"

        # header styling
        for r in (1, 2, 3):
            ws.cell(r, 1).font = header_font
            ws.cell(r, 1).alignment = left
            for c in range(2, max_col + 1):
                cell = ws.cell(r, c)
                cell.font = header_font
                cell.alignment = center

        # date formats: row 3 (Datenstand) and column A from row 4 down
        for c in range(2, max_col + 1):
            ws.cell(3, c).number_format = "yyyy-mm-dd"

        for r in range(4, max_row + 1):
            ws.cell(r, 1).number_format = "yyyy-mm-dd"
            ws.cell(r, 1).alignment = left

        # numeric formats for data area (optional but helps)
        for r in range(4, max_row + 1):
            for c in range(2, max_col + 1):
                cell = ws.cell(r, c)
                if isinstance(cell.value, (int, float)) and cell.value is not None:
                    cell.number_format = "0.0######"
                cell.alignment = center

        # widths
        ws.column_dimensions["A"].width = 13
        for c in range(2, max_col + 1):
            ws.column_dimensions[get_column_letter(c)].width = 5

    wb.save(path)


# ======================================================================================
# Main: group raw files by provider-leading substring; write one workbook per provider
# ======================================================================================
xlsx_files = sorted(IN_DIR.glob("*.xlsx"))

groups: dict[str, list[tuple[str, Path]]] = {}
for p in xlsx_files:
    leading, component = parse_filename(p)
    component = COMPONENT_RENAME.get(component, component)
    groups.setdefault(leading, []).append((component, p))

for leading, component_files in groups.items():
    out_path = OUT_DIR / f"{leading}_BIP_Komponenten_YoY.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for component, fpath in sorted(component_files, key=lambda x: x[0]):
            df_raw = read_excel_flex(fpath)
            sheet_mat = long_to_ifo_like_matrix(df_raw, provider_name=leading)
            sheet_mat.to_excel(writer, sheet_name=safe_sheet_name(component), index=False, header=False)

    format_workbook(out_path)
    print(f"Wrote: {out_path}")
