
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        GDP  QoQ Germany Pre-Processing
#
# Author:       ChatGPT 5.2, prompted and ammended by Jan Ole Westphal
#
# Description:  Takes the ifo forecast overview excel and extracts vintage tables for all compontents
#
# ==================================================================================================
# --------------------------------------------------------------------------------------------------


from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
IN_FILE = Path("0_0_Data/0_Forecast_Inputs/1_QoQ_Germany/qoq_GDP_Germany_raw.xlsx")  # input (long format)
OUT_DIR = Path("0_0_Data/0_Forecast_Inputs/1_QoQ_Germany")  # output directory
SHEET_NAME = "GDP"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _safe_filename(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w\-]+", "_", s)  # keep letters/numbers/_/-
    return s.strip("_") or "UNKNOWN"


def build_vintage_matrix(df_inst: pd.DataFrame) -> pd.DataFrame:
    """
    For one institute:
      - columns: Actual Vintage dates (not converted)
      - rows   : Quarter-middle dates (150 quarters starting from first vintage quarter)
      - values : Value
    
    Horizon 0: forecast placed in same quarter as Vintage
    Horizon 1: forecast placed 1 quarter ahead
    Horizon N: forecast placed N quarters ahead
    """
    df = df_inst.copy()

    # Required columns
    required = {"Institute", "Vintage", "Horizon", "Value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {sorted(missing)}")

    # Types
    df["Vintage"] = pd.to_datetime(df["Vintage"], errors="coerce")
    df["Horizon"] = pd.to_numeric(df["Horizon"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # Drop rows that cannot be mapped
    df = df.dropna(subset=["Vintage", "Horizon"])
    df["Horizon"] = df["Horizon"].astype(int)

    # Convert Vintage to quarter period (for calculating target quarters)
    df["Vintage_Q"] = df["Vintage"].dt.to_period("Q")

    # Calculate target quarter: Vintage quarter + Horizon quarters
    df["target_Q"] = df["Vintage_Q"] + df["Horizon"]

    # Get first vintage quarter
    first_vintage_period = df["Vintage_Q"].min()
    
    # Function to convert quarter period to middle date (45 days into quarter)
    def quarter_to_middle(q_period):
        start = q_period.to_timestamp(how="start")
        middle = start + pd.DateOffset(days=45)
        return middle.floor("D")
    
    # Build row index: 150 quarter-middle dates starting from first vintage quarter
    row_dates = [quarter_to_middle(first_vintage_period + i) for i in range(150)]
    
    # Create mapping from target_Q to row date
    q_to_row = {first_vintage_period + i: row_dates[i] for i in range(150)}
    
    # Get sorted unique actual Vintage dates for columns
    vintage_dates = sorted(df["Vintage"].unique())
    
    # Sort df for deterministic duplicate handling (keep last)
    df = df.sort_values(["Vintage", "target_Q", "Horizon"])
    
    # Initialize empty matrix
    mat = pd.DataFrame(index=row_dates, columns=vintage_dates, dtype=float)
    
    # Fill the matrix - last value wins for duplicates
    for _, row in df.iterrows():
        vintage = row["Vintage"]
        target_q = row["target_Q"]
        value = row["Value"]
        
        if target_q in q_to_row and vintage in mat.columns:
            target_row = q_to_row[target_q]
            mat.loc[target_row, vintage] = value
    
    # Drop entirely empty rows at the end
    mat = mat.dropna(how='all')
    
    return mat


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(IN_FILE)

    # If file contains multiple variables, keep GDP only (defensive)
    if "Variable" in df.columns:
        df = df[df["Variable"].astype(str).str.upper().eq("GDP")].copy()

    if df.empty:
        raise ValueError("No rows found after filtering (input may be empty or non-GDP).")

    for inst, g in df.groupby("Institute", dropna=False):
        inst_name = _safe_filename(inst)
        mat = build_vintage_matrix(g)

        out_file = OUT_DIR / f"qoq_forecasts_{inst_name}.xlsx"
        with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
            mat.to_excel(writer, sheet_name=SHEET_NAME, index=True)

        print(f"Wrote: {out_file}  |  shape={mat.shape}")


if __name__ == "__main__":
    main()