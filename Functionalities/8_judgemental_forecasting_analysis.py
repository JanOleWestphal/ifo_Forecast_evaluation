
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Judgemental Forecasting Analysis Module
#
# Author:       Jan Ole Westphal
# Date:         2026-01
#
# Description:  Subprogram to run an econometric analysis on judgemental derivations in german
#               macroeconomic forecasting.
# 
#               Runs all components from Data Processing to Output Processing and Visualizations.         
# ==================================================================================================
# --------------------------------------------------------------------------------------------------



""""
Main Tasks:
- create a measure of judgemental derivations:
    - derivation from the ifoCAST
    - derivation from an AR2-benchmark
    - possibly: derivations from ifo's forecast methodology, if vintage data exists

- Create a measure of net-improvement of derivations
- Classify derivations:
    - direction of the shock
    - direction of the adjustment
    - net improvement

    --> Derivation types: r -> realized value, , b -> benchmark, j -> judgemental forecasts
        - negative shocks, r<b: 
            - r<b<j (overconfidence), 
            - r<j<b (prudent pessimism), 
            - j<r<b; |j-b|<|r-b| (mild overpessimism), 
            - j<r<b |j-b|>|r-b|(strong overpessimism)

        - positive shocks, b<r:
            - overpessimism: j<r<b
            - prudent optimism: j>b>r
            - mild overoptimism: b<r<j; |j-b|<|r-b|
            - strong overoptimism: b<r<j; |j-b|>|r-b|


- Analyze judgement persistence through the Pedersen (2025) methodology, (autoregression of derivaitons)

VISUALIZATIONS:
- Judgemental vs Benchmark error bars by quarter
"""




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        Code begins here                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# ==================================================================================================
#                                           SETUP
# ==================================================================================================

from __future__ import annotations

# Import built-ins
import importlib
import subprocess
import sys
import os
import glob
import re
from pathlib import Path
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from itertools import product
from typing import Union, Dict, Optional, Mapping, Tuple, Dict


# Import libraries
import requests
import pandas as pd
from pandas.tseries.offsets import QuarterBegin
from pandasgui import show  #uncomment this to allow for easier debugging

import numpy as np
from statsmodels.tsa.ar_model import AutoReg


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Patch

import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
# sns.set_theme(style='whitegrid')





# ==================================================================================================
#                                IMPORT CORE CUSTOM FUNCTIONALITIES
# ==================================================================================================

# Ensure project root is in sys.path
wd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if wd not in sys.path:
    sys.path.insert(0, wd)

## Import helperfunctions
from Functionalities.helpers.helperfunctions import *


# --------------------------------------------------------------------------------------------------
# Import Evaluation Functions
# --------------------------------------------------------------------------------------------------

from Functionalities.helpers.evalfunctions import *



# ==================================================================================================
# Import settings from the settings file
# ==================================================================================================

import ifo_forecast_evaluation_settings as settings

# Define the model
models = settings.models
AR_orders = settings.AR_orders
average_horizons = settings.average_horizons
AR_horizons = settings.AR_horizons
forecast_horizon = settings.forecast_horizon

# Format the output
resultfolder_name_n_forecast = settings.resultfolder_name_n_forecast   
naming_convention = settings.naming_convention

# Select timeframes
horizon_limit_year = settings.horizon_limit_year
horizon_limit_quarter = settings.horizon_limit_quarter   

# Select whether to evaluate GVA predictions
run_gva_evaluation = settings.run_gva_evaluation



## Print Module header
print("\nExecuting the Judgemental Derivations Analysis Module ... \n")


# ==================================================================================================
# SETUP OUTOUT FOLDER STRUCTURE
# ==================================================================================================

## Result Folder Paths
result_folder = os.path.join(wd, '5_Judgemental_Derivations_Analysis')


## Subfolder
table_folder = os.path.join(result_folder, '1_Tables')
graph_folder = os.path.join(result_folder, '2_Plots')

## Graph subfolders
graph_folder_derivations = os.path.join(graph_folder, '1_Derivations_and_Improvements')
graph_folder_errors = os.path.join(graph_folder, '2_Error_Comparisons')



## Create if needed
for folder in [result_folder, table_folder, graph_folder, graph_folder_derivations, graph_folder_errors]:
    os.makedirs(folder, exist_ok=True)


## Clear Result Folders
#if settings.clear_result_folders:
#    folder_clear(folder_path)






# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                          LOAD IN DATA                                            #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

# -------------------------------------------------------------------------------------------------#
# Load realized GDP-series
# -------------------------------------------------------------------------------------------------#
eval_path = os.path.join(wd, '0_0_Data', '2_Processed_Data', '2_GDP_Evaluation_series')
qoq_path_first = os.path.join(eval_path, 'first_release_qoq_GDP.xlsx')

## First Releases
qoq_first_eval = pd.read_excel(qoq_path_first, index_col=0)
qoq_first_eval = align_df_to_mid_quarters(qoq_first_eval)  # Align to mid-quarter dates
#show(qoq_first_eval)


# -------------------------------------------------------------------------------------------------#
# Load ifo qoq nowcasts
# -------------------------------------------------------------------------------------------------#
# Path
file_path_ifo_qoq = os.path.join(wd, '0_0_Data', '2_Processed_Data', '3_ifo_qoq_series',
                                  'ifo_qoq_forecasts.xlsx' )

# Load 
ifo_qoq_forecasts = pd.read_excel(file_path_ifo_qoq, index_col=0)

# Build ifo judgemental nowcasts by matching row/column on quarterly level.
def nowcast_builder(df, colname="ifo_judgemental_nowcast"):
    ifo_rows_quarter = pd.to_datetime(df.index).to_period('Q')
    ifo_cols_quarter = pd.to_datetime(df.columns).to_period('Q')

    dtx1_records = []
    for col, col_quarter in zip(df.columns, ifo_cols_quarter):
        matching_rows = np.where(ifo_rows_quarter == col_quarter)[0]
        if len(matching_rows) != 1:
            raise ValueError(
                f"Expected exactly one quarterly row match for column {col} ({col_quarter}), "
                f"found {len(matching_rows)}."
            )

        row_label = df.index[matching_rows[0]]
        dtx1_records.append(
            {
                'column_date': col,
                'matched_row_date': row_label,
                'ifo_judgemental_nowcast': df.loc[row_label, col]
            }
        )

    dtx1 = pd.DataFrame(dtx1_records).set_index('column_date')
    df_out = dtx1[['ifo_judgemental_nowcast']].copy()

    df_out = df_out.rename(columns={'ifo_judgemental_nowcast': colname})
    
    # Align to mid-quarter dates
    df_out = align_df_to_mid_quarters(df_out)

    return df_out

ifo_judgemental_nowcasts = nowcast_builder(ifo_qoq_forecasts)
#show(ifo_judgemental_nowcasts)


# -------------------------------------------------------------------------------------------------#
# Load ifoCAST nowcasts
# -------------------------------------------------------------------------------------------------#
ifoCAST_nowcasts_full_path = os.path.join(
    wd, '0_0_Data', '0_Forecast_Inputs', '2_ifoCAST', 'ifoCAST_nowcasts_full.xlsx')

# Load 
ifoCAST_nowcast = pd.read_excel(ifoCAST_nowcasts_full_path , index_col=0)
ifoCAST_nowcast = align_df_to_mid_quarters(ifoCAST_nowcast)  # Align to mid-quarter dates
#show(ifoCAST_nowcast)

# -------------------------------------------------------------------------------------------------#
# Load AR2-nowcasts
# -------------------------------------------------------------------------------------------------#
# Paths to the folders containing the Excel files
file_path_naive_qoq = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '1_QoQ_Forecast_Tables')

# Load all QoQ naive forecast Excel files into dictionary
naive_qoq_dfs_dict = load_excels_to_dict(file_path_naive_qoq, strip_string='naive_qoq_forecasts_')

# Get the AR2 model
matches = [k for k in naive_qoq_dfs_dict if "AR2" in k]
if not matches:
    raise KeyError("No entry containing 'AR2' found in naive_qoq_dfs_dict. Check Setting file and run Naive Forecaster Module")

df_ar2 = naive_qoq_dfs_dict[matches[0]]
#show(df_ar2)

# Get Nowcasts
AR_nowcasts = nowcast_builder(df_ar2, colname="AR2_nowcast")
#show(AR_nowcasts)







# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                          PROCESS DATA                                            #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# =================================================================================================#
#                                       Merge to joint df                                          #
# =================================================================================================#

## Call merge_quarterly_dfs_dropna() from helperfunctions
joint_nowcast_df = merge_quarterly_dfs_dropna(
    dfs= [qoq_first_eval, ifo_judgemental_nowcasts, ifoCAST_nowcast, AR_nowcasts],
    col_names=['realized', 'judgemental', 'ifoCast', 'naiveAR2']
)

# Re-align to mid-quarter dates to ensure consistency after merge
joint_nowcast_df = align_df_to_mid_quarters(joint_nowcast_df)

#show(joint_nowcast_df)

## Create a clean copy
joint_nowcast_base_df = joint_nowcast_df.copy()



# -------------------------------------------------------------------------------------------------#
# OPTIONAL: filter rows
# -------------------------------------------------------------------------------------------------#

"""NOTE: all rows are indexed by latest date of quarter"""

## Adjust filter if needed, boundary inclusive
joint_nowcast_df = filter_df_by_datetime_index(joint_nowcast_df, '2000-01-01', '2100-01-01')

# Re-align to mid-quarter dates after filtering
joint_nowcast_df = align_df_to_mid_quarters(joint_nowcast_df)

#show(joint_nowcast_df)





# =================================================================================================#
#                                    Create error measures                                         #
# =================================================================================================#

## for ifo judgemental, AR2 and ifoCAST nowcasts

# Loop through cols and substract them from the leading realized values col
def add_error_columns(df, prefix="error"):

    first_col = df.columns[0]

    for col in df.columns[1:]:
        df[f"{prefix}_{first_col}_minus_{col}"] = df[col] - df[first_col] 

    return df

# Call error function
joint_nowcast_df = add_error_columns(joint_nowcast_df)
#show(joint_nowcast_df)

## Save
joint_nowcast_df.to_excel(
    excel_writer=os.path.join(table_folder, "Nowcast_Series_full.xlsx")
)



# =================================================================================================#
#                                   Create derivation measures                                     #
# =================================================================================================#

## from ifoCAST
joint_nowcast_df["derivation_from_ifoCast"] = (
    joint_nowcast_df["judgemental"] - joint_nowcast_df["ifoCast"]
)

## from AR2
joint_nowcast_df["derivation_from_AR"] = (
    joint_nowcast_df["judgemental"] - joint_nowcast_df["naiveAR2"]
)




# =================================================================================================#
#                                    Obtain net improvements                                       #
# =================================================================================================#

# ------------------------------------------------------------------------------------
# Net improvement of judgemental forecast relative to baseline forecasts
#
# Definitions
# ----------
# Linear improvement  :  NI_lin  = |e_baseline| - |e_judgemental|
# Quadratic improvement: NI_quad = e_baseline^2 - e_judgemental^2
#
# Positive values  -> judgement improved the forecast
# Negative values  -> judgement worsened the forecast
#
# Required columns already present:
#   realized
#   judgemental
#   naiveAR2
#   ifoCast
#   error_realized_minus_judgemental
#   error_realized_minus_naiveAR2
#   error_realized_minus_ifoCast
#   derivation_from_ifoCast
#   derivation_from_AR
# ------------------------------------------------------------------------------------

# ---------- Judgement vs ifoCast ----------
joint_nowcast_df["net_improvement_jdg_ifoCast_lin"] = (
    joint_nowcast_df["error_realized_minus_ifoCast"].abs()
    - joint_nowcast_df["error_realized_minus_judgemental"].abs()
)

joint_nowcast_df["net_improvement_jdg_ifoCast_quad"] = (
    joint_nowcast_df["error_realized_minus_ifoCast"]**2
    - joint_nowcast_df["error_realized_minus_judgemental"]**2
)


# ---------- Judgement vs naiveAR2 ----------
joint_nowcast_df["net_improvement_jdg_AR_lin"] = (
    joint_nowcast_df["error_realized_minus_naiveAR2"].abs()
    - joint_nowcast_df["error_realized_minus_judgemental"].abs()
)

joint_nowcast_df["net_improvement_jdg_AR_quad"] = (
    joint_nowcast_df["error_realized_minus_naiveAR2"]**2
    - joint_nowcast_df["error_realized_minus_judgemental"]**2
)

#show(joint_nowcast_df)





# =================================================================================================#
#                                Classify Judgemental Derications                                  #
# =================================================================================================#

"""
r<b: True (negative shock), False; 'r_less_b'
j<b: True (negatve adjustment), False; 'j_less_b'
|j-r|<|b-r| True (judgemental improvement), False; 'jdiff_less_bdiff'
"""

# -------------------------------------------------------------------------------------------------#
# Classification builder function
# -------------------------------------------------------------------------------------------------#


def _classify_derivations(df, b_col: str, suffix: str, r_col: str = "realized", j_col: str = "judgemental"):
    r = df[r_col]
    j = df[j_col]
    b = df[b_col]

    # Ensure NA-safe comparisons: keep pd.NA where any input is missing
    valid = r.notna() & j.notna() & b.notna()

    df[f"r_less_{suffix}"] = pd.Series(np.where(valid, r < b, pd.NA), index=df.index, dtype="boolean")
    df[f"j_less_{suffix}"] = pd.Series(np.where(valid, j < b, pd.NA), index=df.index, dtype="boolean")
    df[f"j_diff_less_{suffix}_diff"] = pd.Series(
        np.where(valid, (j - r).abs() < (b - r).abs(), pd.NA),
        index=df.index,
        dtype="boolean",
    )


# -------------------------------------------------------------------------------------------------#
# Evaluation against the ifoCAST df
# -------------------------------------------------------------------------------------------------#
_classify_derivations(joint_nowcast_df, b_col="ifoCast", suffix="ifoCast")


# -------------------------------------------------------------------------------------------------#
# Evaluation against the AR2 df
# -------------------------------------------------------------------------------------------------#
_classify_derivations(joint_nowcast_df, b_col="naiveAR2", suffix="AR2")

#show(joint_nowcast_df)



# =================================================================================================#
#                                  Split dfs and save results                                      #
# =================================================================================================#

## Split

# Create copy
judgment_eval_df_joint = joint_nowcast_df.copy()

# Columns always included
base_cols = [
    "realized",
    "judgemental",
    "error_realized_minus_judgemental",
]

# ---- ifoCast subset ----
ifo_cols = [c for c in judgment_eval_df_joint .columns if "ifoCast" in c]
judgment_eval_df_ifoCast = judgment_eval_df_joint [base_cols + ifo_cols].copy()
#show(judgment_eval_df_ifoCast)

# ---- AR subset (matches AR / naiveAR2 / derivation_from_AR etc.) ----
ar_cols = [c for c in judgment_eval_df_joint .columns if "AR" in c]
judgment_eval_df_AR = judgment_eval_df_joint [base_cols + ar_cols].copy()


## Save
judgment_eval_df_joint.to_excel(os.path.join(table_folder, "judgemental_derivations_full.xlsx"))
judgment_eval_df_ifoCast.to_excel(os.path.join(table_folder, "judgemental_derivations_ifoCast.xlsx"))
judgment_eval_df_AR.to_excel(os.path.join(table_folder, "judgemental_derivations_AR2.xlsx"))

print('\nData processing complete. Now proceeding to analysis and visualizations!\n')











# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                Judgemental derivations analysis                                  #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# =================================================================================================#
#                                        Builder Functions                                         #
# =================================================================================================#

## HELPER

def _infer_baseline_spec(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer which baseline the df refers to (ifoCast vs AR2) and return column spec.
    Expects one of:
      - ifoCast columns present, incl. derivation_from_ifoCast, r_less_ifoCast
      - AR2 columns present, incl. derivation_from_AR, r_less_AR2 (baseline series is naiveAR2)
    """
    cols = set(df.columns)

    if "ifoCast" in cols or any("ifoCast" in c for c in cols):
        return {
            "baseline_label": "ifoCast",
            "shock_col": "r_less_ifoCast",
            "derivation_col": "derivation_from_ifoCast",
            "ni_lin_col": "net_improvement_jdg_ifoCast_lin",
            "ni_quad_col": "net_improvement_jdg_ifoCast_quad",
        }

    # AR2 case (baseline series column is naiveAR2; classification uses r_less_AR2)
    if "naiveAR2" in cols or any("AR" in c for c in cols) or any("AR2" in c for c in cols):
        return {
            "baseline_label": "AR2",
            "shock_col": "r_less_AR2",
            "derivation_col": "derivation_from_AR",
            "ni_lin_col": "net_improvement_jdg_AR_lin",
            "ni_quad_col": "net_improvement_jdg_AR_quad",
        }

    raise ValueError("Could not infer baseline. Expected ifoCast- or AR(2)-related columns.")


## Statistics Table builder

def _generate_summary_statistics(df: pd.DataFrame, baseline_label: str, shock_filter: Optional[bool] = None) -> dict:
    """
    Generate summary statistics for a judgemental evaluation dataframe.
    
    Args:
        df: DataFrame with columns for shocks, errors, improvements, adjustments
        baseline_label: "ifoCast" or "AR2"
        shock_filter: None for overall, True for negative shocks only, False for positive shocks only
    
    Returns:
        Dictionary with summary statistics
    """
    spec = _infer_baseline_spec(df)
    shock_col = spec["shock_col"]
    ni_lin_col = spec["ni_lin_col"]
    ni_quad_col = spec["ni_quad_col"]
    
    # Determine error columns based on baseline
    if baseline_label == "ifoCast":
        error_jdg_col = "error_realized_minus_judgemental"
        error_baseline_col = "error_realized_minus_ifoCast"
        adjustment_col = "j_less_ifoCast"
        improvement_col = "j_diff_less_ifoCast_diff"
    else:  # AR2
        error_jdg_col = "error_realized_minus_judgemental"
        error_baseline_col = "error_realized_minus_naiveAR2"
        adjustment_col = "j_less_AR2"
        improvement_col = "j_diff_less_AR2_diff"
    
    # Remove rows with NaN values for calculations
    df_clean = df.dropna(subset=[shock_col, error_jdg_col, error_baseline_col, 
                                   ni_lin_col, ni_quad_col, adjustment_col, improvement_col])
    
    # Apply shock filter if specified
    if shock_filter is not None:
        shock_series_bool = df_clean[shock_col].astype(bool)
        if shock_filter:
            # Negative shocks (r < baseline)
            df_clean = df_clean[shock_series_bool]
            subsample_label = "Negative Shocks"
        else:
            # Positive shocks (r >= baseline)
            df_clean = df_clean[~shock_series_bool]
            subsample_label = "Positive Shocks"
    else:
        subsample_label = "Overall"
    
    # Shock counts
    shock_series = df_clean[shock_col].astype(bool)
    negative_shocks = shock_series.sum()
    positive_shocks = (~shock_series).sum()
    
    # Average shock size (absolute error)
    avg_jdg_error = df_clean[error_jdg_col].abs().mean()
    avg_baseline_error = df_clean[error_baseline_col].abs().mean()
    
    # Average net improvements
    avg_ni_lin = df_clean[ni_lin_col].mean()
    avg_ni_quad = df_clean[ni_quad_col].mean()
    
    # Adjustment counts (j_less_baseline)
    adjustment_series = df_clean[adjustment_col].astype(bool)
    adjustments_below_baseline = adjustment_series.sum()
    adjustments_above_baseline = (~adjustment_series).sum()
    
    # Improvement counts (times adjustments led to improvements)
    improvement_series = df_clean[improvement_col].astype(bool)
    successful_improvements = improvement_series.sum()
    unsuccessful_adjustments = (~improvement_series).sum()
    
    return {
        "Baseline": baseline_label,
        "Subsample": subsample_label,
        "Negative Shocks (r < baseline)": int(negative_shocks),
        "Positive Shocks (r >= baseline)": int(positive_shocks),
        "Avg Judgemental Error (abs)": round(avg_jdg_error, 4),
        "Avg Baseline Error (abs)": round(avg_baseline_error, 4),
        "Avg Net Improvement (Linear)": round(avg_ni_lin, 4),
        "Avg Net Improvement (Quadratic)": round(avg_ni_quad, 4),
        "Adjustments Below Baseline (j < b)": int(adjustments_below_baseline),
        "Adjustments Above Baseline (j >= b)": int(adjustments_above_baseline),
        "Adjustments Reducing Error (|j-r| < |b-r|)": int(successful_improvements),
        "Adjustments Increasing Error": int(unsuccessful_adjustments),
        "Total Observations": len(df_clean),
    }


# =================================================================================================#
#                                    Generate Summary Statistics                                   #
# =================================================================================================#

# Define visualization function first
def visualize_summary_statistics(df: pd.DataFrame, save_folder: str | Path) -> None:
    """
    Create comprehensive visualizations of summary statistics.
    
    Args:
        df: Summary statistics DataFrame
        save_folder: Directory to save plots
    """
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    
    # Prepare data: separate by baseline and align subsample order
    df_ifoCast = df[df['Baseline'] == 'ifoCast'].copy()
    df_AR2 = df[df['Baseline'] == 'AR2'].copy()

    preferred_subsample_order = ["Overall", "Negative Shocks", "Positive Shocks"]
    present_subsamples = df["Subsample"].dropna().unique().tolist()
    subsamples = [s for s in preferred_subsample_order if s in present_subsamples]
    subsamples.extend([s for s in present_subsamples if s not in subsamples])

    df_ifoCast = df_ifoCast.set_index("Subsample").reindex(subsamples).reset_index()
    df_AR2 = df_AR2.set_index("Subsample").reindex(subsamples).reset_index()
    
    # ---- PLOT 1: Sample sizes across subsamples ----
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x_pos = np.arange(len(subsamples))
    width = 0.35
    
    ax.bar(x_pos - width/2, df_ifoCast['Total Observations'].values, width, label='ifoCast', alpha=0.8)
    ax.bar(x_pos + width/2, df_AR2['Total Observations'].values, width, label='AR2', alpha=0.8)
    
    ax.set_xlabel('Subsample', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Observations', fontsize=11, fontweight='bold')
    ax.set_title('Sample Sizes by Subsample and Baseline', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subsamples)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_folder / 'summary_sample_sizes.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    # ---- PLOT 2: Average Errors + Linear Improvement (side-by-side baselines) ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    width = 0.25

    # Left: ifoCast baseline
    axes[0].bar(
        x_pos - width,
        df_ifoCast['Avg Judgemental Error (abs)'].values,
        width,
        label='Avg |judgemental - realized|',
        alpha=0.85,
        color='steelblue',
    )
    axes[0].bar(
        x_pos,
        df_ifoCast['Avg Baseline Error (abs)'].values,
        width,
        label='Avg |ifoCast - realized|',
        alpha=0.85,
        color='darkorange',
    )
    axes[0].bar(
        x_pos + width,
        df_ifoCast['Avg Net Improvement (Linear)'].values,
        width,
        label='Avg net improvement (linear)',
        alpha=0.85,
        color='seagreen',
    )
    axes[0].axhline(0, color='black', linestyle='-', linewidth=0.8)
    axes[0].set_xlabel('Subsample', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Average Value', fontsize=11, fontweight='bold')
    axes[0].set_title('ifoCast Baseline', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(subsamples)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend(fontsize=9, loc='upper left')

    # Right: AR2 baseline
    axes[1].bar(
        x_pos - width,
        df_AR2['Avg Judgemental Error (abs)'].values,
        width,
        label='Avg |judgemental - realized|',
        alpha=0.85,
        color='steelblue',
    )
    axes[1].bar(
        x_pos,
        df_AR2['Avg Baseline Error (abs)'].values,
        width,
        label='Avg |AR2 - realized|',
        alpha=0.85,
        color='darkorange',
    )
    axes[1].bar(
        x_pos + width,
        df_AR2['Avg Net Improvement (Linear)'].values,
        width,
        label='Avg net improvement (linear)',
        alpha=0.85,
        color='seagreen',
    )
    axes[1].axhline(0, color='black', linestyle='-', linewidth=0.8)
    axes[1].set_xlabel('Subsample', fontsize=11, fontweight='bold')
    axes[1].set_title('AR2 Baseline', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(subsamples)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend(fontsize=9, loc='upper left')

    fig.suptitle('Average Forecast Errors by Subsample and Baseline', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_folder / 'summary_average_errors_by_baseline.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    # ---- PLOT 3: Net Improvements (Linear) ----
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.bar(x_pos - width/2, df_ifoCast['Avg Net Improvement (Linear)'].values, width, 
           label='ifoCast', alpha=0.8, color='steelblue')
    ax.bar(x_pos + width/2, df_AR2['Avg Net Improvement (Linear)'].values, width, 
           label='AR2', alpha=0.8, color='darkorange')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Subsample', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Linear Net Improvement', fontsize=11, fontweight='bold')
    ax.set_title('Judgemental vs Baseline Linear Improvements by Subsample', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subsamples)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_folder / 'summary_net_improvement_linear.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    # ---- PLOT 4: Net Improvements (Quadratic) ----
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.bar(x_pos - width/2, df_ifoCast['Avg Net Improvement (Quadratic)'].values, width, 
           label='ifoCast', alpha=0.8, color='steelblue')
    ax.bar(x_pos + width/2, df_AR2['Avg Net Improvement (Quadratic)'].values, width, 
           label='AR2', alpha=0.8, color='darkorange')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Subsample', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Quadratic Net Improvement', fontsize=11, fontweight='bold')
    ax.set_title('Judgemental vs Baseline Quadratic Improvements by Subsample', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subsamples)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_folder / 'summary_net_improvement_quadratic.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    # ---- PLOT 5: Adjustment Success Rates ----
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Calculate success rates (percentage of adjustments that reduced error)
    df_ifoCast['Success Rate (%)'] = (df_ifoCast['Adjustments Reducing Error (|j-r| < |b-r|)'] / 
                                       (df_ifoCast['Adjustments Reducing Error (|j-r| < |b-r|)'] + 
                                        df_ifoCast['Adjustments Increasing Error'])) * 100
    df_AR2['Success Rate (%)'] = (df_AR2['Adjustments Reducing Error (|j-r| < |b-r|)'] / 
                                  (df_AR2['Adjustments Reducing Error (|j-r| < |b-r|)'] + 
                                   df_AR2['Adjustments Increasing Error'])) * 100
    
    ax.bar(x_pos - width/2, df_ifoCast['Success Rate (%)'].values, width, 
           label='ifoCast', alpha=0.8, color='green')
    ax.bar(x_pos + width/2, df_AR2['Success Rate (%)'].values, width, 
           label='AR2', alpha=0.8, color='red')
    
    ax.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='50% (Random)')
    ax.set_xlabel('Subsample', fontsize=11, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Adjustment Success Rates (% reducing error) by Subsample', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subsamples)
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_folder / 'summary_adjustment_success_rates.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    # ---- PLOT 6: Adjustment Direction Distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ifoCast
    adjustment_labels = ['Below Baseline\n(j < b)', 'Above Baseline\n(j >= b)']
    for idx, subsample in enumerate(subsamples):
        below = df_ifoCast[df_ifoCast['Subsample'] == subsample]['Adjustments Below Baseline (j < b)'].values[0]
        above = df_ifoCast[df_ifoCast['Subsample'] == subsample]['Adjustments Above Baseline (j >= b)'].values[0]
        
        axes[0].bar(idx, below, label='Below' if idx == 0 else '', alpha=0.8, color='steelblue')
        axes[0].bar(idx, above, bottom=below, label='Above' if idx == 0 else '', alpha=0.8, color='lightcoral')
    
    axes[0].set_ylabel('Number of Adjustments', fontsize=11, fontweight='bold')
    axes[0].set_title('Adjustment Direction Distribution - ifoCast Baseline', fontsize=12, fontweight='bold')
    axes[0].set_xticks(np.arange(len(subsamples)))
    axes[0].set_xticklabels(subsamples)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # AR2
    for idx, subsample in enumerate(subsamples):
        below = df_AR2[df_AR2['Subsample'] == subsample]['Adjustments Below Baseline (j < b)'].values[0]
        above = df_AR2[df_AR2['Subsample'] == subsample]['Adjustments Above Baseline (j >= b)'].values[0]
        
        axes[1].bar(idx, below, label='Below' if idx == 0 else '', alpha=0.8, color='steelblue')
        axes[1].bar(idx, above, bottom=below, label='Above' if idx == 0 else '', alpha=0.8, color='lightcoral')
    
    axes[1].set_ylabel('Number of Adjustments', fontsize=11, fontweight='bold')
    axes[1].set_title('Adjustment Direction Distribution - AR2 Baseline', fontsize=12, fontweight='bold')
    axes[1].set_xticks(np.arange(len(subsamples)))
    axes[1].set_xticklabels(subsamples)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_folder / 'summary_adjustment_directions.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSummary statistics visualizations saved to {save_folder}")


# Generate summary statistics for both baselines with subsamples
summary_stats_list = []

# ifoCast baseline
summary_stats_list.append(_generate_summary_statistics(judgment_eval_df_ifoCast, "ifoCast", shock_filter=None))
summary_stats_list.append(_generate_summary_statistics(judgment_eval_df_ifoCast, "ifoCast", shock_filter=True))
summary_stats_list.append(_generate_summary_statistics(judgment_eval_df_ifoCast, "ifoCast", shock_filter=False))

# AR2 baseline
summary_stats_list.append(_generate_summary_statistics(judgment_eval_df_AR, "AR2", shock_filter=None))
summary_stats_list.append(_generate_summary_statistics(judgment_eval_df_AR, "AR2", shock_filter=True))
summary_stats_list.append(_generate_summary_statistics(judgment_eval_df_AR, "AR2", shock_filter=False))

# Create summary statistics DataFrame
summary_stats_df = pd.DataFrame(summary_stats_list)

print(f"\nSummary Statistics:")
#print("\n" + summary_stats_df.to_string())

# Generate visualizations FIRST
visualize_summary_statistics(summary_stats_df, graph_folder_derivations)

# Save to Excel
summary_stats_output_path = os.path.join(table_folder, "Summary_Statistics.xlsx")
summary_stats_df.to_excel(summary_stats_output_path, index=False, sheet_name="Summary Statistics")
print(f"\nSummary Statistics saved to {summary_stats_output_path}")








# =================================================================================================#
#                                   Analyze forecast persistence                                   #
# =================================================================================================#

## IDEA: use AR2 forecasts to obtain a baseline here











# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                       Visualize Results                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# =================================================================================================#
#                                    Derivation and Improvements                                   #
# =================================================================================================#

# -------------------------------------------------------------------------------------------------#
# Error Bar Plotter
# -------------------------------------------------------------------------------------------------#

## Reformat Helper
def _format_quarterly_index(dt_index) -> list[str]:
    """
    Convert a datetime index to yyyy-Qx format for display.
    """
    def to_quarter_str(ts):
        if pd.isna(ts):
            return "NaN"
        # Determine quarter from month
        quarter = (ts.month - 1) // 3 + 1
        return f"{ts.year}-Q{quarter}"
    
    return [to_quarter_str(ts) for ts in dt_index]

## MAIN PLOTTER FUNCTION
def plot_judgemental_derivations_or_net_improvement(
    df: pd.DataFrame,
    kind: str,  # "derivation" or "net_improvement"
    graph_folder: str | Path,
    header: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    filename_suffix: Optional[str] = None,
    show: bool = False,
    dpi: int = 180,
    y_axis_percentile: Optional[float] = None,  # e.g., 95.0 to truncate at 95th percentile (both tails)
) -> Path | list[Path]:
    """
    Plot either:
      - kind="derivation": judgemental derivations (j - baseline) over the index
      - kind="net_improvement": net improvements (LINEAR and QUADRATIC in separate plots)

    Bars are coloured by "shock" (negative shock): r_less_<baseline> == True
      - Red   : negative shock (realised < baseline)
      - Green : otherwise

    Args:
        df: DataFrame with relevant columns
        kind: "derivation" or "net_improvement"
        graph_folder: Directory to save plots
        header: Custom plot title
        filename_prefix: Custom filename prefix
        filename_suffix: Optional suffix to append to filename (e.g., "_truncated")
        show: Whether to display plots
        dpi: Resolution for saved figures
        y_axis_percentile: If provided (e.g., 95.0), truncate y-axis at percentile bounds 
                          (symmetric around zero). Useful for outlier visualization.

    Returns:
        Path: for "derivation" kind
        list[Path]: for "net_improvement" kind (two plots: linear and quadratic)
    """
    if kind not in {"derivation", "net_improvement"}:
        raise ValueError("kind must be either 'derivation' or 'net_improvement'.")

    spec = _infer_baseline_spec(df)
    baseline_label = spec["baseline_label"]
    shock_col = spec["shock_col"]

    graph_folder = Path(graph_folder)
    graph_folder.mkdir(parents=True, exist_ok=True)

    # X axis: convert to positional indices for consistent handling
    x = df.index
    x_labels = _format_quarterly_index(x)
    pos = np.arange(len(df))

    # Shock colouring (NA-safe)
    shock = df[shock_col].astype("boolean")
    colours = np.where(shock.fillna(False).to_numpy(), "red", "green")

    # Legend patches for shock definition
    shock_legend = [
        Patch(facecolor="red", edgecolor="none", label=f"Negative shock: realised < {baseline_label}"),
        Patch(facecolor="green", edgecolor="none", label=f"Positive shock: realised ≥ {baseline_label}"),
    ]

    if kind == "derivation":
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.axhline(0.0, linewidth=1.0)

        y_col = spec["derivation_col"]
        y = df[y_col].to_numpy()

        # Bar plot (colour by shock) - use positional indices for consistent handling
        ax.bar(pos, y, color=colours)

        # Tick labels: format as yyyy-Qx
        ax.set_xticks(pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        # Title/labels
        title = header or f"Judgemental derivations vs {baseline_label}"
        ax.set_title(title)
        ax.set_ylabel(f"Derivation (judgemental - {baseline_label})")

        # Apply y-axis truncation if requested
        if y_axis_percentile is not None:
            _apply_percentile_truncation(ax, y, y_axis_percentile)

        # Intelligent legend
        ax.legend(handles=shock_legend, loc="best", frameon=True)

        # Filename
        prefix = filename_prefix or "judgemental_derivations"
        suffix = f"_{filename_suffix}" if filename_suffix else ""
        out_path = graph_folder / f"{prefix}_vs_{baseline_label}{suffix}.png"

        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return out_path

    else:  # net_improvement - create TWO separate plots
        lin_col = spec["ni_lin_col"]
        quad_col = spec["ni_quad_col"]

        y_lin = df[lin_col].to_numpy()
        y_quad = df[quad_col].to_numpy()

        out_paths = []
        prefix = filename_prefix or "net_improvement"
        suffix = f"_{filename_suffix}" if filename_suffix else ""

        # --- PLOT 1: LINEAR IMPROVEMENT ---
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.axhline(0.0, linewidth=1.0)

        ax.bar(pos, y_lin, color=colours, label="Net improvement (linear)")

        # Tick labels: format as yyyy-Qx
        ax.set_xticks(pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        title = header or f"Net improvement (linear) of judgemental forecast vs {baseline_label}"
        ax.set_title(title)
        ax.set_ylabel("Improvement (>0 is better than baseline)")

        # Apply y-axis truncation if requested
        if y_axis_percentile is not None:
            _apply_percentile_truncation(ax, y_lin, y_axis_percentile)

        # Intelligent legend: metric legend + shock explanation
        metric_legend = ax.legend(loc="upper left", frameon=True)
        ax.add_artist(metric_legend)
        ax.legend(handles=shock_legend, loc="best", frameon=True)

        out_path_lin = graph_folder / f"{prefix}_linear_jdg_vs_{baseline_label}{suffix}.png"
        fig.tight_layout()
        fig.savefig(out_path_lin, dpi=dpi, bbox_inches="tight")
        out_paths.append(out_path_lin)

        if show:
            plt.show()
        else:
            plt.close(fig)

        # --- PLOT 2: QUADRATIC IMPROVEMENT ---
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.axhline(0.0, linewidth=1.0)

        ax.bar(pos, y_quad, color=colours, alpha=0.7, label="Net improvement (quadratic)")

        # Tick labels: format as yyyy-Qx
        ax.set_xticks(pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        title = header or f"Net improvement (quadratic) of judgemental forecast vs {baseline_label}"
        ax.set_title(title)
        ax.set_ylabel("Improvement (>0 is better than baseline)")

        # Apply y-axis truncation if requested
        if y_axis_percentile is not None:
            _apply_percentile_truncation(ax, y_quad, y_axis_percentile)

        # Intelligent legend: metric legend + shock explanation
        metric_legend = ax.legend(loc="upper left", frameon=True)
        ax.add_artist(metric_legend)
        ax.legend(handles=shock_legend, loc="best", frameon=True)

        out_path_quad = graph_folder / f"{prefix}_quadratic_jdg_vs_{baseline_label}.png"
        fig.tight_layout()
        fig.savefig(out_path_quad, dpi=dpi, bbox_inches="tight")
        out_paths.append(out_path_quad)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return out_paths

## Plotter Zoomer
def _apply_percentile_truncation(ax, data: np.ndarray, percentile: float) -> None:
    """
    Truncate y-axis symmetrically based on percentile.
    
    Args:
        ax: Matplotlib axis object
        data: Numeric data array
        percentile: Percentile threshold (0-100). E.g., 95 means truncate tails beyond 95th percentile.
    """
    # Handle NaN values
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) == 0:
        return
    
    # Calculate bounds symmetrically
    lower_bound = np.percentile(clean_data, 100 - percentile)
    upper_bound = np.percentile(clean_data, percentile)
    
    # Ensure symmetric margins around zero if both bounds have same sign
    if lower_bound >= 0:
        margin = upper_bound * (1 - percentile / 100)
        lower_bound = -margin
    elif upper_bound <= 0:
        margin = abs(lower_bound) * (1 - percentile / 100)
        upper_bound = margin
    
    ax.set_ylim(lower_bound, upper_bound)


# =================================================================================================#
#                                         Generate Plots                                           #
# =================================================================================================#

# -------------------------------------------------------------------------------------------------#
# judgemental derivations against the ifoCAST df
# -------------------------------------------------------------------------------------------------#

# expects joint_nowcast_ifoCast_df from earlier subsetting step
_ = plot_judgemental_derivations_or_net_improvement(
    df=judgment_eval_df_ifoCast,
    kind="derivation",
    graph_folder=graph_folder_derivations,
    header="Judgemental derivations vs ifoCast",
    filename_prefix="judgemental_derivations",
    show=False,
)

# -------------------------------------------------------------------------------------------------#
# judgemental derivations against the AR2 df
# -------------------------------------------------------------------------------------------------#

# expects joint_nowcast_AR_df from earlier subsetting step
_ = plot_judgemental_derivations_or_net_improvement(
    df=judgment_eval_df_AR,
    kind="derivation",
    graph_folder=graph_folder_derivations,
    header="Judgemental derivations vs AR(2)",
    filename_prefix="judgemental_derivations",
    show=False,
)

# -------------------------------------------------------------------------------------------------#
# Net improvement against the ifoCAST df (TRUNCATED at 95th percentile)
# -------------------------------------------------------------------------------------------------#

_ = plot_judgemental_derivations_or_net_improvement(
    df=judgment_eval_df_ifoCast,
    kind="net_improvement",
    graph_folder=graph_folder_derivations,
    header=None,  # custom headers now per metric (linear/quadratic)
    filename_prefix="net_improvement",
    filename_suffix="t95p",  # will append to filename
    show=False,
    y_axis_percentile=95.0,  # Truncate at 95th percentile for outlier visibility
)

# -------------------------------------------------------------------------------------------------#
# Net improvement against the ifoCAST df (FULL, no truncation)
# -------------------------------------------------------------------------------------------------#

_ = plot_judgemental_derivations_or_net_improvement(
    df=judgment_eval_df_ifoCast,
    kind="net_improvement",
    graph_folder=graph_folder_derivations,
    header=None,  # custom headers now per metric (linear/quadratic)
    filename_prefix="net_improvement",
    filename_suffix="full",  # will append to filename
    show=False,
    y_axis_percentile=None,  # No truncation - show full range
)

# -------------------------------------------------------------------------------------------------#
# Net improvement against the AR2 df (TRUNCATED at 95th percentile)
# -------------------------------------------------------------------------------------------------#

_ = plot_judgemental_derivations_or_net_improvement(
    df=judgment_eval_df_AR,
    kind="net_improvement",
    graph_folder=graph_folder_derivations,
    header=None,  # custom headers now per metric (linear/quadratic)
    filename_prefix="net_improvement",
    filename_suffix="t95p",  # will append to filename
    show=False,
    y_axis_percentile=95.0,  # Truncate at 95th percentile for outlier visibility
)

# -------------------------------------------------------------------------------------------------#
# Net improvement against the AR2 df (FULL, no truncation)
# -------------------------------------------------------------------------------------------------#

_ = plot_judgemental_derivations_or_net_improvement(
    df=judgment_eval_df_AR,
    kind="net_improvement",
    graph_folder=graph_folder_derivations,
    header=None,  # custom headers now per metric (linear/quadratic)
    filename_prefix="net_improvement",
    filename_suffix="full",  # will append to filename
    show=False,
    y_axis_percentile=None,  # No truncation - show full range
)





# =================================================================================================#
#                          Judgemental vs Benchmark Error Bars by Quarter                          #
# =================================================================================================#

# -------------------------------------------------------------------------------------------------#
# Error Bar Series Plotter
# -------------------------------------------------------------------------------------------------#

def plot_error_comparison(
    df: pd.DataFrame,
    error_col_jdg: str,  # e.g., "error_realized_minus_judgemental"
    error_col_benchmark: str,  # e.g., "error_realized_minus_ifoCast"
    benchmark_label: str,  # e.g., "ifoCast" or "AR2"
    graph_folder: str | Path,
    filename_prefix: Optional[str] = None,
    filename_suffix: Optional[str] = None,
    show: bool = False,
    dpi: int = 180,
    y_axis_percentile: Optional[float] = None,
) -> Path:
    """
    Plot judgemental vs benchmark errors side-by-side by quarter.
    
    Bars are coloured by "shock" classification (r_less_<benchmark>):
      - Red   : negative shock (realised < benchmark)
      - Green : otherwise
    
    Args:
        df: DataFrame with error columns and shock classification
        error_col_jdg: Column name for judgemental error
        error_col_benchmark: Column name for benchmark error
        benchmark_label: Label for benchmark (e.g., "ifoCast", "AR2")
        graph_folder: Directory to save plot
        filename_prefix: Custom filename prefix
        filename_suffix: Optional suffix to append to filename
        show: Whether to display plot
        dpi: Resolution for saved figures
        y_axis_percentile: Optional truncation at percentile (e.g., 95.0)
    
    Returns:
        Path to saved plot
    """
    spec = _infer_baseline_spec(df)
    shock_col = spec["shock_col"]
    
    graph_folder = Path(graph_folder)
    graph_folder.mkdir(parents=True, exist_ok=True)
    
    # X axis
    x = df.index
    x_labels = _format_quarterly_index(x)
    pos = np.arange(len(df))
    
    # Data
    y_jdg = df[error_col_jdg].to_numpy()
    y_bench = df[error_col_benchmark].to_numpy()
    
    # Shock colouring (NA-safe)
    shock = df[shock_col].astype("boolean")
    colours = np.where(shock.fillna(False).to_numpy(), "red", "green")
    
    # Legend patches
    shock_legend = [
        Patch(facecolor="red", edgecolor="none", label=f"Negative shock: realised < {benchmark_label}"),
        Patch(facecolor="green", edgecolor="none", label=f"Positive shock: realised ≥ {benchmark_label}"),
    ]
    
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axhline(0.0, linewidth=1.0)
    
    # Plot side-by-side bars
    width = 0.38
    ax.bar(pos - width/2, y_jdg, width=width, color=colours, label="Judgemental error", alpha=0.9)
    ax.bar(pos + width/2, y_bench, width=width, color=colours, label=f"{benchmark_label} error", alpha=0.6)
    
    # Tick labels: format as yyyy-Qx
    ax.set_xticks(pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    
    title = f"Judgemental vs {benchmark_label} Forecast Errors by Quarter"
    ax.set_title(title)
    ax.set_ylabel("Error (forecast - realized)")
    
    # Apply y-axis truncation if requested
    if y_axis_percentile is not None:
        combined_data = np.concatenate([y_jdg[~np.isnan(y_jdg)], y_bench[~np.isnan(y_bench)]])
        _apply_percentile_truncation(ax, combined_data, y_axis_percentile)
    
    # Intelligent legend: metric legend + shock explanation
    metric_legend = ax.legend(loc="upper left", frameon=True)
    ax.add_artist(metric_legend)
    ax.legend(handles=shock_legend, loc="best", frameon=True)
    
    # Filename
    prefix = filename_prefix or "error_comparison"
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    out_path = graph_folder / f"{prefix}_{benchmark_label}{suffix}.png"
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return out_path


# -------------------------------------------------------------------------------------------------#
# Evaluation against the ifoCAST df
# -------------------------------------------------------------------------------------------------#

# Truncated version
_ = plot_error_comparison(
    df=judgment_eval_df_ifoCast,
    error_col_jdg="error_realized_minus_judgemental",
    error_col_benchmark="error_realized_minus_ifoCast",
    benchmark_label="ifoCast",
    graph_folder=graph_folder_errors,
    filename_prefix="error_comparison",
    filename_suffix="t95p",
    show=False,
    y_axis_percentile=95.0,
)

# Full version
_ = plot_error_comparison(
    df=judgment_eval_df_ifoCast,
    error_col_jdg="error_realized_minus_judgemental",
    error_col_benchmark="error_realized_minus_ifoCast",
    benchmark_label="ifoCast",
    graph_folder=graph_folder_errors,
    filename_prefix="error_comparison",
    filename_suffix="full",
    show=False,
    y_axis_percentile=None,
)


# -------------------------------------------------------------------------------------------------#
# Evaluation against the AR2 df
# -------------------------------------------------------------------------------------------------#

# Truncated version
_ = plot_error_comparison(
    df=judgment_eval_df_AR,
    error_col_jdg="error_realized_minus_judgemental",
    error_col_benchmark="error_realized_minus_naiveAR2",
    benchmark_label="AR2",
    graph_folder=graph_folder_errors,
    filename_prefix="error_comparison",
    filename_suffix="t95p",
    show=False,
    y_axis_percentile=95.0,
)

# Full version
_ = plot_error_comparison(
    df=judgment_eval_df_AR,
    error_col_jdg="error_realized_minus_judgemental",
    error_col_benchmark="error_realized_minus_naiveAR2",
    benchmark_label="AR2",
    graph_folder=graph_folder_errors,
    filename_prefix="error_comparison",
    filename_suffix="full",
    show=False,
    y_axis_percentile=None,
)














# --------------------------------------------------------------------------------------------------
print(f" \n ifo Judgemental Forecasting Analysis Module complete! \n",f"Find Result Graphs in {graph_folder} and \nResult Tables in {table_folder}\n")
# --------------------------------------------------------------------------------------------------





# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#
