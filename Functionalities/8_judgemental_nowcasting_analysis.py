
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Judgemental Nowcasting Analysis Module
#
# Author:       Jan Ole Westphal
# Date:         2026-01
#
# Description:  Subprogram to run an econometric analysis on judgemental derivations in german
#               macroeconomic nowcasting.
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
import statsmodels.api as sm
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
result_folder = os.path.join(wd, '5_Judgemental_Derivations_Analysis', '1_Nowcasting')


## Subfolder
main_analysis_folder = os.path.join(result_folder, '0_Main_Analysis')
main_analysis_graphs_folder = os.path.join(main_analysis_folder, 'Graphs')
main_analysis_tables_folder = os.path.join(main_analysis_folder, 'Tables')

table_folder = os.path.join(result_folder, '1_Tables_EDA')
graph_folder = os.path.join(result_folder, '2_Plots_EDA')



## Graph subfolders
graph_folder_derivations = os.path.join(graph_folder, '1_Derivations_and_Improvements')
graph_folder_errors = os.path.join(graph_folder, '2_Error_Comparisons')



## Create if needed
for folder in [result_folder, table_folder, graph_folder, graph_folder_derivations, graph_folder_errors, main_analysis_folder, main_analysis_graphs_folder, main_analysis_tables_folder]:
    os.makedirs(folder, exist_ok=True)

# Additional subfolders for organized plots
#graph_folder_summary = os.path.join(graph_folder_derivations, 'summary_statistics')
graph_folder_net_improvement = os.path.join(graph_folder_derivations, 'net_improvement')

for folder in [ graph_folder_net_improvement]:
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




## HELPER: Build nowcasts by matching row/column on quarterly level.
def nowcast_builder(df, colname="ifo_judgemental_nowcast"):

    #show(df)

    # Convert row and column labels to quarterly Periods for robust matching
    ifo_rows_quarter = pd.to_datetime(df.index).to_period('Q')
    ifo_cols_quarter = pd.to_datetime(df.columns).to_period('Q')

    # Collect records for the output DataFrame
    records = []
    for col, col_quarter in zip(df.columns, ifo_cols_quarter):
        # Find rows whose quarter equals the column's quarter
        matching_rows = np.where(ifo_rows_quarter == col_quarter)[0]

        # Expect exactly one matching row per column; otherwise signal an error
        if len(matching_rows) != 1:
            raise ValueError(
                f"Expected exactly one quarterly row match for column {col} ({col_quarter}), "
                f"found {len(matching_rows)}."
            )

        # Get the row label (original index) and the corresponding value
        row_label = df.index[matching_rows[0]]
        records.append({
            'column_date': col,
            'matched_row_date': row_label,
            colname: df.loc[row_label, col]
        })

    # Build output DataFrame indexed by the original column dates
    out = pd.DataFrame(records).set_index('column_date')
    df_out = out[[colname]].copy()

    # Align to mid-quarter dates for downstream compatibility
    df_out = align_df_to_mid_quarters(df_out)

    #show(df_out)

    return df_out

# Path
file_path_ifo_qoq = os.path.join(wd, '0_0_Data', '2_Processed_Data', '3_ifo_qoq_series',
                                  'ifo_qoq_forecasts.xlsx' )

# Load 
ifo_qoq_forecasts = pd.read_excel(file_path_ifo_qoq, index_col=0)

# Extract nowcasts
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
# Load AR2-nowcasts AND AVERAGE-nowcasts
# -------------------------------------------------------------------------------------------------#

# Paths to the folders containing the Excel files
file_path_naive_qoq = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '1_QoQ_Forecast_Tables')

# Load all QoQ naive forecast Excel files into dictionary
naive_qoq_dfs_dict = load_excels_to_dict(file_path_naive_qoq, strip_string='naive_qoq_forecasts_')

# Define target naive models
naive_target_models = ['AR2', 'AVERAGE_1', 'AVERAGE_10', 'AVERAGE_FULL']
naive_nowcasts_dict = {}

for model_name in naive_target_models:
    # Use regex boundary match to avoid e.g. "AVERAGE_1" matching "AVERAGE_10_9"
    pattern = re.compile(rf'^{re.escape(model_name)}(_|$)')
    matches = [k for k in naive_qoq_dfs_dict if pattern.match(k)]
    if matches:
        print(f"Found naive forecast: {model_name}")
        df_model = naive_qoq_dfs_dict[matches[0]]
        
        # Get Nowcasts
        # Create column name like "naiveAR2", "naiveAVERAGE_1"
        col_name_naive = f"naive{model_name}"
        nowcasts = nowcast_builder(df_model, colname=col_name_naive)
        #print(f"Debug: {col_name_naive} head:")
        #print(nowcasts.head())
        naive_nowcasts_dict[model_name] = nowcasts
        #show(nowcasts)
    else:
        print(f"Warning: {model_name} not found in naive forecasts. Proceeding without it.")


# Ensure existence of naive forecasts
if not naive_nowcasts_dict:
    raise ValueError("No naive forecast models found (AR2, AVERAGE_10, AVERAGE_FULL, etc.). Cannot proceed with analysis. Check settings file and re-run Naive Forecaster.")





# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                          PROCESS DATA                                            #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# =================================================================================================#
#                                       Merge to joint df                                          #
# =================================================================================================#

# Prepare lists for merging
dfs_to_merge = [qoq_first_eval, ifo_judgemental_nowcasts, ifoCAST_nowcast]
col_names_merge = ['realized', 'judgemental', 'ifoCast']

# Add naive models if they exist
for model_name, df_nowcast in naive_nowcasts_dict.items():
    dfs_to_merge.append(df_nowcast)
    col_names_merge.append(f"naive{model_name}")

## Call merge_quarterly_dfs_dropna() from helperfunctions
joint_nowcast_df = merge_quarterly_dfs_dropna(
    dfs=dfs_to_merge,
    col_names=col_names_merge
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

## from Naive Models (AR2, Average, etc.)
for model_name in naive_nowcasts_dict.keys():
    col_name = f"naive{model_name}"
    # Check if column exists (it should after merge)
    if col_name in joint_nowcast_df.columns:
        joint_nowcast_df[f"derivation_from_{model_name}"] = (
            joint_nowcast_df["judgemental"] - joint_nowcast_df[col_name]
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
#   naive{Model}
#   ifoCast
#   error_realized_minus_judgemental
#   error_realized_minus_naive{Model}
#   error_realized_minus_ifoCast
#   derivation_from_ifoCast
#   derivation_from_{Model}
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


# ---------- Judgement vs Naive Models ----------
for model_name in naive_nowcasts_dict.keys():

    col_name = f"naive{model_name}"
    if col_name in joint_nowcast_df.columns:
        joint_nowcast_df[f"net_improvement_jdg_{model_name}_lin"] = (
            joint_nowcast_df[f"error_realized_minus_{col_name}"].abs()
            - joint_nowcast_df["error_realized_minus_judgemental"].abs()
        )

        joint_nowcast_df[f"net_improvement_jdg_{model_name}_quad"] = (
            joint_nowcast_df[f"error_realized_minus_{col_name}"]**2
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
# Evaluation against Naive Models
# -------------------------------------------------------------------------------------------------#
for model_name in naive_nowcasts_dict.keys():
    col_name = f"naive{model_name}"
    if col_name in joint_nowcast_df.columns:
        _classify_derivations(joint_nowcast_df, b_col=col_name, suffix=model_name)

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
ifo_cols = [c for c in judgment_eval_df_joint.columns if "ifoCast" in c]
judgment_eval_df_ifoCast = judgment_eval_df_joint[base_cols + ifo_cols].copy()
#show(judgment_eval_df_ifoCast)

judgment_eval_df_joint.to_excel(os.path.join(table_folder, "judgemental_derivations_full.xlsx"))
judgment_eval_df_ifoCast.to_excel(os.path.join(table_folder, "judgemental_derivations_ifoCast.xlsx"))


# ---- Naive Models subsets ----
judgment_eval_dfs = {}
for model_name in naive_nowcasts_dict.keys():
    # Identify model specific columns: contain model_name but not others
    # Strategy: columns containing model_name. 
    # Use exact checks to avoid substring matches if needed, but naive{model_name} is distinct enough usually.
    # Be careful with AVERAGE_1 vs AVERAGE_10.
    
    model_specific_cols = []
    for c in judgment_eval_df_joint.columns:
        if model_name in c:
            # Handle potential overlaps if any (e.g. AVERAGE_1 in AVERAGE_10)
            # If model_name is "AVERAGE_1", skip "AVERAGE_10"
            if model_name == "AVERAGE_1" and "AVERAGE_10" in c:
                continue
            model_specific_cols.append(c)
            
    df_subset = judgment_eval_df_joint[base_cols + model_specific_cols].copy()
    judgment_eval_dfs[model_name] = df_subset
    
    # Save
    df_subset.to_excel(os.path.join(table_folder, f"judgemental_derivations_{model_name}.xlsx"))





















# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                            BASELINE ANALYSIS: Judgemental Derivations                            #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# =================================================================================================#
#                                        Builder Functions                                         #
# =================================================================================================#

## HELPER

def _infer_baseline_spec(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer which baseline the df refers to (ifoCast vs AR2 vs AVERAGE etc.) and return column spec.
    """
    cols = set(df.columns)

    # ifoCast
    if "ifoCast" in cols or any("ifoCast" in c for c in cols):
        return {
            "baseline_label": "ifoCast",
            "shock_col": "r_less_ifoCast",
            "derivation_col": "derivation_from_ifoCast",
            "ni_lin_col": "net_improvement_jdg_ifoCast_lin",
            "ni_quad_col": "net_improvement_jdg_ifoCast_quad",
            "error_baseline_col": "error_realized_minus_ifoCast",
            "adjustment_col": "j_less_ifoCast",
            "improvement_col": "j_diff_less_ifoCast_diff",
        }

    # Naive Models
    # Look for derivation_from_{Model} column
    for c in cols:
        if c.startswith("derivation_from_"):
             model_part = c.replace("derivation_from_", "")
             if model_part == "ifoCast": continue
             
             # Check if corresponding shock classification exists to confirm it's a valid baseline set
             if f"r_less_{model_part}" in cols:
                 return {
                    "baseline_label": model_part,
                    "shock_col": f"r_less_{model_part}",
                    "derivation_col": f"derivation_from_{model_part}",
                    "ni_lin_col": f"net_improvement_jdg_{model_part}_lin",
                    "ni_quad_col": f"net_improvement_jdg_{model_part}_quad",
                    "error_baseline_col": f"error_realized_minus_naive{model_part}",
                    "adjustment_col": f"j_less_{model_part}",
                    "improvement_col": f"j_diff_less_{model_part}_diff",
                 }

    raise ValueError(f"Could not infer baseline. Columns found: {cols}")


## Statistics Table builder

def _generate_summary_statistics(df: pd.DataFrame, baseline_label: str, shock_filter: Optional[bool] = None) -> dict:
    """
    Generate summary statistics for a judgemental evaluation dataframe.
    """
    spec = _infer_baseline_spec(df)
    shock_col = spec["shock_col"]
    ni_lin_col = spec["ni_lin_col"]
    ni_quad_col = spec["ni_quad_col"]
    error_baseline_col = spec.get("error_baseline_col") 
    
    # If not in spec (old logic compat), standardizing inference
    if not error_baseline_col:
        # Fallback for ifoCast
        if baseline_label == "ifoCast":
            error_baseline_col = "error_realized_minus_ifoCast"
        else: # AR2 old fallback
            error_baseline_col = "error_realized_minus_naiveAR2"

    error_jdg_col = "error_realized_minus_judgemental"
    adjustment_col = spec.get("adjustment_col")
    improvement_col = spec.get("improvement_col")
    
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
    Dynamics version: adapts to available baselines.
    """
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    
    baselines = df['Baseline'].unique()
    num_baselines = len(baselines)
    
    preferred_subsample_order = ["Overall", "Negative Shocks", "Positive Shocks"]
    present_subsamples = df["Subsample"].dropna().unique().tolist()
    subsamples = [s for s in preferred_subsample_order if s in present_subsamples]
    subsamples.extend([s for s in present_subsamples if s not in subsamples])
    
    # helper to get data for a baseline aligned to subsamples
    def get_data(baseline, col):
        sub_df = df[df['Baseline'] == baseline].set_index("Subsample").reindex(subsamples)
        return sub_df[col].values

    # Colors for baselines
    # Use pyplot.colormaps to avoid MatplotlibDeprecationWarning from get_cmap
    cmap = plt.colormaps['tab10']
    colors = {b: cmap(i) for i, b in enumerate(baselines)}
    
    x_pos = np.arange(len(subsamples))
    # width depending on number of baselines. total width = 0.8?
    width = 0.8 / num_baselines
    
    # ---- PLOT 1: Sample sizes across subsamples ----
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for i, base in enumerate(baselines):
        offset = (i - (num_baselines - 1) / 2) * width
        vals = get_data(base, 'Total Observations')
        ax.bar(x_pos + offset, vals, width, label=base, alpha=0.8, color=colors[base])

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
    
    # ---- PLOT 2: Average Errors (subplots per baseline) ----
    # 3 bars per subsample: Jdg Error, Baseline Error, Improvement
    fig, axes = plt.subplots(1, num_baselines, figsize=(7*num_baselines, 5), sharey=True)
    if num_baselines == 1: axes = [axes]
    
    sub_width = 0.25
    
    for i, base in enumerate(baselines):
        ax = axes[i]
        
        err_jdg = get_data(base, 'Avg Judgemental Error (abs)')
        err_base = get_data(base, 'Avg Baseline Error (abs)')
        ni_lin = get_data(base, 'Avg Net Improvement (Linear)')
        
        ax.bar(x_pos - sub_width, err_jdg, sub_width, label='Avg |judgemental - realized|', alpha=0.85, color='steelblue')
        ax.bar(x_pos, err_base, sub_width, label='Avg |baseline - realized|', alpha=0.85, color='darkorange')
        ax.bar(x_pos + sub_width, ni_lin, sub_width, label='Avg net improvement (linear)', alpha=0.85, color='seagreen')
        
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Subsample', fontsize=11, fontweight='bold')
        ax.set_title(f'{base} Baseline', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subsamples)
        ax.grid(axis='y', alpha=0.3)
        if i == 0:
            ax.set_ylabel('Average Value', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9, loc='upper left')

    fig.suptitle('Average Forecast Errors by Subsample and Baseline', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_folder / 'summary_average_errors_by_baseline.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    # ---- PLOT 3: Net Improvements (Linear) ----
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for i, base in enumerate(baselines):
        offset = (i - (num_baselines - 1) / 2) * width
        vals = get_data(base, 'Avg Net Improvement (Linear)')
        ax.bar(x_pos + offset, vals, width, label=base, alpha=0.8, color=colors[base])
    
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
    # similar logic
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, base in enumerate(baselines):
        offset = (i - (num_baselines - 1) / 2) * width
        vals = get_data(base, 'Avg Net Improvement (Quadratic)')
        ax.bar(x_pos + offset, vals, width, label=base, alpha=0.8, color=colors[base])
        
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
    
    for i, base in enumerate(baselines):
        offset = (i - (num_baselines - 1) / 2) * width
        
        succ = get_data(base, 'Adjustments Reducing Error (|j-r| < |b-r|)')
        fail = get_data(base, 'Adjustments Increasing Error')
        tot = succ + fail
        # handle div 0
        rate = np.divide(succ, tot, out=np.zeros_like(succ, dtype=float), where=tot!=0) * 100
        
        ax.bar(x_pos + offset, rate, width, label=base, alpha=0.8, color=colors[base])
    
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
    fig, axes = plt.subplots(1, num_baselines, figsize=(7*num_baselines, 5), sharey=True)
    if num_baselines == 1: axes = [axes]

    for i, base in enumerate(baselines):
        ax = axes[i]
        
        below = get_data(base, 'Adjustments Below Baseline (j < b)')
        above = get_data(base, 'Adjustments Above Baseline (j >= b)')
        
        for idx in range(len(subsamples)):
             ax.bar(idx, below[idx], label='Below' if idx == 0 else '', alpha=0.8, color='steelblue')
             ax.bar(idx, above[idx], bottom=below[idx], label='Above' if idx == 0 else '', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Subsample', fontsize=11, fontweight='bold')
        ax.set_title(f'Adjustment Direction - {base} Baseline', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subsamples)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        if i == 0:
            ax.set_ylabel('Number of Adjustments', fontsize=11, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(save_folder / 'summary_adjustment_directions.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSummary statistics visualizations saved to {save_folder}")


# Generate summary statistics for baselines with subsamples
summary_stats_list = []

# ifoCast baseline
summary_stats_list.append(_generate_summary_statistics(judgment_eval_df_ifoCast, "ifoCast", shock_filter=None))
summary_stats_list.append(_generate_summary_statistics(judgment_eval_df_ifoCast, "ifoCast", shock_filter=True))
summary_stats_list.append(_generate_summary_statistics(judgment_eval_df_ifoCast, "ifoCast", shock_filter=False))

# Naive baselines
for model_name, df_sub in judgment_eval_dfs.items():
    summary_stats_list.append(_generate_summary_statistics(df_sub, model_name, shock_filter=None))
    summary_stats_list.append(_generate_summary_statistics(df_sub, model_name, shock_filter=True))
    summary_stats_list.append(_generate_summary_statistics(df_sub, model_name, shock_filter=False))

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

# Dictionary of all dataframes to iterate over for plotting
all_models_dfs = {'ifoCast': judgment_eval_df_ifoCast}
if naive_nowcasts_dict:
    all_models_dfs.update(judgment_eval_dfs)

for model_name, df_eval in all_models_dfs.items():
    if df_eval is None or df_eval.empty:
        continue
    
    print(f"Generating plots for baseline: {model_name}")

    # -------------------------------------------------------------------------------------------------#
    # Judgemental derivations
    # -------------------------------------------------------------------------------------------------#
    try:
        # Save to main derivations and net_improvement folders (no per-baseline folders)
        derivations_folder = graph_folder_derivations
        netimp_folder = graph_folder_net_improvement

        os.makedirs(derivations_folder, exist_ok=True)
        os.makedirs(netimp_folder, exist_ok=True)

        plot_judgemental_derivations_or_net_improvement(
            df=df_eval,
            kind="derivation",
            graph_folder=derivations_folder,
            header=f"Judgemental derivations vs {model_name}",
            filename_prefix="judgemental_derivations",
            show=False,
        )
    except Exception as e:
        print(f"  Error plotting derivations for {model_name}: {e}")

    # -------------------------------------------------------------------------------------------------#
    # Net improvement (TRUNCATED at 95th percentile)
    # -------------------------------------------------------------------------------------------------#
    try:
        plot_judgemental_derivations_or_net_improvement(
            df=df_eval,
            kind="net_improvement",
            graph_folder=netimp_folder,
            header=None,
            filename_prefix="net_improvement",
            filename_suffix=f"t95p",
            show=False,
            y_axis_percentile=95.0,
        )
    except Exception as e:
        print(f"  Error plotting net improvement (truncated) for {model_name}: {e}")

    # -------------------------------------------------------------------------------------------------#
    # Net improvement (FULL, no truncation)
    # -------------------------------------------------------------------------------------------------#
    try:
        plot_judgemental_derivations_or_net_improvement(
            df=df_eval,
            kind="net_improvement",
            graph_folder=netimp_folder,
            header=None,
            filename_prefix="net_improvement",
            filename_suffix=f"full",
            show=False,
            y_axis_percentile=None,
        )
    except Exception as e:
        print(f"  Error plotting net improvement (full) for {model_name}: {e}")



# =================================================================================================#
#                          Judgemental vs Benchmark Error Bars Time Series                         #
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
# Run Error Comparison Plots for all models
# -------------------------------------------------------------------------------------------------#

for model_name, df_eval in all_models_dfs.items():
    if df_eval is None or df_eval.empty:
        continue
        
    # Determine error column for benchmark
    if model_name == "ifoCast":
        error_col_benchmark = "error_realized_minus_ifoCast"
    else:
        # Construct error column: error_realized_minus_naive{Model}
        error_col_benchmark = f"error_realized_minus_naive{model_name}"
    
    # Truncated version
    try:
        # Save error plots to main error folder
        error_baseline_folder = graph_folder_errors
        os.makedirs(error_baseline_folder, exist_ok=True)

        plot_error_comparison(
            df=df_eval,
            error_col_jdg="error_realized_minus_judgemental",
            error_col_benchmark=error_col_benchmark,
            benchmark_label=model_name,
            graph_folder=error_baseline_folder,
            filename_prefix="error_comparison",
            filename_suffix="t95p",
            show=False,
            y_axis_percentile=95.0,
        )
    except Exception as e:
        print(f"  Error plotting error comparison (truncated) for {model_name}: {e}")

    # Full version
    try:
        plot_error_comparison(
            df=df_eval,
            error_col_jdg="error_realized_minus_judgemental",
            error_col_benchmark=error_col_benchmark,
            benchmark_label=model_name,
            graph_folder=error_baseline_folder,
            filename_prefix="error_comparison",
            filename_suffix="full",
            show=False,
            y_axis_percentile=None,
        )
    except Exception as e:
        print(f"  Error plotting error comparison (full) for {model_name}: {e}")













# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                        MAIN IDENTIFICATION ANALYSIS: Judgemental Derivations                     #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


"""
Model to be estimated: GDP forecasting as a compination of an autoregressive component,
observable covariates and a judgement component.

"""



# =================================================================================================#
#                                 Get core evaluation dataframe(s)                                 #
# =================================================================================================#


## Get function to retrieve the equation components

def transform_eval_dataframe(df, naive_col_name):
    """
    Transform evaluation dataframe by calculating signals relative to naive baseline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns ['realized', naive_col_name, 'ifoCast', 'judgemental']
    naive_col_name : str
        Name of the naive baseline column (e.g., 'naiveAR2', 'naiveAVERAGE_10')
    
    Returns:
    --------
    pd.DataFrame
        Transformed dataframe with columns ['realized', naive_col_name, 'ifoCast_signal', 'judgemental_signal']
    """
    # Create a copy to avoid modifying the original
    df_transformed = df.copy()
    
    # Calculate signals: realized - naive for the last two columns

    df_transformed['ifoCast_signal'] = df_transformed['ifoCast'] - df_transformed[naive_col_name]
    df_transformed['judgemental_signal'] = df_transformed['judgemental'] - df_transformed['ifoCast_signal'] - df_transformed[naive_col_name]
    df_transformed['error'] = df_transformed['realized'] - df_transformed['judgemental']
    
    # Drop the original ifoCast and judgemental columns
    df_transformed = df_transformed.drop(columns=['ifoCast', 'judgemental'])
    
    return df_transformed



# Apply to all dataframes
if naive_nowcasts_dict and "AR2" in naive_nowcasts_dict:
    equation_eval_df_AR2 = joint_nowcast_df[['realized', 'naiveAR2', 'ifoCast', 'judgemental']].copy()
    equation_eval_df_AR2 = transform_eval_dataframe(equation_eval_df_AR2, 'naiveAR2')
else:
    print("WARNING: AR2 baseline not available, skipping AR2-specific analyses; check settings file and Re-Run Naive Forecaster")


if naive_nowcasts_dict and "AVERAGE_1" in naive_nowcasts_dict:
    equation_eval_df_AVERAGE_1 = joint_nowcast_df[['realized', 'naiveAVERAGE_1', 'ifoCast', 'judgemental']].copy()
    equation_eval_df_AVERAGE_1 = transform_eval_dataframe(equation_eval_df_AVERAGE_1, 'naiveAVERAGE_1')
else:
    print("WARNING: AVERAGE_1 baseline not available, skipping AVERAGE_1-specific analyses; check settings file and Re-Run Naive Forecaster")


if naive_nowcasts_dict and "AVERAGE_10" in naive_nowcasts_dict:
    equation_eval_df_AVERAGE_10 = joint_nowcast_df[['realized', 'naiveAVERAGE_10', 'ifoCast', 'judgemental']].copy()
    equation_eval_df_AVERAGE_10 = transform_eval_dataframe(equation_eval_df_AVERAGE_10, 'naiveAVERAGE_10')
else:
    print("WARNING: AVERAGE_10 baseline not available, skipping AVERAGE_10-specific analyses; check settings file and Re-Run Naive Forecaster")


if naive_nowcasts_dict and "AVERAGE_FULL" in naive_nowcasts_dict:   
    equation_eval_df_AVERAGE_FULL = joint_nowcast_df[['realized', 'naiveAVERAGE_FULL', 'ifoCast', 'judgemental']].copy()
    equation_eval_df_AVERAGE_FULL = transform_eval_dataframe(equation_eval_df_AVERAGE_FULL, 'naiveAVERAGE_FULL')
else:   
    print("WARNING: AVERAGE_FULL baseline not available, skipping AVERAGE_FULL-specific analyses; check settings file and Re-Run Naive Forecaster")








# =================================================================================================#
#                                  Signals Analysis & OLS                                          #
# =================================================================================================#

def run_signals_analysis(df, model_name, naive_col, save_folder_graphs, save_folder_tables):
    """
    Run visualization, summary stats, and OLS regressions for the signals decomposition.
    """
    print(f"\n--- Running Signals Analysis for {model_name} ---")
    
    # 1. Visualization
    # ----------------
    # Plot components as bar time series (excluding error)
    cols_to_plot = ['realized', naive_col, 'ifoCast_signal', 'judgemental_signal']
    colors = ['black', 'gray', 'tab:blue', 'tab:orange']
    
    fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(12, 10), sharex=True)
    if len(cols_to_plot) == 1: axes = [axes]

    x_labels = _format_quarterly_index(df.index)
    pos = np.arange(len(df))
    
    for ax, col, color in zip(axes, cols_to_plot, colors):
        if col in df.columns:
            ax.bar(pos, df[col], color=color, alpha=0.7)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.set_ylabel(col)
            ax.grid(axis='y', alpha=0.3)
        
    axes[-1].set_xticks(pos)
    axes[-1].set_xticklabels(x_labels, rotation=45, ha='right')
    fig.suptitle(f'Signals Decomposition - {model_name}', fontsize=14)
    fig.tight_layout()
    
    plot_path = os.path.join(save_folder_graphs, f"Signals_Optimization_Plots_{model_name}.png")
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plots to {plot_path}")

    # 2. Summary Statistics
    # ---------------------
    stats_df = df.describe()
    stats_path = os.path.join(save_folder_tables, f"Signals_Stats_{model_name}.xlsx")
    stats_df.to_excel(stats_path)
    print(f"Saved stats to {stats_path}")

    # 3. Covariance / Correlation
    # ---------------------------
    cov_cols = [c for c in ['realized', naive_col, 'ifoCast_signal', 'judgemental_signal', 'error'] if c in df.columns]
    df_cov = df[cov_cols].dropna()

    # Save covariance table
    cov_matrix = df_cov.cov()
    cov_path = os.path.join(save_folder_tables, f"Covariance_Matrix_{model_name}.xlsx")
    cov_matrix.to_excel(cov_path)
    print(f"Saved covariance matrix to {cov_path}")

    # Save correlation table
    corr_matrix = df_cov.corr()
    corr_path = os.path.join(save_folder_tables, f"Correlation_Matrix_{model_name}.xlsx")
    corr_matrix.to_excel(corr_path)
    print(f"Saved correlation matrix to {corr_path}")

    # Seaborn heatmap of covariance matrix (upper triangle only; show diagonal)
    mask_cov = np.tril(np.ones_like(cov_matrix, dtype=bool), k=-1)
    fig_cov, ax_cov = plt.subplots(figsize=(8, 6))
    sns.heatmap(cov_matrix, mask=mask_cov, annot=True, fmt=".4f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax_cov)
    ax_cov.set_title(f"Covariance Matrix – {model_name}", fontsize=13, fontweight="bold")
    fig_cov.tight_layout()
    cov_plot_path = os.path.join(save_folder_graphs, f"Covariance_Heatmap_{model_name}.png")
    fig_cov.savefig(cov_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig_cov)
    print(f"Saved covariance heatmap to {cov_plot_path}")

    # Seaborn heatmap of correlation matrix (upper triangle only; show diagonal)
    mask_corr = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1)
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, mask=mask_corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax_corr)
    ax_corr.set_title(f"Correlation Matrix – {model_name}", fontsize=13, fontweight="bold")
    fig_corr.tight_layout()
    corr_plot_path = os.path.join(save_folder_graphs, f"Correlation_Heatmap_{model_name}.png")
    fig_corr.savefig(corr_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig_corr)
    print(f"Saved correlation heatmap to {corr_plot_path}")

    # 4. OLS Regressions
    # ------------------
    results_list = []
    
    # Ensure temporary columns for Model 4 don't pollute the next run if we modify df in place
    # Work on a copy OR clean up
    df_ols = df.copy()

    # Helper to run OLS and store results
    def run_ols_and_store(y_col_name, x_cols, hypothesis_str, y_data=None, equation_label="", individual_hypotheses=None):
        if y_data is None:
            y = df_ols[y_col_name]
        else:
            y = y_data
            
        X = df_ols[x_cols]
        
        # Handle NA
        combined = pd.concat([y, X], axis=1)
        combined_clean = combined.dropna()
        if combined_clean.empty:
            print(f"  Warning: No valid data for {equation_label}")
            return
            
        y_clean = combined_clean.iloc[:, 0]
        X_clean = combined_clean.iloc[:, 1:]
        
        model = sm.OLS(y_clean, X_clean)
        results = model.fit()
        
        # Joint Hypothesis Testing (F-test)
        try:
            f_test = results.f_test(hypothesis_str)
            f_val = f_test.fvalue.item() if hasattr(f_test.fvalue, "item") else f_test.fvalue
            f_p = f_test.pvalue.item() if hasattr(f_test.pvalue, "item") else f_test.pvalue
            reject = f_p < 0.05
        except Exception as e:
            f_val = np.nan
            f_p = np.nan
            reject = f"Error: {e}"

        # Store per-regressor info
        for term in x_cols:
            # Individual hypothesis test (t-test)
            # Default H0 is usually 1.0 for these signal models, or 0.0 for intercept/residuals
            h0_val = 0.0
            if individual_hypotheses and term in individual_hypotheses:
                h0_val = individual_hypotheses[term]
            
            try:
                t_test_res = results.t_test(f"{term} = {h0_val}")
                t_p_val = t_test_res.pvalue.item() if hasattr(t_test_res.pvalue, "item") else t_test_res.pvalue
            except:
                t_p_val = np.nan

            res_dict = {
                "Model_Baseline": model_name,
                "Equation": equation_label,
                "Term": term,
                "Coefficient": results.params[term],
                "StdErr": results.bse[term],
                "P-Value": results.pvalues[term],
                "H0_Value": h0_val,
                "Indiv_Test_P_Value": t_p_val,
                "Joint_Test_Hypothesis": hypothesis_str,
                "Joint_Test_F_Stat": f_val,
                "Joint_Test_P_Value": f_p,
                "Joint_Reject_5pct": reject
            }
            results_list.append(res_dict)

    # Model 1 (realized on AR, statSignal, judgSignal) is not identigied
    
    if naive_col in df_ols.columns:
        # Model 2: (realized - naive) ~ ifoCast_signal + judgemental_signal (No Intercept)
        # H0: betas = 1
        y_m2 = df_ols['realized'] - df_ols[naive_col]
        cols_m2 = ['ifoCast_signal', 'judgemental_signal']
        hyp_m2 = "ifoCast_signal = 1, judgemental_signal = 1"
        indiv_hyp_m2 = {'ifoCast_signal': 1.0, 'judgemental_signal': 1.0}
        run_ols_and_store(None, cols_m2, hyp_m2, y_data=y_m2, equation_label="2. (Realized - Naive) ~ Signals", individual_hypotheses=indiv_hyp_m2)

        # Model 3: (realized - naive) ~ intercept + ifoCast_signal + judgemental_signal
        # H0: intercept=0, betas = 1
        # Ensure intercept column exists for the regression
        df_ols['const'] = 1
        y_m3 = df_ols['realized'] - df_ols[naive_col]
        cols_m3 = ['const', 'ifoCast_signal', 'judgemental_signal']
        hyp_m3 = "const = 0, ifoCast_signal = 1, judgemental_signal = 1"
        indiv_hyp_m3 = {'const': 0.0, 'ifoCast_signal': 1.0, 'judgemental_signal': 1.0}
        run_ols_and_store(None, cols_m3, hyp_m3, y_data=y_m3, equation_label="3. (Realized - Naive) ~ Intercept + Signals", individual_hypotheses=indiv_hyp_m3)

        # Model 4: (realized - naive - ifoCast_signal) ~ judgemental_signal
        # H0: beta = 1
        y_m4 = df_ols['realized'] - df_ols[naive_col] - df_ols['ifoCast_signal']
        cols_m4 = ['judgemental_signal']
        hyp_m4 = "judgemental_signal = 1"
        indiv_hyp_m4 = {'judgemental_signal': 1.0}
        run_ols_and_store(None, cols_m4, hyp_m4, y_data=y_m4, equation_label="4. (Realized - Naive - ifoSignal) ~ JudgSignal", individual_hypotheses=indiv_hyp_m4)

        # Model 5: (realized - naive - ifoCast_signal) ~ intercept + judgemental_signal
        # H0: intercept=0, beta=1
        y_m5 = y_m4  # Same LHS
        cols_m5 = ['const', 'judgemental_signal']
        hyp_m5 = "const = 0, judgemental_signal = 1"
        indiv_hyp_m5 = {'const': 0.0, 'judgemental_signal': 1.0}
        run_ols_and_store(None, cols_m5, hyp_m5, y_data=y_m5, equation_label="5. (Realized - Naive - ifoSignal) ~ Intercept + JudgSignal", individual_hypotheses=indiv_hyp_m5)

        # Model 6: (realized - naive - ifoCast_signal) ~ intercept + judgemental_signal + judgemental_signal^2
        # H0: intercept=0, beta=1, beta_sq=0
        # Construct RHS with squared term and intercept
        df_ols['judgemental_signal_sq'] = df_ols['judgemental_signal'] ** 2
        cols_m6 = ['const', 'judgemental_signal', 'judgemental_signal_sq']
        hyp_m6 = "const = 0, judgemental_signal = 1, judgemental_signal_sq = 0"
        indiv_hyp_m6 = {'const': 0.0, 'judgemental_signal': 1.0, 'judgemental_signal_sq': 0.0}
        run_ols_and_store(None, cols_m6, hyp_m6, y_data=y_m5, equation_label="6. Efficiency Test (with Intercept & Sq)", individual_hypotheses=indiv_hyp_m6)

    return results_list

# Collection of all results
all_ols_results = []

# List of potential models to analyze
# Check if they exist in local scope
potential_models = [
    ("AR2", "naiveAR2"),
    ("AVERAGE_1", "naiveAVERAGE_1"),
    ("AVERAGE_10", "naiveAVERAGE_10"),
    ("AVERAGE_FULL", "naiveAVERAGE_FULL")
]

for name, col in potential_models:
    var_name = f"equation_eval_df_{name}"
    if var_name in locals():
        df_model = locals()[var_name]
        # Run analysis
        try:
            res = run_signals_analysis(df_model, name, col, main_analysis_graphs_folder, main_analysis_tables_folder)
            all_ols_results.extend(res)
        except Exception as e:
            print(f"Error running analysis for {name}: {e}")

# Save OLS Results
if all_ols_results:
    ols_df = pd.DataFrame(all_ols_results)
    ols_output_path = os.path.join(main_analysis_tables_folder, "Signals_OLS_Results.xlsx")
    ols_df.to_excel(ols_output_path, index=False)
    print(f"\nAll OLS results saved to {ols_output_path}")













# --------------------------------------------------------------------------------------------------
print(f" \n ifo Judgemental Forecasting Analysis Module complete! \n",f"Find Result Graphs in {graph_folder} and \nResult Tables in {table_folder}\n")
# --------------------------------------------------------------------------------------------------





# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#
