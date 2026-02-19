
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Judgemental Forecasting Analysis Module
#
# Author:       Jan Ole Westphal
# Date:         2026-01
#
# Description:  Subprogram to run an econometric analysis on judgemental derivations in german
#               macroeconomic forecasting with evaluation at multiple horizons (h=0 to h=6).
# 
#               Evaluates forecasts issued at different vintage dates against realized outcomes
#               at horizons 0 through 6 quarters ahead.
#
#               Runs all components from Data Processing to Output Processing and Visualizations.         
# ==================================================================================================
# --------------------------------------------------------------------------------------------------



""""
Main Tasks:
- Create measures of judgemental derivations at multiple horizons (h=0 to h=6):
    - derivation from the ifoCAST (NOTE: ifoCAST evaluation currently toggled OFF)
    - derivation from an AR2-benchmark
    - derivation from average models

- Create measures of net-improvement of derivations across horizons
- Classify derivations by:
    - direction of the shock
    - direction of the adjustment
    - net improvement success

- Analyze judgment quality as a function of forecast horizon
- Compare absolute vs relative accuracy across horizons

VISUALIZATIONS:
- Judgemental vs Benchmark error bars by horizon
- Derivations and net improvements across horizons
- Summary statistics by horizon
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
from typing import Union, Dict, Optional, Mapping, Tuple


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

from Functionalities.helpers.judgemental_evalfunctions import (
    add_error_columns,
    _classify_derivations,
    _infer_baseline_spec,
    _generate_summary_statistics,
    visualize_summary_statistics,
    _format_quarterly_index,
    _format_quarterly_index_with_horizon,
    _apply_percentile_truncation,
    plot_judgemental_derivations_or_net_improvement,
    plot_error_comparison,
    transform_eval_dataframe,
    run_signals_analysis,
)



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

# Component evaluation settings (optional)
evaluate_forecast_components = getattr(settings, "evaluate_forecast_components", False)
included_components = getattr(settings, "included_components", [])


# ==================================================================================================
#                                  CONFIGURATION VARIABLES
# ==================================================================================================

# TODO: Currently ifoCAST evaluation is disabled. Set to True to enable.
# NOTE: When enabled, implementation must include conditional expectation estimator
#       for horizon-h forecasts. Currently placeholder - requires conditional mean logic.
EVALUATE_IFOCAST = False

# For horizons, number of periods ahead to evaluate (0 = nowcast, up to 6)
MAX_HORIZON = 6


## Print Module header
print("\nExecuting the Judgemental Forecasting Analysis Module (Multi-Horizon) ... \n")


# ==================================================================================================
#                                    HORIZON LOADING HELPERS                                      #
# ==================================================================================================

def extract_horizon_forecasts(
    forecast_df: pd.DataFrame,
    *,
    colname: str,
    max_horizon: int = 6,
) -> Dict[int, pd.DataFrame]:
    """
    Extract horizon-specific forecasts from a forecast matrix.

    For each column (forecast vintage date), match the row with the same quarter (h=0)
    and then the next rows for horizons h=1..max_horizon.
    Output DataFrames are indexed by the target quarter date (row date).
    """
    if forecast_df.empty:
        return {}

    row_quarters = pd.to_datetime(forecast_df.index).to_period("Q")
    col_quarters = pd.to_datetime(forecast_df.columns).to_period("Q")

    horizon_records: Dict[int, list[dict]] = {h: [] for h in range(max_horizon + 1)}

    for col, col_quarter in zip(forecast_df.columns, col_quarters):
        matching_rows = np.where(row_quarters == col_quarter)[0]

        if len(matching_rows) != 1:
            raise ValueError(
                f"Expected exactly one quarterly row match for column {col} ({col_quarter}), "
                f"found {len(matching_rows)}."
            )

        start_idx = matching_rows[0]

        for h in range(max_horizon + 1):
            target_idx = start_idx + h
            if target_idx >= len(forecast_df.index):
                continue

            target_row = forecast_df.index[target_idx]
            horizon_records[h].append(
                {
                    "target_date": target_row,
                    colname: forecast_df.loc[target_row, col],
                }
            )

    horizons_dict: Dict[int, pd.DataFrame] = {}
    for h, records in horizon_records.items():
        if not records:
            continue

        df_out = pd.DataFrame(records).set_index("target_date")
        df_out = align_df_to_mid_quarters(df_out)
        horizons_dict[h] = df_out[[colname]].copy()

    return horizons_dict


## Clear Result Folders
#if settings.clear_result_folders:
#    folder_clear(folder_path)




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                          LOAD IN DATA                                            #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

# -------------------------------------------------------------------------------------------------#
# Load realized GDP series
# -------------------------------------------------------------------------------------------------#
eval_path = os.path.join(wd, '0_0_Data', '2_Processed_Data', '2_evaluation_series')
qoq_path_first = os.path.join(eval_path, 'first_release_qoq_GDP.xlsx')

## First Releases
qoq_first_eval = pd.read_excel(qoq_path_first, index_col=0)
qoq_first_eval = align_df_to_mid_quarters(qoq_first_eval)  # Align to mid-quarter dates
#show(qoq_first_eval)


# -------------------------------------------------------------------------------------------------#
# Load ifo qoq forecasts (with horizons)
# -------------------------------------------------------------------------------------------------#

# NOTE: For forecasting with horizons, we need to structure data as follows:
# For each forecast vintage date on the COLUMN, we extract 7 consecutive quarters
# starting from the same quarter as the forecast date (h=0) through 6 quarters ahead (h=6).
#
# Example: If forecast issued on 2010Q1 (column), we extract:
#   h=0: realized value for 2010Q1 (row matching column quarter)
#   h=1: realized value for 2010Q2 (row below that)
#   h=2: realized value for 2010Q3 (row 2 below)
#   ... and so on up to h=6
#
# TODO: Implement horizon loading logic below once forecast data structure is finalized

# Path to forecast files
file_path_ifo_qoq = os.path.join(wd, '0_0_Data', '2_Processed_Data', '3_ifo_qoq_series',
                                  'ifo_qoq_forecasts.xlsx' )

# Load 
ifo_qoq_forecasts = pd.read_excel(file_path_ifo_qoq, index_col=0)

ifo_judgemental_forecasts = extract_horizon_forecasts(
    ifo_qoq_forecasts,
    colname="judgemental",
    max_horizon=MAX_HORIZON,
)

# -------------------------------------------------------------------------------------------------#
# Load ifo component forecasts (optional)
# -------------------------------------------------------------------------------------------------#

ifo_qoq_forecasts_components = {}
ifo_component_forecasts = {}

if evaluate_forecast_components:
    file_path_ifo_qoq_components = os.path.join(
        wd, '0_0_Data', '2_Processed_Data', '3_gdp_component_forecast'
    )

    ifo_qoq_forecasts_components = load_ifo_component_forecasts(
        file_path_ifo_qoq_components,
        included_components=included_components,
    )

    for comp_name, comp_df in ifo_qoq_forecasts_components.items():
        ifo_component_forecasts[comp_name] = extract_horizon_forecasts(
            comp_df,
            colname="judgemental",
            max_horizon=MAX_HORIZON,
        )
        print(f"Loaded ifo component forecasts (h=0..{MAX_HORIZON}): {comp_name}")


# -------------------------------------------------------------------------------------------------#
# Load ifoCAST forecasts (currently disabled)
# -------------------------------------------------------------------------------------------------#

# TODO: When EVALUATE_IFOCAST=True, implement conditional expectation estimator
#       Current ifoCAST contains nowcasts only. For horizon-h forecasts, need:
#       - Conditional mean of future realized values given current information
#       - Multi-step ahead forecast structure
#
# if EVALUATE_IFOCAST:
#     ifoCAST_forecasts_full_path = os.path.join(
#         wd, '0_0_Data', '0_Forecast_Inputs', '2_ifoCAST', 'ifoCAST_forecasts_full.xlsx')
#     ifoCAST_forecasts = pd.read_excel(ifoCAST_forecasts_full_path, index_col=0)
#     ifoCAST_forecasts = align_df_to_mid_quarters(ifoCAST_forecasts)
# else:
#     print("INFO: ifoCAST forecasting evaluation is disabled (EVALUATE_IFOCAST=False)")
#     ifoCAST_forecasts = None


# -------------------------------------------------------------------------------------------------#
# Load AR2 forecasts AND AVERAGE forecasts (with horizons)
# -------------------------------------------------------------------------------------------------#

# Paths to the folders containing the Excel files
file_path_naive_qoq = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '1_QoQ_Forecast_Tables')

# Load all QoQ naive forecast Excel files into dictionary
naive_qoq_dfs_dict = load_excels_to_dict(file_path_naive_qoq, strip_string='naive_qoq_forecasts_')

# Define target naive models
naive_target_models = ['AR2', 'AVERAGE_1', 'AVERAGE_10', 'AVERAGE_FULL']
naive_forecasts_dict = {}

for model_name in naive_target_models:
    pattern = re.compile(rf'^{re.escape(model_name)}(_|$)')
    matches = [k for k in naive_qoq_dfs_dict if pattern.match(k)]
    if not matches:
        print(f"Warning: {model_name} not found in naive forecasts. Proceeding without it.")
        continue

    df_model = naive_qoq_dfs_dict[matches[0]]
    naive_forecasts_dict[model_name] = extract_horizon_forecasts(
        df_model,
        colname=f"naive{model_name}",
        max_horizon=MAX_HORIZON,
    )


# Ensure existence of naive forecasts
if not naive_forecasts_dict:
    raise ValueError(
        "No naive forecast models found (AR2, AVERAGE_10, AVERAGE_FULL, etc.). "
        "Cannot proceed with analysis. Check settings file and re-run Naive Forecaster."
    )


# -------------------------------------------------------------------------------------------------#
# Load naive component forecasts (optional)
# -------------------------------------------------------------------------------------------------#

component_naive_qoq_dfs_dict = {}
naive_component_forecasts = {}

if evaluate_forecast_components:
    file_path_component_qoq = os.path.join(
        wd, '0_0_Data', '3_Naive_Forecaster_Data', '3_QoQ_Component_Forecast_Tables'
    )

    component_naive_qoq_dfs_dict = load_component_naive_forecasts(
        file_path_component_qoq,
        included_components=included_components,
        drop_ar2_components=["PRIVCON"],
    )

    for comp_name, model_dict in component_naive_qoq_dfs_dict.items():
        naive_component_forecasts[comp_name] = {}
        for model_name, df_model in model_dict.items():
            naive_component_forecasts[comp_name][model_name] = extract_horizon_forecasts(
                df_model,
                colname=f"naive{model_name}",
                max_horizon=MAX_HORIZON,
            )
        print(f"Loaded naive component forecasts (h=0..{MAX_HORIZON}): {comp_name}")




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                    OUTPUT FOLDER SETUP                                           #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

# Base output folder for forecasting analysis (differs from nowcasting)
base_output_folder = os.path.join(wd, '5_Judgemental_Forecasts_Derivations_Analysis')

os.makedirs(base_output_folder, exist_ok=True)

# Create main and components subfolders
main_folder = os.path.join(base_output_folder, '0_Main')
components_folder = os.path.join(base_output_folder, '1_Components')
os.makedirs(main_folder, exist_ok=True)
os.makedirs(components_folder, exist_ok=True)

# Create subfolders for each baseline model in main folder
# Output structure: 5_Judgemental_Forecasts_Derivations_Analysis/0_Main/{baseline_model}/
model_output_folders = {}
for model_name in naive_target_models:
    model_folder = os.path.join(main_folder, model_name)
    os.makedirs(model_folder, exist_ok=True)
    model_output_folders[model_name] = model_folder

    # Create subdirectories for EDA and main analysis
    table_folder_EDA = os.path.join(model_folder, '0_EDA_Tables')
    graph_folder_EDA = os.path.join(model_folder, '0_EDA_Graphs')
    graph_folder_EDA_derivations = os.path.join(model_folder, '1_Derivations_Graphs')
    graph_folder_EDA_errors = os.path.join(model_folder, '2_Errors_Graphs')
    graph_folder_EDA_net_improvement = os.path.join(model_folder, '3_Net_Improvement_Graphs')
    
    main_analysis_tables_folder = os.path.join(model_folder, '4_Main_Analysis_Tables')
    main_analysis_graphs_folder = os.path.join(model_folder, '5_Main_Analysis_Graphs')

    for folder in [table_folder_EDA, graph_folder_EDA, graph_folder_EDA_derivations,
                   graph_folder_EDA_errors, graph_folder_EDA_net_improvement,
                   main_analysis_tables_folder, main_analysis_graphs_folder]:
        os.makedirs(folder, exist_ok=True)

# Create component folders for each component and baseline model upfront
# This ensures the folder structure exists even if there's no data to process
if evaluate_forecast_components and included_components:
    for component_name in included_components:
        for model_name in naive_target_models:
            comp_model_folder = os.path.join(components_folder, component_name, model_name)
            
            # Create all subdirectories matching the main analysis structure
            comp_folders_to_create = [
                os.path.join(comp_model_folder, '0_EDA_Tables'),
                os.path.join(comp_model_folder, '0_EDA_Graphs'),
                os.path.join(comp_model_folder, '1_Derivations_Graphs'),
                os.path.join(comp_model_folder, '2_Errors_Graphs'),
                os.path.join(comp_model_folder, '3_Net_Improvement_Graphs'),
                os.path.join(comp_model_folder, '4_Main_Analysis_Tables'),
                os.path.join(comp_model_folder, '5_Main_Analysis_Graphs'),
            ]
            
            for folder in comp_folders_to_create:
                os.makedirs(folder, exist_ok=True)


# Define a function to create component folders
def setup_component_folder(component_name, baseline_model):
    """Create folder structure for a specific component and baseline model"""
    comp_model_folder = os.path.join(components_folder, component_name, baseline_model)
    
    # Create subdirectories matching the main analysis structure
    comp_table_eda_folder = os.path.join(comp_model_folder, '0_EDA_Tables')
    comp_graph_eda_folder = os.path.join(comp_model_folder, '0_EDA_Graphs')
    comp_derivations_folder = os.path.join(comp_model_folder, '1_Derivations_Graphs')
    comp_errors_folder = os.path.join(comp_model_folder, '2_Errors_Graphs')
    comp_net_improvement_folder = os.path.join(comp_model_folder, '3_Net_Improvement_Graphs')
    comp_main_tables_folder = os.path.join(comp_model_folder, '4_Main_Analysis_Tables')
    comp_main_graphs_folder = os.path.join(comp_model_folder, '5_Main_Analysis_Graphs')
    
    for folder in [comp_model_folder, comp_table_eda_folder, comp_graph_eda_folder,
                   comp_derivations_folder, comp_errors_folder, comp_net_improvement_folder,
                   comp_main_tables_folder, comp_main_graphs_folder]:
        os.makedirs(folder, exist_ok=True)
    
    return {
        'table_eda': comp_table_eda_folder,
        'graph_eda': comp_graph_eda_folder,
        'graph_derivations': comp_derivations_folder,
        'graph_errors': comp_errors_folder,
        'graph_net_improvement': comp_net_improvement_folder,
        'table_main': comp_main_tables_folder,
        'graph_main': comp_main_graphs_folder,
    }


print(f"Output folders created in: {base_output_folder}")
print(f"  Main GDP Analysis:")
print(f"    - AR2")
print(f"    - AVERAGE_1")
print(f"    - AVERAGE_10")
print(f"    - AVERAGE_FULL")
if evaluate_forecast_components and included_components:
    print(f"  Component Analysis:")
    for comp in included_components:
        print(f"    - {comp}")




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                               FORECAST EVALUATION PIPELINE                                      #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


def run_forecast_evaluation_pipeline(
    judgemental_forecasts_dict: Dict[int, pd.DataFrame],
    naive_forecasts_dict: Dict[str, Dict[int, pd.DataFrame]],
    ifoCast_forecasts_dict: Optional[Dict[int, pd.DataFrame]],
    realized_df: pd.DataFrame,
    baseline_model_name: str,
    output_folders: Dict[str, str],
    max_horizon: int = 6,
    evaluate_ifoCast: bool = False,
    time_filter_start: str = '2000-01-01',
    time_filter_end: str = '2100-01-01'
) -> None:
    """
    Run evaluation pipeline for judgemental forecasts across multiple horizons.
    
    Parameters
    ----------
    judgemental_forecasts_dict : Dict[int, pd.DataFrame]
        Dictionary mapping horizon h -> DataFrame of judgemental forecasts at h
    naive_forecasts_dict : Dict[str, Dict[int, pd.DataFrame]]
        Dictionary mapping model_name -> {horizon -> forecast DataFrame}
    ifoCast_forecasts_dict : Optional[Dict[int, pd.DataFrame]]
        Dictionary mapping horizon -> ifoCAST DataFrame, or None if not evaluating
    realized_df : pd.DataFrame
        DataFrame of realized values
    baseline_model_name : str
        Name of baseline model (e.g., "AR2", "AVERAGE_10")
    output_folders : Dict[str, str]
        Dictionary with keys: 'table_eda', 'graph_eda', 'graph_derivations', 'graph_errors',
                             'graph_net_improvement', 'table_main', 'graph_main'
    max_horizon : int
        Maximum horizon to evaluate (default 6)
    evaluate_ifoCast : bool
        Whether to include ifoCAST in evaluation
    time_filter_start : str
        Start date for filtering (format: YYYY-MM-DD)
    time_filter_end : str
        End date for filtering (format: YYYY-MM-DD)
    
    Returns
    -------
    None
        Results are saved to output folders
    
    Notes
    -----
    TODO: Implementation of full horizon-based evaluation pipeline.
    Current outline:
    1. For each horizon h (0 to max_horizon):
       a. Extract judgemental, naive, and (optionally) ifoCAST forecasts at horizon h
       b. Merge with realized values at corresponding dates
       c. Compute error measures
       d. Compute derivation measures
       e. Compute net improvement measures
       f. Classify derivations
    2. Generate horizon-specific summary statistics
    3. Generate horizon-aware visualizations
    4. Aggregate results across horizons for cross-horizon comparison
    """
    
    print(f"\nRunning forecast evaluation pipeline for baseline model: {baseline_model_name}")
    print(f"Evaluating horizons: h=0 to h={max_horizon}")

    table_folder_eda = output_folders['table_eda']
    graph_folder_eda = output_folders['graph_eda']
    graph_folder_derivations = output_folders['graph_derivations']
    graph_folder_errors = output_folders['graph_errors']
    graph_folder_net_improvement = output_folders['graph_net_improvement']
    table_folder_main = output_folders['table_main']
    graph_folder_main = output_folders['graph_main']

    summary_stats_list: list[dict] = []

    baseline_horizons = naive_forecasts_dict.get(baseline_model_name, {})
    if not baseline_horizons:
        print(f"  WARNING: No naive forecasts found for baseline {baseline_model_name}")
        return

    for h in range(max_horizon + 1):
        if h not in judgemental_forecasts_dict or h not in baseline_horizons:
            print(f"  Skipping horizon h={h}: missing data")
            continue

        judgemental_df = judgemental_forecasts_dict[h]
        baseline_df = baseline_horizons[h]

        dfs_to_merge = [realized_df, judgemental_df, baseline_df]
        col_names_merge = ["realized", "judgemental", f"naive{baseline_model_name}"]

        if evaluate_ifoCast and ifoCast_forecasts_dict and h in ifoCast_forecasts_dict:
            dfs_to_merge.append(ifoCast_forecasts_dict[h])
            col_names_merge.append("ifoCast")

        joint_df = merge_quarterly_dfs_dropna(
            dfs=dfs_to_merge,
            col_names=col_names_merge,
        )

        joint_df = align_df_to_mid_quarters(joint_df)
        joint_df = filter_df_by_datetime_index(joint_df, time_filter_start, time_filter_end)
        joint_df = align_df_to_mid_quarters(joint_df)

        joint_df = add_error_columns(joint_df)

        if evaluate_ifoCast and "ifoCast" in joint_df.columns:
            joint_df["derivation_from_ifoCast"] = joint_df["judgemental"] - joint_df["ifoCast"]
            joint_df["net_improvement_jdg_ifoCast_lin"] = (
                joint_df["error_realized_minus_ifoCast"].abs()
                - joint_df["error_realized_minus_judgemental"].abs()
            )
            joint_df["net_improvement_jdg_ifoCast_quad"] = (
                joint_df["error_realized_minus_ifoCast"] ** 2
                - joint_df["error_realized_minus_judgemental"] ** 2
            )
            _classify_derivations(joint_df, b_col="ifoCast", suffix="ifoCast")

        joint_df[f"derivation_from_{baseline_model_name}"] = (
            joint_df["judgemental"] - joint_df[f"naive{baseline_model_name}"]
        )

        joint_df[f"net_improvement_jdg_{baseline_model_name}_lin"] = (
            joint_df[f"error_realized_minus_naive{baseline_model_name}"].abs()
            - joint_df["error_realized_minus_judgemental"].abs()
        )

        joint_df[f"net_improvement_jdg_{baseline_model_name}_quad"] = (
            joint_df[f"error_realized_minus_naive{baseline_model_name}"] ** 2
            - joint_df["error_realized_minus_judgemental"] ** 2
        )

        _classify_derivations(joint_df, b_col=f"naive{baseline_model_name}", suffix=baseline_model_name)

        horizon_table_path = os.path.join(table_folder_eda, f"Forecast_Series_h{h}.xlsx")
        joint_df.to_excel(horizon_table_path)

        base_cols = [
            "realized",
            "judgemental",
            "error_realized_minus_judgemental",
        ]

        model_cols = [c for c in joint_df.columns if baseline_model_name in c]
        eval_df = joint_df[base_cols + model_cols].copy()

        for shock_filter in (None, True, False):
            stats = _generate_summary_statistics(eval_df, baseline_model_name, shock_filter=shock_filter)
            stats["Horizon"] = h
            summary_stats_list.append(stats)

        try:
            plot_judgemental_derivations_or_net_improvement(
                df=eval_df,
                kind="derivation",
                graph_folder_EDA=graph_folder_derivations,
                header=f"Judgemental derivations vs {baseline_model_name} (h={h})",
                filename_prefix="judgemental_derivations",
                filename_suffix=f"h{h}",
                show=False,
            )
        except Exception as e:
            print(f"  Error plotting derivations for h={h}: {e}")

        try:
            plot_judgemental_derivations_or_net_improvement(
                df=eval_df,
                kind="net_improvement",
                graph_folder_EDA=graph_folder_net_improvement,
                header=None,
                filename_prefix="net_improvement",
                filename_suffix=f"h{h}_t95p",
                show=False,
                y_axis_percentile=95.0,
            )
        except Exception as e:
            print(f"  Error plotting net improvement for h={h}: {e}")

        try:
            plot_error_comparison(
                df=eval_df,
                error_col_jdg="error_realized_minus_judgemental",
                error_col_benchmark=f"error_realized_minus_naive{baseline_model_name}",
                benchmark_label=baseline_model_name,
                graph_folder=graph_folder_errors,
                filename_prefix="error_comparison",
                filename_suffix=f"h{h}_t95p",
                show=False,
                y_axis_percentile=95.0,
            )
        except Exception as e:
            print(f"  Error plotting error comparison for h={h}: {e}")

        horizon_summary_df = pd.DataFrame(
            [s for s in summary_stats_list if s.get("Horizon") == h]
        )
        if not horizon_summary_df.empty:
            horizon_summary_path = os.path.join(table_folder_eda, f"Summary_Statistics_h{h}.xlsx")
            horizon_summary_df.to_excel(horizon_summary_path, index=False)
            horizon_summary_folder = os.path.join(graph_folder_derivations, f"h{h}_summary")
            visualize_summary_statistics(horizon_summary_df, horizon_summary_folder)

    if summary_stats_list:
        summary_stats_df = pd.DataFrame(summary_stats_list)
        summary_stats_path = os.path.join(table_folder_main, "Summary_Statistics_All_Horizons.xlsx")
        summary_stats_df.to_excel(summary_stats_path, index=False)
    







# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                    MAIN EXECUTION BLOCK                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

"""
Main evaluation execution:

1. For each baseline model (AR2, AVERAGE_1, AVERAGE_10, AVERAGE_FULL):
   a. Run forecast evaluation pipeline
   b. Generate model-specific results
   
2. Aggregate and compare results across baseline models
"""

print("\n" + "="*80)
print("FULL GDP ANALYSIS")
print("="*80)

# Placeholder: Loop over baseline models
for baseline_model in naive_target_models:
    print(f"\n>>> Processing baseline model: {baseline_model}")
    print(f"    Output folder: {model_output_folders.get(baseline_model, 'Not set')}")

    if baseline_model in naive_forecasts_dict:
        run_forecast_evaluation_pipeline(
            judgemental_forecasts_dict=ifo_judgemental_forecasts,
            naive_forecasts_dict=naive_forecasts_dict,
            ifoCast_forecasts_dict=None,
            realized_df=qoq_first_eval,
            baseline_model_name=baseline_model,
            output_folders={
                'table_eda': os.path.join(model_output_folders[baseline_model], '0_EDA_Tables'),
                'graph_eda': os.path.join(model_output_folders[baseline_model], '0_EDA_Graphs'),
                'graph_derivations': os.path.join(model_output_folders[baseline_model], '1_Derivations_Graphs'),
                'graph_errors': os.path.join(model_output_folders[baseline_model], '2_Errors_Graphs'),
                'graph_net_improvement': os.path.join(model_output_folders[baseline_model], '3_Net_Improvement_Graphs'),
                'table_main': os.path.join(model_output_folders[baseline_model], '4_Main_Analysis_Tables'),
                'graph_main': os.path.join(model_output_folders[baseline_model], '5_Main_Analysis_Graphs'),
            },
            max_horizon=MAX_HORIZON,
            evaluate_ifoCast=EVALUATE_IFOCAST,
        )
    else:
        print(f"    WARNING: {baseline_model} not available in naive forecasts. Skipping.")


# Run analysis on Components (if enabled)
if evaluate_forecast_components and included_components:
    print("\n" + "="*80)
    print("COMPONENT ANALYSIS")
    print("="*80)
    
    for component_name in included_components:
        print(f"\n--- Processing Component: {component_name} ---\n")
        
        # Check if we have judgemental forecasts for this component
        if component_name not in ifo_component_forecasts:
            print(f"    WARNING: No judgemental forecasts for component {component_name}, skipping.")
            continue
        
        jdg_comp_forecasts = ifo_component_forecasts[component_name]
        component_naive_dict = naive_component_forecasts.get(component_name, {})
        
        # For each baseline model, run component evaluation
        for baseline_model in naive_target_models:
            # Use pattern matching to find the model (e.g., 'AR2' matches 'AR2_FULL_9')
            pattern = re.compile(rf'^{re.escape(baseline_model)}(_|$)')
            model_matches = [k for k in component_naive_dict if pattern.match(k)]
            
            if not model_matches:
                print(f"    Skipping {baseline_model} for component {component_name} - no data")
                continue
            
            # Use the first matching model key
            actual_model_key = model_matches[0]
            print(f"  >>> Processing {baseline_model} for component {component_name} (using {actual_model_key})")
            
            # Get component-specific folders for this baseline model
            comp_folders = setup_component_folder(component_name, baseline_model)
            
            # Run evaluation pipeline for this component and baseline model
            try:
                run_forecast_evaluation_pipeline(
                    judgemental_forecasts_dict=jdg_comp_forecasts,
                    naive_forecasts_dict={baseline_model: component_naive_dict[actual_model_key]},
                    ifoCast_forecasts_dict=None,
                    realized_df=qoq_first_eval,
                    baseline_model_name=baseline_model,
                    output_folders={
                        'table_eda': comp_folders['table_eda'],
                        'graph_eda': comp_folders['graph_eda'],
                        'graph_derivations': comp_folders['graph_derivations'],
                        'graph_errors': comp_folders['graph_errors'],
                        'graph_net_improvement': comp_folders['graph_net_improvement'],
                        'table_main': comp_folders['table_main'],
                        'graph_main': comp_folders['graph_main'],
                    },
                    max_horizon=MAX_HORIZON,
                    evaluate_ifoCast=EVALUATE_IFOCAST,
                )
            except Exception as e:
                print(f"    Error processing component {component_name} with baseline {baseline_model}: {e}")


print("\n" + "="*80)
print("FORECAST EVALUATION COMPLETE")
print("="*80)


# --------------------------------------------------------------------------------------------------
print(f" \n ifo Judgemental Forecasting Analysis Module (Horizons) Complete! \n")
print(f"Output structured in: {base_output_folder}")
print(f"  Main GDP Analysis in: {main_folder}")
print(f"    - AR2")
print(f"    - AVERAGE_1")
print(f"    - AVERAGE_10")
print(f"    - AVERAGE_FULL")
if evaluate_forecast_components and included_components:
    print(f"  Component Analysis in: {components_folder}")
    for comp in included_components:
        print(f"    - {comp}")
# --------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#
