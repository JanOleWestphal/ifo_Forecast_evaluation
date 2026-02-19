
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

from Functionalities.helpers.judgemental_evalfunctions import (
    add_error_columns,
    _classify_derivations,
    _infer_baseline_spec,
    _generate_summary_statistics,
    visualize_summary_statistics,
    _format_quarterly_index,
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



## Print Module header
print("\nExecuting the Judgemental Derivations Analysis Module ... \n")



# ==================================================================================================
# SETUP OUTPUT FOLDER STRUCTURE
# ==================================================================================================

## Base Result Folder Paths
result_folder = os.path.join(wd, '5_Judgemental_Nowcasts_Derivations_Analysis')
components_folder = os.path.join(result_folder, '1_Components')

## Main Analysis Folders (Full GDP)
main_analysis_folder = os.path.join(result_folder, '0_Main_Analysis')
main_eda_tables_folder = os.path.join(main_analysis_folder, 'EDA_Tables')
main_eda_graphs_folder = os.path.join(main_analysis_folder, 'EDA_Graphs')
main_eda_derivations_folder = os.path.join(main_eda_graphs_folder, 'Derivations')
main_eda_errors_folder = os.path.join(main_eda_graphs_folder, 'Errors')
main_eda_net_improvement_folder = os.path.join(main_eda_graphs_folder, 'NI')
main_analysis_graphs_folder = os.path.join(main_analysis_folder, 'Graphs')
main_analysis_tables_folder = os.path.join(main_analysis_folder, 'Tables')

## Create all required folders for main analysis
for folder in [result_folder, main_analysis_folder, main_eda_tables_folder, main_eda_graphs_folder,
               main_eda_derivations_folder, main_eda_errors_folder, main_eda_net_improvement_folder,
               main_analysis_graphs_folder, main_analysis_tables_folder, components_folder]:
    os.makedirs(folder, exist_ok=True)

# Define a function to create component folders
def setup_component_folder(component_name):
    """Create folder structure for a specific component"""
    comp_folder = os.path.join(components_folder, component_name)
    comp_eda_tables_folder = os.path.join(comp_folder, 'EDA_Tables')
    comp_eda_graphs_folder = os.path.join(comp_folder, 'EDA_Graphs')
    comp_eda_derivations_folder = os.path.join(comp_eda_graphs_folder, 'Derivations')
    comp_eda_errors_folder = os.path.join(comp_eda_graphs_folder, 'Errors')
    comp_eda_net_improvement_folder = os.path.join(comp_eda_graphs_folder, 'NI')
    comp_analysis_graphs_folder = os.path.join(comp_folder, 'Graphs')
    comp_analysis_tables_folder = os.path.join(comp_folder, 'Tables')
    
    for folder in [comp_folder, comp_eda_tables_folder, comp_eda_graphs_folder,
                   comp_eda_derivations_folder, comp_eda_errors_folder, comp_eda_net_improvement_folder,
                   comp_analysis_graphs_folder, comp_analysis_tables_folder]:
        os.makedirs(folder, exist_ok=True)
    
    return {
        'folder': comp_folder,
        'eda_tables': comp_eda_tables_folder,
        'eda_graphs': comp_eda_graphs_folder,
        'eda_derivations': comp_eda_derivations_folder,
        'eda_errors': comp_eda_errors_folder,
        'eda_net_improvement': comp_eda_net_improvement_folder,
        'analysis_graphs': comp_analysis_graphs_folder,
        'analysis_tables': comp_analysis_tables_folder,
    }



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
eval_path = os.path.join(wd, '0_0_Data', '2_Processed_Data', '2_evaluation_series')
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
# Load ifo component forecasts (optional)
# -------------------------------------------------------------------------------------------------#

ifo_qoq_forecasts_components = {}
ifo_component_nowcasts = {}

if evaluate_forecast_components:
    file_path_ifo_qoq_components = os.path.join(
        wd, '0_0_Data', '2_Processed_Data', '3_gdp_component_forecast'
    )

    ifo_qoq_forecasts_components = load_ifo_component_forecasts(
        file_path_ifo_qoq_components,
        included_components=included_components,
    )

    for comp_name, comp_df in ifo_qoq_forecasts_components.items():
        ifo_component_nowcasts[comp_name] = nowcast_builder(comp_df, colname="judgemental")
        print(f"Loaded ifo component nowcasts: {comp_name}")


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
# Load naive component forecasts (optional)
# -------------------------------------------------------------------------------------------------#

component_naive_qoq_dfs_dict = {}
naive_component_nowcasts = {}

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
        naive_component_nowcasts[comp_name] = {}
        for model_name, df_model in model_dict.items():
            col_name_naive = f"naive{model_name}"
            naive_component_nowcasts[comp_name][model_name] = nowcast_builder(
                df_model,
                colname=col_name_naive,
            )
        print(f"Loaded naive component nowcasts: {comp_name}")













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

# Call error function
joint_nowcast_df = add_error_columns(joint_nowcast_df)
#show(joint_nowcast_df)



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




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                    NOTE: All data processing, analysis, and visualization                        #
#                         is now handled by run_complete_nowcasting_analysis()                     #
#                         See function definition below                                             #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# =================================================================================================#
#                                Classify Judgemental Derications                                  #
# =================================================================================================#

"""
r<b: True (negative shock), False; 'r_less_b'
j<b: True (negatve adjustment), False; 'j_less_b'
|j-r|<|b-r| True (judgemental improvement), False; 'jdiff_less_bdiff'
"""

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

base_cols = [
    "realized",
    "judgemental",
    "error_realized_minus_judgemental",
]

# ---- ifoCast subset ----
ifo_cols = [c for c in judgment_eval_df_joint.columns if "ifoCast" in c]
judgment_eval_df_ifoCast = judgment_eval_df_joint[base_cols + ifo_cols].copy()

# Note: All table saves are now handled by run_complete_nowcasting_analysis() function





















# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                            BASELINE ANALYSIS: Judgemental Derivations                            #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# =================================================================================================#
#                                    Generate Summary Statistics                                   #
# =================================================================================================#

# NOTE: Summary statistics generation is now handled in run_complete_nowcasting_analysis()


# =================================================================================================#
#                                       Visualize Results                                          #
# =================================================================================================#

# NOTE: All visualizations are now handled in run_complete_nowcasting_analysis()


# =================================================================================================#
#                               COMPREHENSIVE ANALYSIS FUNCTION



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
# =================================================================================================#
#                               COMPREHENSIVE ANALYSIS FUNCTION
# =================================================================================================#

def run_complete_nowcasting_analysis(
    joint_df,
    naive_nowcasts_dict_in,
    ifo_component_nowcasts_in,
    naive_component_nowcasts_in,
    eda_tables_folder,
    eda_graphs_folder,
    eda_derivations_folder,
    eda_errors_folder,
    eda_net_improvement_folder,
    analysis_graphs_folder,
    analysis_tables_folder,
    analysis_label="Full GDP"
):
    """
    Run the complete nowcasting analysis pipeline on a dataset (full GDP or component).
    
    Parameters:
    -----------
    joint_df : pd.DataFrame
        Merged nowcast dataframe with all forecasts
    naive_nowcasts_dict_in : dict
        Dictionary of naive forecast models
    ifo_component_nowcasts_in : dict
        Dictionary of ifo component nowcasts (if applicable)
    naive_component_nowcasts_in : dict
        Dictionary of naive component nowcasts (if applicable)
    eda_tables_folder : str
        Path to save EDA tables
    eda_graphs_folder : str
        Path to save EDA graphs
    eda_derivations_folder : str
        Path to save derivation plots
    eda_errors_folder : str
        Path to save error comparison plots
    eda_net_improvement_folder : str
        Path to save net improvement plots
    analysis_graphs_folder : str
        Path to save main analysis graphs
    analysis_tables_folder : str
        Path to save main analysis tables
    analysis_label : str
        Label for analysis (e.g., "Full GDP", "Component: PRIVCON")
    """
    
    print(f"\n--- Running analysis for {analysis_label} ---\n")
    
    # =================================================================================================#
    #                                    Create error measures                                         #
    # =================================================================================================#
    
    df_eval = joint_df.copy()
    df_eval = add_error_columns(df_eval)
    
    # Save nowcast series
    df_eval.to_excel(os.path.join(eda_tables_folder, f"Nowcast_Series_{analysis_label.replace(' ', '_')}.xlsx"))
    
    # =================================================================================================#
    #                                   Create derivation measures                                     #
    # =================================================================================================#
    
    # from ifoCast
    df_eval["derivation_from_ifoCast"] = df_eval["judgemental"] - df_eval["ifoCast"]
    
    # from Naive Models (AR2, Average, etc.)
    for model_name in naive_nowcasts_dict_in.keys():
        col_name = f"naive{model_name}"
        if col_name in df_eval.columns:
            df_eval[f"derivation_from_{model_name}"] = df_eval["judgemental"] - df_eval[col_name]
    
    # =================================================================================================#
    #                                    Obtain net improvements                                       #
    # =================================================================================================#
    
    # Judgement vs ifoCast
    df_eval["net_improvement_jdg_ifoCast_lin"] = (
        df_eval["error_realized_minus_ifoCast"].abs() - df_eval["error_realized_minus_judgemental"].abs()
    )
    df_eval["net_improvement_jdg_ifoCast_quad"] = (
        df_eval["error_realized_minus_ifoCast"]**2 - df_eval["error_realized_minus_judgemental"]**2
    )
    
    # Judgement vs Naive Models
    for model_name in naive_nowcasts_dict_in.keys():
        col_name = f"naive{model_name}"
        if col_name in df_eval.columns:
            df_eval[f"net_improvement_jdg_{model_name}_lin"] = (
                df_eval[f"error_realized_minus_{col_name}"].abs()
                - df_eval["error_realized_minus_judgemental"].abs()
            )
            df_eval[f"net_improvement_jdg_{model_name}_quad"] = (
                df_eval[f"error_realized_minus_{col_name}"]**2
                - df_eval["error_realized_minus_judgemental"]**2
            )
    
    # =================================================================================================#
    #                                Classify Judgemental Derivations                                  #
    # =================================================================================================#
    
    _classify_derivations(df_eval, b_col="ifoCast", suffix="ifoCast")
    
    for model_name in naive_nowcasts_dict_in.keys():
        col_name = f"naive{model_name}"
        if col_name in df_eval.columns:
            _classify_derivations(df_eval, b_col=col_name, suffix=model_name)
    
    # =================================================================================================#
    #                                  Split dfs and save results                                      #
    # =================================================================================================#
    
    judgment_eval_df_joint = df_eval.copy()
    
    base_cols = ["realized", "judgemental", "error_realized_minus_judgemental"]
    
    # ifoCast subset
    ifo_cols = [c for c in judgment_eval_df_joint.columns if "ifoCast" in c]
    judgment_eval_df_ifoCast = judgment_eval_df_joint[base_cols + ifo_cols].copy()
    judgment_eval_df_joint.to_excel(os.path.join(eda_tables_folder, "judgemental_derivations_full.xlsx"))
    judgment_eval_df_ifoCast.to_excel(os.path.join(eda_tables_folder, "judgemental_derivations_ifoCast.xlsx"))
    
    # Naive Models subsets
    judgment_eval_dfs = {}
    for model_name in naive_nowcasts_dict_in.keys():
        model_specific_cols = []
        for c in judgment_eval_df_joint.columns:
            if model_name in c:
                if model_name == "AVERAGE_1" and "AVERAGE_10" in c:
                    continue
                model_specific_cols.append(c)
        
        df_subset = judgment_eval_df_joint[base_cols + model_specific_cols].copy()
        judgment_eval_dfs[model_name] = df_subset
        df_subset.to_excel(os.path.join(eda_tables_folder, f"judgemental_derivations_{model_name}.xlsx"))
    
    # =================================================================================================#
    #                                    Generate Summary Statistics                                   #
    # =================================================================================================#
    
    summary_stats_list = []
    
    summary_stats_list.append(_generate_summary_statistics(judgment_eval_df_ifoCast, "ifoCast", shock_filter=None))
    summary_stats_list.append(_generate_summary_statistics(judgment_eval_df_ifoCast, "ifoCast", shock_filter=True))
    summary_stats_list.append(_generate_summary_statistics(judgment_eval_df_ifoCast, "ifoCast", shock_filter=False))
    
    for model_name, df_sub in judgment_eval_dfs.items():
        summary_stats_list.append(_generate_summary_statistics(df_sub, model_name, shock_filter=None))
        summary_stats_list.append(_generate_summary_statistics(df_sub, model_name, shock_filter=True))
        summary_stats_list.append(_generate_summary_statistics(df_sub, model_name, shock_filter=False))
    
    summary_stats_df = pd.DataFrame(summary_stats_list)
    
    visualize_summary_statistics(summary_stats_df, eda_graphs_folder)
    summary_stats_output_path = os.path.join(eda_tables_folder, "Summary_Statistics.xlsx")
    summary_stats_df.to_excel(summary_stats_output_path, index=False, sheet_name="Summary Statistics")
    
    # =================================================================================================#
    #                                    Visualize Results                                            #
    # =================================================================================================#
    
    all_models_dfs = {'ifoCast': judgment_eval_df_ifoCast}
    if naive_nowcasts_dict_in:
        all_models_dfs.update(judgment_eval_dfs)
    
    for model_name, df_eval_mod in all_models_dfs.items():
        if df_eval_mod is None or df_eval_mod.empty:
            continue
        
        # Judgemental derivations
        try:
            plot_judgemental_derivations_or_net_improvement(
                df=df_eval_mod,
                kind="derivation",
                graph_folder_EDA=eda_derivations_folder,
                header=f"Judgemental derivations vs {model_name}",
                filename_prefix=f"{analysis_label.replace(' ', '_')}_judgemental_derivations",
                show=False,
            )
        except Exception as e:
            print(f"  Error plotting derivations for {model_name}: {e}")
        
        # Net improvement (truncated)
        try:
            plot_judgemental_derivations_or_net_improvement(
                df=df_eval_mod,
                kind="net_improvement",
                graph_folder_EDA=eda_net_improvement_folder,
                header=None,
                filename_prefix=f"{analysis_label.replace(' ', '_')}_net_improvement",
                filename_suffix="t95p",
                show=False,
                y_axis_percentile=95.0,
            )
        except Exception as e:
            print(f"  Error plotting net improvement (truncated) for {model_name}: {e}")
        
        # Net improvement (full)
        try:
            plot_judgemental_derivations_or_net_improvement(
                df=df_eval_mod,
                kind="net_improvement",
                graph_folder_EDA=eda_net_improvement_folder,
                header=None,
                filename_prefix=f"{analysis_label.replace(' ', '_')}_net_improvement",
                filename_suffix="full",
                show=False,
                y_axis_percentile=None,
            )
        except Exception as e:
            print(f"  Error plotting net improvement (full) for {model_name}: {e}")
        
        # Error comparisons
        if model_name == "ifoCast":
            error_col_benchmark = "error_realized_minus_ifoCast"
        else:
            error_col_benchmark = f"error_realized_minus_naive{model_name}"
        
        try:
            plot_error_comparison(
                df=df_eval_mod,
                error_col_jdg="error_realized_minus_judgemental",
                error_col_benchmark=error_col_benchmark,
                benchmark_label=model_name,
                graph_folder=eda_errors_folder,
                filename_prefix=f"{analysis_label.replace(' ', '_')}_error_comparison",
                filename_suffix="t95p",
                show=False,
                y_axis_percentile=95.0,
            )
        except Exception as e:
            print(f"  Error plotting error comparison (truncated) for {model_name}: {e}")
        
        try:
            plot_error_comparison(
                df=df_eval_mod,
                error_col_jdg="error_realized_minus_judgemental",
                error_col_benchmark=error_col_benchmark,
                benchmark_label=model_name,
                graph_folder=eda_errors_folder,
                filename_prefix=f"{analysis_label.replace(' ', '_')}_error_comparison",
                filename_suffix="full",
                show=False,
                y_axis_percentile=None,
            )
        except Exception as e:
            print(f"  Error plotting error comparison (full) for {model_name}: {e}")
    
    # =================================================================================================#
    #                                  Signals Analysis & OLS                                         #
    # =================================================================================================#
    
    all_ols_results = []
    
    # For components, model names may have suffixes like AVERAGE_1_9
    # Extract the base model names from naive_nowcasts_dict_in
    base_model_names = set()
    for key in naive_nowcasts_dict_in.keys():
        # Extract base name: "AVERAGE_1_9" -> "AVERAGE_1", "AR2" -> "AR2"
        base_name = key.split('_')[0] + ('_' + key.split('_')[1] if len(key.split('_')) > 1 and key.split('_')[1].isdigit() else '')
        base_model_names.add(base_name)
    
    potential_models = [
        ("AR2", "naiveAR2"),
        ("AVERAGE_1", "naiveAVERAGE_1"),
        ("AVERAGE_10", "naiveAVERAGE_10"),
        ("AVERAGE_FULL", "naiveAVERAGE_FULL")
    ]
    
    # Filter to only process models that are actually present in this analysis
    for name, col_base in potential_models:
        if name not in base_model_names:
            continue
        
        # Look for exact match first, then try suffixed versions
        matching_col = None
        if col_base in df_eval.columns:
            matching_col = col_base
        else:
            # Try to find a column like "naiveAVERAGE_1_<suffix>"
            pattern = re.compile(rf"^{re.escape(col_base)}_\d+$")
            for av_col in [c for c in df_eval.columns if c.startswith('naive')]:
                if pattern.match(av_col):
                    matching_col = av_col
                    break
        
        if matching_col:
            # Create transformed dataframe for this baseline
            df_model = df_eval[['realized', matching_col, 'ifoCast', 'judgemental']].copy()
            df_model = transform_eval_dataframe(df_model, matching_col)
            
            try:
                res = run_signals_analysis(df_model, f"{analysis_label}_{name}", matching_col, analysis_graphs_folder, analysis_tables_folder)
                all_ols_results.extend(res)
            except Exception as e:
                print(f"Error running signal analysis for {name}: {e}")
    
    if all_ols_results:
        ols_df = pd.DataFrame(all_ols_results)
        ols_output_path = os.path.join(analysis_tables_folder, f"Signals_OLS_Results_{analysis_label.replace(' ', '_')}.xlsx")
        ols_df.to_excel(ols_output_path, index=False)






# =================================================================================================#
#                                       Execute Analysis                                           #
# =================================================================================================#

# Run analysis on Full GDP
print("\n" + "="*100)
print("FULL GDP ANALYSIS")
print("="*100)

run_complete_nowcasting_analysis(
    joint_df=joint_nowcast_df,
    naive_nowcasts_dict_in=naive_nowcasts_dict,
    ifo_component_nowcasts_in=ifo_component_nowcasts,
    naive_component_nowcasts_in=naive_component_nowcasts,
    eda_tables_folder=main_eda_tables_folder,
    eda_graphs_folder=main_eda_graphs_folder,
    eda_derivations_folder=main_eda_derivations_folder,
    eda_errors_folder=main_eda_errors_folder,
    eda_net_improvement_folder=main_eda_net_improvement_folder,
    analysis_graphs_folder=main_analysis_graphs_folder,
    analysis_tables_folder=main_analysis_tables_folder,
    analysis_label="Full_GDP"
)

# Run analysis on Components (if enabled)
if evaluate_forecast_components and included_components:
    print("\n" + "="*100)
    print("COMPONENT ANALYSIS")
    print("="*100)
    
    for component_name in included_components:
        print(f"\n--- Processing Component: {component_name} ---\n")
        
        # Get component-specific folders
        comp_folders = setup_component_folder(component_name)
        
        # Prepare component data
        dfs_to_merge_comp = [qoq_first_eval]
        col_names_merge_comp = ['realized']
        
        # Add component ifo forecasts
        if component_name in ifo_component_nowcasts:
            dfs_to_merge_comp.append(ifo_component_nowcasts[component_name])
            col_names_merge_comp.append('judgemental')
        
        # Add ifoCAST (assumed same for all components or not applicable)
        dfs_to_merge_comp.append(ifoCAST_nowcast)
        col_names_merge_comp.append('ifoCast')
        
        # Add component naive forecasts
        component_naive_dict = {}
        if component_name in naive_component_nowcasts:
            for model_name, df_naive in naive_component_nowcasts[component_name].items():
                component_naive_dict[model_name] = df_naive
                dfs_to_merge_comp.append(df_naive)
                # df_naive already has column name like 'naiveAR2' from nowcast_builder
                col_names_merge_comp.append(f"naive{model_name}")
        
        # Merge component data
        if len(dfs_to_merge_comp) > 1:
            joint_component_df = merge_quarterly_dfs_dropna(
                dfs=dfs_to_merge_comp,
                col_names=col_names_merge_comp
            )
            joint_component_df = align_df_to_mid_quarters(joint_component_df)
            joint_component_df = filter_df_by_datetime_index(joint_component_df, '2000-01-01', '2100-01-01')
            joint_component_df = align_df_to_mid_quarters(joint_component_df)
            
            # Run analysis
            run_complete_nowcasting_analysis(
                joint_df=joint_component_df,
                naive_nowcasts_dict_in=component_naive_dict,
                ifo_component_nowcasts_in={},  # Already included
                naive_component_nowcasts_in={},  # Already included
                eda_tables_folder=comp_folders['eda_tables'],
                eda_graphs_folder=comp_folders['eda_graphs'],
                eda_derivations_folder=comp_folders['eda_derivations'],
                eda_errors_folder=comp_folders['eda_errors'],
                eda_net_improvement_folder=comp_folders['eda_net_improvement'],
                analysis_graphs_folder=comp_folders['analysis_graphs'],
                analysis_tables_folder=comp_folders['analysis_tables'],
                analysis_label=f"Component_{component_name}"
            )
        else:
            print(f"  Warning: Insufficient data for component {component_name}, skipping.")














# --------------------------------------------------------------------------------------------------
print(f" \n ifo Judgemental Nowcasting Analysis Module complete! \n")
print(f"Find Full GDP Analysis:")
print(f"  - EDA Graphs in {main_eda_graphs_folder}")
print(f"  - EDA Tables in {main_eda_tables_folder}")
print(f"  - Main Analysis in {main_analysis_folder}\n")
if evaluate_forecast_components and included_components:
    print(f"Find Components Analysis in {components_folder}\n")
# --------------------------------------------------------------------------------------------------





# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#
