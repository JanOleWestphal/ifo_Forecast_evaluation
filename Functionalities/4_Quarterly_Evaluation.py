
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Quarterly Evaluation
#
# Author:       Jan Ole Westphal
# Date:         2026-02
#
# Description:  Subprogram to evaluate quarterly forecasts of both ifo and the Naive Forecaster,
#               possibly other providers as well          
# ==================================================================================================
# --------------------------------------------------------------------------------------------------


"""
Remaining Bugs/Issues:
- Dynamic naming of graphs should be improved
"""




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        Code begins here                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

print("\n Executing the Quarterly Evaluation Module ... \n")


# ==================================================================================================
#                                           SETUP
# ==================================================================================================

# Import built-ins
import importlib
import subprocess
import sys
import os
import glob
import re
from datetime import datetime, date
from itertools import product
from typing import Union, Dict, Optional


# Import libraries
import requests
import pandas as pd
from pandasgui import show  #uncomment this to allow for easier debugging
import numpy as np
from statsmodels.tsa.ar_model import AutoReg


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

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

# Select Format of Output Graphs
QoQ_eval_n_bars = settings.QoQ_eval_n_bars


# Select timeframes
horizon_limit_year = settings.horizon_limit_year
horizon_limit_quarter = settings.horizon_limit_quarter    

# Define the horizon of first releases which should be evaluated:
first_release_lower_limit_year = settings.first_release_lower_limit_year
first_release_lower_limit_quarter = settings.first_release_lower_limit_quarter

first_release_upper_limit_year = settings.first_release_upper_limit_year
first_release_upper_limit_quarter = settings.first_release_upper_limit_quarter


## Check whether component evaluation has been defined correctly in the settings file:
included_components = settings.included_components

# Try-except component name matching
try:
    allowed = {'GDP','PRIVCON','PUBCON','CONSTR', 'EQUIPMENT','OPA','INVINV','DOMUSE','TRDBAL','EXPORT','IMPORT'}

    invalid = set(settings.included_components) - allowed
    if invalid:
        raise ValueError

except Exception:
    print("WARNING: spelling error in settings.included_components, re-evalaute your input")







# ==================================================================================================
# SETUP OUTOUT FOLDER STRUCTURE
# ==================================================================================================

## Result Folder Paths
table_folder = os.path.join(wd, '1_Result_Tables')
graph_folder = os.path.join(wd, '2_Result_Graphs')

component_result_folder = os.path.join(wd, '3_Component_Results')



## Create if needed
for folder in [table_folder, graph_folder, component_result_folder ]:
    os.makedirs(folder, exist_ok=True)


## Clear Result Folders
for component_name in included_components:

    # Loop through the component folders which are called in this instance, leaves previously called subfolders intact
    comp_folder = os.path.join(component_result_folder, component_name)
    os.makedirs(comp_folder, exist_ok=True)

    if settings.clear_result_folders:
        folder_clear(comp_folder)












# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                 Load in data to be evaluated                                     #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#



# ==================================================================================================
#                                         ifo FORECASTS
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Load ifo forecasts - GDP
# --------------------------------------------------------------------------------------------------

# Path
file_path_ifo_qoq = os.path.join(wd, '0_0_Data', '2_Processed_Data', '3_ifo_qoq_series',
                                  'ifo_qoq_forecasts.xlsx' )

# Load 
ifo_qoq_forecast_df = pd.read_excel(file_path_ifo_qoq, index_col=0)


# --------------------------------------------------------------------------------------------------
# Load ifo forecasts - components
# --------------------------------------------------------------------------------------------------

if settings.evaluate_forecast_components:
    # Path
    file_path_ifo_qoq_components = os.path.join(wd, '0_0_Data', '2_Processed_Data', '3_gdp_component_forecast')

    # Load - manually parse component names from filenames
    ifo_qoq_forecasts_components = {}

    if os.path.exists(file_path_ifo_qoq_components):
        component_files = glob.glob(os.path.join(file_path_ifo_qoq_components, 'qoq_forecast_data*.xlsx'))
        for file in component_files:
            filename = os.path.basename(file)
            # Extract component name from 'qoq_forecast_data_COMPONENT.xlsx'
            comp_name = filename.replace('qoq_forecast_data_', '').replace('.xlsx', '')
            
            if comp_name in included_components:
                ifo_qoq_forecasts_components[comp_name] = pd.read_excel(file, index_col=0)
                print(f"  Loaded ifo forecasts for component: {comp_name}")

            else: 
                print(f"  Skipped ifo forecasts for component: {comp_name}; change settings.included_components to include this component in the evaluation.")

else:
    print("\nComponent forecast evaluation is turned off; to turn on, change settings.evaluate_forecast_components to True\n")






# ==================================================================================================
#                                        NAIVE FORECASTS
# ==================================================================================================

"""
def load_excels_to_dict(folder_path, strip_string=None):
    excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
    dfs = {}
    for file in excel_files:
        name = os.path.splitext(os.path.basename(file))[0]
        
        # Strip the custom string if provided
        if strip_string:
            name = name.replace(strip_string, '')
        
        dfs[name] = pd.read_excel(file, index_col=0)
    return dfs
"""

# --------------------------------------------------------------------------------------------------
# Naive GDP forecast 
# --------------------------------------------------------------------------------------------------

# Paths to the folders containing the Excel files
file_path_naive_qoq = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '1_QoQ_Forecast_Tables')

# Load all QoQ naive forecast Excel files into dictionary
naive_qoq_dfs_dict = load_excels_to_dict(file_path_naive_qoq, strip_string='naive_qoq_forecasts_')



# --------------------------------------------------------------------------------------------------
# Naive component forecasts
# --------------------------------------------------------------------------------------------------

file_path_component_qoq = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '3_QoQ_Component_Forecast_Tables')

## Load component naive forecasts with structure: naive_qoq_forecasts_COMPONENT_MODEL.xlsx
def load_component_naive_forecasts(
    file_path_component_qoq: str, *, 
    included_components=None,                          # default → global fallback
    drop_ar2_components: list[str] | None = None,
    pattern: str = "naive_qoq_forecasts_*.xlsx"
) -> dict:

    # Use global included_components if not explicitly passed
    if included_components is None:
        included_components = globals().get("included_components", None)

    # Create filter object
    drop_ar2_components = set(drop_ar2_components or [])

    # Convert included_components to set for faster lookup (if provided)
    included_components = set(included_components) if included_components is not None else None

    # Define output
    out: dict = {}


    if os.path.exists(file_path_component_qoq):
        component_files = glob.glob(os.path.join(file_path_component_qoq, pattern))

        # Loop over all files in folder
        for file in component_files:
            filename = os.path.basename(file)

            # Extract component and model from 'naive_qoq_forecasts_COMPONENT_MODEL.xlsx'
            # e.g., 'naive_qoq_forecasts_CONSTR_AR2_50_9.xlsx' -> component='CONSTR', model='AR2_50_9'
            # split only once because model names may contain underscores
            parts = filename.replace('naive_qoq_forecasts_', '').replace('.xlsx', '').split('_', 1)
            if len(parts) != 2:
                continue

            # Define dynamic naming objects
            component, model = parts[0], parts[1]

            # Apply filter: skip if component is in drop_ar2_components and model starts with "AR2"
            if component in drop_ar2_components and model.startswith("AR2"):
                print(f"  Skipped naive component forecast (drop AR2): {component} - {model}")
                continue

            # Apply include-filter if provided
            if included_components is not None and component not in included_components:
                print(f"  Skipped naive component forecast: {component}.")
                continue
            
            # Ensure component container exists
            if component not in out:
                out[component] = {}

            # Always read and assign the model
            df = pd.read_excel(file, index_col=0)
            out[component][model] = df

            print(f"  Loaded naive component forecast: {component} - {model}")

    return out


## Load component naive forecasts, drop exploding time series:
if settings.evaluate_forecast_components:
    component_naive_qoq_dfs_dict = load_component_naive_forecasts(file_path_component_qoq,
        drop_ar2_components=["PRIVCON"])




# ==================================================================================================
#                                        EVALUATION DATA
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Load Evaluation Data - main GDP analysis
# --------------------------------------------------------------------------------------------------

eval_path = os.path.join(wd, '0_0_Data', '2_Processed_Data', '2_GDP_Evaluation_series')

## First Releases
qoq_path_first = os.path.join(eval_path, 'first_release_qoq_GDP.xlsx')
qoq_first_eval = pd.read_excel(qoq_path_first, index_col=0)
qoq_first_eval = align_df_to_mid_quarters(qoq_first_eval)  # Align to mid-quarter dates
#show(qoq_first_eval)


## Latest Releases
qoq_path_latest= os.path.join(eval_path, 'latest_release_qoq_GDP.xlsx')
qoq_latest_eval = pd.read_excel(qoq_path_latest, index_col=0)
qoq_latest_eval = align_df_to_mid_quarters(qoq_latest_eval)  # Align to mid-quarter dates
#show(qoq_latest_eval)


## Revision
qoq_path_rev = os.path.join(eval_path, 'revision_qoq_GDP.xlsx')
qoq_rev = pd.read_excel(qoq_path_rev, index_col=0)
qoq_rev = align_df_to_mid_quarters(qoq_rev)  # Align to mid-quarter dates



# --------------------------------------------------------------------------------------------------
# Load Evaluation Data - component analysis
# --------------------------------------------------------------------------------------------------


if settings.evaluate_forecast_components:

    # Setup
    component_eval_path = os.path.join(wd, '0_0_Data', '2_Processed_Data', '2_component_Evaluation_series')
    component_first_eval_dict = {}

    # Load evaluation data for each component
    if os.path.exists(component_eval_path):
        for comp in included_components:
            eval_file = os.path.join(component_eval_path, f'first_release_qoq_{comp}.xlsx')
            if os.path.exists(eval_file):
                try:
                    component_first_eval_dict[comp] = pd.read_excel(eval_file, index_col=0)
                    print(f"  Loaded evaluation data for component: {comp}")
                except Exception as e:
                    print(f"  ⚠ Could not load evaluation data for {comp}: {e}")










# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                          MATCH FORECAST AVAILABILITY, APPLY FILTERS                              #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# =================================================================================================#
#                                       MAIN GDP ANALYSIS                                          #
# =================================================================================================#

# -------------------------------------------------------------------------------------------------#
#                                         ifo Forecasts
# -------------------------------------------------------------------------------------------------#

## Filter for subset analysis
ifo_qoq_forecast_subset = filter_first_release_limit(ifo_qoq_forecast_df)


# -------------------------------------------------------------------------------------------------#
#                                  Subset Naive to ifo Forecasts
# -------------------------------------------------------------------------------------------------#

## Execute the filter function: drop all naive qoq which do not have a corresponding ifo qoq forecast
naive_qoq_dfs_dict = match_ifo_naive_forecasts_dates(ifo_qoq_forecast_df, naive_qoq_dfs_dict)

## Retain full dataset 
naive_qoq_dfs_dict_subset = naive_qoq_dfs_dict.copy()

## Filter for subset analysis
for key, val in naive_qoq_dfs_dict_subset.items():
    val = filter_first_release_limit(val)
    naive_qoq_dfs_dict_subset[key] = val







# =================================================================================================#
#                                      COMPONENT ANALYSIS                                          #
# =================================================================================================#


## Switch of if not needed:
if settings.evaluate_forecast_components:

    # ---------------------------------------------------------------------------------------------#
    #                                   Component ifo Forecasts
    # ---------------------------------------------------------------------------------------------#

    ## Filter for subset analysis
    ifo_qoq_forecasts_components_subset = {}

    for comp_name in included_components:
        # Apply filtering to each component's ifo qoq forecast data
        ifo_qoq_forecasts_components_subset[comp_name] = filter_first_release_limit(ifo_qoq_forecasts_components[comp_name])


    # ---------------------------------------------------------------------------------------------#
    #                                Subset Naive to ifo Forecasts
    # ---------------------------------------------------------------------------------------------#

    ## Match naive components to available ifo components:
    for comp_name in included_components:
        if comp_name in ifo_qoq_forecasts_components and comp_name in component_naive_qoq_dfs_dict:
            ifo_df = ifo_qoq_forecasts_components[comp_name]
            naive_dict = component_naive_qoq_dfs_dict[comp_name]

            filtered_naive_comp = match_ifo_naive_forecasts_dates(ifo_df, naive_dict)
            
            component_naive_qoq_dfs_dict[comp_name] = filtered_naive_comp
            #[show(val) for key, val in filtered_naive_comp.items()]


    ## Filter for subset analysis
    component_naive_qoq_dfs_dict_subset = {}

    # Loop over all components
    for comp_name in included_components:

        # Loop over the component dictionaries, than apply the filter to each model's dataframe
        if comp_name in component_naive_qoq_dfs_dict:
            component_naive_qoq_dfs_dict_subset[comp_name] = {
                model_name: filter_first_release_limit(model_df)
                for model_name, model_df in component_naive_qoq_dfs_dict[comp_name].items()
            }









# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                       MAIN GDP EVALUATION PIPELINE                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

"""
GENERAL PIPELINE:
- Define Savepaths and clear if needed
- create_qoq_evaluation_df
- collapse_quarterly_prognosis
- get_qoq_error_series
- get_qoq_error_statistics_table
- Plotter functions
"""

def qoq_error_evaluation_pipeline(ifo_qoq_df, naive_qoq_dict, 
                                  ifo_qoq_error_path=None, ifo_qoq_table_path=None, naive_qoq_error_path=None, naive_qoq_table_path=None,
                                  subset_mode=False, sd_filter_mode=False, main_mode=True,
                                  sd_cols=5, sd_threshold=0.05):



    # ==================================================================================================
    #                                          DYNAMIC NAMING
    # ==================================================================================================

    ## Dynamic result naming
    filter_str = "_sd_filtered" if sd_filter_mode else ""
    subset_str = f"_{first_release_lower_limit_year}{first_release_lower_limit_quarter}_{first_release_upper_limit_year}{first_release_upper_limit_quarter}" if subset_mode else ""




    # --------------------------------------------------------------------------------------------------
    # ==================================================================================================
    #                                       GET FORECAST TABLES
    # ==================================================================================================
    # --------------------------------------------------------------------------------------------------

    # ==================================================================================================
    # Savepaths
    # ==================================================================================================

    if main_mode:

        ## ifo
        # Error Series
        ifo_qoq_error_path = os.path.join(wd, '0_1_Output_Data', f'4_ifo_qoq_error_series{subset_str}')
        # Error Tables
        ifo_qoq_table_path = os.path.join(table_folder, f'2_ifo_qoq_evaluations{filter_str}{subset_str}')

        ## Naive
        # Error Series
        naive_qoq_error_path = os.path.join(wd, '0_1_Output_Data', f'4_naive_forecaster_qoq_error_series{subset_str}')
        # Error table
        naive_qoq_table_path = os.path.join(table_folder, f'3_naive_forecaster_qoq_evaluations{filter_str}{subset_str}')


        ## Create if needed
        for folder in [ifo_qoq_error_path, ifo_qoq_table_path, naive_qoq_error_path, naive_qoq_table_path]:
            os.makedirs(folder, exist_ok=True)

        ## Clear
        if settings.clear_result_folders:
            for folder in [ifo_qoq_error_path, ifo_qoq_table_path, naive_qoq_error_path, naive_qoq_table_path]:
                folder_clear(folder)






    # ==================================================================================================
    # ifo QoQ FORECASTS
    # ==================================================================================================

    # --------------------------------------------------------------------------------------------------
    # First Release
    # --------------------------------------------------------------------------------------------------

    ## Get Error Series
    ifo_qoq_forecasts_eval_first = create_qoq_evaluation_df(ifo_qoq_df, qoq_first_eval)
    #show(ifo_qoq_forecasts_eval_first)
    ifo_qoq_forecasts_eval_first_collapsed = collapse_quarterly_prognosis(ifo_qoq_forecasts_eval_first)
    #show(ifo_qoq_forecasts_eval_first_collapsed)

    ifo_qoq_errors_first = get_qoq_error_series(ifo_qoq_forecasts_eval_first_collapsed,
                                        ifo_qoq_error_path, 
                                        file_name=f"ifo_qoq_errors_first_eval{subset_str}.xlsx")
    if sd_filter_mode:
        ifo_qoq_errors_first = drop_outliers(ifo_qoq_errors_first, sd_cols=sd_cols, sd_threshold=sd_threshold)


    ## Get Table
    ifo_qoq_error_table_first = get_qoq_error_statistics_table(ifo_qoq_errors_first,                                                            
                                                            'first_eval', ifo_qoq_table_path, 
                                                            f'ifo_qoq_forecast_error_table_first_eval{filter_str}{subset_str}.xlsx')


    # --------------------------------------------------------------------------------------------------
    # Latest Release
    # --------------------------------------------------------------------------------------------------

    ## Get Error Series
    ifo_qoq_forecasts_eval_latest = create_qoq_evaluation_df(ifo_qoq_df, qoq_latest_eval)
    ifo_qoq_forecasts_eval_latest_collapsed = collapse_quarterly_prognosis(ifo_qoq_forecasts_eval_latest)

    ifo_qoq_errors_latest = get_qoq_error_series(ifo_qoq_forecasts_eval_latest_collapsed,
                                                ifo_qoq_error_path, 
                                                file_name=f"ifo_qoq_errors_latest_eval{subset_str}.xlsx")
    if sd_filter_mode:
        ifo_qoq_errors_latest = drop_outliers(ifo_qoq_errors_latest, sd_cols=sd_cols, sd_threshold=sd_threshold)

    ## Get Table
    ifo_qoq_error_table_latest = get_qoq_error_statistics_table(ifo_qoq_errors_latest,
                                                                'latest_eval', ifo_qoq_table_path, 
                                                            f'ifo_qoq_forecast_error_table_latest_eval{filter_str}{subset_str}.xlsx')






    # ==================================================================================================
    # NAIVE QoQ FORECASTS
    # ==================================================================================================

    # --------------------------------------------------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------------------------------------------------

    # Store Resuls in a dictionary
    naive_qoq_first_eval_dfs = {}
    naive_qoq_first_eval_dfs_collapsed = {} 

    naive_qoq_latest_eval_dfs = {}
    naive_qoq_latest_eval_dfs_collapsed = {} 

    # Loop over all available models
    for name, df in naive_qoq_dict.items():
        
        ## Evaluate against first release
        naive_qoq_first_eval_dfs[name] = create_qoq_evaluation_df(df, qoq_first_eval)
        naive_qoq_first_eval_dfs_collapsed[name] = collapse_quarterly_prognosis(naive_qoq_first_eval_dfs[name])

        ## Evaluate against latest release
        naive_qoq_latest_eval_dfs[name] = create_qoq_evaluation_df(df, qoq_latest_eval)
        naive_qoq_latest_eval_dfs_collapsed[name] = collapse_quarterly_prognosis(naive_qoq_latest_eval_dfs[name])


    # --------------------------------------------------------------------------------------------------
    # Get Error Series
    # --------------------------------------------------------------------------------------------------

    ## First Release

    # Get Result Dictionaries
    naive_qoq_first_eval_error_series_dict = {}
    naive_qoq_first_eval_error_tables_dict = {}

    # Run evaluation loop over all models
    for name, df in naive_qoq_first_eval_dfs_collapsed.items():

        naive_qoq_first_eval_error_series_dict[name] = get_qoq_error_series(df,
                                                            naive_qoq_error_path, 
                                                            file_name=f"{name}_qoq_errors_first_eval{subset_str}.xlsx")

        if sd_filter_mode:
            naive_qoq_first_eval_error_series_dict[name] = drop_outliers(
                naive_qoq_first_eval_error_series_dict[name],
                sd_cols=sd_cols,
                sd_threshold=sd_threshold
            )

        naive_qoq_first_eval_error_tables_dict[name] = get_qoq_error_statistics_table(
                                                            naive_qoq_first_eval_error_series_dict[name],
                                                            'first_eval', naive_qoq_table_path, 
                                                        f'{name}_qoq_forecast_error_table_first_eval{filter_str}{subset_str}.xlsx')
    
    #show(next(iter(naive_qoq_first_eval_error_tables_dict.values())))


    ## Evaluate against latest releases

    # Get Result Dictionaries
    naive_qoq_latest_eval_error_series_dict = {}
    naive_qoq_latest_eval_error_tables_dict = {}

    # Run evaluation loop over all models
    for name, df in naive_qoq_latest_eval_dfs_collapsed.items():
        naive_qoq_latest_eval_error_series_dict[name] = get_qoq_error_series(df,
                                                            naive_qoq_error_path, 
                                                            file_name=f"{name}_qoq_errors_latest_eval{subset_str}.xlsx")

        if sd_filter_mode:
            naive_qoq_latest_eval_error_series_dict[name] = drop_outliers(
                naive_qoq_latest_eval_error_series_dict[name],
                sd_cols=sd_cols,
                sd_threshold=sd_threshold
            )

        naive_qoq_latest_eval_error_tables_dict[name] = get_qoq_error_statistics_table(
                                                            naive_qoq_latest_eval_error_series_dict[name],
                                                        'latest_eval', naive_qoq_table_path, 
                                                        f'{name}_qoq_forecast_error_table_latest_eval{filter_str}{subset_str}.xlsx')













    # -------------------------------------------------------------------------------------------------#
    # =================================================================================================#
    #                                  Visualizing Error Measures                                      #
    # =================================================================================================#
    # -------------------------------------------------------------------------------------------------#

    print("Visualizing error statistics (this takes a significant amount of time) ...  \n")



    # ==================================================================================================
    #                                 PLOT AND SAVE RESULTS OF INTEREST
    # ==================================================================================================


    # --------------------------------------------------------------------------------------------------
    # Error Time Series
    # --------------------------------------------------------------------------------------------------
    print("Plotting Error Time Series ...")

    ## Savepaths

    # ifo
    first_eval_error_series_path_ifo = os.path.join(graph_folder, f'1_QoQ_Error_Series{filter_str}{subset_str}', f'0_First_Evaluation_ifo{filter_str}{subset_str}')
    first_eval_error_series_path = os.path.join(graph_folder, f'1_QoQ_Error_Series{filter_str}{subset_str}', f'0_First_Evaluation_joint{filter_str}{subset_str}')
    latest_eval_error_series_path = os.path.join(graph_folder, f'1_QoQ_Error_Series{filter_str}{subset_str}', f'1_Latest_Evaluation{filter_str}{subset_str}')

    os.makedirs(first_eval_error_series_path_ifo, exist_ok=True)
    os.makedirs(first_eval_error_series_path, exist_ok=True)
    os.makedirs(latest_eval_error_series_path, exist_ok=True)


    ## Clear
    if settings.clear_result_folders:

        for folder in [first_eval_error_series_path_ifo, first_eval_error_series_path, latest_eval_error_series_path]:
            
            folder_clear(folder)



    ## Create and Save Plots
    """
    plot_forecast_timeseries(*args, df_eval=None, title_prefix=None, figsize=(12, 8), 
                                show=False, save_path=None, save_name_prefix=None, select_quarters=None)
    """

    plot_forecast_timeseries(ifo_qoq_forecasts_eval_first, 
                            df_eval=qoq_first_eval, title_prefix=None, figsize=(12, 8), linestyle=None,
                                show=False, 
                                save_path=first_eval_error_series_path_ifo, save_name_prefix=f'ifo_First_Eval{filter_str}{subset_str}_', select_quarters=None)

    plot_forecast_timeseries(ifo_qoq_forecasts_eval_first, naive_qoq_first_eval_dfs, 
                            df_eval=qoq_first_eval, title_prefix=None, figsize=(12, 8), linestyle=None,
                                show=False, 
                                save_path=first_eval_error_series_path, save_name_prefix=f'First_Eval{filter_str}{subset_str}_', select_quarters=None)


    plot_forecast_timeseries(ifo_qoq_forecasts_eval_latest, naive_qoq_latest_eval_dfs, 
                            df_eval=qoq_latest_eval, title_prefix=None, figsize=(12, 8), linestyle=None,
                                show=False, 
                                save_path=latest_eval_error_series_path, save_name_prefix=f'Latest_Eval{filter_str}{subset_str}_', select_quarters=None)





    # --------------------------------------------------------------------------------------------------
    # Error Scatter Plots
    # --------------------------------------------------------------------------------------------------
    print("Plotting Error Scatter Plots ...")

    ## Savepaths
    first_eval_error_line_path = os.path.join(graph_folder, f'1_QoQ_Error_Scatter{filter_str}{subset_str}', f'0_First_Evaluation{filter_str}{subset_str}')
    latest_eval_error_line_path = os.path.join(graph_folder, f'1_QoQ_Error_Scatter{filter_str}{subset_str}', f'1_Latest_Evaluation{filter_str}{subset_str}')

    os.makedirs(first_eval_error_line_path, exist_ok=True)
    os.makedirs(latest_eval_error_line_path, exist_ok=True)


    ## Clear
    if settings.clear_result_folders:

        for folder in [first_eval_error_line_path, latest_eval_error_line_path]:
            
            folder_clear(folder)


    ## Create and Save Plots

    # First Evaluation
    plot_error_lines(ifo_qoq_errors_first, 
                    show=False, 
                    n_bars = QoQ_eval_n_bars,
                    save_path=first_eval_error_line_path,
                    save_name=f'ifo_QoQ_First_Eval{filter_str}{subset_str}_Error_Scatter.png')

    plot_error_lines(ifo_qoq_errors_first, naive_qoq_first_eval_error_series_dict,
                    show=False, 
                    n_bars = QoQ_eval_n_bars,
                    save_path=first_eval_error_line_path,
                    save_name=f'Joint_QoQ_First_Eval{filter_str}{subset_str}_Error_Scatter.png')
                    

    # Latest Evaluation
    plot_error_lines(ifo_qoq_errors_latest, 
                    show=False, 
                    n_bars = QoQ_eval_n_bars,
                    save_path=latest_eval_error_line_path,
                    save_name=f'ifo_QoQ_Latest_Eval{filter_str}{subset_str}_Error_Scatter.png')

    plot_error_lines(ifo_qoq_errors_latest, naive_qoq_latest_eval_error_series_dict,
                    show=False, 
                    n_bars = QoQ_eval_n_bars,
                    save_path=latest_eval_error_line_path,
                    save_name=f'Joint_QoQ_Latest_Eval{filter_str}{subset_str}_Error_Scatter.png')





    # --------------------------------------------------------------------------------------------------
    # Error Bar Plots
    # --------------------------------------------------------------------------------------------------
    print("Plotting Error Bar Plots ...")


    ## Savepaths
    first_eval_error_bars_path = os.path.join(graph_folder, f'1_QoQ_Error_Bars{filter_str}{subset_str}', f'0_First_Evaluation{filter_str}{subset_str}')
    latest_eval_error_bars_path = os.path.join(graph_folder, f'1_QoQ_Error_Bars{filter_str}{subset_str}', f'1_Latest_Evaluation{filter_str}{subset_str}')

    os.makedirs(first_eval_error_bars_path, exist_ok=True)
    os.makedirs(latest_eval_error_bars_path, exist_ok=True)

    
    ## Clear
    if settings.clear_result_folders:

        for folder in [first_eval_error_bars_path, latest_eval_error_bars_path]:
            
            folder_clear(folder)


    ## Create and Save Plots

    # First Evaluation
    for metric in ['ME', 'MAE', 'MSE', 'RMSE', 'SE']:
        plot_quarterly_metrics(ifo_qoq_error_table_first, naive_qoq_first_eval_error_tables_dict, 
                            
                            metric_col=metric,
                            scale_by_n=False, show=False, 
                            n_bars = QoQ_eval_n_bars,

                            save_path=first_eval_error_bars_path,
                            save_name=f'Joint_Quarterly_{metric}_first_eval{filter_str}{subset_str}.png'
                                )


    # Latest Evaluation
    for metric in ['ME', 'MAE', 'MSE', 'RMSE', 'SE']:
        plot_quarterly_metrics(ifo_qoq_error_table_latest, naive_qoq_latest_eval_error_tables_dict, 
                            
                            metric_col=metric,
                            scale_by_n=False, show=False, 
                            n_bars = QoQ_eval_n_bars,

                            save_path=latest_eval_error_bars_path,
                            save_name=f'Joint_Quarterly_{metric}_latest_eval{filter_str}{subset_str}.png'
                                )









# =================================================================================================#
#                                    EXECUTE THE GDP PIPELINE                                      #
# =================================================================================================#


if settings.evaluate_quarterly_gdp_forecasts:

    # Call print command
    print("\n\n" + "="*100)
    print(' '*38 +"MAIN GDP EVALUATION")
    print("="*100 + "\n")



    # --------------------------------------------------------------------------------------------------
    # Full time series
    # --------------------------------------------------------------------------------------------------

    ## Unfiltered Data
    print(" \nAnalysing the full error series ... \n")
    qoq_error_evaluation_pipeline(ifo_qoq_df=ifo_qoq_forecast_df,
                                naive_qoq_dict= naive_qoq_dfs_dict,
                                sd_filter_mode=False)

    ## Filtered Errors: Drop Outliers as resulting from crisis events, e.g. 2009 and Covid Quarters
    if settings.drop_outliers:
        print(" \nDropping Outliers from Error Series before Re-evaluation ... \n")
        qoq_error_evaluation_pipeline(ifo_qoq_df=ifo_qoq_forecast_df,
                                naive_qoq_dict= naive_qoq_dfs_dict,
                                sd_filter_mode=True, sd_cols=5, sd_threshold=settings.sd_threshold)



    # --------------------------------------------------------------------------------------------------
    # Filtered time series (e.g. from 2010-Q1 - 2017-Q1)
    # --------------------------------------------------------------------------------------------------

    ## Unfiltered Data
    print(f" \nAnalysing error series from {first_release_lower_limit_year}-{first_release_lower_limit_quarter} to {first_release_upper_limit_year}-{first_release_upper_limit_quarter}... \n")
    qoq_error_evaluation_pipeline(ifo_qoq_df=ifo_qoq_forecast_subset,
                                naive_qoq_dict= naive_qoq_dfs_dict_subset,
                                sd_filter_mode=False, subset_mode=True)

    ## Filtered Errors: Drop Outliers as resulting from crisis events, e.g. 2009 and Covid Quarters
    if settings.filter_outliers_within_eval_intervall:
        print(" \nDropping Outliers from Error Series before Re-evaluation ... \n")
        qoq_error_evaluation_pipeline(ifo_qoq_df=ifo_qoq_forecast_subset,
                                naive_qoq_dict= naive_qoq_dfs_dict_subset,
                                sd_filter_mode=True, subset_mode=True, sd_cols=5, sd_threshold=settings.sd_threshold)











# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#        COMPONENTS:  Define and call the evaluation pipeline on filtered and unfiltered data      #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# ==================================================================================================
# Component-level evaluation pipeline
# ==================================================================================================

def qoq_error_evaluation_pipeline_components(component_name, 
                                                ifo_qoq_df_components, naive_qoq_dict_components, 
                                                component_eval_dict,
                                                subset_mode=False, sd_filter_mode=False):
    """
    Evaluate component-level forecasts with results stored in component-specific folders
    """
    
    # ==================================================================================================
    # Dynamic Naming
    # ==================================================================================================
    
    filter_str = "_sd_filtered" if sd_filter_mode else ""
    subset_str = f"_{first_release_lower_limit_year}{first_release_lower_limit_quarter}_{first_release_upper_limit_year}{first_release_upper_limit_quarter}" if subset_mode else ""
    
    # ==================================================================================================
    # Component-specific folder paths
    # ==================================================================================================
    
    comp_folder = os.path.join(component_result_folder, component_name)
    comp_table_folder = os.path.join(comp_folder, 'Tables')
    comp_graph_folder = os.path.join(comp_folder, 'Graphs')
    comp_data_folder = os.path.join(comp_folder, 'Data')
    
    # Create folders
    for folder in [comp_table_folder, comp_graph_folder, comp_data_folder]:
        os.makedirs(folder, exist_ok=True)
    

    
    # ==================================================================================================
    # Savepaths
    # ==================================================================================================
    
    # ifo
    ifo_qoq_error_path = os.path.join(comp_data_folder, f'ifo_err{subset_str}')
    ifo_qoq_table_path = os.path.join(comp_table_folder, f'ifo_tbl{filter_str}{subset_str}')
    
    # Naive
    naive_qoq_error_path = os.path.join(comp_data_folder, f'naive_err{subset_str}')
    naive_qoq_table_path = os.path.join(comp_table_folder, f'naive_tbl{filter_str}{subset_str}')
    
    # Create if needed
    for folder in [ifo_qoq_error_path, ifo_qoq_table_path, naive_qoq_error_path, naive_qoq_table_path]:
        os.makedirs(folder, exist_ok=True)
    
    # Clear
    if settings.clear_result_folders:
        for folder in [ifo_qoq_error_path, ifo_qoq_table_path, naive_qoq_error_path, naive_qoq_table_path]:
            folder_clear(folder)




    
    # ==================================================================================================
    # ifo QoQ FORECASTS
    # ==================================================================================================
    
    
    ## Get Error Series
    ifo_qoq_forecasts_eval_first = create_qoq_evaluation_df(ifo_qoq_df_components, component_eval_dict)
    ifo_qoq_forecasts_eval_first_collapsed = collapse_quarterly_prognosis(ifo_qoq_forecasts_eval_first)
    
    ifo_qoq_errors_first = get_qoq_error_series(ifo_qoq_forecasts_eval_first_collapsed,
                                            ifo_qoq_error_path, 
                                            file_name=f"ifo_qoq_errors_first_eval{subset_str}.xlsx")
    if sd_filter_mode:
        ifo_qoq_errors_first = drop_outliers(ifo_qoq_errors_first, sd_cols=5, sd_threshold=settings.sd_threshold)
    
    ## Get Table
    ifo_qoq_error_table_first = get_qoq_error_statistics_table(ifo_qoq_errors_first,                                                            
                                                            'first_eval', ifo_qoq_table_path, 
                                                            f'ifo_qoq_forecast_error_table_first_eval{filter_str}{subset_str}.xlsx')
    



    # ==================================================================================================
    # NAIVE QoQ FORECASTS
    # ==================================================================================================
    
    # --------------------------------------------------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------------------------------------------------
    
    naive_qoq_first_eval_dfs = {}
    naive_qoq_first_eval_dfs_collapsed = {}
    
    # Loop over all available models
    for name, df in naive_qoq_dict_components.items():
        
        ## Evaluate against first release
        naive_qoq_first_eval_dfs[name] = create_qoq_evaluation_df(df, component_eval_dict)
        naive_qoq_first_eval_dfs_collapsed[name] = collapse_quarterly_prognosis(naive_qoq_first_eval_dfs[name])
    
    # --------------------------------------------------------------------------------------------------
    # Get Error Series
    # --------------------------------------------------------------------------------------------------
    
    naive_qoq_first_eval_error_series_dict = {}
    naive_qoq_first_eval_error_tables_dict = {}
    
    # Run evaluation loop over all models
    for name, df in naive_qoq_first_eval_dfs_collapsed.items():
        
        naive_qoq_first_eval_error_series_dict[name] = get_qoq_error_series(df,
                                                            naive_qoq_error_path, 
                                                            file_name=f"{name}_qoq_errors_first_eval{subset_str}.xlsx")
        
        if sd_filter_mode:
            naive_qoq_first_eval_error_series_dict[name] = drop_outliers(
                naive_qoq_first_eval_error_series_dict[name],
                sd_cols=5,
                sd_threshold=settings.sd_threshold
            )
        
        naive_qoq_first_eval_error_tables_dict[name] = get_qoq_error_statistics_table(
                                                            naive_qoq_first_eval_error_series_dict[name],
                                                            'first_eval', naive_qoq_table_path, 
                                                        f'{name}_qoq_forecast_error_table_first_eval{filter_str}{subset_str}.xlsx')
        


    
    # ==================================================================================================
    # VISUALIZATIONS
    # ==================================================================================================
    
    print(f"\n Visualizing component {component_name} error statistics ...")
    
    # --------------------------------------------------------------------------------------------------
    # Error Time Series
    # --------------------------------------------------------------------------------------------------
    
    first_eval_error_series_path_ifo = os.path.join(comp_graph_folder, f'ts_ifo{filter_str}{subset_str}')
    first_eval_error_series_path = os.path.join(comp_graph_folder, f'ts_jnt{filter_str}{subset_str}')
    
    # Ensure all parent directories exist
    os.makedirs(first_eval_error_series_path_ifo, exist_ok=True)
    os.makedirs(first_eval_error_series_path, exist_ok=True)
    
    
    plot_forecast_timeseries(ifo_qoq_forecasts_eval_first, 
                            df_eval=component_eval_dict, title_prefix=None, figsize=(12, 8), linestyle=None,
                                show=False, 
                                save_path=first_eval_error_series_path_ifo, save_name_prefix=f'ifo_ts{filter_str}{subset_str}_', select_quarters=None)
    
    plot_forecast_timeseries(ifo_qoq_forecasts_eval_first, naive_qoq_first_eval_dfs, 
                            df_eval=component_eval_dict, title_prefix=None, figsize=(12, 8), linestyle=None,
                                show=False, 
                                save_path=first_eval_error_series_path, save_name_prefix=f'jnt_ts{filter_str}{subset_str}_', select_quarters=None)
    
    # --------------------------------------------------------------------------------------------------
    # Error Scatter Plots
    # --------------------------------------------------------------------------------------------------
    
    first_eval_error_line_path = os.path.join(comp_graph_folder)
    os.makedirs(first_eval_error_line_path, exist_ok=True)

    
    plot_error_lines(ifo_qoq_errors_first, 
                    show=False, 
                    n_bars=QoQ_eval_n_bars,
                    save_path=first_eval_error_line_path,
                    save_name=f'ifo_sc{filter_str}{subset_str}.png')
    
    plot_error_lines(ifo_qoq_errors_first, naive_qoq_first_eval_error_series_dict,
                    show=False, 
                    n_bars=QoQ_eval_n_bars,
                    save_path=first_eval_error_line_path,
                    save_name=f'jnt_sc{filter_str}{subset_str}.png')
    
    # --------------------------------------------------------------------------------------------------
    # Error Bar Plots
    # --------------------------------------------------------------------------------------------------
    
    first_eval_error_bars_path = os.path.join(comp_graph_folder)
    os.makedirs(first_eval_error_bars_path, exist_ok=True)
    
    
    for metric in ['ME', 'MAE', 'MSE', 'RMSE', 'SE']:
        plot_quarterly_metrics(ifo_qoq_error_table_first, naive_qoq_first_eval_error_tables_dict, 
                            
                            metric_col=metric,
                            scale_by_n=False, show=False, 
                            n_bars=QoQ_eval_n_bars,
                            save_path=first_eval_error_bars_path,
                            save_name=f'Joint_{metric}{filter_str}{subset_str}.png')






# ==================================================================================================
# RUN COMPONENT EVALUATION FOR EACH COMPONENT
# ==================================================================================================

if settings.evaluate_forecast_components:

    ## Initial Print Statements    
    print("\n\n" + "="*100)
    print(' '*30 +"COMPONENT-LEVEL EVALUATION")
    print("="*100 + "\n")

    print(f"\nEvaluating all {len(included_components)} selected components ...\n")


    ## Loop over all 
    for comp_name in included_components:
        
        # Check if component data exists
        has_ifo = comp_name in ifo_qoq_forecasts_components
        has_naive = comp_name in component_naive_qoq_dfs_dict
        has_eval = comp_name in component_first_eval_dict


        # Console Printing for tracking
        """
        print(f"  {comp_name}: ifo={has_ifo}, naive={has_naive}, eval={has_eval}")
        
        if not (has_ifo and has_naive and has_eval):
            print(f"    -> Skipping (missing data)")
            continue
        
        print(f"    -> Evaluating component")
        """
        
        # --------------------------------------------------------------------------------------------------
        # Full time series (unfiltered)
        # --------------------------------------------------------------------------------------------------
        
        qoq_error_evaluation_pipeline_components(
            comp_name,
            ifo_qoq_df_components=ifo_qoq_forecasts_components[comp_name],
            naive_qoq_dict_components=component_naive_qoq_dfs_dict[comp_name],
            component_eval_dict=component_first_eval_dict[comp_name],
            subset_mode=False,
            sd_filter_mode=False
        )
        
        # --------------------------------------------------------------------------------------------------
        # Full time series (filtered outliers if enabled)
        # --------------------------------------------------------------------------------------------------
        
        if settings.run_component_filter:
            print(f"\n\n     - Dropping outliers for {comp_name}... \n\n ")
            qoq_error_evaluation_pipeline_components(
                comp_name,
                ifo_qoq_df_components=ifo_qoq_forecasts_components[comp_name],
                naive_qoq_dict_components=component_naive_qoq_dfs_dict[comp_name],
                component_eval_dict=component_first_eval_dict[comp_name],
                subset_mode=False,
                sd_filter_mode=True
            )
        
        # --------------------------------------------------------------------------------------------------
        # Filtered time series (if run_component_filter is enabled)
        # --------------------------------------------------------------------------------------------------
        
        if settings.run_component_filter:
            print(f"\n\n     - Running filtered timeframe analysis for {comp_name}... \n\n ")
            qoq_error_evaluation_pipeline_components(
                comp_name,
                ifo_qoq_df_components=ifo_qoq_forecasts_components_subset[comp_name],
                naive_qoq_dict_components=component_naive_qoq_dfs_dict_subset[comp_name],
                component_eval_dict=component_first_eval_dict[comp_name],
                subset_mode=True,
                sd_filter_mode=False
            )
            
            if settings.filter_outliers_within_eval_intervall:
                print(f"\n\n    - Dropping outliers within filtered timeframe for {comp_name}... \n\n ")
                qoq_error_evaluation_pipeline_components(
                    comp_name,
                    ifo_qoq_df_components=ifo_qoq_forecasts_components_subset[comp_name],
                    naive_qoq_dict_components=component_naive_qoq_dfs_dict_subset[comp_name],
                    component_eval_dict=component_first_eval_dict[comp_name],
                    subset_mode=True,
                    sd_filter_mode=True
                )



# --------------------------------------------------------------------------------------------------
print(f" \n Quarterly Evaluation Module complete! \n",f"Find Result Graphs in {graph_folder}n\nResult Tables in {table_folder} and \n Component Evaluation Results in {component_result_folder}\n")
# --------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#