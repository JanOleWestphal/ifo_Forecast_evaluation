
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
- ifo errors in the nowcast are faulty at least from Q2-2020 onwards, reasons unclear
- filter for naive forecasts with unavailable horizons to the ifo forecasts is faulty, needs to be ammended, not a first order concern

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
evaluation_limit_year = settings.evaluation_limit_year
evaluation_limit_quarter = settings.evaluation_limit_quarter    

# Define the horizon of first releases which should be evaluated:
first_release_lower_limit_year = settings.first_release_lower_limit_year
first_release_lower_limit_quarter = settings.first_release_lower_limit_quarter

first_release_upper_limit_year = settings.first_release_upper_limit_year
first_release_upper_limit_quarter = settings.first_release_upper_limit_quarter





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
#if settings.clear_result_folders:
#    folder_clear(folder_path)












# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                 Load in data to be evaluated                                     #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

print("\n Loading Evaluation Data ...  \n")


# ==================================================================================================
# Load Data 
# ==================================================================================================


# --------------------------------------------------------------------------------------------------
# Load ifo forecasts - GDP
# --------------------------------------------------------------------------------------------------

# Path
file_path_ifo_qoq = os.path.join(wd, '0_0_Data', '2_Processed_Data', '3_ifo_qoq_series',
                                  'ifo_qoq_forecasts.xlsx' )

# Load 
ifo_qoq_forecasts = pd.read_excel(file_path_ifo_qoq, index_col=0)





# --------------------------------------------------------------------------------------------------
# Load ifo forecasts - components
# --------------------------------------------------------------------------------------------------

# Path
file_path_ifo_qoq_components = os.path.join(wd, '0_0_Data', '2_Processed_Data', '3_gdp_component_forecast')

# Load - manually parse component names from filenames
ifo_qoq_forecasts_components = {}
if os.path.exists(file_path_ifo_qoq_components):
    component_files = glob.glob(os.path.join(file_path_ifo_qoq_components, 'ifo_qoq_forecasts_*.xlsx'))
    for file in component_files:
        filename = os.path.basename(file)
        # Extract component name from 'ifo_qoq_forecasts_COMPONENT.xlsx'
        comp_name = filename.replace('ifo_qoq_forecasts_', '').replace('.xlsx', '')
        ifo_qoq_forecasts_components[comp_name] = pd.read_excel(file, index_col=0)
        print(f"  Loaded ifo forecasts for component: {comp_name}")




# --------------------------------------------------------------------------------------------------
# Load naive forecasts 
# --------------------------------------------------------------------------------------------------

## MAIN GDP ANALYSIS

# Paths to the folders containing the Excel files
file_path_naive_qoq = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '1_QoQ_Forecast_Tables')

# Load all QoQ naive forecast Excel files into dictionary
naive_qoq_dfs_dict = load_excels_to_dict(file_path_naive_qoq, strip_string='naive_qoq_forecasts_')




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


## COMPONENT ANALYSIS
file_path_component_qoq = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '3_QoQ_Component_Forecast_Tables')

# Load component naive forecasts with structure: naive_qoq_forecasts_COMPONENT_MODEL.xlsx
component_naive_qoq_dfs_dict = {}

if os.path.exists(file_path_component_qoq):
    component_files = glob.glob(os.path.join(file_path_component_qoq, 'naive_qoq_forecasts_*.xlsx'))
    
    for file in component_files:
        filename = os.path.basename(file)
        # Extract component and model from 'naive_qoq_forecasts_COMPONENT_MODEL.xlsx'
        # e.g., 'naive_qoq_forecasts_CONSTR_AR2_50_9.xlsx' -> component='CONSTR', model='AR2_50_9'
        parts = filename.replace('naive_qoq_forecasts_', '').replace('.xlsx', '').split('_', 1)
        
        if len(parts) == 2:
            component = parts[0]
            model = parts[1]
            
            if component not in component_naive_qoq_dfs_dict:
                component_naive_qoq_dfs_dict[component] = {}
            
            df = pd.read_excel(file, index_col=0)
            component_naive_qoq_dfs_dict[component][model] = df
            print(f"  Loaded component forecast: {component} - {model}")




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

component_eval_path = os.path.join(wd, '0_0_Data', '2_Processed_Data', '2_component_Evaluation_series')
component_first_eval_dict = {}

# Load evaluation data for each component
if os.path.exists(component_eval_path):
    component_names_check = ['GDP', 'PRIVCON', 'PUBCON', 'CONSTR', 'OPA', 'INVINV', 'DOMUSE', 'TRDBAL', 'EXPORT', 'IMPORT']
    for comp in component_names_check:
        eval_file = os.path.join(component_eval_path, f'first_release_qoq_{comp}.xlsx')
        if os.path.exists(eval_file):
            try:
                component_first_eval_dict[comp] = pd.read_excel(eval_file, index_col=0)
                print(f"  Loaded evaluation data for component: {comp}")
            except Exception as e:
                print(f"  ⚠ Could not load evaluation data for {comp}: {e}")










# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                              Build joint evaluation dataframes                                   #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


def filter_naive_for_ifo_forecasts(ifo_qoq_forecast_df=ifo_qoq_forecasts, naive_qoq_dfs_dict=naive_qoq_dfs_dict):

    # ==================================================================================================
    # Select whether to omit Naive Forecaster Observations if no ifo Forecast is available
    # ==================================================================================================

    ## Execute the filter function: drop all naive qoq which do not have a corresponding ifo qoq forecast
    naive_qoq_dfs_dict = match_ifo_naive_forecasts_dates(ifo_qoq_forecast_df, naive_qoq_dfs_dict)


    ## LEGACY CODE: full dataset without evaluation timeframe filtering
    # Store full dataset for full Error Time Series later on
    #ifo_qoq_forecasts_full = ifo_qoq_forecast_df.copy()

    # Store full dataset for full Error Time Series later on
    #naive_qoq_dfs_dict_full = naive_qoq_dfs_dict.copy()



    # ==================================================================================================
    # Select Evaluation timeframe
    # ==================================================================================================

    # Apply row and col selection from helperfunctions:
    ifo_qoq_forecast_df = filter_first_release_limit(ifo_qoq_forecast_df)
    ifo_qoq_forecast_df = filter_evaluation_limit(ifo_qoq_forecast_df)

    for key, val in naive_qoq_dfs_dict.items():
        val = filter_first_release_limit(val)
        val = filter_evaluation_limit(val)
        naive_qoq_dfs_dict[key] = val


    ## Filter naive forecasts accordingly
    naive_qoq_dfs_dict = match_ifo_naive_forecasts_dates(ifo_qoq_forecast_df, naive_qoq_dfs_dict)

    return naive_qoq_dfs_dict




# ==================================================================================================
#  MAIN GDP EVALUATION
# ==================================================================================================

naive_qoq_dfs_dict = filter_naive_for_ifo_forecasts(ifo_qoq_forecasts, naive_qoq_dfs_dict)












# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                         Data Pipeline for the full scope Error Series                            #
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










# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#             MAIN:  Run the evaluation pipeline on filtered and unfiltered data                   #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


if settings.evaluate_quarterly_gdp_forecasts:
    # --------------------------------------------------------------------------------------------------
    # Full time series
    # --------------------------------------------------------------------------------------------------

    ## Unfiltered Data
    print(" \nAnalysing the full error series ... \n")
    qoq_error_evaluation_pipeline(ifo_qoq_df=ifo_qoq_forecasts,
                                naive_qoq_dict= naive_qoq_dfs_dict,
                                sd_filter_mode=False)

    ## Filtered Errors: Drop Outliers as resulting from crisis events, e.g. 2009 and Covid Quarters
    if settings.drop_outliers:
        print(" \nDropping Outliers from Error Series before Re-evaluation ... \n")
        qoq_error_evaluation_pipeline(ifo_qoq_df=ifo_qoq_forecasts,
                                naive_qoq_dict= naive_qoq_dfs_dict,
                                sd_filter_mode=True, sd_cols=5, sd_threshold=settings.sd_threshold)



    # --------------------------------------------------------------------------------------------------
    # Filtered time series (e.g. from 2010-Q1 - 2017-Q1)
    # --------------------------------------------------------------------------------------------------

    ## Unfiltered Data
    print(f" \nAnalysing error series from {first_release_lower_limit_year}-{first_release_lower_limit_quarter} to {first_release_upper_limit_year}-{first_release_upper_limit_quarter}... \n")
    qoq_error_evaluation_pipeline(ifo_qoq_df=ifo_qoq_forecasts,
                                naive_qoq_dict= naive_qoq_dfs_dict,
                                sd_filter_mode=False, subset_mode=True)

    ## Filtered Errors: Drop Outliers as resulting from crisis events, e.g. 2009 and Covid Quarters
    if settings.filter_outliers_within_eval_intervall:
        print(" \nDropping Outliers from Error Series before Re-evaluation ... \n")
        qoq_error_evaluation_pipeline(ifo_qoq_df=ifo_qoq_forecasts,
                                naive_qoq_dict= naive_qoq_dfs_dict,
                                sd_filter_mode=True, subset_mode=True, sd_cols=5, sd_threshold=settings.sd_threshold)










# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#          COMPONENTS:  Run the evaluation pipeline on filtered and unfiltered data                #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


if settings.evaluate_forecast_components:
    
    print("\n\n" + "="*100)
    print("             COMPONENT-LEVEL EVALUATION")
    print("="*100 + "\n")
    
    # ==================================================================================================
    # Component Names and Setup
    # ==================================================================================================
    
    component_names = ['GDP', 'PRIVCON', 'PUBCON', 'CONSTR', 'OPA', 'INVINV', 'DOMUSE', 'TRDBAL', 'EXPORT', 'IMPORT']
    
    # ==================================================================================================
    # Build joint evaluation dataframes for components
    # ==================================================================================================


    """
    NOTE: this code is faulty - ammend to filter for naive forecast horizons unavailable to the ifo forecast

    def filter_naive_for_ifo_components(ifo_components_dict, naive_components_dict):

        filtered_naive = {}
        
        for comp_name in component_names:
            if comp_name in ifo_components_dict and comp_name in naive_components_dict:
                ifo_df = ifo_components_dict[comp_name]
                naive_dict = naive_components_dict[comp_name]
                #show(ifo_df)
                #show(naive_dict)
                
                # Execute the filter function
                filtered_naive_comp = match_ifo_naive_forecasts_dates(ifo_df, naive_dict)
                
                # Apply row and col selection
                ifo_df_filtered = filter_first_release_limit(ifo_df)
                ifo_df_filtered = filter_evaluation_limit(ifo_df_filtered)
                
                for key, val in filtered_naive_comp.items():
                    val = filter_first_release_limit(val)
                    val = filter_evaluation_limit(val)
                    filtered_naive_comp[key] = val
                
                # Filter naive forecasts accordingly
                filtered_naive_comp = match_ifo_naive_forecasts_dates(ifo_df_filtered, filtered_naive_comp)
                #show(filtered_naive_comp)
                
                filtered_naive[comp_name] = filtered_naive_comp
        
        return filtered_naive
    
    # Build filtered component dicts
    component_naive_qoq_dfs_dict_filtered = component_naive_qoq_dfs_dict
    """
    """
    filter_naive_for_ifo_components(
        ifo_qoq_forecasts_components, 
        component_naive_qoq_dfs_dict
    )"""
    
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
        
        # Clear if needed
        if settings.clear_result_folders:
            for folder in [comp_table_folder, comp_graph_folder, comp_data_folder]:
                folder_clear(folder)
        
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
        
        print(f"Visualizing component {component_name} error statistics ...")
        
        # --------------------------------------------------------------------------------------------------
        # Error Time Series
        # --------------------------------------------------------------------------------------------------
        
        first_eval_error_series_path_ifo = os.path.join(comp_graph_folder, f'ts_ifo{filter_str}{subset_str}')
        first_eval_error_series_path = os.path.join(comp_graph_folder, f'ts_jnt{filter_str}{subset_str}')
        
        # Ensure all parent directories exist
        os.makedirs(first_eval_error_series_path_ifo, exist_ok=True)
        os.makedirs(first_eval_error_series_path, exist_ok=True)
        
        if settings.clear_result_folders:
            for folder in [first_eval_error_series_path_ifo, first_eval_error_series_path]:
                folder_clear(folder)
        
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
        
        first_eval_error_line_path = os.path.join(comp_graph_folder, f'sc{filter_str}{subset_str}')
        
        os.makedirs(first_eval_error_line_path, exist_ok=True)
        
        if settings.clear_result_folders:
            folder_clear(first_eval_error_line_path)
        
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
        
        first_eval_error_bars_path = os.path.join(comp_graph_folder, f'1_QoQ_Error_Bars{filter_str}{subset_str}')
        
        os.makedirs(first_eval_error_bars_path, exist_ok=True)
        
        if settings.clear_result_folders:
            folder_clear(first_eval_error_bars_path)
        
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
    
    print(f"\nEvaluating all {len(component_names)} components ...\n")
    
    for comp_name in component_names:
        
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
            print(f"    - Dropping outliers for {comp_name}...")
            qoq_error_evaluation_pipeline_components(
                comp_name,
                ifo_qoq_df_components=ifo_qoq_forecasts_components[comp_name],
                naive_qoq_dict=component_naive_qoq_dfs_dict[comp_name],
                component_eval_dict=component_first_eval_dict[comp_name],
                subset_mode=False,
                sd_filter_mode=True
            )
        
        # --------------------------------------------------------------------------------------------------
        # Filtered time series (if run_component_filter is enabled)
        # --------------------------------------------------------------------------------------------------
        
        if settings.run_component_filter:
            print(f"    - Running filtered timeframe analysis for {comp_name}...")
            qoq_error_evaluation_pipeline_components(
                comp_name,
                ifo_qoq_df_components=ifo_qoq_forecasts_components[comp_name],
                naive_qoq_dict_components=component_naive_qoq_dfs_dict[comp_name],
                component_eval_dict=component_first_eval_dict[comp_name],
                subset_mode=True,
                sd_filter_mode=False
            )
            
            if settings.filter_outliers_within_eval_intervall:
                print(f"    - Dropping outliers within filtered timeframe for {comp_name}...")
                qoq_error_evaluation_pipeline_components(
                    comp_name,
                    ifo_qoq_df_components=ifo_qoq_forecasts_components[comp_name],
                    naive_qoq_dict=component_naive_qoq_dfs_dict[comp_name],
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