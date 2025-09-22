
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Quarterly Evaluation
#
# Author:       Jan Ole Westphal
# Date:         2025-07
#
# Description:  Subprogram to evaluate quarterly forecasts of both ifo and the Naive Forecaster,
#               possibly other providers as well          
# ==================================================================================================
# --------------------------------------------------------------------------------------------------




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



## Create if needed
for folder in [table_folder, graph_folder]:
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
# Load ifo forecasts
# --------------------------------------------------------------------------------------------------

# Path
file_path_ifo_qoq = os.path.join(wd, '0_0_Data', '2_Processed_Data', '3_ifo_qoq_series',
                                  'ifo_qoq_forecasts.xlsx' )

# Load 
ifo_qoq_forecasts = pd.read_excel(file_path_ifo_qoq, index_col=0)




# --------------------------------------------------------------------------------------------------
# Load naive forecasts
# --------------------------------------------------------------------------------------------------

# Paths to the folders containing the Excel files
file_path_naive_qoq = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '1_QoQ_Forecast_Tables')

# Load all QoQ naive forecast Excel files into dictionary
naive_qoq_dfs_dict = load_excels_to_dict(file_path_naive_qoq, strip_string='naive_qoq_forecasts_')
#show(naive_qoq_dfs_dict)

"""
def load_excels_to_dict(folder_path):
    excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
    dfs = {}
    for file in excel_files:
        name = os.path.splitext(os.path.basename(file))[0]
        dfs[name] = pd.read_excel(file, index_col=0)
    return dfs
"""



# --------------------------------------------------------------------------------------------------
# Load Evaluation Data
# --------------------------------------------------------------------------------------------------

eval_path = os.path.join(wd, '0_0_Data', '2_Processed_Data', '2_Evaluation_series')

## First Releases
qoq_path_first = os.path.join(eval_path, 'first_release_qoq_GDP.xlsx')
qoq_first_eval = pd.read_excel(qoq_path_first, index_col=0)
#show(qoq_first_eval)


## Latest Releases
qoq_path_latest= os.path.join(eval_path, 'latest_release_qoq_GDP.xlsx')
qoq_latest_eval = pd.read_excel(qoq_path_latest, index_col=0)
#show(qoq_latest_eval)


## Revision
qoq_path_rev = os.path.join(eval_path, 'revision_qoq_GDP.xlsx')
qoq_rev = pd.read_excel(qoq_path_rev, index_col=0)







# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                              Build joint evaluation dataframes                                   #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#



# ==================================================================================================
# Select whether to omit Naive Forecaster Observations if no ifo Forecast is available
# ==================================================================================================

if settings.match_ifo_naive_dates:

    for key, naive_df in naive_qoq_dfs_dict.items():
        
        # Convert to datetime
        naive_cols_dt = pd.to_datetime(naive_df.columns)
        naive_rows_dt = pd.to_datetime(naive_df.index)
        ifo_cols_dt = pd.to_datetime(ifo_qoq_forecasts.columns)
        ifo_rows_dt = pd.to_datetime(ifo_qoq_forecasts.index)

        # Build sets of year-quarter pairs for IFO
        ifo_col_quarters = {(d.year, d.quarter) for d in ifo_cols_dt}
        ifo_row_quarters = {(d.year, d.quarter) for d in ifo_rows_dt}

        # Keep only valid naive columns (year-quarter match)
        valid_cols = [
            col for col in naive_df.columns
            if (pd.to_datetime(col).year, pd.to_datetime(col).quarter) in ifo_col_quarters
        ]

        # Start filtered df
        filtered_df = pd.DataFrame(index=naive_df.index)

        for col in valid_cols:
            col_dt = pd.to_datetime(col)

            # For this col, filter rows by year-quarter match with IFO rows
            valid_rows = [
                row for row in naive_df.index
                if (pd.to_datetime(row).year, pd.to_datetime(row).quarter) in ifo_row_quarters
            ]

            # Assign truncated series
            filtered_df[col] = naive_df.loc[valid_rows, col]

        # Save back
        naive_qoq_dfs_dict[key] = filtered_df


# ==================================================================================================
# Store full dataset for full Error Time Series
# ==================================================================================================

ifo_qoq_forecasts_full = ifo_qoq_forecasts.copy()
naive_qoq_dfs_dict_full = naive_qoq_dfs_dict.copy()



# ==================================================================================================
# Select Evaluation timeframe
# ==================================================================================================

# Apply row and col selection from helperfunctions:
ifo_qoq_forecasts = filter_first_release_limit(ifo_qoq_forecasts)
ifo_qoq_forecasts = filter_evaluation_limit(ifo_qoq_forecasts)

for key, val in naive_qoq_dfs_dict.items():
    val = filter_first_release_limit(val)
    val = filter_evaluation_limit(val)
    naive_qoq_dfs_dict[key] = val











# ==================================================================================================
# ifo QoQ FORECASTS
# ==================================================================================================

## Evalaute against First Release
ifo_qoq_forecasts_eval_first = create_qoq_evaluation_df(ifo_qoq_forecasts, qoq_first_eval)
ifo_qoq_forecasts_eval_first_collapsed = collapse_quarterly_prognosis(ifo_qoq_forecasts_eval_first)

#show(ifo_qoq_forecasts_eval_first_collapsed)

## Evalaute against Latest Release
ifo_qoq_forecasts_eval_latest = create_qoq_evaluation_df(ifo_qoq_forecasts, qoq_latest_eval)
ifo_qoq_forecasts_eval_latest_collapsed = collapse_quarterly_prognosis(ifo_qoq_forecasts_eval_latest)





# ==================================================================================================
# NAIVE QoQ FORECASTS
# ==================================================================================================

# Store Resuls in a dictionary
naive_qoq_first_eval_dfs = {}
naive_qoq_first_eval_dfs_collapsed = {} 

naive_qoq_latest_eval_dfs = {}
naive_qoq_latest_eval_dfs_collapsed = {} 

# Loop over all available models
for name, df in naive_qoq_dfs_dict.items():
    
    ## Evaluate against first release
    naive_qoq_first_eval_dfs[name] = create_qoq_evaluation_df(df, qoq_first_eval)
    naive_qoq_first_eval_dfs_collapsed[name] = collapse_quarterly_prognosis(naive_qoq_first_eval_dfs[name])

    ## Evaluate against latest release
    naive_qoq_latest_eval_dfs[name] = create_qoq_evaluation_df(df, qoq_latest_eval)
    naive_qoq_latest_eval_dfs_collapsed[name] = collapse_quarterly_prognosis(naive_qoq_latest_eval_dfs[name])












# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                    Analyzing Error Measures                                      #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


print("Computing error statistics ...  \n")



# ==================================================================================================
# ifo QoQ FORECASTS
# ==================================================================================================

## Define path and Filename
# Error Series
ifo_qoq_error_path = os.path.join(wd, '0_1_Output_Data', '4_ifo_qoq_error_series')
os.makedirs(ifo_qoq_error_path, exist_ok=True)

# Error Tables
ifo_qoq_table_path = os.path.join(table_folder, '2_ifo_qoq_evaluations')
os.makedirs(ifo_qoq_table_path, exist_ok=True)

## Clear
if settings.clear_result_folders:

    for folder in [ifo_qoq_error_path, ifo_qoq_table_path]:
        
        folder_clear(folder)


# --------------------------------------------------------------------------------------------------
# Evaluate the ifo qoq Forecasts
# --------------------------------------------------------------------------------------------------

## Evaluate against first releases
ifo_qoq_errors_first = get_qoq_error_series(
                                        ifo_qoq_forecasts_eval_first_collapsed, 
                                        ifo_qoq_error_path, 
                                        file_name="ifo_qoq_errors_first_eval.xlsx")

ifo_qoq_error_table_first = get_qoq_error_statistics_table(ifo_qoq_errors_first,
                                                           'first_eval', ifo_qoq_table_path, 
                                                           'ifo_qoq_forecast_error_table_first_eval.xlsx')
#show(ifo_qoq_error_table_first)

## Evaluate against latest releases
ifo_qoq_errors_latest = get_qoq_error_series(
                                        ifo_qoq_forecasts_eval_latest_collapsed, 
                                        ifo_qoq_error_path, 
                                        file_name="ifo_qoq_errors_latest_eval.xlsx")

ifo_qoq_error_table_latest = get_qoq_error_statistics_table(ifo_qoq_errors_latest,
                                                           'latest_eval', ifo_qoq_table_path, 
                                                           'ifo_qoq_forecast_error_table_latest_eval.xlsx')







# ==================================================================================================
# NAIVE QoQ FORECASTS
# ==================================================================================================

## Define path
# Error Series
naive_qoq_error_path = os.path.join(wd, '0_1_Output_Data', '4_naive_forecaster_qoq_error_series')
os.makedirs(naive_qoq_error_path, exist_ok=True)

# Error table
naive_qoq_table_path = os.path.join(table_folder, '3_naive_forecaster_qoq_evaluations')
os.makedirs(ifo_qoq_table_path, exist_ok=True)

## Clear
if settings.clear_result_folders:

    for folder in [naive_qoq_error_path, naive_qoq_table_path]:
        
        folder_clear(folder)



# --------------------------------------------------------------------------------------------------
# Evaluate the naive qoq Forecasts
# --------------------------------------------------------------------------------------------------


## Evaluate against first releases

# Get Result Dictionaries
naive_qoq_first_eval_error_series_dict = {}
naive_qoq_first_eval_error_tables_dict = {}

# Run evaluation loop over all models
for name, df in naive_qoq_first_eval_dfs_collapsed.items():
    naive_qoq_first_eval_error_series_dict[name] = get_qoq_error_series(
                                        df, 
                                        naive_qoq_error_path, 
                                        file_name=f"{name}_qoq_errors_first_eval.xlsx")

    naive_qoq_first_eval_error_tables_dict[name] = get_qoq_error_statistics_table(
                                                        naive_qoq_first_eval_error_series_dict[name],
                                                        'first_eval',naive_qoq_table_path, 
                                                        f'{name}_qoq_forecast_error_table_first_eval.xlsx')

#show(next(iter(naive_qoq_first_eval_error_tables_dict.values())))



## Evaluate against latest releases

# Get Result Dictionaries
naive_qoq_latest_eval_error_series_dict = {}
naive_qoq_latest_eval_error_tables_dict = {}

# Run evaluation loop over all models
for name, df in naive_qoq_latest_eval_dfs_collapsed.items():
    naive_qoq_latest_eval_error_series_dict[name] = get_qoq_error_series(
                                        df, 
                                        naive_qoq_error_path, 
                                        file_name=f"{name}_qoq_errors_latest_eval.xlsx")

    naive_qoq_latest_eval_error_tables_dict[name] = get_qoq_error_statistics_table(
                                                        naive_qoq_latest_eval_error_series_dict[name],
                                                        'latest_eval', naive_qoq_table_path, 
                                                        f'{name}_qoq_forecast_error_table_latest_eval.xlsx')











# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                         Data Pipeline for the full scope Error Series                            #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

def qoq_error_evaluation_pipeline(filter_mode=False, sd_cols=5, sd_threshold=0.05, 
                                  ifo_qoq_errors_first= ifo_qoq_errors_first, ifo_qoq_errors_latest= ifo_qoq_errors_latest,
                                  naive_qoq_first_eval_error_series_dict=naive_qoq_first_eval_error_series_dict,
                                  naive_qoq_latest_eval_error_series_dict=naive_qoq_latest_eval_error_series_dict,):


    # ==================================================================================================
    #                                          DYNAMIC NAMING
    # ==================================================================================================

    ## Dynamic result naming
    filter_str = "_sd_filtered" if filter_mode else ""




    # ==================================================================================================
    #                                       FILTER ERRORS
    # ==================================================================================================

    if filter_mode:

        ## Filter error Series

        # ifo
        ifo_qoq_errors_first = drop_outliers(ifo_qoq_errors_first, sd_cols=sd_cols, sd_threshold=sd_threshold)
        ifo_qoq_errors_latest = drop_outliers(ifo_qoq_errors_latest, sd_cols=sd_cols, sd_threshold=sd_threshold)

        # naive — loop over dictionary structure
        naive_qoq_first_eval_error_series_dict = {
            name: drop_outliers(df, sd_cols=sd_cols, sd_threshold=sd_threshold)
            for name, df in naive_qoq_first_eval_error_series_dict.items()
        }
        naive_qoq_latest_eval_error_series_dict = {
            name: drop_outliers(df, sd_cols=sd_cols, sd_threshold=sd_threshold)
            for name, df in naive_qoq_latest_eval_error_series_dict.items()
        }








    # ==================================================================================================
    #                                       GET FORECAST TABLES
    # ==================================================================================================

    # --------------------------------------------------------------------------------------------------
    # ifo QoQ FORECASTS
    # --------------------------------------------------------------------------------------------------

    ## Evalaute against First Release
    ifo_qoq_forecasts_eval_first_full = create_qoq_evaluation_df(ifo_qoq_forecasts_full, qoq_first_eval)
    ifo_qoq_forecasts_eval_first_collapsed_full = collapse_quarterly_prognosis(ifo_qoq_forecasts_eval_first_full)

    #show(ifo_qoq_forecasts_eval_first_full)

    ## Evalaute against Latest Release
    ifo_qoq_forecasts_eval_latest_full = create_qoq_evaluation_df(ifo_qoq_forecasts_full, qoq_latest_eval)
    ifo_qoq_forecasts_eval_latest_collapsed_full = collapse_quarterly_prognosis(ifo_qoq_forecasts_eval_latest_full)


    # --------------------------------------------------------------------------------------------------
    # NAIVE QoQ FORECASTS
    # --------------------------------------------------------------------------------------------------

    # Store Resuls in a dictionary
    naive_qoq_first_eval_dfs_full = {}
    naive_qoq_first_eval_dfs_collapsed_full = {} 

    naive_qoq_latest_eval_dfs_full = {}
    naive_qoq_latest_eval_dfs_collapsed_full = {} 

    # Loop over all available models
    for name, df in naive_qoq_dfs_dict_full.items():
        
        ## Evaluate against first release
        naive_qoq_first_eval_dfs_full[name] = create_qoq_evaluation_df(df, qoq_first_eval)
        naive_qoq_first_eval_dfs_collapsed_full[name] = collapse_quarterly_prognosis(naive_qoq_first_eval_dfs_full[name])

        ## Evaluate against latest release
        naive_qoq_latest_eval_dfs_full[name] = create_qoq_evaluation_df(df, qoq_latest_eval)
        naive_qoq_latest_eval_dfs_collapsed_full[name] = collapse_quarterly_prognosis(naive_qoq_latest_eval_dfs_full[name])




    # ==================================================================================================
    # ifo QoQ FORECASTS
    # ==================================================================================================


    # --------------------------------------------------------------------------------------------------
    # Evaluate the ifo qoq Forecasts
    # --------------------------------------------------------------------------------------------------

    ## Evaluate against first releases
    ifo_qoq_errors_first_full = get_qoq_error_series(ifo_qoq_forecasts_eval_first_collapsed_full)
    if filter_mode:
        ifo_qoq_errors_first_full = drop_outliers(ifo_qoq_errors_first_full, sd_cols=sd_cols, sd_threshold=sd_threshold)

    # Get Table
    ifo_qoq_error_table_first_full = get_qoq_error_statistics_table(ifo_qoq_errors_first_full)
    #show(ifo_qoq_error_table_first_full)

    ## Evaluate against latest releases
    ifo_qoq_errors_latest_full = get_qoq_error_series(ifo_qoq_forecasts_eval_latest_collapsed_full)
    if filter_mode:
        ifo_qoq_errors_latest_full = drop_outliers(ifo_qoq_errors_latest_full, sd_cols=sd_cols, sd_threshold=sd_threshold)

    # Get Table
    ifo_qoq_error_table_latest_full = get_qoq_error_statistics_table(ifo_qoq_errors_latest_full)



    # --------------------------------------------------------------------------------------------------
    # Evaluate the naive qoq Forecasts
    # --------------------------------------------------------------------------------------------------


    ## Evaluate against first releases

    # Get Result Dictionaries
    naive_qoq_first_eval_error_series_dict_full = {}
    naive_qoq_first_eval_error_tables_dict_full = {}

    # Run evaluation loop over all models
    for name, df in naive_qoq_first_eval_dfs_collapsed_full.items():

        naive_qoq_first_eval_error_series_dict_full[name] = get_qoq_error_series(df)

        if filter_mode:
            naive_qoq_first_eval_error_series_dict_full[name] = drop_outliers(
                naive_qoq_first_eval_error_series_dict_full[name],
                sd_cols=sd_cols,
                sd_threshold=sd_threshold
            )

        naive_qoq_first_eval_error_tables_dict_full[name] = get_qoq_error_statistics_table(
                                                            naive_qoq_first_eval_error_series_dict_full[name])
    
    #show(next(iter(naive_qoq_first_eval_error_tables_dict_full.values())))
        
        



    ## Evaluate against latest releases

    # Get Result Dictionaries
    naive_qoq_latest_eval_error_series_dict_full = {}
    naive_qoq_latest_eval_error_tables_dict_full = {}

    # Run evaluation loop over all models
    for name, df in naive_qoq_latest_eval_dfs_collapsed_full.items():
        naive_qoq_latest_eval_error_series_dict_full[name] = get_qoq_error_series(df)

        if filter_mode:
            naive_qoq_latest_eval_error_series_dict_full[name] = drop_outliers(
                naive_qoq_latest_eval_error_series_dict_full[name],
                sd_cols=sd_cols,
                sd_threshold=sd_threshold
            )

        naive_qoq_latest_eval_error_tables_dict_full[name] = get_qoq_error_statistics_table(
                                                            naive_qoq_latest_eval_error_series_dict_full[name])
















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
    first_eval_error_series_path_ifo = os.path.join(graph_folder, f'1_QoQ_Error_Series{filter_str}', f'0_First_Evaluation_ifo{filter_str}')
    first_eval_error_series_path = os.path.join(graph_folder, f'1_QoQ_Error_Series{filter_str}', f'0_First_Evaluation_joint{filter_str}')
    latest_eval_error_series_path = os.path.join(graph_folder, f'1_QoQ_Error_Series{filter_str}', f'1_Latest_Evaluation{filter_str}')

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

    plot_forecast_timeseries(ifo_qoq_forecasts_eval_first_full, 
                            df_eval=qoq_first_eval, title_prefix=None, figsize=(12, 8), linestyle=None,
                                show=False, 
                                save_path=first_eval_error_series_path_ifo, save_name_prefix=f'ifo_First_Eval{filter_str}_', select_quarters=None)

    plot_forecast_timeseries(ifo_qoq_forecasts_eval_first_full, naive_qoq_first_eval_dfs_full, 
                            df_eval=qoq_first_eval, title_prefix=None, figsize=(12, 8), linestyle=None,
                                show=False, 
                                save_path=first_eval_error_series_path, save_name_prefix=f'First_Eval{filter_str}_', select_quarters=None)


    plot_forecast_timeseries(ifo_qoq_forecasts_eval_latest_full, naive_qoq_latest_eval_dfs_full, 
                            df_eval=qoq_latest_eval, title_prefix=None, figsize=(12, 8), linestyle=None,
                                show=False, 
                                save_path=latest_eval_error_series_path, save_name_prefix=f'Latest_Eval{filter_str}_', select_quarters=None)





    # --------------------------------------------------------------------------------------------------
    # Error Scatter Plots
    # --------------------------------------------------------------------------------------------------
    print("Plotting Error Scatter Plots ...")

    ## Savepaths
    first_eval_error_line_path = os.path.join(graph_folder, f'1_QoQ_Error_Scatter{filter_str}', f'0_First_Evaluation{filter_str}')
    latest_eval_error_line_path = os.path.join(graph_folder, f'1_QoQ_Error_Scatter{filter_str}', f'1_Latest_Evaluation{filter_str}')

    os.makedirs(first_eval_error_line_path, exist_ok=True)
    os.makedirs(latest_eval_error_line_path, exist_ok=True)


    ## Clear
    if settings.clear_result_folders:

        for folder in [first_eval_error_line_path, latest_eval_error_line_path]:
            
            folder_clear(folder)


    ## Create and Save Plots

    # First Evaluation
    plot_error_lines(ifo_qoq_errors_first_full, 
                    show=False, 
                    n_bars = QoQ_eval_n_bars,
                    save_path=first_eval_error_line_path,
                    save_name=f'ifo_QoQ_First_Eval{filter_str}_Error_Scatter.png')

    plot_error_lines(ifo_qoq_errors_first_full, naive_qoq_first_eval_error_series_dict_full,
                    show=False, 
                    n_bars = QoQ_eval_n_bars,
                    save_path=first_eval_error_line_path,
                    save_name=f'Joint_QoQ_First_Eval{filter_str}_Error_Scatter.png')
                    

    # Latest Evaluation
    plot_error_lines(ifo_qoq_errors_latest_full, 
                    show=False, 
                    n_bars = QoQ_eval_n_bars,
                    save_path=latest_eval_error_line_path,
                    save_name=f'ifo_QoQ_Latest_Eval{filter_str}_Error_Scatter.png')

    plot_error_lines(ifo_qoq_errors_latest_full, naive_qoq_latest_eval_error_series_dict_full,
                    show=False, 
                    n_bars = QoQ_eval_n_bars,
                    save_path=latest_eval_error_line_path,
                    save_name=f'Joint_QoQ_Latest_Eval{filter_str}_Error_Scatter.png')





    # --------------------------------------------------------------------------------------------------
    # Error Bar Plots
    # --------------------------------------------------------------------------------------------------
    print("Plotting Error Bar Plots ...")


    ## Savepaths
    first_eval_error_bars_path = os.path.join(graph_folder, f'1_QoQ_Error_Bars{filter_str}', f'0_First_Evaluation{filter_str}')
    latest_eval_error_bars_path = os.path.join(graph_folder, f'1_QoQ_Error_Bars{filter_str}', f'1_Latest_Evaluation{filter_str}')

    os.makedirs(first_eval_error_bars_path, exist_ok=True)
    os.makedirs(latest_eval_error_bars_path, exist_ok=True)

    
    ## Clear
    if settings.clear_result_folders:

        for folder in [first_eval_error_bars_path, latest_eval_error_bars_path]:
            
            folder_clear(folder)


    ## Create and Save Plots

    # First Evaluation
    for metric in ['ME', 'MAE', 'MSE', 'RMSE', 'SE']:
        plot_quarterly_metrics(ifo_qoq_error_table_first_full, naive_qoq_first_eval_error_tables_dict_full, 
                            
                            metric_col=metric,
                            scale_by_n=False, show=False, 
                            n_bars = QoQ_eval_n_bars,

                            save_path=first_eval_error_bars_path,
                            save_name=f'Joint_Quarterly_{metric}_first_eval{filter_str}.png'
                                )


    # Latest Evaluation
    for metric in ['ME', 'MAE', 'MSE', 'RMSE', 'SE']:
        plot_quarterly_metrics(ifo_qoq_error_table_latest_full, naive_qoq_latest_eval_error_tables_dict_full, 
                            
                            metric_col=metric,
                            scale_by_n=False, show=False, 
                            n_bars = QoQ_eval_n_bars,

                            save_path=latest_eval_error_bars_path,
                            save_name=f'Joint_Quarterly_{metric}_latest_eval{filter_str}.png'
                                )










# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#               Run the evaluation pipeline on filtered and unfiltered data                        #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


## Unfiltered Data
print(" \nAnalysing the full error series ... \n")
qoq_error_evaluation_pipeline(filter_mode=False)

## Filtered Errors: Drop Outliers as resulting from crisis events, e.g. 2009 and Covid Quarters
if settings.drop_outliers:
    print(" \nDropping Outliers from Error Series before Re-evaluation ... \n")
    qoq_error_evaluation_pipeline(filter_mode=True, sd_cols=5, sd_threshold=settings.sd_threshold)










# --------------------------------------------------------------------------------------------------
print(f" \n Quarterly Evaluation Module complete! \n",f"Find Result Graphs in {graph_folder} and \nResult Tables in {table_folder}\n")
# --------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#