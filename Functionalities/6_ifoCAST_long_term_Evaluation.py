
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        ifoCAST long-term Evaluation
#
# Author:       Jan Ole Westphal
# Date:         2025-09
#
# Description:  Subprogram to evaluate quarterly forecasts of both the ifoCAST on a longer horizon.        
# ==================================================================================================
# --------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        Code begins here                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

print("\n Executing the ifoCAST long term Evaluation Module ... \n")


# ==================================================================================================
#                                           SETUP
# ==================================================================================================

# Import built-ins
import importlib
import subprocess
import sys
import os
import shutil
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
# Load long-term ifoCAST data
# --------------------------------------------------------------------------------------------------

# Path
ifoCast_longterm_filepath = os.path.join(wd, '0_0_Data', '0_Forecast_Inputs', '2_ifoCAST',
                                  'ifoCast_longterm_Q42019.xlsx' )

# Load Excel
ifoCAST_longterm = pd.read_excel(ifoCast_longterm_filepath)



# Ensure Date is datetime and set as index
ifoCAST_longterm['Date'] = pd.to_datetime(ifoCAST_longterm['Date'])
ifoCAST_longterm = ifoCAST_longterm.set_index('Date')

# Use qoq_growth as column labels and values as diagonal entries
# -> make a diagonal DataFrame
values = ifoCAST_longterm['qoq_growth'].values
dates = ifoCAST_longterm.index

ifoCAST_longterm = pd.DataFrame(
    np.diag(values),
    index=dates,
    columns=dates
)

# Set all non diagonal entries to NaN
ifoCAST_longterm = ifoCAST_longterm.replace(0, np.nan)

#show(ifoCast_longterm)  





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





# ==================================================================================================
# FILTER DATA TO ifoCAST Scope
# ==================================================================================================

def match_ifoCAST_naive_forecasts_dates(ifo_qoq_forecasts, naive_qoq_dfs_dict):

    if settings.match_ifo_naive_dates:

        for key, naive_df in naive_qoq_dfs_dict.items():
            
            # Convert to datetime
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

                # For this col, filter rows by year-quarter match with IFO rows
                valid_rows = [
                    row for row in naive_df.index
                    if (pd.to_datetime(row).year, pd.to_datetime(row).quarter) in ifo_row_quarters
                ]

                # Assign truncated series
                filtered_df[col] = naive_df.loc[valid_rows, col]

            # Save back
            naive_qoq_dfs_dict[key] = filtered_df

        return naive_qoq_dfs_dict



## Match ifoCAST to normal forecast availability
ifoCAST_longterm_dict = {"ifoCAST": ifoCAST_longterm}
ifoCAST_longterm_dict = match_ifoCAST_naive_forecasts_dates(ifo_qoq_forecasts, ifoCAST_longterm_dict)
ifoCAST_longterm = ifoCAST_longterm_dict['ifoCAST']
#show(ifoCAST_longterm)


## match towards ifoCAST availability
# ifo normal forecasts
ifo_qoq_forecasts_dict = {"ifo": ifo_qoq_forecasts}
ifo_qoq_forecasts_dict = match_ifoCAST_naive_forecasts_dates(ifoCAST_longterm, ifo_qoq_forecasts_dict)
ifo_qoq_forecasts = ifo_qoq_forecasts_dict['ifo']

#show(ifo_qoq_forecasts)

# naive forecasts
naive_qoq_dfs_dict = match_ifoCAST_naive_forecasts_dates(ifoCAST_longterm, naive_qoq_dfs_dict)
#show(next(iter(naive_qoq_dfs_dict.values())))







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
                                  naive_mode= False, ifoCAST_mode=False):



    # ==================================================================================================
    #                                          DYNAMIC NAMING
    # ==================================================================================================

    ## Dynamic result naming
    naive_str = "_naive_qoq" if naive_mode else ""
    CAST = "CAST" if ifoCAST_mode else ""




    # --------------------------------------------------------------------------------------------------
    # ==================================================================================================
    #                                       GET FORECAST TABLES
    # ==================================================================================================
    # --------------------------------------------------------------------------------------------------

    # ==================================================================================================
    # Savepaths
    # ==================================================================================================

    ## ifo
    # Error Series
    ifo_qoq_error_path = os.path.join(wd, '0_1_Output_Data', f'6_ifo_qoq_error_series_ifoCASTset{naive_str}')
    # Error Tables
    ifo_qoq_table_path = os.path.join(table_folder, f'6_ifo{CAST}_qoq_evaluations_ifoCASTset{naive_str}')

    ## Naive
    # Error Series
    naive_qoq_error_path = os.path.join(wd, '0_1_Output_Data', f'6_naive_forecaster_qoq_error_series_ifoCASTset{naive_str}')
    # Error table
    naive_qoq_table_path = os.path.join(table_folder, f'6_naive_forecaster_qoq_evaluations_ifoCASTset{naive_str}')


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
    ifo_qoq_forecasts_eval_first_collapsed = collapse_quarterly_prognosis(ifo_qoq_forecasts_eval_first)

    ifo_qoq_errors_first = get_qoq_error_series(ifo_qoq_forecasts_eval_first_collapsed,
                                        ifo_qoq_error_path, 
                                        file_name=f"ifo{CAST}_qoq_errors_first_eval_ifoCASTset{naive_str}.xlsx")



    ## Get Table
    ifo_qoq_error_table_first = get_qoq_error_statistics_table(ifo_qoq_errors_first,                                                            
                                                            'first_eval', ifo_qoq_table_path, 
                                                            f'ifo{CAST}_qoq_forecast_error_table_first_eval_ifoCASTset{naive_str}.xlsx')


    # --------------------------------------------------------------------------------------------------
    # Latest Release
    # --------------------------------------------------------------------------------------------------

    ## Get Error Series
    ifo_qoq_forecasts_eval_latest = create_qoq_evaluation_df(ifo_qoq_df, qoq_latest_eval)
    ifo_qoq_forecasts_eval_latest_collapsed = collapse_quarterly_prognosis(ifo_qoq_forecasts_eval_latest)

    ifo_qoq_errors_latest = get_qoq_error_series(ifo_qoq_forecasts_eval_latest_collapsed,
                                                ifo_qoq_error_path, 
                                                file_name=f"ifo{CAST}_qoq_errors_latest_eval_ifoCASTset{naive_str}.xlsx")


    ## Get Table
    ifo_qoq_error_table_latest = get_qoq_error_statistics_table(ifo_qoq_errors_latest,
                                                                'latest_eval', ifo_qoq_table_path, 
                                                            f'ifo{CAST}_qoq_forecast_error_table_latest_eval_ifoCASTset{naive_str}.xlsx')






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
                                                            file_name=f"{name}_qoq_errors_first_eval_ifoCASTset{naive_str}.xlsx")

        naive_qoq_first_eval_error_tables_dict[name] = get_qoq_error_statistics_table(
                                                            naive_qoq_first_eval_error_series_dict[name],
                                                            'first_eval', naive_qoq_table_path, 
                                                        f'{name}_qoq_forecast_error_table_first_eval_ifoCASTset{naive_str}.xlsx')
    
    #show(next(iter(naive_qoq_first_eval_error_tables_dict.values())))


    ## Evaluate against latest releases

    # Get Result Dictionaries
    naive_qoq_latest_eval_error_series_dict = {}
    naive_qoq_latest_eval_error_tables_dict = {}

    # Run evaluation loop over all models
    for name, df in naive_qoq_latest_eval_dfs_collapsed.items():
        naive_qoq_latest_eval_error_series_dict[name] = get_qoq_error_series(df,
                                                            naive_qoq_error_path, 
                                                            file_name=f"{name}_qoq_errors_latest_eval_ifoCASTset{naive_str}.xlsx")


        naive_qoq_latest_eval_error_tables_dict[name] = get_qoq_error_statistics_table(
                                                            naive_qoq_latest_eval_error_series_dict[name],
                                                        'latest_eval', naive_qoq_table_path, 
                                                        f'{name}_qoq_forecast_error_table_latest_eval_ifoCASTset{naive_str}.xlsx')
















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
    first_eval_error_series_path_ifo = os.path.join(graph_folder, f'6_QoQ_Error_Series_ifoCASTset{naive_str}', f'0_First_Evaluation_ifo_ifoCASTset{naive_str}')
    first_eval_error_series_path = os.path.join(graph_folder, f'6_QoQ_Error_Series_ifoCASTset{naive_str}', f'0_First_Evaluation_joint_ifoCASTset{naive_str}')
    latest_eval_error_series_path = os.path.join(graph_folder, f'6_QoQ_Error_Series_ifoCASTset{naive_str}', f'1_Latest_Evaluation_ifoCASTset{naive_str}')

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
                                save_path=first_eval_error_series_path_ifo, save_name_prefix=f'ifo{CAST}_First_Eval_ifoCASTset_', select_quarters=[0])

    plot_forecast_timeseries(ifo_qoq_forecasts_eval_first, naive_qoq_first_eval_dfs, 
                            df_eval=qoq_first_eval, title_prefix=None, figsize=(12, 8), linestyle=None,
                                show=False, 
                                save_path=first_eval_error_series_path, save_name_prefix=f'First_Eval_ifoCASTset_', select_quarters=[0])


    plot_forecast_timeseries(ifo_qoq_forecasts_eval_latest, naive_qoq_latest_eval_dfs, 
                            df_eval=qoq_latest_eval, title_prefix=None, figsize=(12, 8), linestyle=None,
                                show=False, 
                                save_path=latest_eval_error_series_path, save_name_prefix=f'Latest_Eval_ifoCASTset_', select_quarters=[0])





    # --------------------------------------------------------------------------------------------------
    # Error Scatter Plots
    # --------------------------------------------------------------------------------------------------
    print("Plotting Error Scatter Plots ...")

    ## Savepaths
    first_eval_error_line_path = os.path.join(graph_folder, f'6_QoQ_Error_Scatter_ifoCASTset{naive_str}', f'0_First_Evaluation_ifoCASTset{naive_str}')
    latest_eval_error_line_path = os.path.join(graph_folder, f'6_QoQ_Error_Scatter_ifoCASTset{naive_str}', f'1_Latest_Evaluation_ifoCASTset{naive_str}')

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
                    n_bars = 1,
                    save_path=first_eval_error_line_path,
                    save_name=f'ifo_QoQ_First_Eval_ifoCASTset_Error_Scatter.png')

    plot_error_lines(ifo_qoq_errors_first, naive_qoq_first_eval_error_series_dict,
                    show=False, 
                    n_bars = 1,
                    save_path=first_eval_error_line_path,
                    save_name=f'Joint_QoQ_First_Eval_ifoCASTset_Error_Scatter.png')
                    

    # Latest Evaluation
    plot_error_lines(ifo_qoq_errors_latest, 
                    show=False, 
                    n_bars = 1,
                    save_path=latest_eval_error_line_path,
                    save_name=f'ifo_QoQ_Latest_Eval_ifoCASTset_Error_Scatter.png')

    plot_error_lines(ifo_qoq_errors_latest, naive_qoq_latest_eval_error_series_dict,
                    show=False, 
                    n_bars = 1,
                    save_path=latest_eval_error_line_path,
                    save_name=f'Joint_QoQ_Latest_Eval_ifoCASTset_Error_Scatter.png')





    # --------------------------------------------------------------------------------------------------
    # Error Bar Plots
    # --------------------------------------------------------------------------------------------------
    print("Plotting Error Bar Plots ...")


    ## Savepaths
    first_eval_error_bars_path = os.path.join(graph_folder, f'6_QoQ_Error_Bars_ifoCASTset{naive_str}', f'0_First_Evaluation_ifoCASTset{naive_str}')
    latest_eval_error_bars_path = os.path.join(graph_folder, f'6_QoQ_Error_Bars_ifoCASTset{naive_str}', f'1_Latest_Evaluation_ifoCASTset{naive_str}')

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
                            n_bars = 1,

                            save_path=first_eval_error_bars_path,
                            save_name=f'Joint_Quarterly_{metric}_first_eval_ifoCASTset.png'
                                )


    # Latest Evaluation
    for metric in ['ME', 'MAE', 'MSE', 'RMSE', 'SE']:
        plot_quarterly_metrics(ifo_qoq_error_table_latest, naive_qoq_latest_eval_error_tables_dict, 
                            
                            metric_col=metric,
                            scale_by_n=False, show=False, 
                            n_bars = 1,

                            save_path=latest_eval_error_bars_path,
                            save_name=f'Joint_Quarterly_{metric}_latest_eval_ifoCASTset.png'
                                )
        
  
  
    # --------------------------------------------------------------------------------------------------
    # Clean folder structure
    # --------------------------------------------------------------------------------------------------

    #if ifoCAST_mode==False and naive_mode==False:
    #    shutil.rmtree(naive_qoq_table_path)


    










# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#               Run the evaluation pipeline on ifocAST + ifo & ifoCAST + Naive                     #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


## Unfiltered Data
#print(" \nAnalysing the full error series ... \n")
qoq_error_evaluation_pipeline(ifo_qoq_df=ifo_qoq_forecasts,
                              naive_qoq_dict= ifoCAST_longterm_dict)


qoq_error_evaluation_pipeline(ifo_qoq_df=ifoCAST_longterm,
                              naive_qoq_dict= naive_qoq_dfs_dict,
                              ifoCAST_mode=True,
                              naive_mode=True)











# --------------------------------------------------------------------------------------------------
print(f" \n ifoCAST long term Evaluation Module complete! \n",f"Find Result Graphs in {graph_folder} and \nResult Tables in {table_folder}\n")
# --------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#