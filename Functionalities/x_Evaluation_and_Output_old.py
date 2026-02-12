

# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Evaluation and Output Script
#
# Author:       Jan Ole Westphal
# Date:         2025-06
#
# Description:  This script evaluates the ifo forecasts and outputs the results.   
#               It is called by the main script and relies on the settings defined in the
#               ifo_forecast_evaluation_settings.py file.         
# ==================================================================================================
# --------------------------------------------------------------------------------------------------





# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        Code begins here                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# ==================================================================================================
#                                           SETUP
# ==================================================================================================

print("\n Initiating the evaluation and output module ...  \n")

# Import built-ins
import importlib
import subprocess
import sys
import os
import glob
from datetime import datetime, date


# Import libraries
import requests
import pandas as pd
from pandasgui import show  #uncomment this to allow for easier debugging
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
sns.set_theme(style='whitegrid')

# Ensure project root is in sys.path
wd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if wd not in sys.path:
    sys.path.insert(0, wd)


# Result Folder Paths
table_folder = os.path.join(wd, '1_Result_Tables')
graph_folder = os.path.join(wd, '2_Result_Graphs')





# ==================================================================================================
#                             IMPORT SETTINGS AND CUSTOM FUNCTIONALITIES
# ==================================================================================================

## Import settings from the settings file
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

# Define the horizon of first releases which should be evaluated: available from 1995-Q3 onwards
first_release_limit_year = settings.first_release_limit_year
first_release_limit_quarter = settings.first_release_limit_quarter


## Get helper functions
from Functionalities.helpers.helperfunctions import *






# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                   Load in evaluation data                                        #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

print("\n Loading Evaluation Data ...  \n")

# ==================================================================================================
# Load Data 
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Load Evaluation Data
# --------------------------------------------------------------------------------------------------

eval_path = os.path.join(wd, '0_0_Data', '2_Processed_Data', '2_Evaluation_series')

## First Releases
# Filepaths
#absolute_path_first = os.path.join(eval_path, 'first_release_absolute_GDP.xlsx')
qoq_path_first = os.path.join(eval_path, 'first_release_qoq_GDP.xlsx')
yoy_path_first = os.path.join(eval_path, 'first_release_yoy_GDP.xlsx')

# Load 
#absolute_first_eval= pd.read_excel(absolute_path_first, index_col=0)
qoq_first_eval = pd.read_excel(qoq_path_first, index_col=0)
yoy_first_eval = pd.read_excel(yoy_path_first, index_col=0)

#show(qoq_first_eval)

## Latest Releases
# Filepaths
#absolute_path_latest= os.path.join(eval_path, 'latest_release_absolute_GDP.xlsx')
qoq_path_latest= os.path.join(eval_path, 'latest_release_qoq_GDP.xlsx')
yoy_path_latest = os.path.join(eval_path, 'latest_release_yoy_GDP.xlsx')


# Load 
#absolute_latest_eval= pd.read_excel(absolute_path_latest, index_col=0)
qoq_latest_eval = pd.read_excel(qoq_path_latest, index_col=0)
yoy_latest_eval = pd.read_excel(yoy_path_latest, index_col=0)

#show(yoy_latest_eval)


## Revision
# Filepaths
#absolute_path_rev = os.path.join(eval_path, 'revision_absolute_GDP.xlsx')
qoq_path_rev = os.path.join(eval_path, 'revision_qoq_GDP.xlsx')
yoy_path_rev = os.path.join(eval_path, 'revision_yoy_GDP.xlsx')


# Load 
#absolute_rev = pd.read_excel(absolute_path_rev, index_col=0)
qoq_rev = pd.read_excel(qoq_path_rev, index_col=0)
yoy_rev = pd.read_excel(yoy_path_rev, index_col=0)




# --------------------------------------------------------------------------------------------------
# Load ifo and consensus forecasts
# --------------------------------------------------------------------------------------------------
forecast_path = os.path.join(wd, '0_0_Data', '0_Forecast_Inputs')

## Filepaths
# consensus
consensus_dec_path = os.path.join(forecast_path, 'consensus_december_forecasts.xlsx')
consensus_jan_path = os.path.join(forecast_path, 'consensus_january_forecasts.xlsx')
consensus_jun_path = os.path.join(forecast_path, 'consensus_june_forecasts.xlsx')
consensus_jul_path = os.path.join(forecast_path, 'consensus_july_forecasts.xlsx')

## Load
# consensus
consensus_dec = pd.read_excel(consensus_dec_path, index_col=0, decimal=',')
consensusjan = pd.read_excel(consensus_jan_path, index_col=0, decimal=',')
consensus_jun = pd.read_excel(consensus_jun_path, index_col=0, decimal=',')
consensus_jul = pd.read_excel(consensus_jul_path, index_col=0, decimal=',')

# show(ifo_dec)



# --------------------------------------------------------------------------------------------------
# Load naive forecasts
# --------------------------------------------------------------------------------------------------

# Paths to the folders containing the Excel files
#file_path_dt_qoq = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '0_Combined_QoQ_Forecasts')
file_path_dt_yoy = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '1_YoY_Forecast_Vectors')

## Helper function to load all Excel files in a folder into a dict of DataFrames
def load_excels_to_dict(folder_path):
    excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
    dfs = {}
    for file in excel_files:
        name = os.path.splitext(os.path.basename(file))[0]
        dfs[name] = pd.read_excel(file, index_col=0)
    return dfs

# Load all QoQ and YoY naive forecast Excel files into dicts
#naive_qoq_dfs = load_excels_to_dict(file_path_dt_qoq)
naive_yoy_dfs = load_excels_to_dict(file_path_dt_yoy)












# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                             Analyzing Error Measures - Functions                                 #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

print("Computing error statistics ...  \n")


# ==================================================================================================
#                                        FUNCTIONALITIES
# ==================================================================================================

## Select common cols in both dfs
def match_evaluation_qoq(forecast_df, eval_df):

    """
    Subset forecast_df and eval_df to only matching quarters
    based on their 'date_of_forecast'/'date' columns.
    """

    # Convert dates to yyyy-Qx format for matching
    forecast_quarters = forecast_df['date_of_forecast'].apply(lambda d: f"{d.year}-Q{((d.month - 1) // 3 + 1)}")
    eval_quarters = eval_df['date'].apply(lambda d: f"{d.year}-Q{((d.month - 1) // 3 + 1)}")

    # Find common quarters
    common_quarters = set(forecast_quarters).intersection(set(eval_quarters))

    # Subset both DataFrames
    forecast_df_matched = forecast_df[forecast_quarters.isin(common_quarters)].copy()
    eval_df_matched = eval_df[eval_quarters.isin(common_quarters)].copy()

    return forecast_df_matched, eval_df_matched



## Create YoY evaluation setup
def get_evaluation_yoy(forecast_df, eval_df):

    """
    Create joint df with forecasted and realized YoY growth values, used to compute error statistics:

    Extend forecast_df by adding y_0_eval, y_1_eval, y_minus1_eval columns
    containing the actual evaluation data from eval_df, matched by year.
    
    Only add eval columns if the corresponding y_*_forecast column exists.
    """

    # Convert eval_df index to years
    eval_years = eval_df.index.year

    # Prepare the extended DataFrame
    extended_df = forecast_df.copy()

    # Helper to add eval column if forecast column exists
    def add_eval_col(forecast_col, eval_col_name, year_col_name):
        if forecast_col in forecast_df.columns:
            # Get the year values from forecast_df
            years = forecast_df[year_col_name]
            # Get corresponding eval values by matching year to eval_df index
            eval_values = years.map(
                lambda y: eval_df.loc[eval_df.index.year == y]['value'].iloc[0]
                if any(eval_years == y) else pd.NA
            )
            # Insert eval column *after* forecast_col
            col_idx = extended_df.columns.get_loc(forecast_col) + 1
            extended_df.insert(col_idx, eval_col_name, eval_values)

    # Add y_minus1_eval
    add_eval_col('y_minus1_forecast', 'y_minus1_eval', 'y_minus1')

    # Add y_0_eval
    add_eval_col('y_0_forecast', 'y_0_eval', 'y_0')

    # Add y_1_eval
    add_eval_col('y_1_forecast', 'y_1_eval', 'y_1')

    return extended_df




# --------------------------------------------------------------------------------------------------
# Get Summary Statistics
# --------------------------------------------------------------------------------------------------

## Raw errors
def get_error(forecast_df_matched, eval_df_matched):
    """
    Compute raw errors (forecast - actual) between matched forecast and evaluation DataFrames.
    """
    diff = forecast_df_matched - eval_df_matched
    return diff

## Mean errors
def get_me(forecast_df_matched, eval_df_matched):
    """
    Compute Mean Error (ME) between matched forecast and evaluation DataFrames.
    """
    diff = forecast_df_matched - eval_df_matched
    me = diff.mean()
    return me

## MAE
def get_mae(forecast_df_matched, eval_df_matched):
    """
    Compute Mean Absolute Error (MAE) between matched forecast and evaluation DataFrames.
    """
    diff = forecast_df_matched - eval_df_matched
    mae = diff.abs().mean()
    return mae

## MSE
def get_mse(forecast_df_matched, eval_df_matched):
    """
    Compute Mean Squared Error (MSE) between matched forecast and evaluation DataFrames.
    """
    diff = forecast_df_matched - eval_df_matched
    mse = (diff ** 2).mean()
    return mse

## RMSE
def get_rmse(forecast_df_matched, eval_df_matched):
    """
    Compute Root Mean Squared Error (RMSE) between matched forecast and evaluation DataFrames.
    """
    diff = forecast_df_matched - eval_df_matched
    rmse = np.sqrt((diff ** 2).mean())
    return rmse

## Standard Errors
def get_se(forecast_df_matched, eval_df_matched):
    """
    Compute Standard Error (SE) of the forecast errors for matched DataFrames.
    """
    diff = forecast_df_matched - eval_df_matched
    se = diff.std().mean()
    return se

## n
def get_n(forecast_df_matched, eval_df_matched):
    """
    Compute the number of valid (non-NaN) forecast-evaluation pairs for matched DataFrames.
    """
    n = (forecast_df_matched - eval_df_matched).count().sum()
    return n




# --------------------------------------------------------------------------------------------------
# Build Evaluation Table
# --------------------------------------------------------------------------------------------------

def build_and_save_evaluation_table(forecast_df_matched, eval_df_matched, save_path, file_name):
    """
    Build a joint evaluation table with error measures and save it as an Excel file.

    Parameters:
        forecast_df_matched (pd.DataFrame): Forecasts matched to evaluation data.
        eval_df_matched (pd.DataFrame): Actual values matched to forecasts.
        save_path (str): Path to save the Excel file.
    """
    # Compute error measures
    me = get_me(forecast_df_matched, eval_df_matched)
    mae = get_mae(forecast_df_matched, eval_df_matched)
    mse = get_mse(forecast_df_matched, eval_df_matched)
    rmse = get_rmse(forecast_df_matched, eval_df_matched)
    se = get_se(forecast_df_matched, eval_df_matched)
    n = get_n(forecast_df_matched, eval_df_matched)

    # Build table
    eval_table = pd.DataFrame({
        "ME": [me],
        "MAE": [mae],
        "MSE": [mse],
        "RMSE": [rmse],
        "SE": [se],
        "N": [n]
    })


    # Save to Excel
    eval_table.to_excel(os.path.join(save_path, file_name))

    return eval_table









# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                         CONDUCT EVALUATIONS                                      #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# ==================================================================================================
# BASELINE ifo FORECAST EVALUATION
# ==================================================================================================

ifo_result_path = os.path.join(table_folder, '2_ifo_evaluations')









# ==================================================================================================
# Consensus Forecasts
# ==================================================================================================
"""
Might be very interesting to extend this section by putting up ifo against concensus 
and other institutes here
"""

consensus_result_path = os.path.join(table_folder, '3_consensus_evaluations')

## Could Rewrite this to evaluate the corresponding consensus forecasts
"""

# --------------------------------------------------------------------------------------------------
# December Forecast
# --------------------------------------------------------------------------------------------------

## Get Evaluation df:
ifo_dec_first = get_evaluation_yoy(ifo_dec, yoy_first_eval)
ifo_dec_latest = get_evaluation_yoy(ifo_dec, yoy_latest_eval)

#show(ifo_dec_first)

## y_0 evaluation
ifo_dec_y0_errors_first = build_and_save_evaluation_table(ifo_dec_first['y_0_forecast'], ifo_dec_first['y_0_eval'], 
                                                       ifo_result_path, 
                                                       f'ifo_december_y0_forecast_errors_since_{horizon_limit_year}-Q{horizon_limit_quarter}_first_release.xlsx')

ifo_dec_y0_errors_latest = build_and_save_evaluation_table(ifo_dec_latest['y_0_forecast'], ifo_dec_latest['y_0_eval'], 
                                                       ifo_result_path, 
                                                       f'ifo_december_y0__forecast_errors_since_{horizon_limit_year}-Q{horizon_limit_quarter}_latest_release.xlsx')


## y_1 evaluation
ifo_dec_y1_errors_first = build_and_save_evaluation_table(ifo_dec_first['y_1_forecast'], ifo_dec_first['y_1_eval'], 
                                                       ifo_result_path, 
                                                       f'ifo_december_y1_forecast_errors_since_{horizon_limit_year}-Q{horizon_limit_quarter}_first_release.xlsx')

ifo_dec_y1errors_latest = build_and_save_evaluation_table(ifo_dec_latest['y_1_forecast'], ifo_dec_latest['y_1_eval'], 
                                                       ifo_result_path, 
                                                       f'ifo_december_y1_forecast_errors_since_{horizon_limit_year}-Q{horizon_limit_quarter}_latest_release.xlsx')



# --------------------------------------------------------------------------------------------------
# January Forecast
# --------------------------------------------------------------------------------------------------

## Get Evaluation df:
ifo_jan_first = get_evaluation_yoy(ifo_jan, yoy_first_eval)
ifo_jan_latest = get_evaluation_yoy(ifo_jan, yoy_latest_eval)

## y_minus1 evaluation
ifo_jan_yminus1_errors_first = build_and_save_evaluation_table(ifo_jan_first['y_minus1_forecast'], ifo_jan_first['y_minus1_eval'], 
                                                       ifo_result_path, 
                                                       f'ifo_january_yminus1_forecast_errors_since_{horizon_limit_year}-Q{horizon_limit_quarter}_first_release.xlsx')

ifo_jan_yminus_errors_latest = build_and_save_evaluation_table(ifo_jan_latest['y_minus1_forecast'], ifo_jan_latest['y_minus1_eval'], 
                                                       ifo_result_path, 
                                                       f'ifo_january_yminus1_forecast_errors_since_{horizon_limit_year}-Q{horizon_limit_quarter}_latest_release.xlsx')

## y_0 evaluation
ifo_jan_y0_errors_first = build_and_save_evaluation_table(ifo_jan_first['y_0_forecast'], ifo_jan_first['y_0_eval'], 
                                                       ifo_result_path, 
                                                       f'ifo_january_y0_forecast_errors_since_{horizon_limit_year}-Q{horizon_limit_quarter}_first_release.xlsx')

ifo_jan_y0_errors_latest = build_and_save_evaluation_table(ifo_jan_latest['y_0_forecast'], ifo_jan_latest['y_0_eval'], 
                                                       ifo_result_path, 
                                                       f'ifo_january_y0_forecast_errors_since_{horizon_limit_year}-Q{horizon_limit_quarter}_latest_release.xlsx')

"""

# --------------------------------------------------------------------------------------------------
# June Forecast
# --------------------------------------------------------------------------------------------------

## y_0 evaluation

## y_1 evaluation

# --------------------------------------------------------------------------------------------------
# July Forecast
# --------------------------------------------------------------------------------------------------

## y_0 evaluation

## y_1 evaluation



# ==================================================================================================
# Naive Forecaster Evaluation
# ==================================================================================================

naive_forecaster_result_path = os.path.join(table_folder, '4_naive_forecaster_evaluations')

## y_minus1 evaluation

## y_0 evaluation

## y_1 evaluation






# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        Visualizations                                            #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# ==================================================================================================
#                                     DEFINE PLOTTER FUNCTIONS
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Barplots with CI
# --------------------------------------------------------------------------------------------------

"""
ifo forecast as bars, naive forecasts as dots
"""

# --------------------------------------------------------------------------------------------------
# Time Series
# --------------------------------------------------------------------------------------------------





# ==================================================================================================
#                                            GET PLOTS
# ==================================================================================================


















# --------------------------------------------------------------------------------------------------
print(f" \n \nEvaluation Output Generation complete! \n \n",
      f"Find Result Tables in {table_folder} \n and Result Graphs in {graph_folder} \n")
# --------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#