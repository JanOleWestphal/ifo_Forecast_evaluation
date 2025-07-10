
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
from datetime import datetime, date
from itertools import product


# Import libraries
import requests
import pandas as pd
from pandasgui import show  #uncomment this to allow for easier debugging
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
# import matplotlib.pyplot as plt
# import seaborn as sns

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
evaluation_limit_year = settings.evaluation_limit_year
evaluation_limit_quarter = settings.evaluation_limit_quarter    

# Define the horizon of first releases which should be evaluated: available from 1995-Q3 onwards
first_release_limit_year = settings.first_release_limit_year
first_release_limit_quarter = settings.first_release_limit_quarter







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
# Select whether to omit Naive Forecaster Observations if no ifo Forecast is available
# ==================================================================================================

if settings.match_ifo_naive_dates:
    for key, naive_df in naive_qoq_dfs_dict.items():
        
        # Convert columns to datetime for proper quarter comparison
        naive_cols_dt = pd.to_datetime(naive_df.columns)
        ifo_cols_dt = pd.to_datetime(ifo_qoq_forecasts.columns)
        
        # Find columns in naive_df that are in the same quarter as any column in ifo_qoq_forecasts
        valid_cols = []
        for naive_col in naive_df.columns:
            naive_col_dt = pd.to_datetime(naive_col)
            naive_quarter = (naive_col_dt.year, naive_col_dt.quarter)
            
            # Check if this quarter exists in ifo_qoq_forecasts
            quarter_match = any(
                (ifo_col_dt.year, ifo_col_dt.quarter) == naive_quarter
                for ifo_col_dt in ifo_cols_dt
            )
            
            if quarter_match:
                valid_cols.append(naive_col)
        
        # Filter naive_df to keep only columns that match quarters in ifo_qoq_forecasts
        naive_qoq_dfs_dict[key] = naive_df[valid_cols]






# ==================================================================================================
# Necessary Functions
# ==================================================================================================


## Creating the joint evaluation df
def create_qoq_evaluation_df(qoq_forecast_df, eval_vector):
    """
    For each column in qoq_forecast_df, create two new columns:
      - {col}_eval : the actual value for that quarter (only if forecast is not NA)
      - {col}_diff : forecast minus actual
    
    Matching is done on quarterly frequency.
    Afterward, all columns are cast to strings and sorted alphabetically.
    """

    # 1) Copy & align both to quarter-ends
    fc = qoq_forecast_df.copy()
    fc.index = pd.to_datetime(fc.index).to_period('Q').to_timestamp()

    ev = eval_vector.copy()
    ev.index = pd.to_datetime(ev.index).to_period('Q').to_timestamp()

    # 2) Build new columns into this dict
    new_cols = {}
    for col in fc.columns:
        forecast = fc[col]

        # Squeeze ev → Series if it's a single-column DataFrame
        raw_ev = ev
        if isinstance(raw_ev, pd.DataFrame):
            if raw_ev.shape[1] != 1:
                raise ValueError("eval_vector must have exactly one column")
            raw_ev = raw_ev.iloc[:, 0]

        # Reindex that Series to the forecast dates
        eval_series = raw_ev.reindex(forecast.index)

        # But only keep eval where forecast is not NA
        eval_series = eval_series.where(forecast.notna(), pd.NA)

        # Name them
        new_cols[f"{col}_eval"] = eval_series
        new_cols[f"{col}_diff"] = forecast - eval_series

    # 3) Concatenate everything in one go
    result = pd.concat(
        [fc, pd.DataFrame(new_cols, index=fc.index)],
        axis=1
    )

    # 4) Cast column names to strings & sort
    result.columns = result.columns.astype(str)
    result = result[sorted(result.columns)]

    return result





## Create a df for quarterly forecast evaluation
def collapse_quarterly_prognosis(df):
   """
   Move all cols to the same rows, rename rows Q1-Qx: gets quarterly error measures based on
   forecast horizons
   """
   # Make a copy to avoid modifying the original DataFrame
   result_df = df.copy()
   
   # Process each column
   for col in result_df.columns:
       # Get non-missing values
       non_missing = result_df[col].dropna()
       
       # Reset the column to all NaN first
       result_df[col] = np.nan
       
       # Place non-missing values starting from row 0
       if len(non_missing) > 0:
           result_df.iloc[:len(non_missing), result_df.columns.get_loc(col)] = non_missing.values
   
   # Find the maximum number of non-missing values across all columns
   max_non_missing = 0
   for col in result_df.columns:
       non_missing_count = result_df[col].notna().sum()
       max_non_missing = max(max_non_missing, non_missing_count)
   
   # Keep only the rows that contain data (up to max_non_missing)
   result_df = result_df.iloc[:max_non_missing]
   
   # Rename index to Q1, Q2, Q3, etc.
   result_df.index = [f'Q{i}' for i in range(len(result_df))]
   
   return result_df






# ==================================================================================================
# ifo QoQ FORECASTS
# ==================================================================================================

## Evalaute against First Release
ifo_qoq_forecasts_eval_first = create_qoq_evaluation_df(ifo_qoq_forecasts, qoq_first_eval)
ifo_qoq_forecasts_eval_first_collapsed = collapse_quarterly_prognosis(ifo_qoq_forecasts_eval_first)

#show(ifo_qoq_forecasts_eval_first)

## Evalaute against Latest Release
ifo_qoq_forecasts_eval_latest = create_qoq_evaluation_df(ifo_qoq_forecasts, qoq_latest_eval)
ifo_qoq_forecasts_eval_latest_collapsed = collapse_quarterly_prognosis(ifo_qoq_forecasts_eval_first)





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
# Necessary Function
# ==================================================================================================

def get_qoq_error_series(qoq_eval_df, save_path, file_name="qoq_errors_by_horizon.xlsx"):
    """
    For each forecast horizon (row label like Q0, Q1, ...), extract raw error vectors from columns
    ending in '_diff', align them as columns by forecast horizon, and compute standard evaluation
    metrics for each horizon. Store all resulting evaluation tables in a single Excel sheet.

    Returns the error table

    Parameters:
        qoq_eval_df (pd.DataFrame): A DataFrame containing columns with suffix "_diff" and rows indexed by forecast horizon (e.g., Q0, Q1, ...).
        save_path (str): The directory to save the output Excel file.
        file_name (str): Name of the Excel file.
    """

    # Ensure output directory exists
    os.makedirs(save_path, exist_ok=True)
    #show(qoq_eval_df)

    # Extract only columns ending with "_diff"
    diff_cols = [col for col in qoq_eval_df.columns if col.endswith("_diff")]
    diff_df = qoq_eval_df[diff_cols].copy()

    # Transpose error vectors: Q0-Qx become columns, forecast dates become rows
    error_by_horizon = diff_df.T
    error_by_horizon.columns.name = "forecast_horizon"

    # Initialize writer to collect multiple DataFrames into one sheet
    output_path = os.path.join(save_path, file_name)
    error_by_horizon.to_excel(output_path)

    return error_by_horizon 


def get_qoq_error_statistics_table(error_by_horizon, save_path, release_name, file_name="qoq_error_statistics_table.xlsx"):

    # Initialize writer to collect multiple DataFrames into one sheet
    output_path = os.path.join(save_path, file_name)
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:

        all_tables = []

        for horizon in error_by_horizon.columns:
            forecast_series = error_by_horizon[horizon]

            # Drop NA
            forecast_series = forecast_series.dropna()
            if forecast_series.empty:
                continue

            # Build evaluation table manually
            me = forecast_series.mean()
            mae = forecast_series.abs().mean()
            mse = (forecast_series ** 2).mean()
            rmse = np.sqrt(mse)
            se = forecast_series.std()
            n = forecast_series.count()

            eval_table = pd.DataFrame({
                "ME": [me],
                "MAE": [mae],
                "MSE": [mse],
                "RMSE": [rmse],
                "SE": [se],
                "N": [n]
            }, index=[horizon])

            all_tables.append(eval_table)

        # Concatenate all evaluation tables by horizon into one DataFrame
        full_error_measure_table = pd.concat(all_tables)
        full_error_measure_table.to_excel(writer, sheet_name=f"{release_name}")

    return full_error_measure_table




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



# --------------------------------------------------------------------------------------------------
# Evaluate the ifo qoq Forecasts
# --------------------------------------------------------------------------------------------------

## Evaluate against first releases
ifo_qoq_errors_first = get_qoq_error_series(
                                        ifo_qoq_forecasts_eval_first_collapsed, 
                                        ifo_qoq_error_path, 
                                        file_name="ifo_qoq_errors_first_eval.xlsx")

ifo_qoq_error_table_first = get_qoq_error_statistics_table(ifo_qoq_errors_first,
                                                           ifo_qoq_table_path, 'first_eval,',
                                                           'ifo_qoq_forecast_error_table_first_eval.xlsx')


## Evaluate against latest releases
ifo_qoq_errors_latest = get_qoq_error_series(
                                        ifo_qoq_forecasts_eval_latest_collapsed, 
                                        ifo_qoq_error_path, 
                                        file_name="ifo_qoq_errors_latest_eval.xlsx")

ifo_qoq_error_table_latest = get_qoq_error_statistics_table(ifo_qoq_errors_latest,
                                                           ifo_qoq_table_path, 'latest_eval,',
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
                                                        naive_qoq_table_path, 'first_eval,',
                                                        f'{name}_qoq_forecast_error_table_first_eval.xlsx')



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
                                                        naive_qoq_table_path, 'latest_eval,',
                                                        f'{name}_qoq_forecast_error_table_latest_eval.xlsx')











# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                  Visualizing Error Measures                                      #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

print("Visualizing error statistics ...  \n")



# ==================================================================================================
# Necessary Functions
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Error Time Series
# --------------------------------------------------------------------------------------------------






# --------------------------------------------------------------------------------------------------
# Error Bar Plots
# --------------------------------------------------------------------------------------------------






























# --------------------------------------------------------------------------------------------------
print(f" \n Quarterly Evaluation Module complete! \n",f"Find Result Graphs in {graph_folder} and \nResult Tables in {table_folder}\n")
# --------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#