
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        ifoCAST Evaluation
#
# Author:       Jan Ole Westphal
# Date:         2025-07
#
# Description:  Subprogram to evaluate the performance of the ifoCAST real time forecasting system.
# 
#               Runs all components from Data Processing to Output Processing and Visualizations.         
# ==================================================================================================
# --------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        Code begins here                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

print("\n Executing the ifoCAST Evaluation Module ... \n")


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
from dateutil.relativedelta import relativedelta

from itertools import product
from typing import Union, Dict, Optional


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
#                                          LOAD IN DATA                                            #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

"""
- ifo Forecasts QoQ
- ifo Forecast Release dates
- Evaluation Series
"""

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

#show(ifo_qoq_forecasts)


# --------------------------------------------------------------------------------------------------
# Load ifo forecast dates
# --------------------------------------------------------------------------------------------------

## Path
file_path_ifo_dates = os.path.join(wd, '0_0_Data', '0_Forecast_Inputs', 'ifoForecast_dates.xlsx' )

## Load 
ifo_forecast_dates = pd.read_excel(file_path_ifo_dates, index_col=0)


# Restrict to release date col
ifo_forecast_dates = ifo_forecast_dates[['Veroeffentlichungsdatum']].rename(
    columns={'Veroeffentlichungsdatum': 'forecast_date'})

#show(ifo_forecast_dates)



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
#                                        PROCESS ifoCASTS                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

"""
- Dynamically load the CSV files from the ifoCAST folder
- Dynamically extract the ifoCAST full time series
- Filter the closest prior date to everything in the ifo Forecast Release dates
- Build a df which can be used as input for the error measure and plotter functions
"""


# ==================================================================================================
# LOAD CSVs
# ==================================================================================================


def read_csvs_from_folder(folder_path: str, file_pattern: str = "*.csv") -> pd.DataFrame:
    """
    Read all CSV files from a folder and combine them into a single DataFrame.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing CSV files
    file_pattern : str, default "*.csv"
        Pattern to match CSV files (e.g., "*.csv", "ifoCAST_*.csv")
    
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all CSV data and forecast_target column
    """
    
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, file_pattern))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {folder_path} matching pattern {file_pattern}")
    
    # Sort files for consistent order
    csv_files.sort()
    
    combined_data = []
    column_names = None
    
    for i, file_path in enumerate(csv_files):
        # Extract filename without extension
        filename = os.path.basename(file_path)
        filename_no_ext = os.path.splitext(filename)[0]
        
        # Create forecast_target by stripping 'ifoCAST_' prefix and trailing spaces
        forecast_target = filename_no_ext.replace('ifoCast_', '').strip()
        
        # Read CSV
        if i == 0:
            # First file: read with header to get column names
            df = pd.read_csv(file_path, header=0, sep=';')
            column_names = df.columns.tolist()
        else:
            # Subsequent files: skip header and use established column names
            df = pd.read_csv(file_path, header=0, skiprows=1, names=column_names, sep=';')
        
        # Add forecast_target column
        df['forecast_target'] = forecast_target
        
        combined_data.append(df)
        
    
    # Combine all DataFrames
    final_df = pd.concat(combined_data, ignore_index=True)
    
    return final_df



# --------------------------------------------------------------------------------------------------
# Get full ifoCAST data
# --------------------------------------------------------------------------------------------------

# Get folder, call function
ifocast_folder = os.path.join(wd, '0_0_Data', '0_Forecast_Inputs', '2_ifoCAST')
ifocast_raw = read_csvs_from_folder(ifocast_folder)

print(f"Loaded ifoCAST forecasts from {ifocast_folder}. \n")

#show(ifocast_raw)



# --------------------------------------------------------------------------------------------------
# Extract Evaluation Series and Forecast Values
# --------------------------------------------------------------------------------------------------

## EVALUATION SERIES: Schnellmeldung T+45, DEtailmeldung T+55

## Extract Schnelllmeldung
T45_eval = ifocast_raw[['Datum', 'BIP-Schnellmeldung (T+45)']].copy()

# Rename and set index
T45_eval.rename(columns={'BIP-Schnellmeldung (T+45)': 'GDP_T45_report'}, inplace=True)
T45_eval.set_index(pd.to_datetime(T45_eval['Datum'], format='%d.%m.%Y'), inplace=True)

# Rescale
T45_eval.drop(columns='Datum', inplace=True)
T45_eval.dropna(inplace=True)


## Extract Detailmeldung
T55_eval = ifocast_raw[['Datum', 'BIP-Detailmeldung (T+55)']].copy()

# Rename and set index
T55_eval.rename(columns={'BIP-Detailmeldung (T+55)': 'GDP_T55_report'}, inplace=True)
T55_eval.set_index(pd.to_datetime(T55_eval['Datum'], format='%d.%m.%Y'), inplace=True)

# Rescale
T55_eval.drop(columns='Datum', inplace=True)
T55_eval.dropna(inplace=True)

#show(T45_eval)
#show(T55_eval)


## FORECAST VALUES: Subset the forecast cols, time cols and target col
ifocast_gdp = ifocast_raw[['Datum','BIP-Prognose', 'forecast_target']]

# Drop cols for which no GDP value is available
ifocast_gdp = ifocast_gdp.dropna(subset=['BIP-Prognose'])

#show(ifocast_gdp)




# ==================================================================================================
# Get backcast, nowcast and forecast series
# ==================================================================================================

def split_by_quarter_forecast(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split forecast data into three DataFrames based on quarter timing relative to target.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns ['Datum', 'BIP-Prognose', 'forecast_target']
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ifocast_Qminus1, ifocast_Q0, ifocast_Q1 DataFrames
    """
    
    # Initialize result lists
    qminus1_data = []
    q0_data = []
    q1_data = []
    
    # Convert Datum to datetime
    df = df.copy()
    df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y')
    
    # Process each row
    for _, row in df.iterrows():
        datum = row['Datum']
        forecast_value = row['BIP-Prognose']
        target_str = row['forecast_target']
        
        # Parse target quarter (format: year_Qx)
        try:
            year_str, quarter_str = target_str.split('_')
            target_year = int(year_str)
            target_quarter = int(quarter_str[1])  # Extract number from 'Qx'
        except (ValueError, IndexError):
            print(f"Warning: Could not parse forecast_target '{target_str}', skipping row")
            continue
        
        # Get quarter of the datum
        datum_quarter = (datum.month - 1) // 3 + 1
        datum_year = datum.year
        
        # Calculate quarter difference (positive means datum is after target)
        quarter_diff = (datum_year - target_year) * 4 + (datum_quarter - target_quarter)
        
        # Assign to appropriate list based on quarter difference
        if quarter_diff == 0:
            # Same quarter as target -> Q0
            q0_data.append({'date': datum, 'Q0': forecast_value,
                             'forecast_target': target_str})
        elif quarter_diff == -1:
            # One quarter before target -> Q1
            q1_data.append({'date': datum, 'Q1': forecast_value,
                             'forecast_target': target_str})
        elif quarter_diff == 1:
            # One quarter after target -> Qminus1
            qminus1_data.append({'date': datum, 'Qminus1': forecast_value,
                                  'forecast_target': target_str})
        # else: skip (more than 1 quarter difference)
    
    # Create DataFrames from lists
    ifocast_Qminus1 = pd.DataFrame(qminus1_data)
    ifocast_Q0 = pd.DataFrame(q0_data)
    ifocast_Q1 = pd.DataFrame(q1_data)

    # Set 'date to index
    for df in [ifocast_Qminus1, ifocast_Q0, ifocast_Q1]:
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index, format='%d.%m.%Y')
    
    return ifocast_Qminus1, ifocast_Q0, ifocast_Q1


## Get the three DataFrames
ifocast_Qminus1, ifocast_Q0, ifocast_Q1 = split_by_quarter_forecast(ifocast_gdp)

#show(ifocast_Qminus1)
#show(ifocast_Q0)
#show(ifocast_Q1)










# ==================================================================================================
# Create a joint forecast df
# ==================================================================================================

# Extract the first column from each DataFrame
def horizon_merger(df1, df2, df3=None) -> pd.DataFrame:
    """
    Merge three DataFrames on index, allowing for non-matching indices.
    
    Parameters:
    -----------
    df1, df2, df3 : pd.DataFrame
        DataFrames to merge
    
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with columns from all three input DataFrames
    """
    
    col_Qm1 = df1.iloc[:, [0]]
    col_Q0 = df2.iloc[:, [0]] if df2 is not None else None
    col_Q1 = df3.iloc[:, [0]] if df3 is not None else None

    
    # Merge them on index, allowing for non-matching indices
    merged_df = pd.concat([col_Qm1, col_Q0, col_Q1], axis=1)
    
    return merged_df



## Merge them on index, allowing for non-matching indices

# Full merge
ifoCAst_Qm1_Q0_Q1_full = horizon_merger(ifocast_Qminus1, ifocast_Q0, ifocast_Q1)

#show(ifoCAst_Qm1_Q0_Q1_full)



# --------------------------------------------------------------------------------------------------
# Filter ifoCAST dates towards the ifo Forecast Release dates
# --------------------------------------------------------------------------------------------------

# Define a filter function
def filter_by_forecast_dates(df_1, date_df=ifo_forecast_dates) -> pd.DataFrame:
    """
    For each date in date_df['forecast_date'], return the closest available past
    values from df_1 (index = datetime), one per column. If NA, search further back.
    """


    # THe Nowcast can't be from the future: Truncate to today's date
    date_df = date_df[date_df['forecast_date'] <= pd.Timestamp.today()]

    # Start the filtering process
    result_rows = []

    for forecast_date in date_df['forecast_date']:
        # Get all df_1 dates before or equal to the forecast_date
        valid_dates = df_1.index[df_1.index <= forecast_date]

        if valid_dates.empty:
            # No data available before forecast_date
            result_rows.append(pd.Series({col: pd.NA for col in df_1.columns}, name=forecast_date))
            continue

        row_values = {}
        for col in df_1.columns:
            # Look backwards for this column to find the first non-NA value
            for dt in reversed(valid_dates): # valid_dates is iterated over for each ifo-release
                val = df_1.at[dt, col]
                if pd.notna(val):
                    row_values[col] = val
                    break
            else:
                # No non-NA found
                row_values[col] = pd.NA

        result_rows.append(pd.Series(row_values, name=forecast_date))

    filtered_df = pd.DataFrame(result_rows)
    filtered_df.index.name = 'ifo_forecast_date'

    # Drop rows where the ifoCast was unavailable
    filtered_df.dropna(how='all', inplace=True)

    return filtered_df



## Aply to ifocast df
ifoCAst_Qm1_Q0_Q1_filtered = filter_by_forecast_dates(ifoCAst_Qm1_Q0_Q1_full)

# show(ifoCAst_Qm1_Q0_Q1_full)
# show(ifoCAst_Qm1_Q0_Q1_filtered)



















# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        RUN EVALUATIONS                                           #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

"""
- Get the Error Series and tables: full and filtered
- Plot the error series and the other statistics
"""



# ==================================================================================================
# Get evaluation Objects
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Ifo QoQ Forecasts
# --------------------------------------------------------------------------------------------------


## Evalaute against First Release
ifo_qoq_forecasts_eval_first = create_qoq_evaluation_df(ifo_qoq_forecasts, qoq_first_eval)
ifo_qoq_forecasts_eval_first_collapsed = collapse_quarterly_prognosis(ifo_qoq_forecasts_eval_first)

#show(ifo_qoq_forecasts_eval_first_collapsed)

## Get error series
#Evaluate against first releases
ifo_qoq_errors_first = get_qoq_error_series(ifo_qoq_forecasts_eval_first_collapsed) 





# ==================================================================================================
# Evaluate ifoCAST
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Evaluation function
# --------------------------------------------------------------------------------------------------


def get_ifoCAST_differences(forecast_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rescales forecast_df and calls additional processing functions.
    
    Args:
        forecast_df: DataFrame with datetime index and columns ['Qminus1', 'Q0', 'Q1']
                    representing forecasts for prior, current, and following quarters
        eval_df: DataFrame for evaluation/comparison purposes
        
    Returns:
        pd.DataFrame: Processed forecast differences
    """
    
    # Copy to precent unexpected changes
    forecast_df = forecast_df.copy()
    eval_df = eval_df.copy()

    """
    NOTE: Check this data processing for the full forecast data, there seems to be a bug
    """

    #show(forecast_df)

    # Step I: Rescale forecast_df
    
    # Create a list to store the rescaled data
    rescaled_data = []
    
    # Process each original column (which will become a column in the new df)
    for col_date in forecast_df.index:
        col_name = col_date.strftime('%Y-%m-%d')  # Use date as column name
        
        # For each forecast horizon in the original row
        for quarter_offset, quarter_name in [(-1, 'Qminus1'), (0, 'Q0'), (1, 'Q1')]:
            # Calculate the target date by shifting quarters
            target_date = col_date + pd.DateOffset(months=3*quarter_offset)
            
            # Get the forecast value
            forecast_value = forecast_df.loc[col_date, quarter_name]
            
            rescaled_data.append({
                'target_date': target_date,
                'forecast_date': col_name,
                'forecast_value': forecast_value
            })
    
    # Convert to DataFrame
    rescaled_df = pd.DataFrame(rescaled_data)
    
    # Group by target_date and forecast_date, taking the mean to handle duplicates
    # (merging rows with different dates within the same quarter)
    rescaled_df = rescaled_df.groupby(['target_date', 'forecast_date']).agg({
        'forecast_value': 'mean'
    }).reset_index()
    
    # Pivot to get the final structure: target_date as index, forecast_dates as columns
    final_df = rescaled_df.pivot(index='target_date', columns='forecast_date', values='forecast_value')
    
    # Ensure the index is datetime
    final_df.index = pd.to_datetime(final_df.index)
    final_df.index.name = 'target_date'


    # Collapse rows from the same quarter using the first date as the index
    final_df = final_df.groupby(final_df.index.to_period('Q')).first()
    final_df.index = final_df.index.to_timestamp()

    #show(final_df)


    
    # Step II: Process the rescaled DataFrame with additional functions
    
    # Placeholder for first function call
    qoq_evaluation_df = create_qoq_evaluation_df(final_df, eval_df)
    
    # Placeholder for second function call  
    final_result = collapse_quarterly_prognosis(qoq_evaluation_df, Ifocast_mode=True) # Adjusts for the fact that initial Qminus1 is missing

    #show(final_result)
    
    return final_result













# --------------------------------------------------------------------------------------------------
# Filtered ifoCAST
# --------------------------------------------------------------------------------------------------

# Define names
ifocast_filtered_eval_df_names = ['ifoCAst_Qm1_Q0_Q1_filtered_first',
                          'ifoCAst_Qm1_Q0_Q1_filtered_latest',
                          'ifoCAst_Qm1_Q0_Q1_filtered_fT45',
                          'ifoCAst_Qm1_Q0_Q1_filtered_fT55']

# Eval against first, latest, T45 and T55
eval_dfs = [qoq_first_eval, qoq_latest_eval, T45_eval, T55_eval]

# Loop to create the eval dfs
fitered_eval_dfs = {}

for name, eval_df in zip(ifocast_filtered_eval_df_names, eval_dfs):

    fitered_eval_dfs[name] = get_ifoCAST_differences(ifoCAst_Qm1_Q0_Q1_filtered, eval_df)



# --------------------------------------------------------------------------------------------------
# Full ifoCAST
# --------------------------------------------------------------------------------------------------

# Define names
ifocast_full_eval_df_names = ['ifoCAst_Qm1_Q0_Q1_full_first',
                          'ifoCAst_Qm1_Q0_Q1_full_latest',
                          'ifoCAst_Qm1_Q0_Q1_full_T45',
                          'ifoCAst_Qm1_Q0_Q1_full_T55']


# Loop to create the eval dfs
full_eval_dfs = {}

for name, eval_df in zip(ifocast_full_eval_df_names, eval_dfs):

    show(ifoCAst_Qm1_Q0_Q1_full)
    full_eval_dfs[name] = get_ifoCAST_differences(ifoCAst_Qm1_Q0_Q1_full, eval_df)
    show(full_eval_dfs[name])




# --------------------------------------------------------------------------------------------------
# Naive Forecats
# --------------------------------------------------------------------------------------------------
"""
Could be added here as well
"""







# ==================================================================================================
# Get Error Statistics and Tables
# ==================================================================================================


# --------------------------------------------------------------------------------------------------
# ifo QoQ Forecasts filtered to ifoCAST dates
# --------------------------------------------------------------------------------------------------
"""
Add this
"""



# --------------------------------------------------------------------------------------------------
# Filtered ifoCAST
# --------------------------------------------------------------------------------------------------

## Filepaths for error series and tables
ifoCast_filtered_error_path = os.path.join(wd, '0_1_Output_Data', '5_ifoCAST_error_series_matched')
os.makedirs(ifoCast_filtered_error_path, exist_ok=True)

ifoCAST_filtered_table_path = os.path.join(table_folder, '4_ifoCAST_evaluations_matched')
os.makedirs(ifoCAST_filtered_table_path, exist_ok=True)


## Evaluation and Table Creation
ifoCast_filtered_error_series_names = ['ifoCAst_errors_filtered_first',
                          'ifoCAst_errors_filtered_latest',
                          'ifoCAst_errors_filtered_T45',
                          'ifoCAst_errors_filtered_T55']

ifoCast_filtered_table_names = ['ifoCAst_error_tables_filtered_first',
                          'ifoCAst_error_tables_filtered_latest',
                          'ifoCAst_error_tables_filtered_T45',
                          'ifoCAst_error_tables_filtered_T55']


## Create error series and tables
for error_name, table_name, eval_key in zip(ifoCast_filtered_error_series_names, 
                                            ifoCast_filtered_table_names, ifocast_filtered_eval_df_names):
    
    # Get the error series
    error_series = get_qoq_error_series(fitered_eval_dfs[eval_key], ifoCast_filtered_error_path, file_name = f"{error_name}.xlsx")

    # Create the corresponding error table
    get_qoq_error_statistics_table(error_series, 
                                   error_name.split('_')[-1], 
                                   ifoCAST_filtered_table_path, 
                                   f"{table_name}.xlsx")



# --------------------------------------------------------------------------------------------------
# Full ifoCAST
# --------------------------------------------------------------------------------------------------

## Filepaths for error series and tables
ifoCast_full_error_path = os.path.join(wd, '0_1_Output_Data', '6_ifoCAST_error_series_full')
os.makedirs(ifoCast_full_error_path, exist_ok=True)

# Error Tables
ifoCAST_full_table_path = os.path.join(table_folder, '4_ifoCAST_evaluations_full')
os.makedirs(ifoCAST_full_table_path, exist_ok=True)


## Evaluation and Table Creation
ifoCast_full_error_series_names = ['ifoCAst_errors_full_first',
                          'ifoCAst_errors_full_latest',
                          'ifoCAst_errors_full_T45',
                          'ifoCAst_errors_full_T55']

ifoCast_full_table_names = ['ifoCAst_error_tables_full_first',
                          'ifoCAst_error_tables_full_latest',
                          'ifoCAst_error_tables_full_T45',
                          'ifoCAst_error_tables_full_T55']


## Create error series and tables
for error_name, table_name, eval_key in zip(ifoCast_full_error_series_names, 
                                            ifoCast_full_table_names, ifocast_full_eval_df_names):
    
    # Get the error series
    error_series = get_qoq_error_series(full_eval_dfs[eval_key], ifoCast_full_error_path, file_name = f"{error_name}.xlsx")
    show(error_series)

    # Create the corresponding error table
    get_qoq_error_statistics_table(error_series, 
                                   error_name.split('_')[-1], 
                                   ifoCAST_full_table_path, 
                                   f"{table_name}.xlsx")























# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                     GET VISUALIZATIONS                                           #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


print("Visualizing error statistics ...  \n")






# ==================================================================================================
#                                 PLOT AND SAVE RESULTS OF INTEREST
# ==================================================================================================


# --------------------------------------------------------------------------------------------------
# Error Time Series
# --------------------------------------------------------------------------------------------------

## Savepaths

# ifo
first_eval_error_series_path_ifo = os.path.join(graph_folder, '1_QoQ_Error_Series', '0_First_Evaluation_ifo')
first_eval_error_series_path = os.path.join(graph_folder, '1_QoQ_Error_Series', '0_First_Evaluation_joint')
latest_eval_error_series_path = os.path.join(graph_folder, '1_QoQ_Error_Series', '1_Latest_Evaluation')

os.makedirs(first_eval_error_series_path_ifo, exist_ok=True)
os.makedirs(first_eval_error_series_path, exist_ok=True)
os.makedirs(latest_eval_error_series_path, exist_ok=True)

# Naive


## Create and Save Plots
"""
plot_forecast_timeseries(*args, df_eval=None, title_prefix=None, figsize=(12, 8), 
                             show=False, save_path=None, save_name_prefix=None, select_quarters=None)
"""
"""
plot_forecast_timeseries(ifo_qoq_forecasts_eval_first, 
                         df_eval=qoq_first_eval, title_prefix=None, figsize=(12, 8), linestyle=None,
                             show=False, 
                             save_path=first_eval_error_series_path_ifo, save_name_prefix='ifo_First_Eval_', select_quarters=None)

plot_forecast_timeseries(ifo_qoq_forecasts_eval_first, naive_qoq_first_eval_dfs, 
                         df_eval=qoq_first_eval, title_prefix=None, figsize=(12, 8), linestyle=None,
                             show=False, 
                             save_path=first_eval_error_series_path, save_name_prefix='First_Eval_', select_quarters=None)


plot_forecast_timeseries(ifo_qoq_forecasts_eval_latest, naive_qoq_latest_eval_dfs, 
                         df_eval=qoq_latest_eval, title_prefix=None, figsize=(12, 8), linestyle=None,
                             show=False, 
                             save_path=latest_eval_error_series_path, save_name_prefix='Latest_Eval_', select_quarters=None)





# --------------------------------------------------------------------------------------------------
# Error Scatter Plots
# --------------------------------------------------------------------------------------------------

## Savepaths
first_eval_error_line_path = os.path.join(graph_folder, '1_QoQ_Error_Scatter', '0_First_Evaluation')
latest_eval_error_line_path = os.path.join(graph_folder, '1_QoQ_Error_Scatter', '1_Latest_Evaluation')

os.makedirs(first_eval_error_line_path, exist_ok=True)
os.makedirs(latest_eval_error_line_path, exist_ok=True)


## Create and Save Plots

# First Evaluation
plot_error_lines(ifo_qoq_errors_first, 
                  show=False, 
                  save_path=first_eval_error_line_path,
                  save_name=f'ifo_QoQ_First_Eval_Error_Scatter.png')

plot_error_lines(ifo_qoq_errors_first, naive_qoq_first_eval_error_series_dict,
                  show=False, 
                  save_path=first_eval_error_line_path,
                  save_name=f'Joint_QoQ_First_Eval_Error_Scatter.png')
                 

# Latest Evaluation
plot_error_lines(ifo_qoq_errors_latest, 
                  show=False, 
                  save_path=latest_eval_error_line_path,
                  save_name=f'ifo_QoQ_Latest_Eval_Error_Scatter.png')

plot_error_lines(ifo_qoq_errors_latest, naive_qoq_latest_eval_error_series_dict,
                  show=False, 
                  save_path=latest_eval_error_line_path,
                  save_name=f'Joint_QoQ_Latest_Eval_Error_Scatter.png')





# --------------------------------------------------------------------------------------------------
# Error Bar Plots
# --------------------------------------------------------------------------------------------------


## Savepaths
first_eval_error_bars_path = os.path.join(graph_folder, '1_QoQ_Error_Bars', '0_First_Evaluation')
latest_eval_error_bars_path = os.path.join(graph_folder, '1_QoQ_Error_Bars', '1_Latest_Evaluation')

os.makedirs(first_eval_error_bars_path, exist_ok=True)
os.makedirs(latest_eval_error_bars_path, exist_ok=True)


## Create and Save Plots

# First Evaluation
for metric in ['ME', 'MAE', 'MSE', 'RMSE', 'SE']:
    plot_quarterly_metrics(ifo_qoq_error_table_first, naive_qoq_first_eval_error_tables_dict, 
                           
                        metric_col=metric,
                        scale_by_n=False, show=False, 

                        save_path=first_eval_error_bars_path,
                        save_name=f'Joint_Quarterly_{metric}_first_eval.png'
                            )


# Latest Evaluation
for metric in ['ME', 'MAE', 'MSE', 'RMSE', 'SE']:
    plot_quarterly_metrics(ifo_qoq_error_table_latest, naive_qoq_latest_eval_error_tables_dict, 
                           
                        metric_col=metric,
                        scale_by_n=False, show=False, 

                        save_path=latest_eval_error_bars_path,
                        save_name=f'Joint_Quarterly_{metric}_latest_eval.png'
                            )

"""









# --------------------------------------------------------------------------------------------------
print(f" \n ifoCAST Evaluation Module complete! \n",f"Find Result Graphs in {graph_folder} and \nResult Tables in {table_folder}\n")
# --------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#