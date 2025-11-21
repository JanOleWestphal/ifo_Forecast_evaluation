
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

# Select whether to evaluate GVA predictions
run_gva_evaluation = settings.run_gva_evaluation

# Not used at the moment
"""
# Define the horizon of first releases which should be evaluated: available from 1995-Q3 onwards
first_release_lower_limit_year = settings.first_release_lower_limit_year
first_release_lower_limit_quarter = settings.first_release_lower_limit_quarter

first_release_upper_limit_year = settings.first_release_upper_limit_year
first_release_upper_limit_quarter = settings.first_release_upper_limit_quarter
"""




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
- Evaluation Series (GDP and GVA)
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

## Make sure to truncate the release dates such that they don't exceed available ifo-forecast to avoid bugs

# Get last col name and parse as datetime
last_col = ifo_qoq_forecasts.columns[-1]
last_col_date = pd.to_datetime(last_col)
#show(ifo_forecast_dates)
#print(last_col_date)

# Add 30 days buffer: release date and Datenstand do not always coincide
threshold_date = last_col_date + pd.Timedelta(days=30)

# Apply truncation
ifo_forecast_dates.loc[ifo_forecast_dates['forecast_date'] > threshold_date, 'forecast_date'] = pd.NaT

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
    #show(final_df)
    
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
# BONUS ANALYSIS: get latest reports of QoQ prior to the official BB releases
# --------------------------------------------------------------------------------------------------

# Keep only the last entry per unique forecast_target
last_rows = ifocast_raw.sort_values('forecast_target').groupby('forecast_target', as_index=False).last()

# Select and rename relevant columns
ifocast_last_values = last_rows[['forecast_target', 'BIP-Prognose']].copy()
ifocast_last_values.rename(columns={'BIP-Prognose': 'Q0'}, inplace=True)

# Convert 'forecast_target' (e.g., '2024_Q1') to datetime
periods = pd.PeriodIndex(ifocast_last_values['forecast_target'].str.replace('_', ''), freq='Q')
ifocast_last_values['forecast_target'] = periods.to_timestamp(how='start')

# Set datetime index
ifocast_last_values.set_index('forecast_target', inplace=True)


#show(ifocast_last_values)



# --------------------------------------------------------------------------------------------------
# Restructure ifoCAST data
# --------------------------------------------------------------------------------------------------

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

#show(ifoCAst_Qm1_Q0_Q1_full)
#show(ifoCAst_Qm1_Q0_Q1_filtered)













# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                      Get Evaluation Data                                         #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# --------------------------------------------------------------------------------------------------
# Load Bundesbank GDP and GVA Evaluation Data
# --------------------------------------------------------------------------------------------------

# Set paths
eval_path_gdp = os.path.join(wd, '0_0_Data', '2_Processed_Data', '2_GDP_Evaluation_series')
eval_path_gva = os.path.join(wd, '0_0_Data', '2_Processed_Data', '2_GVA_Evaluation_series')

# Create dictionaries to hold the evaluation data for GDP and GVA
qoq_first_eval_dict = {}
qoq_latest_eval_dict = {}
qoq_rev_dict = {}

for data_name, eval_path in zip(['GDP', 'GVA'], [eval_path_gdp, eval_path_gva]):


    ## First releases
    qoq_path_first = os.path.join(eval_path, f'first_release_qoq_{data_name}.xlsx')
    qoq_first_eval_dict[data_name] = pd.read_excel(qoq_path_first, index_col=0)

    ## Latest releases
    qoq_path_latest = os.path.join(eval_path, f'latest_release_qoq_{data_name}.xlsx')
    qoq_latest_eval_dict[data_name] = pd.read_excel(qoq_path_latest, index_col=0)

    ## Revision
    qoq_path_rev = os.path.join(eval_path, f'revision_qoq_{data_name}.xlsx')
    qoq_rev_dict[data_name] = pd.read_excel(qoq_path_rev, index_col=0)



# --------------------------------------------------------------------------------------------------
# Extract Evaluation Series from ifoCAST CSVs
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












# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                       EVALUATION PIPELINE                                        #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

"""
- Get the Error Series and tables: full and filtered
- Plot the error series and the other statistics
"""



# --------------------------------------------------------------------------------------------------
# Evaluation helperfunction
# --------------------------------------------------------------------------------------------------

def get_ifoCAST_differences(forecast_df: pd.DataFrame, eval_df: pd.DataFrame, full_input=False) -> pd.DataFrame:
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

    #show(forecast_df)

    # Step I: Rescale forecast_df
    
    # Create a list to store the rescaled data
    rescaled_data = []
    
    # Process each original column (which will become a column in the new df)
    for col_date in forecast_df.index:
        col_name = col_date.strftime('%Y-%m-%d')  # Use date as column name
        
        # For each forecast horizon in the original row
        for quarter_offset, quarter_name in [(-1, 'Qminus1'), (0, 'Q0'), (1, 'Q1')]:

            if quarter_name in forecast_df.columns:
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
    
    # Get the evaluation DataFrame
    qoq_evaluation_df = create_qoq_evaluation_df(final_df, eval_df)

    #show(qoq_evaluation_df)
    
    # Collapse into quarterly prognosis structure

    if full_input:
        final_result = collapse_full_ifocast(qoq_evaluation_df)

    else:
        final_result = collapse_quarterly_prognosis(qoq_evaluation_df, Ifocast_mode=True) # Adjusts for the fact that initial Qminus1 is missing

    #show(final_result)
    
    return final_result






# =================================================================================================#
#                                   EVALUATION PIPLINE FUNCTION                                    #
# =================================================================================================#


def ifocast_eval_pipeline(ifocast_df_filtered= ifoCAst_Qm1_Q0_Q1_filtered, 
                          ifocast_df_full= ifoCAst_Qm1_Q0_Q1_full, 
                          ifocast_last=ifocast_last_values, 
                          subset_str="",
                          
                          naive_qoq_dfs_dict=naive_qoq_dfs_dict,
                          ifo_qoq_forecasts=ifo_qoq_forecasts,

                          gdp_mode = True,
                          qoq_first_eval_dict = qoq_first_eval_dict, 
                          qoq_latest_eval_dict = qoq_latest_eval_dict, 
                          T45_eval = T45_eval, 
                          T55_eval = T55_eval):



    # ==================================================================================================
    #                                     GET EVALUATION DATAFRAMES   
    # ==================================================================================================

    # --------------------------------------------------------------------------------------------------
    # Set naming for the GVA case
    # --------------------------------------------------------------------------------------------------

    if not gdp_mode:
        gva_string = '_GVA'
    else:
        gva_string = ''



    # --------------------------------------------------------------------------------------------------
    # Filtered ifoCAST
    # --------------------------------------------------------------------------------------------------

    # Define names
    ifocast_filtered_eval_df_names = [f'ifoCAst_Qm1_Q0_Q1_filtered_first{gva_string}',
                            f'ifoCAst_Qm1_Q0_Q1_filtered_latest{gva_string}',
                            f'ifoCAst_Qm1_Q0_Q1_filtered_T45{gva_string}',
                            f'ifoCAst_Qm1_Q0_Q1_filtered_T55{gva_string}']
    
    # Exclude T45 and T55 for GVA
    if not gdp_mode:
        ifocast_filtered_eval_df_names = ifocast_filtered_eval_df_names[:2] 


    # Eval against first, latest, T45 and T55
    if gdp_mode:
        eval_dfs = [qoq_first_eval_dict['GDP'], qoq_latest_eval_dict['GDP'], T45_eval, T55_eval]

    else:
        eval_dfs = [qoq_first_eval_dict['GVA'], qoq_latest_eval_dict['GVA']]


    # Loop to create the eval dfs
    fitered_eval_dfs = {}

    for name, eval_df in zip(ifocast_filtered_eval_df_names, eval_dfs):

        fitered_eval_dfs[name] = get_ifoCAST_differences(ifocast_df_filtered, eval_df, full_input=False)



    # --------------------------------------------------------------------------------------------------
    # Full ifoCAST
    # --------------------------------------------------------------------------------------------------

    # Define names
    ifocast_full_eval_df_names = [f'ifoCAst_Qm1_Q0_Q1_full_first{gva_string}',
                            f'ifoCAst_Qm1_Q0_Q1_full_latest{gva_string}',
                            f'ifoCAst_Qm1_Q0_Q1_full_T45{gva_string}',
                            f'ifoCAst_Qm1_Q0_Q1_full_T55{gva_string}']
    
    # Exclude T45 and T55 for GVA
    if not gdp_mode:
        ifocast_full_eval_df_names = ifocast_full_eval_df_names[:2] 


    # Loop to create the eval dfs
    full_eval_dfs = {}

    for name, eval_df in zip(ifocast_full_eval_df_names, eval_dfs):

        #show(ifoCAst_Qm1_Q0_Q1_full)
        full_eval_dfs[name] = get_ifoCAST_differences(ifocast_df_full, eval_df, full_input=True)
        #show(full_eval_dfs[name])




    # --------------------------------------------------------------------------------------------------
    # ifoCAST last values before release
    # --------------------------------------------------------------------------------------------------

    # Define names
    ifocast_last_eval_df_names = [f'ifoCAst_last_rep_first{gva_string}',
                            f'ifoCAst_last_rep_latest{gva_string}',
                            f'ifoCAst_last_rep_T45{gva_string}',
                            f'ifoCAst_last_rep_T55{gva_string}']
    
    # Exclude T45 and T55 for GVA
    if not gdp_mode:
        ifocast_last_eval_df_names = ifocast_last_eval_df_names[:2] 

    # Loop to create the eval dfs
    last_rep_eval_dfs = {}

    for name, eval_df in zip(ifocast_last_eval_df_names, eval_dfs):

        last_rep_eval_dfs[name] = get_ifoCAST_differences(ifocast_last, eval_df, full_input=False)






    # --------------------------------------------------------------------------------------------------
    # Naive QoQ Forecasts matched
    # --------------------------------------------------------------------------------------------------

    # Dict to store the results
    naive_qoq_eval_dfs = {}

    # Loop through all naive forecast variants
    for name, naive_df in naive_qoq_dfs_dict.items():

        # Filter for ifoCAST dates
        naive_df_filtered = naive_df.loc[:, 
            naive_df.columns.to_series().dt.to_period('Q').isin(
                ifocast_df_filtered.T.columns.to_series().dt.to_period('Q')
            )
        ]

        # Loop over evaluation horizons
        for eval_name, eval_df in zip(['first', 'latest', 'T45', 'T55'], eval_dfs):

            key = f"{name}_matched_{eval_name}"

            # Create and collapse evaluation DataFrame
            matched_df = create_qoq_evaluation_df(naive_df_filtered, eval_df)
            collapsed_df = collapse_quarterly_prognosis(matched_df)

            # Store in dictionary
            naive_qoq_eval_dfs[key] = collapsed_df








    # --------------------------------------------------------------------------------------------------
    # Ifo QoQ Forecasts matched
    # --------------------------------------------------------------------------------------------------


    ## Filter for ifoCAST dates
    #show(ifo_qoq_forecasts)
    #show(ifocast_df_filtered)
    ifo_qoq_forecasts = ifo_qoq_forecasts.loc[:, 
        ifo_qoq_forecasts.columns.to_series().dt.to_period('Q').isin(
                                    ifocast_df_filtered.T.columns.to_series().dt.to_period('Q'))]
    #show(ifo_qoq_forecasts)

    #show(ifo_qoq_forecasts)

    ## Define names
    ifo_qoq_matched_eval_df_names = [f'ifo_qoq_matched_first{gva_string}',
                            f'ifo_qoq_matched_latest{gva_string}',
                            f'ifo_qoq_matched_T45',
                            f'ifo_qoq_matched_T55']
    
    # Exclude T45 and T55 for GVA
    if not gdp_mode:
        ifo_qoq_matched_eval_df_names = ifo_qoq_matched_eval_df_names[:2] 


    # Loop to create the eval dfs
    ifo_qoq_eval_dfs = {}

    for name, eval_df in zip(ifo_qoq_matched_eval_df_names, eval_dfs):

        # Create the evaluation DataFrames
        ifo_qoq_eval_dfs[name] = create_qoq_evaluation_df(ifo_qoq_forecasts, eval_df)

        # Collapse them into the desired structure
        ifo_qoq_eval_dfs[name] = collapse_quarterly_prognosis(ifo_qoq_eval_dfs[name])












    # ==================================================================================================
    # Get Error Statistics and Tables
    # ==================================================================================================

    # --------------------------------------------------------------------------------------------------
    # Naive QoQ Forecasts → Error Series and Tables
    # --------------------------------------------------------------------------------------------------

    # Paths
    naive_qoq_matched_error_path = os.path.join(wd, '0_1_Output_Data', f'5_naive_qoq_error_series{gva_string}_matched_to_ifoCAST{subset_str}')
    os.makedirs(naive_qoq_matched_error_path, exist_ok=True)

    naive_qoq_matched_table_path = os.path.join(table_folder, f'4_naive_QoQ{gva_string}_matched_to_ifoCAST{subset_str}')
    os.makedirs(naive_qoq_matched_table_path, exist_ok=True)


    ## Clear
    if settings.clear_result_folders:

        for folder in [naive_qoq_matched_error_path, naive_qoq_matched_table_path]:
            
            folder_clear(folder)


    # Dicts to store outputs
    naive_qoq_matched_error_series_dict = {}
    naive_qoq_matched_error_table_dict  = {}

    # Loop over each entry in naive_qoq_eval_dfs
    for eval_key, df in naive_qoq_eval_dfs.items():
        # eval_key example: 'naive_mean_matched_first'
        parts = eval_key.split('_')
        horizon = parts[-1]                       # 'first', 'latest', 'T45', or 'T55'
        prefix  = '_'.join(parts[:-1])            # 'naive_mean_matched'
        
        # Build standardized names
        error_name = f"{prefix}_errors_{horizon}{subset_str}"         # e.g. 'naive_mean_matched_errors_first'
        table_name = f"{prefix}_error_tables_{horizon}{subset_str}"   # e.g. 'naive_mean_matched_error_tables_first'
        
        # 1) Generate and save the error series
        error_series = get_qoq_error_series(
            df,
            naive_qoq_matched_error_path,
            file_name=f"{error_name}.xlsx"
        )
        naive_qoq_matched_error_series_dict[error_name] = error_series

        # 2) Generate and save the statistics table
        error_table = get_qoq_error_statistics_table(
            error_series,
            horizon,
            naive_qoq_matched_table_path,
            f"{table_name}.xlsx"
        )
        naive_qoq_matched_error_table_dict[table_name] = error_table





    # --------------------------------------------------------------------------------------------------
    # ifo QoQ Forecasts filtered to ifoCAST dates
    # --------------------------------------------------------------------------------------------------

    ## Filepaths for error series and tables
    ifo_qoq_matched_error_path = os.path.join(wd, '0_1_Output_Data', f'5_ifo_qoq_error_series{gva_string}_matched_to_ifoCAST{subset_str}')
    os.makedirs(ifo_qoq_matched_error_path, exist_ok=True)

    # Error Tables
    ifo_qoq_matched_table_path = os.path.join(table_folder, f'4_ifo_QoQ{gva_string}_matched_to_ifoCAST{subset_str}')
    os.makedirs(ifo_qoq_matched_table_path, exist_ok=True)

    ## Clear
    if settings.clear_result_folders:

        for folder in [ifo_qoq_matched_error_path, ifo_qoq_matched_table_path]:
            
            folder_clear(folder)


    ## Evaluation and Table Creation
    ifo_qoq_matched_error_series_names = [f'ifo_qoq_matched_errors_first{subset_str}{gva_string}',
                            f'ifo_qoq_matched_errors_latest{subset_str}{gva_string}',
                            f'ifo_qoq_matched_errors_T45{subset_str}',
                            f'ifo_qoq_matched_errors_T55{subset_str}']

    ifo_qoq_matched_table_names = [f'ifo_qoq_matched_error_tables_first{subset_str}{gva_string}',
                            f'ifo_qoq_matched_error_tables_latest{subset_str}{gva_string}',
                            f'ifo_qoq_matched_error_tables_T45{subset_str}',
                            f'ifo_qoq_matched_error_tables_T55{subset_str}']
    
    # Exclude T45 and T55 for GVA
    if not gdp_mode:
        for df in [ifo_qoq_matched_error_series_names, ifo_qoq_matched_table_names]:
            del df[2:]  # Keep only the first two elements




    ## Create error series and tables

    # Dicts to store outputs
    ifo_qoq_matched_error_series_dict = {}
    ifo_qoq_matched_error_table_dict  = {}

    # Create error series and tables
    for error_name, table_name, eval_key in zip(
        ifo_qoq_matched_error_series_names, 
        ifo_qoq_matched_table_names, 
        ifo_qoq_matched_eval_df_names
    ):
        # Get the error series and store it
        error_series = get_qoq_error_series(
            ifo_qoq_eval_dfs[eval_key], 
            ifo_qoq_matched_error_path, 
            file_name=f"{error_name}.xlsx"
        )
        ifo_qoq_matched_error_series_dict[error_name] = error_series

        # Get the statistics table and store it
        error_table = get_qoq_error_statistics_table(
            error_series, 
            error_name.split('_')[-1], 
            ifo_qoq_matched_table_path, 
            f"{table_name}.xlsx"
        )
        ifo_qoq_matched_error_table_dict[table_name] = error_table






    # --------------------------------------------------------------------------------------------------
    # Filtered ifoCAST
    # --------------------------------------------------------------------------------------------------

    ## Filepaths for error series and tables
    ifoCast_filtered_error_path = os.path.join(wd, '0_1_Output_Data', f'5_ifoCAST_error_series{gva_string}_matched{subset_str}')
    os.makedirs(ifoCast_filtered_error_path, exist_ok=True)

    ifoCAST_filtered_table_path = os.path.join(table_folder, f'4_ifoCAST_evaluations{gva_string}_matched{subset_str}')
    os.makedirs(ifoCAST_filtered_table_path, exist_ok=True)

    ## Clear
    if settings.clear_result_folders:

        for folder in [ifoCast_filtered_error_path, ifoCast_filtered_error_path]:
            
            folder_clear(folder)


    ## Evaluation and Table Creation
    ifoCast_filtered_error_series_names = [f'ifoCAst_errors_filtered_first{subset_str}{gva_string}',
                            f'ifoCAst_errors_filtered_latest{subset_str}{gva_string}',
                            f'ifoCAst_errors_filtered_T45{subset_str}',
                            f'ifoCAst_errors_filtered_T55{subset_str}']

    ifoCast_filtered_table_names = [f'ifoCAst_error_tables_filtered_first{subset_str}{gva_string}',
                            f'ifoCAst_error_tables_filtered_latest{subset_str}{gva_string}',
                            f'ifoCAst_error_tables_filtered_T45{subset_str}',
                            f'ifoCAst_error_tables_filtered_T55{subset_str}']
    
    # Exclude T45 and T55 for GVA
    if not gdp_mode:
        for df in [ifoCast_filtered_error_series_names, ifoCast_filtered_table_names]:
            del df[2:]  # Keep only the first two elements



    # Dicts to store outputs
    ifoCast_filtered_error_series_dict = {}
    ifoCast_filtered_error_table_dict  = {}

    ## Create error series and tables
    for error_name, table_name, eval_key in zip(ifoCast_filtered_error_series_names, 
                                ifoCast_filtered_table_names, ifocast_filtered_eval_df_names):

        # Get the error series and store it
        error_series = get_qoq_error_series(
            fitered_eval_dfs[eval_key], 
            ifoCast_filtered_error_path, 
            file_name=f"{error_name}.xlsx"
        )

        ifoCast_filtered_error_series_dict[error_name] = error_series

        # Get the statistics table and store it
        error_table = get_qoq_error_statistics_table(
            error_series, 
            error_name.split('_')[-1], 
            ifoCAST_filtered_table_path, 
            f"{table_name}.xlsx"
        )
        ifoCast_filtered_error_table_dict[table_name] = error_table





    # --------------------------------------------------------------------------------------------------
    # Full ifoCAST
    # --------------------------------------------------------------------------------------------------

    ## Filepaths for error series and tables
    ifoCast_full_error_path = os.path.join(wd, '0_1_Output_Data', f'5_ifoCAST_error_series{gva_string}_full{subset_str}')
    os.makedirs(ifoCast_full_error_path, exist_ok=True)

    # Error Tables
    ifoCAST_full_table_path = os.path.join(table_folder, f'4_ifoCAST_evaluations{gva_string}_full{subset_str}')
    os.makedirs(ifoCAST_full_table_path, exist_ok=True)


    ## Clear
    if settings.clear_result_folders:

        for folder in [ifoCast_full_error_path, ifoCast_full_error_path]:
            
            folder_clear(folder)


    ## Evaluation and Table Creation
    ifoCast_full_error_series_names = [f'ifoCAst_errors_full_first{subset_str}{gva_string}',
                            f'ifoCAst_errors_full_latest{subset_str}{gva_string}',
                            f'ifoCAst_errors_full_T45{subset_str}',
                            f'ifoCAst_errors_full_T55{subset_str}']

    ifoCast_full_table_names = [f'ifoCAst_error_tables_full_first{subset_str}{gva_string}',
                            f'ifoCAst_error_tables_full_latest{subset_str}{gva_string}',
                            f'ifoCAst_error_tables_full_T45{subset_str}',
                            f'ifoCAst_error_tables_full_T55{subset_str}']
    
    # Exclude T45 and T55 for GVA
    if not gdp_mode:
        for df in [ifoCast_full_error_series_names, ifoCast_full_table_names]:
            del df[2:]  # Keep only the first two elements



    # Dicts to store outputs
    ifoCast_full_error_series_dict = {}
    ifoCast_full_error_table_dict  = {}

    # Create error series and tables
    for error_name, table_name, eval_key in zip(
        ifoCast_full_error_series_names, 
        ifoCast_full_table_names, 
        ifocast_full_eval_df_names
    ):
        # Get the error series and store it
        error_series = get_qoq_error_series(
            full_eval_dfs[eval_key], 
            ifoCast_full_error_path, 
            file_name=f"{error_name}.xlsx"
        )
        ifoCast_full_error_series_dict[error_name] = error_series

        # Get the statistics table and store it
        error_table = get_qoq_error_statistics_table(
            error_series, 
            error_name.split('_')[-1], 
            ifoCAST_full_table_path, 
            f"{table_name}.xlsx"
        )
        ifoCast_full_error_table_dict[table_name] = error_table



    # --------------------------------------------------------------------------------------------------
    # ifoCAST last values before release — Evaluation and Table Creation
    # --------------------------------------------------------------------------------------------------

    ## Filepaths for error series and tables
    ifoCast_last_rep_error_path = os.path.join(wd, '0_1_Output_Data', f'5_ifoCAST_error_series{gva_string}_last_rep{subset_str}')
    os.makedirs(ifoCast_last_rep_error_path, exist_ok=True)

    # Error Tables
    ifoCAST_last_rep_table_path = os.path.join(table_folder, f'4_ifoCAST_evaluations{gva_string}_last_rep{subset_str}')
    os.makedirs(ifoCAST_last_rep_table_path, exist_ok=True)

    ## Clear
    if settings.clear_result_folders:

        for folder in [ifoCast_last_rep_error_path , ifoCAST_last_rep_table_path]:
            
            folder_clear(folder)


    ## Evaluation and Table Creation
    ifoCast_last_rep_error_series_names = [f'ifoCAst_errors_last_rep_first{subset_str}{gva_string}',
                                        f'ifoCAst_errors_last_rep_latest{subset_str}{gva_string}',
                                        f'ifoCAst_errors_last_rep_T45{subset_str}',
                                        f'ifoCAst_errors_last_rep_T55{subset_str}']

    ifoCast_last_rep_table_names = [f'ifoCAst_error_tables_last_rep_first{subset_str}{gva_string}',
                                    f'ifoCAst_error_tables_last_rep_latest{subset_str}{gva_string}',
                                    f'ifoCAst_error_tables_last_rep_T45{subset_str}',
                                    f'ifoCAst_error_tables_last_rep_T55{subset_str}']


    # Exclude T45 and T55 for GVA
    if not gdp_mode:
        for df in [ifoCast_last_rep_error_series_names, ifoCast_last_rep_table_names]:
            del df[2:]  # Keep only the first two elements

    # Dicts to store outputs
    ifoCast_last_rep_error_series_dict = {}
    ifoCast_last_rep_error_table_dict  = {}

    # Create error series and tables
    for error_name, table_name, eval_key in zip(
        ifoCast_last_rep_error_series_names, 
        ifoCast_last_rep_table_names, 
        ifocast_last_eval_df_names
    ):
        # Get the error series and store it
        error_series = get_qoq_error_series(
            last_rep_eval_dfs[eval_key], 
            ifoCast_last_rep_error_path, 
            file_name=f"{error_name}.xlsx"
        )
        ifoCast_last_rep_error_series_dict[error_name] = error_series

        # Get the statistics table and store it
        error_table = get_qoq_error_statistics_table(
            error_series, 
            error_name.split('_')[-1], 
            ifoCAST_last_rep_table_path, 
            f"{table_name}.xlsx"
        )
        ifoCast_last_rep_error_table_dict[table_name] = error_table











    # -------------------------------------------------------------------------------------------------#
    # =================================================================================================#
    #                                     GET VISUALIZATIONS                                           #
    # =================================================================================================#
    # -------------------------------------------------------------------------------------------------#


    print("Visualizing error statistics ...  \n")



    # ==================================================================================================
    #                                      Build the Savepaths
    # ==================================================================================================

    ## Parent Folders

    # Scatter, Barplot and Series
    ifoCAST_scatter_path = os.path.join(graph_folder, '2_ifoCAST_Error_Scatter')
    ifoCAST_barplot_path = os.path.join(graph_folder, '2_ifoCAST_Error_Bars')
    ifoCAST_series_path = os.path.join(graph_folder, '2_ifoCAST_Error_Series')

    # Store in Dict
    ifoCAST_graph_folders = {
        'scatter': ifoCAST_scatter_path,
        'barplot': ifoCAST_barplot_path,
        'series': ifoCAST_series_path
    }



    ## Create child folders

    # Subfolder names
    subfolder_names = [
        f'0_First_Evaluation{subset_str}{gva_string}',
        f'1_Latest_Evaluation{subset_str}{gva_string}',
        f'1_T45_Evaluation{subset_str}',
        f'1_T55_EValuation{subset_str}'
    ]

    # Exclude T45 and T55 for GVA
    if not gdp_mode:
        del subfolder_names[2:]  # Keep only the first two elements

    # Create folders and store paths
    ifoCAST_paths = {}

    for graph_type, parent_path in ifoCAST_graph_folders.items():
        ifoCAST_paths[graph_type] = {}
        for sub in subfolder_names:
            full_path = os.path.join(parent_path, sub)
            os.makedirs(full_path, exist_ok=True)

            ## Clear 
            if settings.clear_result_folders:
                folder_clear(full_path)

            ifoCAST_paths[graph_type][sub] = full_path



    ## Iterators for dynamic result naming

    eval_horizons = [f'first', f'latest', 'T45', 'T55']
    
    # Exclude T45 and T55 for GVA
    if not gdp_mode:
        del eval_horizons[2:]  # Keep only the first two elements





    # ==================================================================================================
    #                                      Error Bar Plots
    # ==================================================================================================

    print("Creating and saving ifoCAST error measure visualizations ...\n")

    """
    plot_quarterly_metrics(*args, metric_col='MSE', title=None, figsize=(12, 8), 
                            scale_by_n=True, n_bars=10, show=False,
                            save_path=None, save_name=None)
    """

    ## Loop over metrics and Eval_horizons

    # Keys for dynamic naming
    eval_horizon_keys = [f'first', f'latest', 'T45', 'T55']

    # Exclude T45 and T55 for GVA
    if not gdp_mode:
        del eval_horizon_keys[2:]  # Keep only the first two elements

    # Store Metrics for plotting
    metrics = ['ME', 'MAE', 'MSE', 'RMSE', 'SE']

    # --------------------------------------------------------------------------------------------------
    # Matched ifoCAST and ifo QoQ
    # --------------------------------------------------------------------------------------------------

    # Loop over metrics and evaluation horizons
    for metric in metrics:

            for horizon, foldername in zip(eval_horizon_keys, subfolder_names):
                
                save_path = ifoCAST_paths['barplot'][foldername]

                # Access table entries by horizon key
                ifoCAST_table  = ifoCast_filtered_error_table_dict.get(f'ifoCAst_error_tables_filtered_{horizon}{subset_str}')
                ifoCAST_table = {"ifoCAST": ifoCAST_table} # The plotter function was built for a dict input

                ifo_table    = ifo_qoq_matched_error_table_dict.get(f'ifo_qoq_matched_error_tables_{horizon}{subset_str}')
                
                if ifoCAST_table is None or ifo_table is None:
                    if gdp_mode:
                        print(f"\n WARNING: {horizon} - missing table. This shouldnt happen \n")
                    continue

                # Call plotting function
                plot_quarterly_metrics(
                    ifo_table,
                    ifoCAST_table,
                    metric_col=metric,
                    n_bars = 2,
                    scale_by_n=False,
                    show=False,
                    save_path=save_path,
                    save_name=f'0_ifo_ifoCAST_filterd_Quarterly_{metric}_{horizon}_eval{gva_string}.png'
                )



    # --------------------------------------------------------------------------------------------------
    # Full ifoCAST and matched ifo QoQ
    # --------------------------------------------------------------------------------------------------

    # Loop over metrics and evaluation horizons
    for metric in metrics:

        for horizon, foldername in zip(eval_horizon_keys, subfolder_names):
            
            save_path = ifoCAST_paths['barplot'][foldername]

            # Access table entries by horizon key
            ifoCAST_table  = ifoCast_full_error_table_dict.get(f'ifoCAst_error_tables_full_{horizon}{subset_str}')
            ifoCAST_table = {"ifoCAST": ifoCAST_table} # The plotter function was built for a dict input
            
            ifo_table    = ifo_qoq_matched_error_table_dict.get(f'ifo_qoq_matched_error_tables_{horizon}{subset_str}')
            
            if ifoCAST_table is None or ifo_table is None:
                if gdp_mode:
                    print(f"\n WARNING: {horizon} - missing table. This shouldnt happen \n")
                continue

            # Call plotting function
            plot_quarterly_metrics(
                ifo_table,
                ifoCAST_table,
                metric_col=metric,
                n_bars = 2,
                scale_by_n=False,
                show=False,
                save_path=save_path,
                save_name=f'1_ifo_ifoCAST_full_Quarterly_{metric}_{horizon}_eval{gva_string}.png'
            )



    # --------------------------------------------------------------------------------------------------
    # Plot: ifoCAST Last Values Before Release vs Matched ifo QoQ
    # --------------------------------------------------------------------------------------------------

    # Loop over metrics and evaluation horizons
    for metric in ['ME', 'MAE', 'MSE', 'RMSE', 'SE']:

        for horizon, foldername in zip(eval_horizon_keys, subfolder_names):
            
            save_path = ifoCAST_paths['barplot'][foldername]

            # Access the relevant table entries
            ifoCAST_table = ifoCast_last_rep_error_table_dict.get(f'ifoCAst_error_tables_last_rep_{horizon}{subset_str}{gva_string}')
            # Rename the Qminus1 col to Q0 for consistency of the plotter function
            ifoCAST_table.rename(index={"Qminus1": "Q0"}, inplace=True)
            ifoCAST_table = {"ifoCAST": ifoCAST_table}  # plot function expects a dict

            ifo_table = ifo_qoq_matched_error_table_dict.get(f'ifo_qoq_matched_error_tables_{horizon}{subset_str}{gva_string}')

            if ifoCAST_table is None or ifo_table is None:
                print(f"\n WARNING: {horizon} - missing table. \n")
                continue

            # Call plotting function
            plot_quarterly_metrics(
                ifo_table,
                ifoCAST_table,
                metric_col=metric,
                n_bars=1,
                scale_by_n=False,
                show=False,
                save_path=save_path,
                save_name=f'0_ifo_ifoCAST_lastRep_Quarterly_{metric}_{horizon}_eval{gva_string}.png'
            )










    # ==================================================================================================
    #                                      Error Scatter Plots
    # ==================================================================================================

    print("Creating and saving ifoCAST error scatter plots ...\n")

    """
    plot_error_lines(*args, title: Optional[str] = None, figsize: tuple = (12, 8),
                        n_bars: int = 10, show: bool = False,
                        save_path: Optional[str] = None, save_name: Optional[str] = None):
    """


    # --------------------------------------------------------------------------------------------------
    # Matched ifoCAST and ifo QoQ → plot_error_lines
    # --------------------------------------------------------------------------------------------------

    for horizon, ifo_name, ifoCAST_name, subfolder in zip(
        eval_horizons, ifo_qoq_matched_error_series_names, ifoCast_filtered_error_series_names, subfolder_names):
        
        ifo_error_df = ifo_qoq_matched_error_series_dict.get(ifo_name)
        filtered_error_df = ifoCast_filtered_error_series_dict.get(ifoCAST_name)
        filtered_error_df = {"ifoCAST": filtered_error_df} # The plotter function was built for a dict input
        
        save_path = ifoCAST_paths['scatter'][subfolder]

        plot_error_lines(
            ifo_error_df,
            filtered_error_df,
            ifocast_mode= True,
            n_bars = 2,
            title=f"ifoCAST_filtered_{horizon.capitalize()}",
            show=False,
            save_path=save_path,
            save_name=f"0_ifoCAST_filtered_{horizon}_errors{gva_string}"
        )




    # --------------------------------------------------------------------------------------------------
    # Full ifoCAST and matched ifo QoQ → plot_error_lines
    # --------------------------------------------------------------------------------------------------

    for horizon, ifo_name, ifoCAST_name, subfolder in zip(eval_horizons, 
        ifo_qoq_matched_error_series_names, ifoCast_full_error_series_names, subfolder_names):
        
        ifo_error_df = ifo_qoq_matched_error_series_dict.get(ifo_name)
        full_error_df = ifoCast_full_error_series_dict.get(ifoCAST_name)
        full_error_df = {"ifoCAST": full_error_df} # The plotter function was built for a dict input
        
        save_path = ifoCAST_paths['scatter'][subfolder]

        plot_error_lines(
            ifo_error_df,
            full_error_df,
            n_bars = 2,
            title=f"ifoCAST_full_{horizon.capitalize()}",
            show=False,
            save_path=save_path,
            save_name=f"1_ifoCAST_full_{horizon}_errors{gva_string}"
        )




    # --------------------------------------------------------------------------------------------------
    # Last-Rep ifoCAST and matched ifo QoQ → plot_error_lines
    # --------------------------------------------------------------------------------------------------

    for horizon, ifo_name, ifoCAST_name, subfolder in zip(
        eval_horizons, 
        ifo_qoq_matched_error_series_names, 
        ifoCast_last_rep_error_series_names, 
        subfolder_names
    ):
        # Get the error series
        ifo_error_df = ifo_qoq_matched_error_series_dict.get(ifo_name)
        last_rep_error_df = ifoCast_last_rep_error_series_dict.get(ifoCAST_name)

        # Rename the Qminus1 col to Q0 for consistency of the plotter function
        last_rep_error_df.rename(columns={"Qminus1": "Q0"}, inplace=True)
        last_rep_error_df = {"ifoCAST": last_rep_error_df}  # dict input for plotter

        # Save path
        save_path = ifoCAST_paths['scatter'][subfolder]

        # Plot
        plot_error_lines(
            ifo_error_df,
            last_rep_error_df,
            n_bars=1, # Set this to 1
            title=f"ifoCAST Last Release Errors against {horizon.capitalize()}",
            show=False,
            save_path=save_path,
            save_name=f"1_ifoCAST_lastRep_{horizon}_errors{gva_string}"
        )






    # ==================================================================================================
    #                                      Error Time Series
    # ==================================================================================================

    print("Creating and saving ifoCAST prediction time series ...\n")

    """
    plot_forecast_timeseries(*args, df_eval=None, title_prefix=None, figsize=(12, 8), 
                                show=False, save_path=None, save_name_prefix=None, select_quarters=None)
    """


    """
    # --------------------------------------------------------------------------------------------------
    # Matched ifoCAST and ifo QoQ
    # --------------------------------------------------------------------------------------------------

    # Loop through evaluation horizons
    for horizon, ifo_name, ifoCAST_name, subfolder in zip(
        eval_horizons, ifo_qoq_matched_eval_df_names, ifocast_filtered_eval_df_names, subfolder_names):
        
        # Get forecast data
        ifo_qoq_df = ifo_qoq_eval_dfs.get(ifo_name)
        filtered_df = fitered_eval_dfs.get(ifoCAST_name)
        #show(filtered_df)
        filtered_dict = {"ifoCAST": filtered_df} # The plotter function was built for a dict input
        
        save_path = ifoCAST_paths['series'][subfolder]

        # Plot 
        plot_forecast_timeseries(
            ifo_qoq_df,
            filtered_dict,
            df_eval=None,
            title_prefix=f"ifoCAST_filtered_{horizon.capitalize()}",
            show=False,
            save_path=save_path,
            save_name_prefix=f"0_ifoCAST_filtered_{horizon}_series",
            select_quarters=None
        )


    # --------------------------------------------------------------------------------------------------
    # Full ifoCAST and matched ifo QoQ
    # --------------------------------------------------------------------------------------------------

    # Loop through evaluation horizons
    for horizon, ifo_name, ifoCAST_name, subfolder in zip(
        eval_horizons, ifo_qoq_matched_eval_df_names, ifocast_full_eval_df_names, subfolder_names):
        
        # Get forecast data
        ifo_qoq_df = ifo_qoq_eval_dfs.get(ifo_name)
        full_df = full_eval_dfs.get(ifoCAST_name)
        full_dict = {"ifoCAST": full_df} # The plotter function was built for a dict input
        
        save_path = ifoCAST_paths['series'][subfolder]

        # Plot 
        plot_forecast_timeseries(
            ifo_qoq_df,
            full_dict,
            df_eval=None,
            title_prefix=f"ifoCAST_filtered_{horizon.capitalize()}",
            show=False,
            save_path=save_path,
            save_name_prefix=f"1_ifoCAST_full_{horizon}_series",
            select_quarters=None
        )
    """








# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                     RUN EVALUATIONS - GDP                                        #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#



# ==================================================================================================
#                                    Full ifoCAST since Q1-2020
# ==================================================================================================

print("\nRunning ifoCAST GDP Evaluation Pipeline on full ifoCAST data since Q1-2020 ... \n")
ifocast_eval_pipeline(ifoCAst_Qm1_Q0_Q1_filtered, ifoCAst_Qm1_Q0_Q1_full, ifocast_last_values)





# ==================================================================================================
#                                       Subsetted ifoCAST
# ==================================================================================================

#show(ifoCAst_Qm1_Q0_Q1_filtered)
#show(ifoCAst_Qm1_Q0_Q1_full)
#show(ifocast_last_values)

# --------------------------------------------------------------------------------------------------
#                                        Subset Function
# --------------------------------------------------------------------------------------------------


def subset_ifocast(df: pd.DataFrame, cutoff_date) -> pd.DataFrame:
    """
    Drop all observations in df with index prior to cutoff_date.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a datetime-like index.
    cutoff_date : str or pd.Timestamp
        The cutoff date. All rows before this will be dropped.
    """
    cutoff_date = pd.to_datetime(cutoff_date)
    return df.loc[df.index >= cutoff_date]


# --------------------------------------------------------------------------------------------------
#                                        since Q1-2021
# --------------------------------------------------------------------------------------------------

if settings.run_ifocast_2021_subset:
    print("\nRunning ifoCAST GDP Evaluation Pipeline on subseted ifoCAST data since Q1-2021 ... \n")

    ## Filter ifoCasts
    ifocast_filtered_2021 = subset_ifocast(ifoCAst_Qm1_Q0_Q1_filtered, '2021-01-01')
    ifocast_full_2021 = subset_ifocast(ifoCAst_Qm1_Q0_Q1_full, '2021-01-01')
    ifocast_last_2021 = subset_ifocast(ifocast_last_values, '2021-01-01')

    #show(ifocast_filtered_2021)
    #show(ifocast_full_2021)
    #show(ifocast_last_2021)

    ## Run Eval pipeline
    ifocast_eval_pipeline(ifocast_filtered_2021, ifocast_full_2021, ifocast_last_2021, subset_str='_since_2021')


# --------------------------------------------------------------------------------------------------
#                                        since Q1-2022
# --------------------------------------------------------------------------------------------------

if settings.run_ifocast_2022_subset:
    print("\nRunning ifoCAST GDP Evaluation Pipeline on subseted ifoCAST data since Q1-2022 ... \n")

    ## Filter ifoCasts
    ifocast_filtered_2022 = subset_ifocast(ifoCAst_Qm1_Q0_Q1_filtered, '2022-01-01')
    ifocast_full_2022 = subset_ifocast(ifoCAst_Qm1_Q0_Q1_full, '2022-01-01')
    ifocast_last_2022 = subset_ifocast(ifocast_last_values, '2022-01-01')

    ## Run Eval pipeline
    ifocast_eval_pipeline(ifocast_filtered_2022, ifocast_full_2022, ifocast_last_2022, subset_str='_since_2022')











# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                     RUN EVALUATIONS - GVA                                        #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

if settings.run_gva_evaluation:

    # ==================================================================================================
    #                                    Full ifoCAST since Q1-2020
    # ==================================================================================================

    print("\nRunning ifoCAST GVA-Evaluation Pipeline on full ifoCAST data since Q1-2020 ... \n")
    ifocast_eval_pipeline(ifoCAst_Qm1_Q0_Q1_filtered, ifoCAst_Qm1_Q0_Q1_full, ifocast_last_values, 
                          gdp_mode=False)


    # ==================================================================================================
    #                                       Subsetted ifoCAST
    # ==================================================================================================

    # --------------------------------------------------------------------------------------------------
    #                                        since Q1-2021
    # --------------------------------------------------------------------------------------------------

    if settings.run_ifocast_2021_subset:
        print("\nRunning ifoCAST GVA-Evaluation Pipeline on subseted ifoCAST data since Q1-2021 ... \n")

        ## Run Eval pipeline
        ifocast_eval_pipeline(ifocast_filtered_2021, ifocast_full_2021, ifocast_last_2021, subset_str='_since_2021', 
                              gdp_mode=False)


    # --------------------------------------------------------------------------------------------------
    #                                        since Q1-2022
    # --------------------------------------------------------------------------------------------------

    if settings.run_ifocast_2022_subset:
        print("\nRunning ifoCAST GVA-Evaluation Pipeline on subseted ifoCAST data since Q1-2022 ... \n")

        ## Run Eval pipeline
        ifocast_eval_pipeline(ifocast_filtered_2022, ifocast_full_2022, ifocast_last_2022, subset_str='_since_2022', 
                              gdp_mode=False)












# --------------------------------------------------------------------------------------------------
print(f"\nifoCAST Evaluation Module complete! \n",f"Find Result Graphs in {graph_folder} and \nResult Tables in {table_folder}\n")
# --------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#
