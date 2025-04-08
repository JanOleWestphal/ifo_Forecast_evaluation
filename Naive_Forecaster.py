

# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Naive Forecaster
#
# Author:       Jan Ole Westphal
# Date:         2025-04
#
# Description:  An evaluation model for the ifo economic forecast, pulling the latest Bundesbank 
#               quaterly GDP releases and building naive forecasting models based on this data. For 
#               every quaterly data-series, a new forecast model is estimated. Available models are 
#               described in the Parameter setup section. 
#               Puts out forecast time series into excel, designed for evaluating historic ifo 
#               forecasts against more naive methods, which is done in a different piece of code.            
# ==================================================================================================
# --------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# OPTIONS
# ==================================================================================================
# --------------------------------------------------------------------------------------------------


# ==================================================================================================
# PARAMETER Setup:
#
# wd                   Sets the current working directory
# result_subfolder     Sets the folder name where results of a given run are printed, e.g. 'results'
#                      OVERRIDES WITH EVERY RUN!
#
# model                Sets the forecast model used by the agent, options: 
#                           - Auto-regressiv model of order AR_order with a constant: 'AR' 
#                           - Simple moving average: 'SMA'
#                           - Very naive: just the previous quarter's average for all future 
#                             quarters: 'AVERAGE'
# 
# average_horizon      Sets the amount of previous quarters averaged over in 'SMA' and 'AVERAGE'
# AR_horizon           Sets the time frame on which the AR- models are estimated in quarters. Set to 
#                      a natural number n for looking backwards up to n quaters, set 'FULL' for use 
#                      of full series
# AR_order             Number of lags in the autoregressive model, recommendation: 2
# forecast_horizon     Determines how many quaters into the future predictions are made
#
# ==================================================================================================


# --------------------------------------------------------------------------------------------------
# Folders
# --------------------------------------------------------------------------------------------------

# Set wd as '/path/to/your/project' or r'\path\to\your\project'
wd = r'C:\Users\janol\OneDrive\Desktop\ifo\Konjunkturprognose Evaluierung\Python Workfolder'

# Name Resultfolder, e.g. 'AR2_results'
result_subfolder = 'Results'

# --------------------------------------------------------------------------------------------------
# Define the model
# --------------------------------------------------------------------------------------------------

# Set the agent's forecasting method; options: 'AR', 'AVERAGE', 'SMA'
model = 'AR'

# For average-based models: set time frame over which the agent averages in quarters
average_horizon = 8

# For AR model: set number of lags (sugested: 2)
AR_order = 2

# For AR model: set the memory of the agent, i.e. the timeframe the model is estimated on
AR_horizon = 100

# Set how far the agent looks into the future
forecast_horizon = 12








# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Code begins here
# ==================================================================================================
# --------------------------------------------------------------------------------------------------



# ==================================================================================================
# Check correct parameter settings
# ==================================================================================================

# Validate model type
valid_models = {'AR', 'AVERAGE', 'SMA'}
if model not in valid_models:
    raise ValueError(f"Invalid model '{model}'. Must be one of: {valid_models}")
    
# Validate all horizons are ≥1 (except AR_order which can be 0)
if average_horizon < 1:
    raise ValueError(f"average_horizon ({average_horizon}) must be ≥1")
if AR_horizon < 1:
    raise ValueError(f"AR_horizon ({AR_horizon}) must be ≥1")
if forecast_horizon < 1:
    raise ValueError(f"forecast_horizon ({forecast_horizon}) must be ≥1")
    
# Model-specific validations
if model in ('AR'):
    if AR_order < 0:
        raise ValueError(f"AR_order ({AR_order}) cannot be negative")
    if AR_horizon <= AR_order:
        raise ValueError(
            f"For {model} model, AR_horizon ({AR_horizon}) must be > AR_order ({AR_order})"
            "\n(Need at least AR_order+1 observations to estimate model)"
            )





# ==================================================================================================
# SET-UP
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------

# Import built-ins
import importlib
import subprocess
import sys
import os
from datetime import datetime, timedelta

# Check for libraries
required_packages = [
    'requests',
    'openpyxl',
    'pandas',
    'numpy',
    'statsmodels',
    'matplotlib',
    'seaborn'
]

# Install if needed
for pkg in required_packages:
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        print(f" {pkg} installed.")

# Import libraries
import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
sns.set_theme(style='whitegrid')



# --------------------------------------------------------------------------------------------------
# Working Directory and results folder
# --------------------------------------------------------------------------------------------------

# Set working directory
if wd:
    try:
        os.makedirs(wd, exist_ok=True)
        os.chdir(wd)
        print(f"Working directory set to: {wd} ...")
    except FileNotFoundError:
        print(f"Directory not found: {wd}")
else:
    print("No working directory set.")



# Create or clean the results folder
# Set path
folder_path = os.path.join(wd, result_subfolder)

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Clear the results folder
for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    if os.path.isfile(file_path):
        os.remove(file_path)







# ==================================================================================================
# DATA PROCESSING
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Import Data via Bundesbank API
# --------------------------------------------------------------------------------------------------

# Bundesbank API link
url = 'https://api.statistiken.bundesbank.de/rest/download/BBKRT/Q.DE.Y.A.AG1.CA010.A.I?format=csv&lang=de'

# Output filename
filename = 'Bundesbank_GDP_raw.csv'
filepath = os.path.join(wd, filename)

# Download and save the CSV file
response = requests.get(url)
if response.status_code == 200:
    with open(filepath, 'wb') as f:
        f.write(response.content)
    print(f"Bundesbank data downloaded and saved to: {filepath} ...")
else:
    print(f"Failed to download file. Library: requests,  Status code: {response.status_code} ...")


# Reformat into a usable dataframe

# Load first line to get headers
with open(filename, encoding="utf-8") as f:
    header = f.readline().strip().split(';')

# Load data (stored in row 12 onwards)
df = pd.read_csv(filename, skiprows=11, names=header, sep=';', index_col=0, na_values=['', ' '])

# Convert all data to float
try:
    # Replace commas with dots -> perform float conversion
    df = df.apply(lambda col: col.str.replace(',', '.') if col.dtype == 'object' else col)
    df = df.apply(pd.to_numeric, errors='raise')

except ValueError as e:
    raise ValueError("Possible Processing error: GDP-data can not be converted into floats") from e


# Convert Column Names to Datetime
df.columns = pd.to_datetime(df.columns, format='%d.%m.%Y', errors='raise')

# Inspect
#print(df.head())

print("Raw Data loaded ...")


# --------------------------------------------------------------------------------------------------
# Data Cleaning
# 
#  - Select only the T+55 releases corresponding to the information sets utilized in corresponding 
#    quaterly ifo forecast: extract columns published in Feb, May, Aug, Nov.
#  - Rename Cols to yyyy_0q format; q in 1,2,3,4
#
# --------------------------------------------------------------------------------------------------

# Set the index to a timestamp
df.index = pd.PeriodIndex(df.index, freq='Q').to_timestamp()

# Add ~45 days to shift to the middle of each quarter
df.index += pd.offsets.Day(45)
print("Index set to datetime and middle of the quarter ...")

# Drop columns not in Feb, May, Aug, Nov
months_to_keep = [2, 5, 8, 11]  
df = df.loc[:, df.columns.month.isin(months_to_keep)]

# Inspect
#print(df.head())


# --------------------------------------------------------------------------------------------------
# Create quarterly growth rates: df_qoq (Quarter over Quarter)
# --------------------------------------------------------------------------------------------------

# Calclulate growth % in t as relative forward differences, where .shift(-1) calls t+1 values for t
df_qoq = ((df.shift(-1) - df) / df) *100


# Inspect
# print(df_qoq.head())





# ==================================================================================================
# MODEL ESTIMATION
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
#  Define necessary functionalities
# --------------------------------------------------------------------------------------------------

# Create a function which selects the correct series as inputs in every column iteration: select_col
def select_col(col, horizon):

    """
    Selects the appropriate memory horizon for the forecast models.

    This function retrieves the non-null portion of the GDP growth column from the DataFrame `df_qoq` 
    and selects a subset of data based on the specified `horizon`. If the series is shorter 
    than the horizon, it returns the entire series; otherwise, it returns only the most recent 
    `horizon` number of observations.

    Args:
        col (str): The name of the column in the DataFrame `df_qoq` to be selected.
        horizon (int): The number of most recent observations to select from the column, 
                       AR_horizon or average_horizon

    Returns:
        pd.Series: A pandas Series containing the selected data for the specified column 
        and horizon.
    """

    # Select the non-null part of each column: series
    series = df_qoq[col].dropna()

    # Select the model scope
    if len(series) < horizon:
        data_index = series.index
        data = series.values
    else:
        data_index = series.iloc[-horizon:].index
        data = series.iloc[-horizon:].values

    return data, data_index


# Create a function appending forecasts with correct indexing: index_append(data, forecast_qoq)
def index_append(data, data_index, forecast_qoq, df_qoq_forecast):

    """
    Appends forecasted values to the original data with correct future quarterly indexing.

    This function generates future quarterly timestamps starting from the last available quarter in 
    the historical data, and appends the corresponding forecasted values to the `df_qoq_forecast` 
    DataFrame. The new values are indexed based on the forecast horizon, ensuring that the quarterly 
    frequency is maintained.

    Args:
        data (pd.Series or np.ndarray): Historical values (used for type consistency).
        data_index (pd.DatetimeIndex): Time indices of the historical data.
        forecast_qoq (list or np.ndarray): Forecasted values to append, lengths must match.
        df_qoq_forecast (pd.DataFrame): Existing DataFrame to which forecasted values are appended.

    Returns:
        pd.DataFrame: Updated `df_qoq_forecast` with new forecasts appended. The new entries are 
        indexed by future quarters starting from the last historical date.

    Note:
        - The function requires the `forecast_horizon` (int) to be defined elsewhere.
        - The quarterly index is generated using 'QE', and set to mid-quarter
        - This function modifies the passed `df_qoq_forecast` globally.
    """


    # Set correct index, adjust to middle of quarter
    last_quarter = data_index[-1]
    forecast_index = pd.date_range(start=last_quarter, periods=forecast_horizon + 1, freq='QE')[1:]
    forecast_index -= pd.offsets.Day(45)


    # Create indexed DataFrame
    df_qoq_forecast_new = pd.DataFrame(forecast_qoq, index=forecast_index, columns=[col])

    # Append the forecasted values to df_qoq_forecast
    df_qoq_forecast = pd.concat([df_qoq_forecast, df_qoq_forecast_new])

    return df_qoq_forecast








# --------------------------------------------------------------------------------------------------
# Prepare the Estimation 
# --------------------------------------------------------------------------------------------------

# Create result tables
df_qoq_forecast = pd.DataFrame()

# Clear workspace, if needed: 
if 'AR_summary' in globals():
    del AR_summary
    # making sure that the AR- diagnostics table doesn't show up in other model outputs



# --------------------------------------------------------------------------------------------------
# If-else Filter to choose the correct model: 
#     filter the series, calculate forecasts, append to df_qoq_forecast with correct index
# --------------------------------------------------------------------------------------------------

# Simple AR (with a constant) on previous growth rates within AR_horizon
if model == 'AR':

    print(f"""Calculating an {model}{AR_order} model on the last {AR_horizon} quarters,
    forecasting {forecast_horizon} quarters into the future ...""")

    # Create df for evaluation statistics: AR_summary
    AR_summary = pd.DataFrame() 

    # Iterate over all quarterly datapoints 
    for col in df_qoq.columns:

        # Select memory window
        data, data_index = select_col(col, AR_horizon)


        # Fit the model
        forecaster = AutoReg(data, lags=AR_order)
        results = forecaster.fit()

        # Save diagnostic statistics to AR_summary
        param_series = pd.Series(results.params, name=col) 
        AR_summary = pd.concat([AR_summary, param_series.to_frame().T])


        # Create prediction df: df_qoq_forecast
        forecast_qoq = results.predict(start=len(data) +1 , end=len(data) + forecast_horizon) 

        df_qoq_forecast = index_append(data, data_index, forecast_qoq, df_qoq_forecast)




# Simple moving average of previous growth rates within the average_horizon
elif model == 'SMA':

    print(f"""Calculating forecasts as a simple moving average of the past {average_horizon} quarters
    forecasting {forecast_horizon} quarters into the future ...""")


    # Iterate over all quarterly datapoints 
    for col in df_qoq.columns:

        # Select memory window
        data, data_index = select_col(col, average_horizon)


        # List to collect forecasted values for this column
        forecast_qoq = []
        
        # Create #forecast_horizon prediction elements elements
        for _ in range(forecast_horizon):
            # Compute the average of the current data window: sma
            sma = data.mean()
            # Build forecast list, save results
            forecast_qoq.append(sma)
            # Shift data window
            data = np.append(data, sma)
            data = data[1:]

        
        # Create prediction df: df_qoq_forecast
        df_qoq_forecast = index_append(data, data_index, forecast_qoq, df_qoq_forecast)




# Static average of previous growth rates within the average_horizon
elif model == 'AVERAGE':

    print(f"""Calculating forecasts as the static average of the past {average_horizon} quaters
    forecasting {forecast_horizon} quarters into the future ...""")
    
    # Iterate over all quarterly datapoints 
    for col in df_qoq.columns:

        # Select memory window
        data, data_index = select_col(col, average_horizon)

        
        # Calculate the average, create forecast list
        average = data.mean()
        forecast_qoq = pd.Series([average] * forecast_horizon)


        # Create prediction df: df_qoq_forecast
        df_qoq_forecast = index_append(data, data_index, forecast_qoq, df_qoq_forecast)




# --------------------------------------------------------------------------------------------------
#  If further options are required, put them here. Make sure to update the descriptions under 
#  Parameter Setup and the parameter validation check under valid_models = {} accordingly
# --------------------------------------------------------------------------------------------------

# Error message
else:
    print("This should never be printed, check wether valid_models is still up to date")





# ---------------------------
# Diagnostics
# ---------------------------


print(df_qoq_forecast.head())





# ==================================================================================================
# PROCESS ESTIMATION RESULTS
# ==================================================================================================

# Combine realised data with forecast data
df_qoq_combined = df_qoq.combine_first(df_qoq_forecast)

# Create yearly forecasts on combined data: df_yoy_combined
#df_qoq_factor_combined = df + 1

# Group by year and multiply values for each year
# df_yoy_combined = df_qoq_factor_combined.resample('YE').prod() #NaN if any quarter is missing
df_yoy_combined = df_qoq_combined.resample('YE').prod() #NaN if any quarter is missing

# --------------------------------------------------------------------------------------------------
#  Reset indices to YYYY-Qx and and set column names to YYYY_0q; q in 1,2,3,4
# --------------------------------------------------------------------------------------------------

# Function: Rename column headers
def rename_column(date_str):
    day, month, year = map(int, date_str.split('.'))
    # Determine the quarter
    quarter = ((month - 1) // 3) + 1
    # Rename to the required format
    return f"{year}_0{quarter}"

# Apply to output dataframes
# df.columns = df.columns.map(rename_column)



# Function: rename indices


#Apply to output dataframes





# ==================================================================================================
# SAVE RESULTS AS EXCEL
# ==================================================================================================

# ---------------------------
# QUARTERLY
# ---------------------------

# Save the quarterly growth forecast table of df_qoq_forecast 
filename_qoq_forecast = f'Naive_QoQ_forecasts_{model}.xlsx'
df_qoq_forecast.to_excel(os.path.join(folder_path, filename_qoq_forecast))

# Save the extended quarterly growth tables (realized + forecasted) of df_qoq_combined
filename_qoq_combined = f'Realized_and_forecasted_QoQ_{model}.xlsx'
df_qoq_combined.to_excel(os.path.join(folder_path, filename_qoq_combined))


# ---------------------------
# YEARLY
# ---------------------------

# Save the extended yearly growth tables (realized + forecasted): 
filename_yoy = f'Realized_and_forecasted_YoY_{model}.xlsx'
df_yoy_combined.to_excel(os.path.join(folder_path, filename_yoy))


"""
Could also calculate and save absolute values ...
"""


# Check wether there is an AR_summary, save if yes
if 'AR_summary' in globals():
    filename_AR_summary = 'AR_model_summary_statistics.xlsx'
    AR_summary.to_excel(os.path.join(folder_path, filename_AR_summary))







# --------------------------------------------------------------------------------------------------
print(f" \n Program executed, find results in Subfolder {result_subfolder} of working directory {wd}")
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# ==============
# End of code
# ==============
# --------------------------------------------------------------------------------------------------











# ==================================================================================================
# For later: visualization module
# ==================================================================================================


# --------------------------------------------------------------------------------------------------
# Visualize the results
# --------------------------------------------------------------------------------------------------


# Set which years you want to visualize, as a list [ , ] or None
#vis_years = None

#
# vis_years            Put in a list of years for which the results are visualized, default None