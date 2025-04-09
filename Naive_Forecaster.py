
"""
PROBLEMS:
    - The program creates a backcast and not just a nowcast

FUTURE DEVELOPMENTS:
    - Create Error measures
    - Explore inconsistencies of the Bundesbank Data
    - Visualize these errors, perform analytics
    - Match with ifo and consensus data
    - ...
"""

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



# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        ~  OPTIONS  ~                                             #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# ==================================================================================================
# PARAMETER Setup:
#
# wd                   Sets the current working directory
# resultfolder_name    Sets the folder name where results of a given run are printed, e.g. 'results',
#                      Default naming: 'Results_model_memory-horizon_forecast'
#                      OVERRIDES WITH EVERY RUN!
#                       
# api_pull             Determines whether the data is automatically pulled form the Bundesbank API 
#                      (set = True) or whether a local version is used (set = False). 
#                      ATTENTION: if False, local file must be named 'Bundesbank_GDP_raw.csv'
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
# Folders and Data
# --------------------------------------------------------------------------------------------------

# Set wd as '/path/to/your/project' or r'\path\to\your\project'
wd = r'C:\Users\janol\OneDrive\Desktop\ifo\Konjunkturprognose Evaluierung\Python Workfolder'

# Customize Resultfolder names, e.g. 'AR2_results', suggested: 'Default'
resultfolder_name = 'Default'   #set 'Default' for default: 'Results_model_memory_forecast'

# Decide wether to use the API pull or a local version of the file; True (suggested) or False 
api_pull = True


# --------------------------------------------------------------------------------------------------
# Define the model
# --------------------------------------------------------------------------------------------------

# Set the agent's forecasting method; options: 'AR', 'AVERAGE', 'SMA'
model = 'AVERAGE'

# For AR model: set number of lags (sugested: 2); int
AR_order = 2


# For average-based models: set time frame over which the agent averages in quarters; int or 'FULL'
average_horizon = 2

# For AR model: set the memory of the agent (timeframe the model is estimated on); int or 'FULL'
AR_horizon = 40


# Set how far the agent predicts into the future; int
forecast_horizon = 20

#-------------#
#  Note: If data is released at Quarter 2, only Q1 data is available. Accordingly, forecasting t 
#        periods into the future requires t+1 forecasts, including the so-called nowcast. The above
#        parameter ignores the nowcast, it is adjusted for below.
#-------------#






# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        Code begins here                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#



# ==================================================================================================
# Check correct parameter settings
# ==================================================================================================

# Include the nowcast
forecast_horizon += 1
# In all subsequent print or name calls, forecast_horizon must be reduced by one ...



# Validate model type
valid_models = {'AR', 'AVERAGE', 'SMA'}
if model not in valid_models:
    raise ValueError(f"Invalid model '{model}'. Must be one of: {valid_models}")
    

# Validate all horizons are ≥1 or 'FULL'
if not (isinstance(average_horizon, int) and average_horizon >= 1 or average_horizon == "FULL"):
    raise ValueError(f"average_horizon ({average_horizon}) must be ≥1 or 'FULL'")

if not (isinstance(AR_horizon, int) and AR_horizon >= 1 or AR_horizon == "FULL"):
    raise ValueError(f"AR_horizon ({AR_horizon}) must be ≥1 or 'FULL'")

if not (isinstance(forecast_horizon, int) and forecast_horizon >= 1):
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
from datetime import datetime, date

# Check for libraries
required_packages = [
    'requests',
    'openpyxl',
    'pandas',
    #'pandasgui',
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
#from pandasgui import show  #uncomment this to allow for easier debugging
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
        print(f" \nWorking directory set to: {wd} ... \n")
    except FileNotFoundError:
        print(f"Directory not found: {wd} \n")
else:
    print("No working directory set.")



#Define the result_subfolder

if resultfolder_name == 'Default':

    if model in ['AVERAGE', 'SMA']:
        result_subfolder = f"Results_{model}_{average_horizon}_{forecast_horizon-1}"

    elif model == 'AR':
        result_subfolder = f"Results_{model}{AR_order}_{AR_horizon}_{forecast_horizon-1}"

    # Faulty model selection
    else:
        print("ERROR: wrong naming of outputs, check SAVE RESULTS section")

else:
    result_subfolder = resultfolder_name



# Define Resultfolder path
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
# Import Data via Bundesbank API or locally
# --------------------------------------------------------------------------------------------------

# Output filename
filename = 'Bundesbank_GDP_raw.csv'
filepath = os.path.join(wd, filename)

# Optional API-pull:
if api_pull == True:
    # Bundesbank API link
    url = 'https://api.statistiken.bundesbank.de/rest/download/BBKRT/Q.DE.Y.A.AG1.CA010.A.I?format=csv&lang=de'

    # Download and save the CSV file
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Bundesbank data downloaded and saved to: {filepath} ... \n ")
    else:
        raise FileNotFoundError(f"""Failed to download file. 
                                Library: requests,  Status code: {response.status_code}.
                                Try again or use local version of the file (api_pull = False)""")

elif api_pull == False:
    print('Attention: local version of the data is being used, set api_pull = True for real-time data \n')

else:
    raise ValueError('ERROR: api_pull must be set to either True or False')



# --------------------------------------------------------------------------------------------------
# Reformat into a usable dataframe
# --------------------------------------------------------------------------------------------------

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

print("Raw Data loaded ... \n")


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
#print("Index set to datetime and middle of the quarter ...")

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

print("Data cleaned ... \n")




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
        tuple: Contains two elements:
            - data (np.ndarray): Array of selected values
            - data_index (pd.Index): Corresponding datetime indices

    Notes:
        - The function always returns numpy arrays (not pd.Series) for model compatibility
        - Returned indices maintain alignment with the selected values
        - For horizons exceeding available data, returns all available observations
    """

    # Select the non-null part of each column: series
    series = df_qoq[col].dropna()

    # Select the model scope
    if len(series) < horizon or horizon == 'FULL':
        data_index = series.index
        data = series.values
    else:
        data_index = series.iloc[-horizon:].index
        data = series.iloc[-horizon:].values

    return data, data_index



# Create a function building an indexed dictionary of forecasts

def index_dict(col, data_index, forecast_qoq, qoq_forecast_index_df):

    """
    Creates and appends forecast records to a results DataFrame with proper quarterly indexing.

    This function constructs forecast records by:
    1. Generating future quarterly dates adjusted to mid-quarter
    2. Combining forecast values with metadata
    3. Appending results to a growing collection DataFrame

    Args:
        col: Identifier for the forecast series (typically column name)
        data_index (pd.DatetimeIndex): Output of select_col, index of selected cols
        forecast_qoq (np.ndarray): Array of forecasted quarterly growth values
        qoq_forecast_index_df (pd.DataFrame): Accumulating DataFrame of all forecasts

    Returns:
        pd.DataFrame: Updated version of qoq_forecast_index_df with new records appended

    Notes:
        - Forecast dates are generated as end-of-quarter (QE) then adjusted back 45 days
        - Requires global forecast_horizon variable to determine date range
        - Each record contains:
            * date_of_forecast: Model/run identifier (col)
            * target_date: Mid-quarter date being forecasted
            * predicted_qoq: Forecasted quarterly growth value
        - Maintains immutable operations (returns new DataFrame rather than modifying in-place)
    """


    # Set correct index, adjust to middle of quarter
    last_quarter = data_index[-1]
    forecast_index = pd.date_range(start=last_quarter, periods= forecast_horizon + 2, freq='QE')[2:]
    forecast_index -= pd.offsets.Day(45)

    # Append the forecast date, index and forecast data, merge to df
    forecast_df = pd.DataFrame({
        'date_of_forecast': col,
        'target_date': forecast_index,
        'predicted_qoq': forecast_qoq
    })    

    # Save to result df
    if qoq_forecast_index_df.empty:
        qoq_forecast_index_df = forecast_df.copy()
    else:
        qoq_forecast_index_df = pd.concat([qoq_forecast_index_df, forecast_df], ignore_index=True)

    return qoq_forecast_index_df



# Create a function which selects the AR results and puts them into a df: AR_summary()

def AR_diagnostics(results):

    diagnostics = {}
    
    # Intercept (always at index 0)
    diagnostics['coefficient_Intercept']        = results.params[0]
    diagnostics['p_Intercept']      = results.pvalues[0]
    diagnostics['stderr_Intercept'] = results.bse[0]
    diagnostics['t_Intercept']      = results.tvalues[0]
    
    # Iterate over the lags based on AR_order
    for i in range(1, AR_order + 1):
        diagnostics[f'coefficient_Lag_{i}']        = results.params[i]
        diagnostics[f'p_Lag_{i}']      = results.pvalues[i]
        diagnostics[f'stderr_Lag_{i}'] = results.bse[i]
        diagnostics[f't_Lag_{i}']      = results.tvalues[i]
    
    # Add model level statistics
    diagnostics['AIC'] = results.aic
    diagnostics['BIC'] = results.bic

    return pd.Series(diagnostics, name=col)




# --------------------------------------------------------------------------------------------------
# Prepare the Estimation 
# --------------------------------------------------------------------------------------------------

# Create prediction dataframe: forecast_horizon x n_cols
qoq_forecast_df = pd.DataFrame()

# Create an indexed dictionary of the predictions
qoq_forecast_index_df = pd.DataFrame(columns=['date_of_forecast', 'target_date', 'predicted_qoq'])


# Clear workspace, if needed: 
if 'AR_summary' in globals():
    del AR_summary
    # making sure that the AR- diagnostics table doesn't show up in other model outputs



# --------------------------------------------------------------------------------------------------
# If-else Filter to choose the correct model: 
#     filter the series, calculate forecasts, append to qoq_forecast_df with correct index
# --------------------------------------------------------------------------------------------------


# Simple AR (with a constant) on previous growth rates within AR_horizon
if model == 'AR':

    print(f"""Calculating an {model}{AR_order} model on the last {AR_horizon} quarters, predicting present and forecasting {forecast_horizon-1} quarters into the future ...""")

    #Create the summary statistic df
    AR_summary = pd.DataFrame()
    AR_summary.index.name = 'prediction_date'


    # Iterate over all quarterly datapoints 
    for col in df_qoq.columns:

        # Select memory window
        data, data_index = select_col(col, AR_horizon)


        # Fit the model
        forecaster = AutoReg(data, lags=AR_order)
        results = forecaster.fit()

        # Save model diagnostics to df:
        col_results = AR_diagnostics(results)
        AR_summary = pd.concat([AR_summary, col_results.to_frame().T])

        # Create prediction df: qoq_forecast_df
        forecast_qoq = results.predict(start=len(data) +1 , end=len(data) + forecast_horizon) 

        # Save unindexed predictions
        qoq_forecast_df[col] = forecast_qoq

        # Save indexed predictions: qoq_forecast_index_df
        qoq_forecast_index_df = index_dict(col, data_index, forecast_qoq, qoq_forecast_index_df)





# Simple moving average of previous growth rates within the average_horizon
elif model == 'SMA':

    print(f"""Calculating forecasts as a simple moving average of the past {average_horizon} quarters, predicting present and forecasting {forecast_horizon - 1} quarters into the future ...""")


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

            # Shift data window for the next step
            data = np.append(data, sma)
            data = data[1:]

        
        # Save unindexed predictions
        qoq_forecast_df[col] = forecast_qoq

        # Save indexed predictions: qoq_forecast_index_df
        qoq_forecast_index_df = index_dict(col, data_index, forecast_qoq, qoq_forecast_index_df)




# Static average of previous growth rates within the average_horizon
elif model == 'AVERAGE':

    print(f"""Calculating forecasts as the static average of the past {average_horizon} quarters, predicting present and forecasting {forecast_horizon - 1} quarters into the future ...""")
    
    # Iterate over all quarterly datapoints 
    for col in df_qoq.columns:

        # Select memory window
        data, data_index = select_col(col, average_horizon)

        
        # Calculate the average, create forecast list
        average = data.mean()
        forecast_qoq = pd.Series([average] * forecast_horizon)


        # Save unindexed predictions
        qoq_forecast_df[col] = forecast_qoq

        # Save indexed predictions: qoq_forecast_index_df
        qoq_forecast_index_df = index_dict(col, data_index, forecast_qoq, qoq_forecast_index_df)




# --------------------------------------------------------------------------------------------------
#  If further options are required, put them here. 
#
#  Make sure to update the descriptions under Parameter Setup, the if-clauses under SAVE RESULTS 
#  and under "#Define the model subfolder", and the parameter validation check under valid_models
#  accordingly
# --------------------------------------------------------------------------------------------------


# Error message
else:
    print("This should never be printed, check whether valid_models is still up to date")







# ==================================================================================================
# PROCESS THE ESTIMATION RESULTS
# ==================================================================================================


# --------------------------------------------------------------------------------------------------
#  Build combined observed-forecasted dataframe
# --------------------------------------------------------------------------------------------------

# Store realized-predicted series in list:
series_list = []

# Loop through paired columns from df_qoq and qoq_forecast_df
for col_real, col_forecast in zip(df_qoq.columns, qoq_forecast_df.columns):

    # Select observed values to array
    real_values = df_qoq[col_real].dropna().to_numpy()

    # Get forecasts as array.
    forecast_values = qoq_forecast_df[col_forecast].to_numpy()

    # Concatenate real and forecast data.
    combined = np.concatenate([real_values, forecast_values])

    # Convert the combined array into a Series.
    s = pd.Series(combined, name=col_real)

    # Append the Series to the list.
    series_list.append(s)

# Concatenate all Series columnwise
df_combined_qoq = pd.concat(series_list, axis=1)

# Dynamicaly reset index
start_date = pd.to_datetime(df.index[0])  # Convert to datetime if not already
n_periods = len(df_combined_qoq)

# Generate quarterly datetime index
df_combined_qoq.index = pd.date_range(start=start_date, periods=n_periods, freq='QE')



# --------------------------------------------------------------------------------------------------
#  Create yearly values
# --------------------------------------------------------------------------------------------------

# Get growth factors (1+g), correct scale
df_combined_factor_qoq = 1 + (df_combined_qoq/100)

#show(df_combined_factor)

# Group by year and get yearly growth: (1+g_q1)*(1+g_q2)*(1+ g_q3)*(1+ g_q4), automatic NaN
df_combined_yoy = df_combined_factor_qoq.resample('YE').apply(lambda x: x.prod(skipna=False))

# Transform to growth in percent
df_combined_yoy = (df_combined_yoy - 1) * 100





# --------------------------------------------------------------------------------------------------
#  Reset indices and column names; Set Dates to Excel-friendly format
#
# 4 relevant dataframes: qoq_forecast_index_df, df_combined_qoq, df_combined_yoy, AR_summary
#
#  Renaming: -> indices back to YYYY-Qx format for QoQ, YYYY format for YoY
#            -> colnames to qoq_YYYY_0x and yoy_YYYY_0x
#            -> prediction col names to qoq_YYYY_0x_f
#            -> qoq_forecast_index_df: first col to YYYY_0x, second to YYYY_Qx
#
# --------------------------------------------------------------------------------------------------


# -------------------------------------------#
#         Define renaming functions          #  
# -------------------------------------------#


# Rename indices to YYYY-Qx
def rename_index_qoq(df):
    """
    Renames a DataFrame's datetime index to 'YYYY-Qx' format, where x is the quarter number (1 to 4).

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with renamed index.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index) 
    df.index = [f"{idx.year}-Q{(idx.month - 1)//3 + 1}" for idx in df.index]
    return df

# Rename indices to YYYY
def rename_index_yoy(df):
    """
    Renames a DataFrame's datetime index to 'YYYY-Qx' format, where x is the quarter number (1 to 4).

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with renamed index.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index) 
    df.index = [f"{idx.year}" for idx in df.index]
    return df


# Rename cols to prefix_YYYY_0x
def rename_col(prefix, df):
    """
    Renames datetime-like column names of a DataFrame to 'prefix_YYYY_0x' format,
    where x is the quarter number (1 to 4).

    Args:
        prefix (str): Prefix to prepend to the new column names.
        df (pd.DataFrame): Input DataFrame with datetime-like column names.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    df = df.copy()
    df.columns = [
        f"{prefix}_{col.year}_0{((col.month - 1) // 3 + 1)}"
        if isinstance(col, (pd.Timestamp, datetime)) else col
        for col in df.columns
    ]
    return df






# -------------------------------------------#
#  Apply functions: Rename index and column  #
#--------------------------------------------#

# index and cols: df_combined_qoq, df_combined_yoy
df_combined_qoq = rename_index_qoq(df_combined_qoq)
df_combined_qoq = rename_col('qoq', df_combined_qoq)

df_combined_yoy = rename_index_yoy(df_combined_yoy)
df_combined_yoy = rename_col('yoy', df_combined_yoy)

# index only: AR_summary
if 'AR_summary' in globals():
    AR_summary = rename_index_qoq(AR_summary)



# Manually rescale the values of qoq_forecast_index_df to strings

# Pre-formating to datetime:
qoq_forecast_index_df.iloc[:, 0] = pd.to_datetime(qoq_forecast_index_df.iloc[:, 0])
qoq_forecast_index_df.iloc[:, 1] = pd.to_datetime(qoq_forecast_index_df.iloc[:, 1])

# Reformat data using flexible code for col names:
col0_name = qoq_forecast_index_df.columns[0]
col1_name = qoq_forecast_index_df.columns[1]

# First col to YYYY_0x
qoq_forecast_index_df[col0_name] = qoq_forecast_index_df[col0_name].apply(
    lambda x: f"{x.year}_0{((x.month - 1) // 3 + 1)}"
)

# Second col to YYYY_Qx
qoq_forecast_index_df[col1_name] = qoq_forecast_index_df[col1_name].apply(
    lambda x: f"{x.year}_Q{((x.month - 1) // 3 + 1)}"
)






# ==================================================================================================
# SAVE RESULTS AS EXCEL: df_combined_qoq, df_combined_yoy,  qoq_forecast_index_df
# ==================================================================================================

# If clause for better naming of results
if model in ['AVERAGE', 'SMA']:

    # Full Data qoq
    filename_df_combined_qoq = f'Real_and_Predicted_QoQ_{model}_{average_horizon}_{forecast_horizon-1}.xlsx'
    df_combined_qoq.to_excel(os.path.join(folder_path, filename_df_combined_qoq))

    # Full Data yoy
    filename_df_combined_yoy = f'Real_and_Predicted_YoY_{model}_{average_horizon}_{forecast_horizon-1}.xlsx'
    df_combined_yoy.to_excel(os.path.join(folder_path, filename_df_combined_yoy)) 

    # Indexed Predictions df
    filename_qoq_forecast_index_df = f'Indexed_Forecasts_QoQ_{model}_{average_horizon}_{forecast_horizon-1}.xlsx'
    qoq_forecast_index_df.to_excel(os.path.join(folder_path, filename_qoq_forecast_index_df)) 


elif model == 'AR':

    # Full Data qoq
    filename_df_combined_qoq = f'Real_and_Predicted_QoQ_{model}{AR_order}_{AR_horizon}_{forecast_horizon-1}.xlsx'
    df_combined_qoq.to_excel(os.path.join(folder_path, filename_df_combined_qoq))

    # Full Data yoy
    filename_df_combined_yoy = f'Real_and_Predicted_YoY_{model}_{AR_horizon}_{forecast_horizon-1}.xlsx'
    df_combined_yoy.to_excel(os.path.join(folder_path, filename_df_combined_yoy)) 

    # Indexed Predictions df
    filename_qoq_forecast_index_df = f'Indexed_Forecasts_QoQ_{model}{AR_order}_{AR_horizon}_{forecast_horizon-1}.xlsx'
    qoq_forecast_index_df.to_excel(os.path.join(folder_path, filename_qoq_forecast_index_df)) 


# Faulty model selection
else:
    print("ERROR: wrong naming of outputs, check SAVE RESULTS section")


"""
Could also calculate and save absolute values, if needed ...
"""


# ----------------------------------------#
#    Save Model Summary if Model is AR    #
# ----------------------------------------#

# Check wether there is an AR_summary, save if yes
if 'AR_summary' in globals():
    filename_AR_summary = f'AR{AR_order}_{AR_horizon}_model_statistics.xlsx'
    AR_summary.to_excel(os.path.join(folder_path, filename_AR_summary))




# --------------------------------------------------------------------------------------------------
print(f" \n \nProgram complete, results are in Subfolder {result_subfolder} of working directory {wd} \n")
# --------------------------------------------------------------------------------------------------








# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#
