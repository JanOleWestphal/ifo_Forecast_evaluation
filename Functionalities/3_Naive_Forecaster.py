

# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Naive Forecaster
#
# Author:       Jan Ole Westphal
# Date:         2025-07
#
# Description:  A program for creating simple GDP-growth forecasting time series using the latest 
#               Bundesbank quaterly GDP releases. For every quaterly released data-series, a new 
#               forecast model is estimated.
#               Available models are described in the Parameter setup section. 
#               Puts out forecasts and time series into excel, designed for evaluating historic ifo 
#               forecasts against more naive methods, which is done in a different piece of code.            
# ==================================================================================================
# --------------------------------------------------------------------------------------------------





# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        Code begins here                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

print("\n Executing the Naive Forecaster Module ... \n")

# ==================================================================================================
#                                           SETUP
# ==================================================================================================

# Import built-ins
import importlib
import subprocess
import sys
import os
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

# Define the horizon of first releases which should be evaluated:
first_release_lower_limit_year = settings.first_release_lower_limit_year
first_release_lower_limit_quarter = settings.first_release_lower_limit_quarter

first_release_upper_limit_year = settings.first_release_upper_limit_year
first_release_upper_limit_quarter = settings.first_release_upper_limit_quarter


# ==================================================================================================
# Check correct parameter settings
# ==================================================================================================

# Include the nowcast
forecast_horizon = forecast_horizon + 1


# Check for correct naming convention
if naming_convention not in ['published', 'data']:
    raise ValueError(f"Invalid naming_convention: '{naming_convention}'. Must be 'published' or 'data'")

# Validate model type
valid_models = {'AR', 'AVERAGE', 'GLIDING_AVERAGE'}
for model in models:
    if model not in valid_models:
        raise ValueError(f"Invalid model '{model}'. Must be one of: {valid_models}")
    

# Validate all horizons are ≥1 or 'FULL'
for average_horizon in average_horizons:
    if not (isinstance(average_horizon, int) 
            and average_horizon >= 1 
            or average_horizon == "FULL"):
        raise ValueError(f"average_horizon ({average_horizon}) must be ≥1 or 'FULL'")

for AR_horizon in AR_horizons:
    if not (isinstance(AR_horizon, int) and AR_horizon >= 1 or AR_horizon == "FULL"):
        raise ValueError(f"AR_horizon ({AR_horizon}) must be ≥1 or 'FULL'")

if not (isinstance(forecast_horizon, int) and forecast_horizon >= 1):
    raise ValueError(f"forecast_horizon ({forecast_horizon}) must be ≥1")

    
# Model-specific validations: AR_lags and horizons
for model in models: 
    if model == 'AR':
        for AR_order in AR_orders:
            if AR_order < 0:
                raise ValueError(f"AR_order ({AR_order}) cannot be negative")
            for AR_horizon in AR_horizons:
                if AR_horizon != "FULL" and AR_horizon <= AR_order:
                    raise ValueError(
                        f"For {model} model, AR_horizon ({AR_horizon}) must be > AR_order ({AR_order})"
                        "\n(Need at least AR_order+1 observations to estimate model)"
                    )






# ==================================================================================================
# Setup the folder Structure
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Working Directory and results folder: Old Output
# --------------------------------------------------------------------------------------------------

## Define the result_subfolder path under wd\1_Result_Tables\1_Naive_Forecaster
base_result_folder = os.path.join(wd, '1_Result_Tables', '1_Naive_Forecaster_Outputs')
os.makedirs(base_result_folder, exist_ok=True)

## Clear
if settings.clear_result_folders:
    folder_clear(base_result_folder)

if resultfolder_name_n_forecast == 'Default':

    for model in models:
        if model in ['AVERAGE', 'GLIDING_AVERAGE']:
            result_subfolder = os.path.join(
                base_result_folder,
                f"Results_{model}_{average_horizon}_{forecast_horizon-1}"
            )

        elif model == 'AR':
            result_subfolder = os.path.join(
                base_result_folder,
                f"Results_{model}{AR_order}_{AR_horizon}_{forecast_horizon-1}"
            )

        else:
            print("ERROR: wrong naming of outputs, check SAVE RESULTS section")

else:
    # Use user-defined folder under the same base path
    result_subfolder = os.path.join(base_result_folder, resultfolder_name_n_forecast)




# --------------------------------------------------------------------------------------------------
# Data Outputs
# --------------------------------------------------------------------------------------------------

## Parent Folder
base_path = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data')
folder_path = os.path.join(base_path, result_subfolder)

## Define Resultfolder path
file_path_dt_qoq = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '0_Combined_QoQ_Forecasts')
file_path_dt_yoy = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '0_Combined_YoY_Forecasts')

file_path_forecasts_qoq = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '1_QoQ_Forecast_Tables')
file_path_forecasts_qoq_2 = os.path.join(wd, '0_1_Output_Data', '3_QoQ_Forecast_Tables')

file_path_forecasts_yoy = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '1_YoY_Forecast_Vectors')
file_path_forecasts_yoy_2 = os.path.join(wd, '0_1_Output_Data', '2_YoY_Forecast_Vectors')

# Create if needed
for folder in [base_path, folder_path, 
               file_path_dt_qoq, file_path_dt_yoy,
               file_path_forecasts_qoq, file_path_forecasts_qoq_2,
               file_path_forecasts_yoy, file_path_forecasts_yoy_2]:
    
    os.makedirs(folder, exist_ok=True)


# --------------------------------------------------------------------------------------------------
# Clear Result Folders
# --------------------------------------------------------------------------------------------------

## Clear
if settings.clear_result_folders:

    for folder in [folder_path, file_path_dt_qoq, file_path_dt_yoy,
                   file_path_forecasts_qoq, file_path_forecasts_qoq_2, 
                   file_path_forecasts_yoy, file_path_forecasts_yoy_2]:
        
        folder_clear(folder)
















# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        DATA PREPARATION                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

# ==================================================================================================
# LOAD PROCESSED DATA 
# ==================================================================================================

# Define directory
input_dir = os.path.join(wd, '0_0_Data', '2_Processed_Data', '1_GDP_series')

# Define file paths
df_path = os.path.join(input_dir, 'absolute_combined_GDP.xlsx')
df_qoq_path = os.path.join(input_dir, 'qoq_combined_GDP_data.xlsx')

# Load files
df = pd.read_excel(df_path, index_col=0)      
df_qoq = pd.read_excel(df_qoq_path, index_col=0)



# ==================================================================================================
# Select Evaluation timeframe
# ==================================================================================================

# Apply row and col selection from helperfunctions:
"""
for df in [df, df_qoq]:
    df = filter_first_release_limit(df)
    df = filter_evaluation_limit(df)
"""




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                   MAIN EXECUTION FUNCTIONS - Preparation and estimation                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

"""
Defines the naive forecast estimation workflow in order to make the for-loop  below more concise
"""

# ==================================================================================================
# PREPARE THE REGRESSION: creating DFs to store results
# ==================================================================================================
def prep_forecast_objects(df_qoq, forecast_horizon):
    """
    Prepare forecast objects with pre-allocated structure to prevent fragmentation.
    
    Parameters:
    -----------
    df_qoq : pd.DataFrame
        The quarterly data DataFrame containing the columns to forecast
    forecast_horizon : int
        Number of periods to forecast ahead
    
    Returns:
    --------
    tuple: (qoq_forecast_df, qoq_forecast_index_df)
    """
    # Pre-allocate prediction dataframe with known dimensions
    # This prevents fragmentation by creating the full structure upfront
    qoq_forecast_df = pd.DataFrame(
        index=range(forecast_horizon),
        columns=df_qoq.columns,
        dtype=float  # Specify dtype for better memory efficiency
    )
    
    # Create indexed dictionary of predictions
    qoq_forecast_index_df = pd.DataFrame(columns=['date_of_forecast', 'target_date', 'predicted_qoq'])
    
    # Clear workspace, if needed:
    """
    if 'AR_summary' in globals():
        del AR_summary
        # making sure that the AR- diagnostics table doesn't show up in other model outputs
    """
    
    return qoq_forecast_df, qoq_forecast_index_df



# ==================================================================================================
# MODEL ESTIMATION HELPERS: select memory horizon, build result df, get diagnostics
# ==================================================================================================


# --------------------------------------------------------------------------------------------------
# Create a function which selects the correct series as inputs in every column iteration: select_col
# --------------------------------------------------------------------------------------------------

def select_col(df_qoq, col, horizon):

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


# --------------------------------------------------------------------------------------------------
# Create a function building an indexed dictionary of forecasts
# --------------------------------------------------------------------------------------------------

def index_dict(col, data_index, forecast_qoq, qoq_forecast_index_df, forecast_horizon):

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
    forecast_index = pd.date_range(start=last_quarter, periods= forecast_horizon + 1, freq='QE')[1:]
        # This shift happens because the input to this function is qoq and not raw data, accordingly,
        # the last quarter where growth is available lies two quarters behind the publishing date, 
        # with the forecasting beginning at one quarter behind the publishing date
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


# --------------------------------------------------------------------------------------------------
# Create a function which selects the AR results and puts them into a df: AR_summary()
# --------------------------------------------------------------------------------------------------

def AR_diagnostics(col, results, AR_order):

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






# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                   MAIN EXECUTION FUNCTIONS - Output Generation and Processing                    #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

# ==================================================================================================
# GET QUARTERLY FORECASTS (same structure as the ifo quarterly forecasts)
# ==================================================================================================

def retrieve_qoq_predictions(qoq_forecast_df):
    """
    I: Set the index equal to colnames, shift every col such that the col name matches the row name on
    the first observation

    II: Store
    """

    ## ---------------------------------------------------------------------------
    ## Reshape the qoq Forecasts into the same format the ifo forecasts are in
    ## ---------------------------------------------------------------------------

    # Make a copy to avoid modifying the original DataFrame
    naive_qoq_forecasts = qoq_forecast_df.copy()

    # Convert columns to datetime if they aren't already
    columns_datetime = pd.to_datetime(naive_qoq_forecasts.columns)

    # Ensure the DataFrame index is datetime first
    naive_qoq_forecasts.index = pd.to_datetime(naive_qoq_forecasts.index, errors='coerce')

    # Drop rows where coercion failed (e.g. if index was pure integers)
    naive_qoq_forecasts = naive_qoq_forecasts[naive_qoq_forecasts.index.notna()]

    # Now extend the index safely
    columns_datetime = pd.to_datetime(naive_qoq_forecasts.columns)
    start_date = columns_datetime.min()
    max_shift_needed = len(columns_datetime)
    total_periods_needed = len(naive_qoq_forecasts) + max_shift_needed

    new_index = pd.date_range(start=start_date, periods=total_periods_needed, freq='3ME')

    # Safe union + sorting
    naive_qoq_forecasts = naive_qoq_forecasts.reindex(
    index=new_index.union(naive_qoq_forecasts.index)
    ).sort_index()

    #show(naive_qoq_forecasts)

    # Match on quarterly basis: Ensure index and columns are datetime (quarterly aligned)
    naive_qoq_forecasts.index = pd.to_datetime(naive_qoq_forecasts.index).to_period('Q').to_timestamp()
    naive_qoq_forecasts.columns = pd.to_datetime(naive_qoq_forecasts.columns).to_period('Q').to_timestamp()

    # Shift each column so that its first non-NA value aligns with its column date
    for col in naive_qoq_forecasts.columns:
        col_date = pd.to_datetime(col).to_period('Q').to_timestamp()

        # Check if column date exists in index
        if col_date in naive_qoq_forecasts.index:
            target_row = naive_qoq_forecasts.index.get_loc(col_date)
            naive_qoq_forecasts[col] = naive_qoq_forecasts[col].shift(target_row)

    # Drop empty rows
    naive_qoq_forecasts = naive_qoq_forecasts.dropna(how='all')

    #show(naive_qoq_forecasts)



    ## ---------------------------------------------------------------------------
    ## Store the Results
    ## ---------------------------------------------------------------------------

    # Store to two different locations
    for path in [file_path_forecasts_qoq, file_path_forecasts_qoq_2]:

        # Model-based dynamic naming
        if model in ['AVERAGE', 'GLIDING_AVERAGE']:
            qoq_forecast_name = f'naive_qoq_forecasts_{model}_{average_horizon}_{forecast_horizon-1}.xlsx'

        elif model == 'AR':
            qoq_forecast_name = f'naive_qoq_forecasts_{model}{AR_order}_{AR_horizon}_{forecast_horizon-1}.xlsx'

        # Store to path
        naive_qoq_forecasts.to_excel(os.path.join(path, qoq_forecast_name))


    return naive_qoq_forecasts









# ==================================================================================================
# JOIN OUTPUT WITH PUBLISHED VALUES: concatenate, rename indices, store results into wd
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Join Predicted and Published values 
# --------------------------------------------------------------------------------------------------

def join_forecaster_output(df_qoq, qoq_forecast_df):

    ## -------------------------------------------------
    ##  Build combined observed-forecasted dataframe
    ## -------------------------------------------------

    df_qoq = df_qoq.copy()
    qoq_forecast_df = qoq_forecast_df.copy()

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

    # Dynamicaly reset index, offset one quarter because first one is NA
    start_date = pd.to_datetime(df.index[0]) + pd.DateOffset(months=3)
    n_periods = len(df_combined_qoq)

    # Generate quarterly datetime index
    df_combined_qoq.index = pd.date_range(start=start_date, periods=n_periods, freq='QE')


    ## -------------------------------------------------
    ##  Create yearly values
    ## -------------------------------------------------

    df_combined_yoy = get_yoy(df_combined_qoq)


    ## Return
    return df_combined_qoq, df_combined_yoy



# --------------------------------------------------------------------------------------------------
# Save prediction time series with datetime indexing
# --------------------------------------------------------------------------------------------------

def save_dt_indexed_results(df_combined_qoq, df_combined_yoy):


    # If clause for dynamic naming of results
    if model in ['AVERAGE', 'GLIDING_AVERAGE']:

        # Full Data qoq
        filename_df_combined_qoq = f'dt_full_qoq{model}_{average_horizon}_{forecast_horizon-1}.xlsx'

        # Full Data yoy
        filename_df_combined_yoy = f'dt_full_yoy_{model}_{average_horizon}_{forecast_horizon-1}.xlsx' 

    elif model == 'AR':

        # Full Data qoq
        filename_df_combined_qoq = f'dt_full_qoq_{model}{AR_order}_{AR_horizon}_{forecast_horizon-1}.xlsx'

        # Full Data yoy
        filename_df_combined_yoy = f'dt_full_yoy_{model}_{AR_horizon}_{forecast_horizon-1}.xlsx'
        

    ## Store
    df_combined_qoq.to_excel(os.path.join(file_path_dt_qoq, filename_df_combined_qoq))
    df_combined_yoy.to_excel(os.path.join(file_path_dt_yoy, filename_df_combined_yoy)) 



# --------------------------------------------------------------------------------------------------
# Build and store YoY forecast Excels
# --------------------------------------------------------------------------------------------------

def get_yoy_forecast_series(df_combined_yoy, summer = False, winter = False):

    """
    Creates a df which matches the structure of the ifo and consensus forecast inputs and saves it 
    as an excel to <file_path_forecasts_yoy>


    """

    df_combined_yoy = df_combined_yoy.copy()

    # Rescale Index to yearly values
    df_combined_yoy.index = pd.to_datetime(df_combined_yoy.index).year
    #show(df_combined_yoy)

    ## Loop through release dates to extract forecasts
    records = []
    for col in df.columns:
        date = pd.to_datetime(col)
        yr = date.year

        # base record
        rec = {"date_of_forecast": date}

        y1 = yr + 1
        rec.update({
            "y_0":        yr,
            "y_0_forecast":      df_combined_yoy.at[yr, col],
            "y_1":        y1,
            "y_1_forecast":      df_combined_yoy.at[y1, col]
        })

        records.append(rec)

    # Build DataFrame, set forecast date as index
    yoy_forecast_series = pd.DataFrame.from_records(records).set_index("date_of_forecast")

    ## Filter if needed
    # Ensure index is datetime
    yoy_forecast_series.index = pd.to_datetime(yoy_forecast_series.index)

    # Filter based on seasonal parameters
    if summer:
        # Filter for Q2 (April, May, June - months 4, 5, 6)
        yoy_forecast_series = yoy_forecast_series[yoy_forecast_series.index.month.isin([4, 5, 6])]
    elif winter:
        # Filter for Q4 (October, November, December - months 10, 11, 12)
        yoy_forecast_series = yoy_forecast_series[yoy_forecast_series.index.month.isin([10, 11, 12])]

    # Dynamic Naming with seasonal suffix
    seasonal_suffix = "_full"
    if summer:
        seasonal_suffix = "_summer"
    elif winter:
        seasonal_suffix = "_winter"

    if model in ['AVERAGE', 'GLIDING_AVERAGE']:
        filename_yoy_forecast_series = f'forecast_series_yoy_{model}_{average_horizon}_{forecast_horizon-1}{seasonal_suffix}.xlsx'
    elif model == 'AR':
        filename_yoy_forecast_series = f'forecast_series_yoy_{model}{AR_order}_{AR_horizon}_{forecast_horizon-1}{seasonal_suffix}.xlsx'

    # Store to two locations
    yoy_forecast_series.to_excel(os.path.join(file_path_forecasts_yoy, filename_yoy_forecast_series), index=True)
    yoy_forecast_series.to_excel(os.path.join(file_path_forecasts_yoy_2, filename_yoy_forecast_series), index=True)

    return yoy_forecast_series




# --------------------------------------------------------------------------------------------------
# OLD REFERENCE OUTPUT: Save prediction time series as excel for evaluation 
# --------------------------------------------------------------------------------------------------

def save_renamed_results(df_combined_qoq, df_combined_yoy, qoq_forecast_index_df, AR_summary=None):

    # ==============================================================================================
    #  Reset indices and column names; Set Dates to Excel-friendly format
    #
    # 4 relevant dataframes: qoq_forecast_index_df, df_combined_qoq, df_combined_yoy, AR_summary
    #
    #  Renaming: 
    #            -> Applies switch accounting for whether cols are named by publishment data 
    #               or by last data point 
    #            -> indices back to YYYY-Qx format for QoQ, YYYY format for YoY
    #            -> colnames to qoq_YYYY_0x and yoy_YYYY_0x
    #            -> prediction col names to qoq_YYYY_0x_f
    #            -> qoq_forecast_index_df: first col to YYYY_0x, second to YYYY_Qx
    #
    # ==============================================================================================



    ## -------------------------------------------------
    ##     qoq_forecast_index_df: Manually rename
    ## -------------------------------------------------

    # Pre-formating to datetime:
    qoq_forecast_index_df.iloc[:, 0] = pd.to_datetime(qoq_forecast_index_df.iloc[:, 0])
    qoq_forecast_index_df.iloc[:, 1] = pd.to_datetime(qoq_forecast_index_df.iloc[:, 1])

    # Define colnames
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



    ## -------------------------------------------------
    ##               Rename combined dfs
    ## -------------------------------------------------

    ## Indices
    df_combined_qoq = rename_index_qoq(df_combined_qoq)
    # yoy has already been renamed at this stage

    if 'AR_summary' in globals():
        AR_summary = rename_index_qoq(AR_summary)


    ## Columns
    # QoQ and YoY
    if naming_convention == 'published':
        df_combined_qoq = rename_col_publish('qoq', df_combined_qoq)
        df_combined_yoy = rename_col_publish('yoy', df_combined_yoy)

    elif naming_convention == 'data':
        df_combined_qoq = rename_col_data('qoq', df_combined_qoq)
        df_combined_yoy= rename_col_data('yoy', df_combined_yoy)
    else: 
        print("This should never be printed, check whether naming_convention is still up to date")



    ## -------------------------------------------------
    ##          SAVE RESULTS AS EXCEL
    ## -------------------------------------------------

    # df_combined_qoq, df_combined_yoy,  qoq_forecast_index_df

    # If clause for better naming of results
    if model in ['AVERAGE', 'GLIDING_AVERAGE']:

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



    # ----------------------------------------#
    #    Save Model Summary if Model is AR    #
    # ----------------------------------------#

    # Check wether there is an AR_summary, save if yes
    if 'AR_summary' in globals():
        filename_AR_summary = f'AR{AR_order}_{AR_horizon}_model_statistics.xlsx'
        AR_summary.to_excel(os.path.join(folder_path, filename_AR_summary))







# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                              Define the Output Processing Workflow
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

def process_and_save_results(df_qoq, qoq_forecast_df, qoq_forecast_index_df, AR_summary):

            ## Process
            df_combined_qoq, df_combined_yoy = join_forecaster_output(df_qoq, qoq_forecast_df)

            ## Store Output

            # qoq Time Series
            retrieve_qoq_predictions(qoq_forecast_df)

            # DateTime indexed results
            save_dt_indexed_results(df_combined_qoq, df_combined_yoy)

            # YoY Forecasts
            yoy_forecast_series = get_yoy_forecast_series(df_combined_yoy)
            yoy_forecast_series_summer = get_yoy_forecast_series(df_combined_yoy, summer=True)
            yoy_forecast_series_winter = get_yoy_forecast_series(df_combined_yoy, winter=True)

            # Reformated combined time series
            save_renamed_results(df_combined_qoq, df_combined_yoy, qoq_forecast_index_df, AR_summary)















# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                   RUN THE NAIVE FORECASTS                                        #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


for model in models:

    # =============================================================================================#
    # Simple AR (with a constant) on previous growth rates within AR_horizon
    # =============================================================================================#

    if model == 'AR':

        for AR_order, AR_horizon in product(AR_orders, AR_horizons):

            ## Start Execution in the inner loop
            print(f""" Calculating an {model}{AR_order} model on the last {AR_horizon} quarters, predicting present and forecasting {forecast_horizon-1} quarters into the future ... \n""")

            ## Get the models
            #Create the summary statistic df
            AR_summary = pd.DataFrame()
            AR_summary.index.name = 'prediction_date'


            # Iterate over all quarterly datapoints 
            forecast_cols = {}
            index_dfs = []
            summary_rows = []

            # Iterate over all quarterly datapoints 
            for col in df_qoq.columns:

                # Select memory window
                data, data_index = select_col(df_qoq, col, AR_horizon)

                # Fit the model
                forecaster = AutoReg(data, lags=AR_order)
                results = forecaster.fit()

                # Collect model diagnostics
                col_results = AR_diagnostics(col, results, AR_order)
                summary_rows.append(col_results)

                # Generate predictions
                forecast_qoq = results.predict(start=len(data), end=len(data) + forecast_horizon - 1)
                forecast_cols[col] = pd.Series(forecast_qoq)


                # Collect indexed forecast DataFrames
                index_dfs.append(index_dict(col, data_index, forecast_qoq, pd.DataFrame(), forecast_horizon))

            # Combine diagnostics and forecasts
            AR_summary = pd.concat([r.to_frame().T for r in summary_rows], ignore_index=True)
            qoq_forecast_df = pd.concat(forecast_cols, axis=1)
            qoq_forecast_index_df = pd.concat(index_dfs, ignore_index=True)
            # show(qoq_forecast_df)


            ## Process and Save results
            process_and_save_results(df_qoq, qoq_forecast_df, qoq_forecast_index_df, AR_summary)








    # =============================================================================================#
    # Simple moving average of previous growth rates within the average_horizon
    # =============================================================================================#

    elif model == 'GLIDING_AVERAGE':

        for average_horizon in average_horizons:

            print(f""" Calculating forecasts as a moving average of the past {average_horizon} quarters, predicting present and forecasting {forecast_horizon - 1} quarters into the future ... \n""")
      
            ## Prepare the forecasting
            qoq_forecast_df, qoq_forecast_index_df = prep_forecast_objects(df_qoq, forecast_horizon)


            # Iterate over all quarterly datapoints 
            for col in df_qoq.columns:

                # Select memory window
                data, data_index = select_col(df_qoq, col, average_horizon)


                # List to collect forecasted values for this column
                forecast_qoq = []
                
                # Create #forecast_horizon prediction elements elements
                for _ in range(forecast_horizon):
                    # Compute the average of the current data window: GLIDING_AVERAGE
                    GLIDING_AVERAGE = data.mean()

                    # Build forecast list, save results
                    forecast_qoq.append(GLIDING_AVERAGE)

                    # Shift data window for the next step
                    data = np.append(data, GLIDING_AVERAGE)
                    data = data[1:]

                
                # Save unindexed predictions
                qoq_forecast_df[col] = forecast_qoq

                # Save indexed predictions: qoq_forecast_index_df
                qoq_forecast_index_df = index_dict(col, data_index, forecast_qoq, qoq_forecast_index_df, forecast_horizon)


            ## Process and Save results
            process_and_save_results(df_qoq, qoq_forecast_df, qoq_forecast_index_df, AR_summary)




    # =============================================================================================#
    # Static average of previous growth rates within the average_horizon
    # =============================================================================================#

    elif model == 'AVERAGE':

        for average_horizon in average_horizons:

            print(f""" Calculating forecasts as the static average of the past {average_horizon} quarters, predicting present and forecasting {forecast_horizon - 1} quarters into the future ... \n""")

            
            ## Prepare the forecasting
            qoq_forecast_df, qoq_forecast_index_df = prep_forecast_objects(df_qoq, forecast_horizon)


            # Iterate over all quarterly datapoints 
            for col in df_qoq.columns:

                # Select memory window
                data, data_index = select_col(df_qoq, col, average_horizon)

                
                # Calculate the average, create forecast list
                average = data.mean()
                forecast_qoq = pd.Series([average] * forecast_horizon)


                # Save unindexed predictions
                qoq_forecast_df[col] = forecast_qoq

                # Save indexed predictions: qoq_forecast_index_df
                qoq_forecast_index_df = index_dict(col, data_index, forecast_qoq, qoq_forecast_index_df, forecast_horizon)


            ## Process and Save results
            process_and_save_results(df_qoq, qoq_forecast_df, qoq_forecast_index_df, AR_summary)




    # ----------------------------------------------------------------------------------------------
    #  If further options are required, put them here. 
    #
    #  If you do, make sure to update the descriptions under Parameter Setup, the if-clauses under 
    #  SAVE RESULTS and under "#Define the model subfolder", and the parameter validation check under
    #  valid_models accordingly
    # ----------------------------------------------------------------------------------------------


    # Error message
    else:
        print("This should never be printed, check whether valid_models is still up to date")








# --------------------------------------------------------------------------------------------------
print(f" \n Naive Forecaster complete! \n",f"Find Results in {base_path}\n")
# --------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#