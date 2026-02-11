
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Helperfunctions
#
# Description:  This script sets up all functions usable across scripts, mainly for data processing,
#               QoQ and YoY conversion, Model construction in the Naive Forecaster and index renaming
# ==================================================================================================
# --------------------------------------------------------------------------------------------------

from __future__ import annotations

# Import built-ins
import importlib
import subprocess
import sys
import os
import glob
from datetime import datetime, date
from typing import Union, Dict, Optional, Mapping,  Iterable, List, Sequence



# Import libraries
import requests
import pandas as pd
from pandasgui import show  #uncomment this to allow for easier debugging
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
# import matplotlib.pyplot as plt
# import seaborn as sns

import ifo_forecast_evaluation_settings as settings





# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                      FOLDER PREPARATION                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

# Clearing Function
def folder_clear(folder_path): 
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                       Data Processing                                            #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# ==================================================================================================
# Flexible df merger
# ==================================================================================================

def merge_quarterly_dfs_dropna(
    dfs: Sequence[pd.DataFrame],
    col_names: Sequence[str],
    *,
    quarter: str = "Q",            # "Q" for quarter-end, "QS" for quarter-start
    join: str = "outer",           # "outer" (union) or "inner" (intersection)
    dropna: bool = True,
    tz_aware_ok: bool = True,
) -> pd.DataFrame:
    """
    Merge multiple n×1 time series into a single DataFrame on a common quarterly index,
    enforcing a common quarterly date standard for ALL inputs.

    What this does
    -------------
    1) Converts each input index to a quarterly PeriodIndex (freq='Q').
    2) Converts that PeriodIndex back to a DatetimeIndex using a common standard:
       - quarter="Q"  -> quarter-end timestamps (e.g. 2024-03-31, 2024-06-30, ...)
       - quarter="QS" -> quarter-start timestamps (e.g. 2024-01-01, 2024-04-01, ...)
    3) Renames each series to the provided `col_names`.
    4) Concatenates all series column-wise and optionally drops any row with missing data.

    Parameters
    ----------
    dfs : sequence of pandas.DataFrame
        Each element must be n×1 and have an index convertible to datetime or period.
    col_names : sequence of str
        Column names for the merged output. Must match len(dfs).
    quarter : {"Q", "QS"}, default="Q"
        Quarterly timestamp standard to enforce:
        - "Q": quarter-end
        - "QS": quarter-start
    join : {"outer", "inner"}, default="outer"
        How to align indices when concatenating:
        - "outer": union of quarters across series
        - "inner": intersection of quarters across series
    dropna : bool, default=True
        If True, drop all rows that have at least one missing value after merging.
    tz_aware_ok : bool, default=True
        If True, timezone-aware DatetimeIndex inputs are converted to naive timestamps
        (timezone removed) before quarterly conversion.

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame with a common quarterly DatetimeIndex and named columns.
    """
    if len(dfs) != len(col_names):
        raise ValueError("dfs and col_names must have the same length.")

    if quarter not in {"Q", "QS"}:
        raise ValueError("quarter must be either 'Q' (quarter-end) or 'QS' (quarter-start).")

    # how parameter for to_timestamp
    how = "end" if quarter == "Q" else "start"

    series_list = []
    for df, name in zip(dfs, col_names):
        if df.shape[1] != 1:
            raise ValueError("Each DataFrame must be n×1 (exactly one column).")

        out = df.copy()

        # --- convert index to datetime/period safely ---
        if isinstance(out.index, pd.PeriodIndex):
            # Ensure quarterly periods
            periods = out.index.asfreq("Q")
        else:
            # DatetimeIndex or other -> coerce to datetime
            dt = pd.to_datetime(out.index)

            # If timezone-aware, optionally drop timezone info
            if hasattr(dt, "tz") and dt.tz is not None:
                if not tz_aware_ok:
                    raise ValueError("Timezone-aware DatetimeIndex found; set tz_aware_ok=True to allow.")
                dt = dt.tz_convert(None)

            periods = dt.to_period("Q")

        # --- enforce common quarterly timestamp standard ---
        out.index = periods.to_timestamp(how=how)

        # rename column
        out.columns = [name]
        series_list.append(out)

    merged = pd.concat(series_list, axis=1, join=join)

    if dropna:
        merged = merged.dropna(how="any")

    return merged



# ==================================================================================================
# Helper functions for mid-quarterly alignment
# ==================================================================================================

def normalize_to_mid_quarter(dates):
    """
    Convert any datetime-like input to mid-quarter dates.
    Q1 -> Feb 15, Q2 -> May 15, Q3 -> Aug 15, Q4 -> Nov 15
    
    Args:
        dates: DatetimeIndex, Series, or Index
    
    Returns:
        DatetimeIndex with normalized mid-quarter dates
    """
    dt_index = pd.to_datetime(dates)
    quarters = dt_index.to_period('Q')
    
    # Map quarter to mid-month (15th of middle month)
    mid_months = {1: 2, 2: 5, 3: 8, 4: 11}
    
    def get_mid_quarter_date(period):
        year = period.year
        month = mid_months[period.quarter]
        return pd.Timestamp(year=year, month=month, day=15)
    
    return pd.DatetimeIndex([get_mid_quarter_date(q) for q in quarters])


def align_df_to_mid_quarters(df):
    """Normalize DataFrame index to mid-quarter dates."""
    df_copy = df.copy()
    df_copy.index = normalize_to_mid_quarter(df.index)
    return df_copy




# ==================================================================================================
# Select Evaluation timeframe
# ==================================================================================================

## Filter columns: only keep columns >= first_release_lower_limit_year/quarter
def filter_first_release_limit(df, 
                               
                               first_release_lower_limit_year = settings.first_release_lower_limit_year,
                                first_release_lower_limit_quarter = settings.first_release_lower_limit_quarter,

                                first_release_upper_limit_year = settings.first_release_upper_limit_year,
                                first_release_upper_limit_quarter = settings.first_release_upper_limit_quarter
                                ):

    # Filter cols prior to first release limit
    col_mask = [

        # Filter lower limit
        ((col.year > first_release_lower_limit_year) or
        (col.year == first_release_lower_limit_year and (col.quarter >= first_release_lower_limit_quarter)))

        and # Filter upper limit
        ((col.year < first_release_upper_limit_year) or
        (col.year == first_release_upper_limit_year and (col.quarter <= first_release_upper_limit_quarter)))


        for col in df.columns
    ]

    # Apply filter
    df = df.loc[:, col_mask]

    #show(df)

    return df



## Filter rows: only keep rows >= evaluation_limit_year/quarter
def filter_evaluation_limit(df, 
                                evaluation_limit_year = settings.evaluation_limit_year,
                                evaluation_limit_quarter = settings.evaluation_limit_quarter):

    # Filter rows prior to evaluation limit
    row_mask = [
        (idx.year > evaluation_limit_year) or
        (idx.year == evaluation_limit_year and (idx.quarter >= evaluation_limit_quarter))
        for idx in df.index
    ]

    # Apply
    df = df.loc[row_mask, :]

    #show(df)

    return df





# --------------------------------------------------------------------------------------------------
# Processing the values of Bundesbank GDP: get numerics, rescale index to datetime
# --------------------------------------------------------------------------------------------------

def process_BB_GDP(df, col_convert=True, col_subset = True):

    ## Convert all data to float
    try:
        # Replace commas with dots -> perform float conversion
        df = df.apply(lambda col: col.str.replace(',', '.') if col.dtype == 'object' else col)
        df = df.apply(pd.to_numeric, errors='raise')

    except ValueError as e:
        raise ValueError("Possible Processing error: GDP-data can not be converted into floats") from e


    ## Convert Column Names to Datetime, filter duplicates
    if col_convert:
        # Rescale cols to datetime
        df.columns = pd.to_datetime(df.columns, format='%d.%m.%Y', errors='raise')

    if col_subset: 
        # Drop columns not in Feb, May, Aug, Nov
        months_to_keep = [2, 5, 8, 11]  
        df = df.loc[:, df.columns.month.isin(months_to_keep)]

    ## Set the index to a timestamp
    df.index = pd.PeriodIndex(df.index, freq='Q').to_timestamp()

    ## Add ~45 days to shift to the middle of each quarter
    df.index += pd.offsets.Day(45)
    #print("Index set to datetime and middle of the quarter ...")

    return df



# --------------------------------------------------------------------------------------------------
# Create quarterly growth rates: df_qoq (Quarter over Quarter)
# --------------------------------------------------------------------------------------------------

def get_qoq(df):

    # Calclulate change % in t as relative backward differences, where .shift(1) calls t-1 values for t
    df_out = ((df - df.shift(1)) / df.shift(1)) *100

    print("Calculating quarter over quarter changes (relative to previous quarter)... \n")

    return df_out


# --------------------------------------------------------------------------------------------------
# Create yearly growth rates or changes: df_yoy (Year over Year)
# --------------------------------------------------------------------------------------------------

def get_yoy(df):

    """
    Make sure to apply this function to a qoq data frame
    """
    # Get growth factors (1+g), correct scale
    df_factor = 1 + (df/100)

    # Use this to get Q4 vs Q4 YoY growth
    """
    # Calculation by geometric mean
    # Group by year and get yearly growth: (1+g_q1)*(1+g_q2)*(1+ g_q3)*(1+ g_q4), automatic NaN
    df_combined_yoy = df_combined_factor_qoq.resample('YE').apply(lambda x: x.prod(skipna=False))

    # Transform to growth in percent
    df_combined_yoy = (df_combined_yoy - 1) * 100
    """

    ## avg(Q1+Q2+Q3+Q4) YoY growth	
    # base_level_gdp = 100
    df_out = df_factor.cumprod(axis=0) #* base_level_gdp

    # take yearly arithmetic mean of GDP
    df_out = df_out.resample('YE').sum(min_count=4)

    # Calculate YoY growth rates
    df_out = df_out.pct_change(fill_method=None) * 100

    print(" Calculating year over year changes (relative to previous year)... \n \n")

    return df_out




# ==================================================================================================
# Dynamically extract prelimiary results of previous scripts
# ==================================================================================================

## Helper function to load all Excel files in a folder into a dict of DataFrames
def load_excels_to_dict(folder_path, strip_string=None, filter=None):
    excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
    dfs = {}
    for file in excel_files:

        ## OPTIONAL: filter files by string in filename
        filename = os.path.basename(file)
        # Skip if filename does not match filter
        if isinstance(filter, str) and filter not in filename:
            continue


        name = os.path.splitext(os.path.basename(file))[0]
        
        # Strip the custom string if provided
        if strip_string:
            name = name.replace(strip_string, '')
        
        df = pd.read_excel(file, index_col=0)
        # Align to mid-quarter dates if index contains datetime-like data
        try:
            df = align_df_to_mid_quarters(df)
        except (TypeError, ValueError):
            # If alignment fails, keep original index
            pass
        dfs[name] = df
    return dfs










# -------------------------------------------------------------------------------------------------#
# ==================================================================================================
#                                       INDEX RENAMING
# ==================================================================================================
# -------------------------------------------------------------------------------------------------#


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
    Renames a DataFrame's datetime index to 'YYYY' format.

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with renamed index.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index) 
    df.index = [f"{idx.year}" for idx in df.index]
    return df


# Rename cols to prefix_YYYY_0x based on available date of data publishment
def rename_col_publish(prefix, df):
    """
    Renames datetime-like column names of a DataFrame to '<prefix>_YYYY_0x' format,
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



# Rename cols to prefix_YYYY_0x based on date of last available datapoint
def rename_col_data(prefix, df):
    """
    
    Renames datetime-like column names of a DataFrame to '<prefix>_YYYY_0x' format,
    where x is the quarter number (1 to 4). Does this quick and dirty by shifting the col datetime
    back by one quarter.

    Args:
        prefix (str): Prefix to prepend to the new column names.
        df (pd.DataFrame): Input DataFrame with datetime-like column names.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    df = df.copy()
    # Quick and dirty offset
    df.columns -= pd.offsets.Day(90)
    df.columns = [
        f"{prefix}_{col.year}_0{((col.month - 1) // 3 + 1)}"
        if isinstance(col, (pd.Timestamp, datetime)) else col
        for col in df.columns
    ]
    return df









# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                             Analyzing Error Measures - Functions                                 #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# ==================================================================================================
#                                       Data Preparation
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




# ==================================================================================================
#                                       Error Measures
# ==================================================================================================

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


