
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Helperfunctions
#
# Description:  This script sets up all functions usable across scripts, mainly for data processing,
#               QoQ and YoY conversion, Model construction in the Naive Forecaster and index renaming
# ==================================================================================================
# --------------------------------------------------------------------------------------------------


# Import built-ins
import importlib
import subprocess
import sys
import os
from datetime import datetime, date


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
#                                       Data Processing                                            #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


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

    print("Calculating year over year changes (relative to previous year)... \n")

    return df_out










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


