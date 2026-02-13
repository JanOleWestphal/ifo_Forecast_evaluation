
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        evalfunctions
#
# Author:       Jan Ole Westphal
# Date:         2025-07
#
# Description:  Subprogram defining all functions used for evaluating the forecasts; used 
#               in 4_Quarterly_Evaluation and 5_ifoCAST_Evaluation         
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

from __future__ import annotations

# ======================================
# Built-ins / standard library
# ======================================
import glob
import importlib
import math
import os
import re
import subprocess
import sys
from datetime import date, datetime
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# ======================================
# Third-party libraries
# ======================================
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import requests

from scipy.stats import norm

# ======================================
# Project imports
# ======================================
import ifo_forecast_evaluation_settings as settings


# -------------------------------------------------------------------------------------------------#
# Model ordering (plots)
# -------------------------------------------------------------------------------------------------#

def model_barplot_sort_key(model_name: str) -> tuple:
    """Return a sort key for model labels used in barplots.

    Ordering rule (requested):
      1) ifo
      2) ifoCAST (if applicable)
      3) AR models by number of lags, descending (AR3 before AR2)
      4) AVERAGE models by horizon, descending (AVERAGE_FULL before AVERAGE_10 before AVERAGE_1)
      5) everything else last (stable/lexicographic)

    Notes
    -----
    Model names in this project commonly look like:
      - 'AR2_50_9', 'AR3_40_9'
      - 'AVERAGE_1_9', 'AVERAGE_10_9', 'AVERAGE_FULL_9'
      - 'ifo', 'ifoCAST'
    """

    name = "" if model_name is None else str(model_name)
    lower = name.lower()

    # 1) ifo
    if lower == "ifo":
        return (0, 0, lower)

    # 2) ifoCAST
    if "ifocast" in lower:
        return (1, 0, lower)

    # 3) AR(p) models
    m_ar = re.search(r"ar\s*(\d+)", lower)
    if m_ar:
        p = int(m_ar.group(1))
        return (2, -p, lower)

    # 4) AVERAGE models
    if lower.startswith("average"):
        # expected: AVERAGE_<horizon>_<forecast_horizon> (horizon is int or 'FULL')
        parts = lower.split("_")
        horizon_token = parts[1] if len(parts) >= 2 else ""
        if horizon_token == "full":
            horizon_rank = 10**9
        else:
            try:
                horizon_rank = int(horizon_token)
            except ValueError:
                horizon_rank = -1
        return (3, -horizon_rank, lower)

    return (99, 0, lower)


def sort_models_for_barplots(model_names: Sequence[Any]) -> list[Any]:
    """Sort a list-like of model names according to `model_barplot_sort_key`."""
    names = list(model_names)
    return sorted(names, key=lambda x: model_barplot_sort_key(str(x)))








# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                   DATA PROCESSING FUNCTIONS                                      #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------#
# Table filter
# -------------------------------------------------------------------------------------------------#

def filter_df_by_datetime_index(
    df: pd.DataFrame,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    
    """
    Filter a DataFrame by applying lower and/or upper datetime bounds to its index.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a DatetimeIndex (or index convertible to datetime).
    start : str or pandas.Timestamp, optional
        Lower bound (inclusive). Rows with index < start are dropped.
    end : str or pandas.Timestamp, optional
        Upper bound (inclusive). Rows with index > end are dropped.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame.

    Notes
    -----
    - Bounds are inclusive.
    - If start or end is None, that side is left unfiltered.
    - The index is coerced to pandas.DatetimeIndex if needed.
    """

    out = df.copy()
    out.index = pd.to_datetime(out.index)

    if start is not None and start != "":
        start = pd.to_datetime(start)
        out = out.loc[out.index >= start]

    if end is not None and end != "":
        end = pd.to_datetime(end)
        out = out.loc[out.index <= end]

    return out




# ==================================================================================================
#                   SELECT ONLY FORECASTS AVAILABLE IN THE PRIMARY OBJECT OF INTEREST
# ==================================================================================================

def match_ifo_naive_forecasts_dates(ifo_qoq_forecasts, naive_qoq_dfs_dict):
    """
    This function matches the dates of the naive forecaster dataframes to the dates of the ifo forecasts.
    It removes any rows and columns from the naive forecaster dataframes that do not have a corresponding
    entry in the ifo forecasts dataframe.
    Parameters:
    ifo_qoq_forecasts (pd.DataFrame): DataFrame containing ifo quarterly forecasts with datetime index and columns.
    naive_qoq_dfs_dict (dict): Dictionary of DataFrames containing naive quarterly forecasts.
    Returns:
    dict: Updated dictionary of DataFrames with matched dates.
    """
    if settings.match_ifo_naive_dates:
        for key, naive_df in naive_qoq_dfs_dict.items():
        
            # Convert to datetime
            naive_df.index = pd.to_datetime(naive_df.index)
            naive_df.columns = pd.to_datetime(naive_df.columns)
            
            ifo_df = ifo_qoq_forecasts.copy()
            ifo_df.columns = pd.to_datetime(ifo_df.columns)
            ifo_df.index = pd.to_datetime(ifo_df.index)
            
            # Normalize both to quarterly level for matching
            naive_col_quarters = pd.PeriodIndex(naive_df.columns, freq='Q')
            naive_index_quarters = pd.PeriodIndex(naive_df.index, freq='Q')
            ifo_col_quarters = pd.PeriodIndex(ifo_df.columns, freq='Q')
            ifo_index_quarters = pd.PeriodIndex(ifo_df.index, freq='Q')
            
            # Build sets of quarters for matching
            ifo_col_quarter_set = set(ifo_col_quarters)
            ifo_index_quarter_set = set(ifo_index_quarters)
            
            # Filter columns: keep naive columns that match ifo quarters
            valid_col_mask = naive_col_quarters.isin(ifo_col_quarter_set)
            valid_cols = naive_df.columns[valid_col_mask]
            
            # Filter rows: keep naive rows that match ifo quarters
            valid_row_mask = naive_index_quarters.isin(ifo_index_quarter_set)
            valid_rows = naive_df.index[valid_row_mask]
            
            # Apply filters
            filtered_df = naive_df.loc[valid_rows, valid_cols]
            
            # Now match ifo_df columns to filtered_df columns on quarterly basis
            # Map ifo quarters to ifo columns
            ifo_quarter_to_col = dict(zip(ifo_col_quarters, ifo_df.columns))
            
            # For each column in filtered_df, find corresponding ifo column
            for col in filtered_df.columns:
                col_quarter = pd.Period(col, freq='Q')
                if col_quarter in ifo_quarter_to_col:
                    ifo_col = ifo_quarter_to_col[col_quarter]
                    # Get rows where ifo_df has non-NaN values for this quarter's column
                    ifo_col_data = ifo_df[ifo_col]
                    
                    # Match rows on quarterly basis
                    for row in filtered_df.index:
                        row_quarter = pd.Period(row, freq='Q')
                        # Find corresponding ifo row
                        ifo_row_matches = ifo_df.index[ifo_index_quarters == row_quarter]
                        if len(ifo_row_matches) > 0:
                            ifo_row = ifo_row_matches[0]
                            # If ifo doesn't have a value, set filtered_df to NaN
                            if pd.isna(ifo_col_data.loc[ifo_row]):
                                filtered_df.loc[row, col] = np.nan
                        else:
                            # No matching row in ifo_df
                            filtered_df.loc[row, col] = np.nan
            
            # Save back
            naive_qoq_dfs_dict[key] = filtered_df
        
        return naive_qoq_dfs_dict
    
    return naive_qoq_dfs_dict

"""
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
    """



# ==================================================================================================
#                                   RESHAPE DATAFRAMES FOR EVALUATION
# ==================================================================================================


# --------------------------------------------------------------------------------------------------
# Creating the joint evaluation df
# --------------------------------------------------------------------------------------------------

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



# --------------------------------------------------------------------------------------------------
# Create a df for quarterly forecast evaluation
# --------------------------------------------------------------------------------------------------

## ---------------------------------------------------------------------------
## General Version for the ifo QoQ and matched ifoCAST evaluation
## ---------------------------------------------------------------------------

def collapse_quarterly_prognosis(df, Ifocast_mode=False):
    """
    Move all cols to the same rows, rename rows Q1-Qx: gets quarterly error measures based on
    forecast horizons. If Ifocast_mode=True, shift first three columns down by 1 and label index as Qminus1, Q0, Q1, ...
    """



    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()

    
    for col in result_df.columns:
        # Drop NA
        non_missing = result_df[col].dropna()

        # Flatten non-scalar values
        clean_values = []
        for v in non_missing.values:
            if isinstance(v, (list, np.ndarray, pd.Series)):
                if len(v) > 0:
                    clean_values.append(float(v[0]))  # Take first element
                else:
                    clean_values.append(np.nan)
            elif v is None:
                clean_values.append(np.nan)
            else:
                clean_values.append(float(v))

        # Assign clean float values
        result_df[col] = np.nan
        result_df.iloc[:len(clean_values), result_df.columns.get_loc(col)] = clean_values

    
    # Find the maximum number of non-missing values across all columns
    max_non_missing = 0
    for col in result_df.columns:
        non_missing_count = result_df[col].notna().sum()
        max_non_missing = max(max_non_missing, non_missing_count)
    
    # Keep only the rows that contain data (up to max_non_missing)
    result_df = result_df.iloc[:max_non_missing]


    # If Ifocast_mode is True: rename index and shift first three columns
    """This is done because the data strucutre is almost identical to the one used in the qoq evaluation,
    but for the initial observation, Qminus is missing """
    if Ifocast_mode:
        result_df.index = ['Qminus1'] + [f'Q{i}' for i in range(max_non_missing - 1)]

        # Shift first three columns downward by one (introduce NA at top)
        for col in result_df.columns[:3]:
            result_df[col] = result_df[col].shift(1)

    else:
        # Default index naming: Q0, Q1, ...
        result_df.index = [f'Q{i}' for i in range(max_non_missing)]

    return result_df


## ---------------------------------------------------------------------------
## Special version for ifoCAST: cannot collapse to first rows as reporting scope is varying
## ---------------------------------------------------------------------------

def collapse_full_ifocast(df):
    """
    Collapse quarterly forecast data into proper quarter brackets (Qminus1, Q0, Q1) 
    based on datetime comparison between rows and columns.
    
    Args:
        df: DataFrame with pd.datetime, 'datetime_diff', 'datetime_eval' columns,
            followed by forecast data columns with datetime information in column names
    
    Returns:
        DataFrame with rows Qminus1, Q0, Q1 containing properly aligned quarterly data
    """

    # Ensure the DataFrame has a datetime index
    df.index = pd.to_datetime(df.index)

    # Create result dataframe with same columns as input
    result_df = pd.DataFrame(columns=df.columns, index=['Qminus1', 'Q0', 'Q1'])

    ## ---------------------------------------------------------------------------
    ## Helper Functions to extract datetime and quarter information
    ## ---------------------------------------------------------------------------

    # Helper function to extract datetime from column name (assuming first part is datetime)
    def extract_datetime_from_col(col_name):
        try:
            # Assuming datetime is at the beginning of column name
            # You may need to adjust this based on your actual column naming convention
            datetime_str = col_name.split('_')[0] if '_' in col_name else col_name
            return pd.to_datetime(datetime_str)
        except:
            return None
    
    # Helper function to get quarter from datetime
    def get_quarter_year(dt):
        if pd.isna(dt):
            return None
        return (dt.year, dt.quarter)
    
    ## ---------------------------------------------------------------------------
    ## Loop through all cols and then through all rows to determine data structure  
    ## ---------------------------------------------------------------------------
           
    # Process each column
    for col_idx, col in enumerate(df.columns):
            
        # Extract datetime from column name
        col_datetime = extract_datetime_from_col(col)
            
        col_quarter_year = get_quarter_year(col_datetime)
        
        # Process each row in this column
        for row_idx, row in df.iterrows():
            if pd.isna(df.loc[row_idx, col]):
                continue
                
            # Get row datetime from index
            if pd.isna(row_idx):
                continue
                
            row_quarter_year = get_quarter_year(row_idx)
            
            if row_quarter_year is None or col_quarter_year is None:
                continue
            
            # Calculate quarter difference
            row_year, row_quarter = row_quarter_year
            col_year, col_quarter = col_quarter_year
            
            # Convert to total quarters for comparison
            row_total_quarters = row_year * 4 + row_quarter
            col_total_quarters = col_year * 4 + col_quarter
            
            quarter_diff = col_total_quarters - row_total_quarters
            
            # Assign to appropriate result row based on quarter difference
            if quarter_diff == -1:  # Column is one quarter behind row
                target_row = 'Q1'
            elif quarter_diff == 0:  # Same quarter
                target_row = 'Q0'
            elif quarter_diff == 1:  # Column is one quarter ahead of row
                target_row = 'Qminus1'
            else:

                print(f"\n \nWarning: Value outside the Qminus1-Q0-Q1 range: {col} at {row_idx} with diff {quarter_diff} \n")

                continue  # Skip if difference is not -1, 0, or 1
            
            # Clean and assign the value
            value = df.loc[row_idx, col]
            if isinstance(value, (list, np.ndarray, pd.Series)):
                if len(value) > 0:
                    clean_value = float(value[0])
                else:
                    clean_value = np.nan
            elif value is None:
                clean_value = np.nan
            else:
                clean_value = float(value)
            
            result_df.loc[target_row, col] = clean_value
    
    return result_df







# ==================================================================================================
#                                    CALCULATE ERROR MEASURES
# ==================================================================================================


# --------------------------------------------------------------------------------------------------
# Compute the error series by forecast horizon
# --------------------------------------------------------------------------------------------------

def get_qoq_error_series(qoq_eval_df, save_path=None, file_name=None):
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
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    #show(qoq_eval_df)

    # Extract only columns ending with "_diff"
    diff_cols = [col for col in qoq_eval_df.columns if col.endswith("_diff")]
    diff_df = qoq_eval_df[diff_cols].copy()

    # Transpose error vectors: Q0-Qx become columns, forecast dates become rows
    error_by_horizon = diff_df.T
    error_by_horizon.columns.name = "forecast_horizon"

    # Initialize writer to collect multiple DataFrames into one sheet
    if save_path is not None and file_name is not None:
        output_path = os.path.join(save_path, file_name)
        error_by_horizon.to_excel(output_path)

    return error_by_horizon 



# --------------------------------------------------------------------------------------------------
# DECIDE WHETHER TO DROP ERRORS EXCEEDING A CERTAIN THRESHHOLD
# --------------------------------------------------------------------------------------------------

def drop_outliers(error_series, sd_cols=4, sd_threshold=3):
    """
    For the first `sd_cols` columns, drop values exceeding `sd_threshold * sd(col)`.
    For all later columns, use the last computed sd for thresholding.
    Prints how many observations were dropped per column.

    Parameters
    ----------
    error_series : pd.DataFrame
        DataFrame with datetime rows and Qx columns.
    sd_cols : int
        Number of columns to compute individual standard deviations for.
    sd_threshold : float
        Threshold multiplier for determining outliers.

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers replaced by NaN.
    """

    print(f"\nDropping outliers in Error df of shape {error_series.shape} when exceeding sd x {sd_threshold}")

    # Setup
    result = error_series.copy()
    last_sd = None

    # Get col-wise standard errors, option to use the lastest sd for later cols with little observations
    for i, col in enumerate(error_series.columns):
        if i < sd_cols:
            col_sd = error_series[col].std(skipna=True)
            last_sd = col_sd
        else:
            col_sd = last_sd

        threshold = sd_threshold * col_sd
        mask = result[col].abs() > threshold
        dropped_count = mask.sum()

        if dropped_count > 0:
            print(f"{col}: Dropped {dropped_count} observations "
                  f"(>{sd_threshold} × {col_sd:.4f} = {threshold:.4f})")
        else:
            print(f"{col}: No observations dropped")

        result[col] = result[col].mask(mask)

    return result





# --------------------------------------------------------------------------------------------------
# Get an Excel/df with error statistics by forecast horizon
# --------------------------------------------------------------------------------------------------

def get_qoq_error_statistics_table(error_by_horizon, release_name=None, save_path=None, file_name=None):
    """
    Compute error statistics by forecast horizon and optionally save to Excel.

    Parameters
    ----------
    error_by_horizon : pd.DataFrame
        DataFrame with forecast horizons as columns and time indices as rows.
    release_name : str, optional
        Sheet name for Excel output.
    save_path : str, optional
        Folder path where file should be saved.
    file_name : str, optional
        Name of the output Excel file.

    Returns
    -------
    pd.DataFrame
        Concatenated error statistics table by forecast horizon.
    """

    all_tables = []

    for horizon in error_by_horizon.columns:
        forecast_series = error_by_horizon[horizon].dropna()
        if forecast_series.empty:
            continue

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

    full_error_measure_table = pd.concat(all_tables)

    # Save only if path and name are provided
    if save_path and file_name:
        output_path = os.path.join(save_path, file_name)
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            full_error_measure_table.to_excel(writer, sheet_name=str(release_name or "results"))

    return full_error_measure_table






# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                 NOWCAST EVAL DATA STRUCTURE                                      #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#



# -------------------------------------------------------------------------------------------------#
# Error series generator
# -------------------------------------------------------------------------------------------------#

def error_columns_summary_nowcast(df, prefix="error"):
    """
    Compute error statistics for all columns starting with `prefix`.

    Statistics computed per column:
        ME   : mean error
        MAE  : mean absolute error
        MSE  : mean squared error
        RMSE : root mean squared error
        SE   : standard error of the mean error
        N    : number of non-missing observations

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing error columns.
    prefix : str, default="error"
        Prefix identifying error columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by error column name with columns:
        ["ME", "MAE", "MSE", "RMSE", "SE", "N"]
    """

    rows = []

    for col in df.columns:
        if not col.startswith(prefix):
            continue

        e = df[col].dropna()
        n = len(e)

        if n == 0:
            stats = dict(ME=np.nan, MAE=np.nan, MSE=np.nan,
                         RMSE=np.nan, SE=np.nan, N=0)
        else:
            me = e.mean()
            mse = (e ** 2).mean()
            stats = dict(
                ME=me,
                MAE=e.abs().mean(),
                MSE=mse,
                RMSE=np.sqrt(mse),
                SE=e.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                N=n,
            )

        rows.append(pd.Series(stats, name=col))

    summary_df = pd.DataFrame(rows)

    # Rescale and Rename
    summary_df["N"] = summary_df["N"].astype("Int64")
    summary_df.index = summary_df.index.astype(str).str.replace(
        "^error_realized_minus_", "", regex=True
    )

    return summary_df











# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                       STATISTICAL TESTS                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#



# --------------------------------------------------------------------------------------------------
# Diebold-Mariano
# --------------------------------------------------------------------------------------------------

def dm_test(e1, e2, h=1, loss="mse", alternative="two-sided", hln=True):
    """
    Diebold–Mariano (DM) test for equal predictive accuracy between two forecasts.

    This function implements the Diebold–Mariano test using a loss differential
    (MSE or MAE) and a HAC (Newey–West–type) variance estimator to account for
    autocorrelation in multi-step-ahead forecast errors.

    Parameters
    ----------
    e1 : array-like
        Forecast errors from model 1 (actual minus forecast).
    e2 : array-like
        Forecast errors from model 2 (actual minus forecast).
    h : int, default=1
        Forecast horizon. For h > 1, the loss differential is assumed to be
        autocorrelated up to lag h-1.
    loss : {"mse", "mae"}, default="mse"
        Loss function used to form the loss differential:
        - "mse": squared error loss
        - "mae": absolute error loss
    alternative : {"two-sided", "less", "greater"}, default="two-sided"
        Alternative hypothesis:
        - "two-sided": E[d] != 0
        - "less":      E[d] < 0  (model 1 has lower expected loss)
        - "greater":   E[d] > 0  (model 2 has lower expected loss)
    hln : bool, default=True
        Whether to apply the Harvey–Leybourne–Newbold small-sample correction
        (recommended for h > 1).

    Returns
    -------
    dict
        Dictionary containing:
        - "stat":   DM test statistic
        - "pvalue": p-value under the chosen alternative
        - "mean_d": sample mean of the loss differential
        - "T":      number of forecast observations
        - "h":      forecast horizon
    """

    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    assert e1.shape == e2.shape
    T = len(e1)

    # Construct loss differential d_t
    if loss.lower() == "mse":
        d = e1**2 - e2**2
    elif loss.lower() == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    dbar = d.mean()

    # Sample autocovariance of d_t at lag 'lag'
    def autocov(x, lag):
        x0 = x - x.mean()
        if lag == 0:
            return np.dot(x0, x0) / T
        return np.dot(x0[lag:], x0[:-lag]) / T

    # HAC variance of the sample mean of d_t
    gamma0 = autocov(d, 0)
    var_dbar = gamma0
    for lag in range(1, h):
        var_dbar += 2 * autocov(d, lag)
    var_dbar /= T

    dm = dbar / sqrt(var_dbar)

    # Harvey–Leybourne–Newbold small-sample correction
    if hln and h > 1:
        adj = sqrt((T + 1 - 2*h + (h*(h-1))/T) / T)
        dm *= adj

    # p-value computation
    if alternative == "two-sided":
        p = 2 * (1 - norm.cdf(abs(dm)))
    elif alternative == "less":
        p = norm.cdf(dm)
    elif alternative == "greater":
        p = 1 - norm.cdf(dm)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    return {"stat": dm, "pvalue": p, "mean_d": dbar, "T": T, "h": h}






# --------------------------------------------------------------------------------------------------
# Clark-West
# --------------------------------------------------------------------------------------------------


def cw_test(y, yhat1, yhat2, h=1):
    """
    Clark–West (CW) test for equal predictive accuracy of nested forecast models.

    This test adjusts the standard MSPE comparison to account for the bias that
    arises when comparing a smaller (benchmark) model to a larger nested model.
    It is appropriate only when model 2 nests model 1.

    The test is typically conducted as a one-sided test with the alternative
    hypothesis that the larger (nested) model has lower MSPE.

    Parameters
    ----------
    y : array-like
        Realized values of the target variable.
    yhat1 : array-like
        Forecasts from the benchmark (smaller) model.
    yhat2 : array-like
        Forecasts from the nested (larger) model.
    h : int, default=1
        Forecast horizon. For h > 1, HAC variance estimation is used with
        truncation lag h-1.

    Returns
    -------
    dict
        Dictionary containing:
        - "stat":              CW test statistic
        - "pvalue_one_sided":  one-sided p-value for the null of no improvement
        - "mean_f":            sample mean of the CW-adjusted loss differential
        - "T":                 number of forecast observations
        - "h":                 forecast horizon
    """

    # Reformat and check shapes
    y = np.asarray(y, dtype=float)
    yhat1 = np.asarray(yhat1, dtype=float)
    yhat2 = np.asarray(yhat2, dtype=float)
    assert y.shape == yhat1.shape == yhat2.shape
    T = len(y)

    # Forecast errors
    e1 = y - yhat1
    e2 = y - yhat2

    # CW-adjusted loss differential f_t
    f = e1**2 - e2**2 + (yhat1 - yhat2)**2
    fbar = f.mean()

    # Sample autocovariance of f_t
    def autocov(x, lag):
        x0 = x - x.mean()
        if lag == 0:
            return np.dot(x0, x0) / T
        return np.dot(x0[lag:], x0[:-lag]) / T

    # HAC variance of the sample mean of f_t
    gamma0 = autocov(f, 0)
    var_fbar = gamma0
    for lag in range(1, h):
        var_fbar += 2 * autocov(f, lag)
    var_fbar /= T

    cw = fbar / sqrt(var_fbar)

    # One-sided p-value: H1 = nested model improves MSPE
    p_one_sided = 1 - norm.cdf(cw)

    return {"stat": cw, "pvalue_one_sided": p_one_sided, "mean_f": fbar, "T": T, "h": h}





# --------------------------------------------------------------------------------------------------
# Output Tables for DM and CW tests
# --------------------------------------------------------------------------------------------------

def results_dicts_to_latex(
    results: Union[Mapping[str, Mapping[str, Any]], Sequence[Mapping[str, Any]]],
    *,
    # --- flexible annotations / labels ---
    row_labels: Optional[Sequence[str]] = None,
    row_keys: Optional[Sequence[str]] = None,
    label_key: str = "label",
    # --- selecting + ordering metrics from each dict ---
    include: Optional[Sequence[str]] = None,
    rename: Optional[Mapping[str, str]] = None,
    sort_rows_by: Optional[str] = None,
    # --- formatting ---
    digits: int = 3,
    sci_threshold: float = 1e6,
    nan_as: str = "",
    # --- LaTeX table controls ---
    caption: Optional[str] = None,
    latex_label: Optional[str] = None,
    booktabs: bool = True,
    align: Optional[str] = None,
    escape_underscores: bool = True,
    # --- optional “double header line” compact style ---
    double_header: bool = True,
    # --- output ---
    save_path: Optional[str] = None,
) -> str:
    """
    Combine a flexible number of result dictionaries (e.g. DM/CW outputs) into a LaTeX table
    and optionally save it to disk.

    Parameters
    ----------
    results
        Either:
        (A) dict-of-dicts: {row_name: result_dict, ...}
        (B) list/tuple of result_dicts.
    save_path
        If provided, the LaTeX table is written to this path (should end in '.tex').

    All other parameters control formatting, selection, and annotation of the table.

    Returns
    -------
    str
        LaTeX table as a string.
    """

    def _is_nan(x: Any) -> bool:
        try:
            return x is None or (isinstance(x, float) and math.isnan(x))
        except Exception:
            return x is None

    def _fmt(x: Any) -> str:
        if _is_nan(x):
            return nan_as
        if isinstance(x, (int,)) and not isinstance(x, bool):
            return str(x)
        if isinstance(x, (float,)) and not isinstance(x, bool):
            ax = abs(x)
            if ax != 0 and ax >= sci_threshold:
                return f"{x:.{digits}e}"
            return f"{x:.{digits}f}"
        return str(x)

    def _esc(s: str) -> str:
        return s.replace("_", "\\_") if escape_underscores else s

    def _header_wrap(s: str) -> str:
        if not double_header:
            return _esc(s)
        if "_" in s:
            a, b = s.split("_", 1)
            return _esc(a) + r"\\ " + _esc(b)
        return _esc(s)

    # --- normalize input ---
    rows: List[Tuple[str, Mapping[str, Any]]] = []

    if isinstance(results, Mapping):
        items = list(results.items())
        if row_labels is None:
            rows = [(str(k), v) for k, v in items]
        else:
            if len(row_labels) != len(items):
                raise ValueError("row_labels length must match results.")
            rows = [(str(lbl), v) for lbl, (_, v) in zip(row_labels, items)]
    else:
        dicts = list(results)
        if row_labels is not None:
            if len(row_labels) != len(dicts):
                raise ValueError("row_labels length must match results.")
            rows = [(str(lbl), d) for lbl, d in zip(row_labels, dicts)]
        else:
            rows = [(str(d.get(label_key, f"Row {i+1}")), d) for i, d in enumerate(dicts)]

    # --- infer columns ---
    if row_keys is None:
        key_union = set().union(*(d.keys() for _, d in rows))
        preferred = ["stat", "pvalue", "pvalue_one_sided", "mean_d", "mean_f", "T", "h"]
        row_keys = [k for k in preferred if k in key_union] + \
                   sorted(k for k in key_union if k not in preferred)

    if include is not None:
        row_keys = [k for k in row_keys if k in set(include)]

    if rename is None:
        rename = {}

    # --- sorting ---
    if sort_rows_by is not None:
        rows = sorted(
            rows,
            key=lambda x: (
                _is_nan(x[1].get(sort_rows_by)),
                x[1].get(sort_rows_by, float("inf"))
            )
        )

    # --- LaTeX assembly ---
    headers = [""] + [rename.get(k, k) for k in row_keys]

    if align is None:
        align = "l" + "r" * len(row_keys)

    if booktabs:
        top, mid, bot = r"\toprule", r"\midrule", r"\bottomrule"
    else:
        top = mid = bot = r"\hline"

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
    ]

    if caption is not None:
        lines.append(rf"\caption{{{_esc(caption)}}}")
    if latex_label is not None:
        lines.append(rf"\label{{{latex_label}}}")

    lines.extend([
        rf"\begin{{tabular}}{{{align}}}",
        top,
        " & ".join([""] + [_header_wrap(h) for h in headers[1:]]) + r" \\",
        mid
    ])

    for lbl, d in rows:
        row = [_esc(lbl)] + [_fmt(d.get(k)) for k in row_keys]
        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        bot,
        r"\end{tabular}",
        r"\end{table}"
    ])

    latex_str = "\n".join(lines)

    # --- write to disk if requested ---
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(latex_str)

    return latex_str












# --------------------------
# Example usage
# --------------------------
# dm_res = dm_test(e1, e2, h=4, loss="mse")
# cw_res = cw_test(y, yhat1, yhat2, h=4)
#
# table = results_dicts_to_latex(
#     [dm_res, cw_res],
#     row_labels=["DM: ModelA vs ModelB", "CW: Bench vs Nested"],
#     include=["stat", "pvalue", "pvalue_one_sided", "T", "h"],
#     rename={"stat": "TestStat", "pvalue": "p-value", "pvalue_one_sided": "p-value_one_sided"},
#     caption="Forecast accuracy tests",
#     latex_label="tab:dm_cw",
#     digits=3,
#     double_header=True
# )
# print(table)

















# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                       PLOTTER FUNCTIONS                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#



# --------------------------------------------------------------------------------------------------
# Error Time Series
# --------------------------------------------------------------------------------------------------

def plot_forecast_timeseries(*args, df_eval=None, title_prefix=None,
                             figsize=(12, 8), linestyle='-', linewidth=1.5, evaluation_legend_name='Realized Values',
                             show=False, save_path=None, save_name_prefix=None, select_quarters=None):
    """
    Create time series plots comparing forecast horizons across multiple DataFrames.
    
    Parameters:
    -----------
    *args : DataFrame or dict
        Variable number of DataFrames or dictionaries containing DataFrames
    df_eval : pd.DataFrame
        1D datetime-indexed DataFrame serving as evaluation/ground truth
    title_prefix : str, optional
        Prefix for plot titles
    figsize : tuple, default (12, 8)
        Figure size (width, height)
    show : bool, default False
        Whether to display plots
    save_path : str, optional
        Path to save figures
    save_name_prefix : str, optional
        Prefix for saved figure names
    select_quarters : list, optional
        List of quarter indices to display (e.g., [0,1,2,3,4,6,9])
        If None, displays all quarters Q0-Q9
    
    Returns:
    --------
    dict: Dictionary containing all created figures
    """

    ## ---------------------------------------------------------------------------
    ## DATA PREPROCESSING
    ## ---------------------------------------------------------------------------

    ## LOAD IN DF TO PLOT
    
    # Collect all DataFrames and their names
    dfs_to_plot = []
    
    for arg in args:
        if isinstance(arg, dict):

            # If it's a dictionary, add all DataFrames in it
            for name, df in arg.items():
                # Drop Qminus1 row for the ifoCAST case
                if 'Qminus1' in df.index:
                    df = df.drop(index='Qminus1')

                # Save the dfs
                dfs_to_plot.append((name, df))


        elif isinstance(arg, pd.DataFrame):

            # Drop Qminus1 row for the ifoCAST case
            if 'Qminus1' in arg.index:
                arg = arg.drop(index='Qminus1')

            # If it's a DataFrame, assume it's the ifo forecast
            dfs_to_plot.append(('ifo', arg))
        else:
            raise ValueError("Arguments must be DataFrames or dictionaries containing DataFrames")
    



    ## PROCESS EACH DATAFRAME TO CREATE Q0-Q9 TIME SERIES

    # Drop Qminus1 row for the ifoCAST case
    
    processed_dfs = {}
    
    for name, df in dfs_to_plot:
        # Filter columns: keep timestamp columns, drop _diff and _eval columns
        timestamp_cols = []
        for col in df.columns:
            if not (col.endswith('_diff') or col.endswith('_eval')):
                try:
                    # Try to convert to datetime to verify it's a timestamp column
                    pd.to_datetime(col)
                    timestamp_cols.append(col)
                except (ValueError, TypeError):
                    # Skip non-timestamp columns
                    continue
        
        if not timestamp_cols:
            print(f"Warning: No valid timestamp columns found in DataFrame '{name}'")
            continue
        
        # Convert column names to datetime for proper sorting
        timestamp_cols_dt = [(col, pd.to_datetime(col)) for col in timestamp_cols]
        timestamp_cols_dt.sort(key=lambda x: x[1])  # Sort by datetime
        timestamp_cols = [col for col, _ in timestamp_cols_dt]
        
        # Pre-allocate Q0-Q9 DataFrames to prevent fragmentation
        q_dfs = {}
        for q in range(10):
            q_dfs[f'Q{q}'] = pd.DataFrame(index=pd.DatetimeIndex([]), columns=['value'], dtype=float)
        
        # Process each timestamp column
        for col in timestamp_cols:
            series = df[col].dropna()  # Remove NA values
            col_datetime = pd.to_datetime(col)
            
            # Assign each non-NA value to the corresponding Q dataframe
            for i, (idx, value) in enumerate(series.items()):
                """
                if i == 'minus1': #Special case for ifoCAST evaluation
                    q_key = f'Qminus1'
                    # Calculate target date: index date + forecast horizon
                    target_date = idx - pd.DateOffset(months=3)  # Quarterly offset
                    
                    # Create new row and append to avoid fragmentation
                    new_row = pd.DataFrame({'value': [value]}, index=[target_date])
                    q_dfs[q_key] = pd.concat([q_dfs[q_key], new_row])
                """

                if i < 10:  # Only process first 10 non-NA values (Q0-Q9)
                    q_key = f'Q{i}'
                    # Calculate target date: index date + forecast horizon
                    target_date = idx + pd.DateOffset(months=3*i)  # Quarterly offset
                    
                    # Create new row and append to avoid fragmentation
                    new_row = pd.DataFrame({'value': [value]}, index=[target_date])
                    q_dfs[q_key] = pd.concat([q_dfs[q_key], new_row])
        
        # Sort indices and remove duplicates
        for q in range(10):
            q_key = f'Q{q}'
            if not q_dfs[q_key].empty:
                q_dfs[q_key] = q_dfs[q_key].sort_index()
                q_dfs[q_key] = q_dfs[q_key][~q_dfs[q_key].index.duplicated(keep='first')]
        
        processed_dfs[name] = q_dfs
    
    

    # ———— Determine overall start date ————
    # Prefer the earliest Q0 from the 'ifo' series, else global minimum
    ifo_q0_start = None
    global_q0_start = None
    for name, q_dfs in processed_dfs.items():
        if 'Q0' in q_dfs and not q_dfs['Q0'].empty:
            q0_start = q_dfs['Q0'].index.min()
            if name.lower() == 'ifo':
                ifo_q0_start = q0_start
            if global_q0_start is None or q0_start < global_q0_start:
                global_q0_start = q0_start
    earliest_start = ifo_q0_start if ifo_q0_start is not None else global_q0_start

    # ———— Also determine the last available date across all Q‑series ————
    latest_end = None
    for name, q_dfs in processed_dfs.items():
        for q_df in q_dfs.values():
            if not q_df.empty:
                last_date = q_df.index.max()
                # pick the most recent (largest) last_date across all series
                if latest_end is None or last_date > latest_end:
                    latest_end = last_date

    # ———— Apply the two‑sided cut‑off to every Q‑series ————
    if earliest_start is not None and latest_end is not None:
        for name, q_dfs in processed_dfs.items():
            for q_key, q_df in q_dfs.items():
                if not q_df.empty:
                    q_dfs[q_key] = q_df[
                        (q_df.index >= earliest_start) &
                        (q_df.index <= latest_end)
                    ]

    # ———— Filter evaluation data to [earliest_start, latest_end] ————
    eval_filtered = None
    if df_eval is not None and earliest_start is not None and latest_end is not None:
        eval_filtered = df_eval[
            (df_eval.index >= earliest_start) &
            (df_eval.index <= latest_end)
        ]
    elif df_eval is not None and earliest_start is not None:
        eval_filtered = df_eval[df_eval.index >= earliest_start]
    elif df_eval is not None:
        eval_filtered = df_eval




    ## Prepare Plots
    
    # Determine which quarters to plot
    quarters_to_plot = list(range(10)) if select_quarters is None else select_quarters


    ## Create color palette for plotting
    def get_color_palette(n_series):
        """Generate colors for Q0-Q9 series"""
        if n_series <= 10:
            # Use a colormap for Q0-Q9
            cmap = plt.get_cmap("tab10")
            return [cmap(i) for i in range(n_series)]
        else:
            # Use viridis for more series
            cmap = plt.get_cmap("viridis")
            return [cmap(i/n_series) for i in range(n_series)]
    

    ## Get dict to store plots
    figures = {}


    ## Ensure save path exists if saving is requested
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)



    ## ---------------------------------------------------------------------------
    ## Plot 1: For each input DataFrame, plot all Q0-Q9 against evaluation
    ## ---------------------------------------------------------------------------

    for name, q_dfs in processed_dfs.items():
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot evaluation data first (if provided)
        if eval_filtered is not None:
            ax.plot(eval_filtered.index, eval_filtered.iloc[:, 0], 
                   color='black', linewidth=2, label=evaluation_legend_name, alpha=0.8)
        
        # Plot Q0-Q9 series (only selected quarters)
        colors = get_color_palette(len(quarters_to_plot))
        color_idx = 0
        for q in quarters_to_plot:
            q_key = f'Q{q}'
            if q_key in q_dfs and not q_dfs[q_key].empty:

                # Align the Q0-Q0 forecasts with their target dates
                shifted_index = q_dfs[q_key].index.to_period('Q') - q -1  # shift by q quarters
                shifted_index = shifted_index.to_timestamp(how='end')  # convert back to datetime

                ax.plot(shifted_index, q_dfs[q_key]['value'], 
                       color=colors[color_idx], marker='o', linewidth=linewidth, linestyle=linestyle,
                       markersize=4, label=q_key, alpha=0.7)
                color_idx += 1
        
        # Customize plot
        #ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('QoQ GDP Growth', fontsize=12)
        plot_title = f'{title_prefix} - {name} Predictions vs Realized Values' if title_prefix else f'{name} Predictions vs Realized Values'
        ax.set_title(plot_title, fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        # Save figure
        if save_path and save_name_prefix:
            save_name = f"{save_name_prefix}_{name}_horizons.png"
            fig.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches='tight')
        
        figures[f'{name}_horizons'] = fig

        plt.close()
    

    ## ---------------------------------------------------------------------------
    ## Plot 2: For each Q0-Q9, plot across all input DataFrames (only selected quarters)
    ## ---------------------------------------------------------------------------

    for q in quarters_to_plot:
        q_key = f'Q{q}'
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot evaluation data first (if provided)
        if eval_filtered is not None:
            ax.plot(eval_filtered.index, eval_filtered.iloc[:, 0], 
                   color='black', linewidth=2, label=evaluation_legend_name, alpha=0.8)
        
        # Plot Q series from each DataFrame
        colors = get_color_palette(len(processed_dfs))
        for i, (name, q_dfs) in enumerate(processed_dfs.items()):
            if q_key in q_dfs and not q_dfs[q_key].empty:
                # Create legend label
                if 'ifo' in name.lower():
                    legend_label = f'ifo'# {q_key} ahead'
                elif any(tag in name.lower() for tag in ['ar', 'sma', 'average', 'ifocast']):
                    # Strip trailing underscore + number
                    legend_label = f"{re.sub(r'_\d+$', '', name)}" #{q_key} ahead"
                else:
                    legend_label = name

                
                # Align the Q0-Q0 forecasts with their target dates
                shifted_index = q_dfs[q_key].index.to_period('Q') - q -1 # shift by q quarters
                shifted_index = shifted_index.to_timestamp(how='end')  # convert back to datetime
                
                ax.plot(shifted_index, q_dfs[q_key]['value'], 
                       color=colors[i], marker='o', linewidth=linewidth, linestyle=linestyle,
                       markersize=4, label=legend_label, alpha=0.7)
        
        # Customize plot
        #ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('QoQ GDP Growth', fontsize=12)
        plot_title = f'{title_prefix} - {q_key} Forecasts vs. Realized Values' if title_prefix else f'{q_key} Forecasts vs. Realized Values'
        ax.set_title(plot_title, fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        # Save figure
        if save_path and save_name_prefix:
            save_name = f"{save_name_prefix}_{q_key}_comparison.png"
            fig.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches='tight')
        
        figures[f'{q_key}_comparison'] = fig

        plt.close()
    

    ## ---------------------------------------------------------------------------
    ## Plot 3: Combined plot showing all selected quarters from all DataFrames
    ## ---------------------------------------------------------------------------

    if len(quarters_to_plot) > 0:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot evaluation data first (if provided)
        if eval_filtered is not None:
            ax.plot(
                eval_filtered.index, eval_filtered.iloc[:, 0],
                color='black', linewidth=3, label=evaluation_legend_name, alpha=0.9
            )
        
        # Prepare base colors for each model
        model_names = list(processed_dfs.keys())
        base_colors = {}
        for name in model_names:
            if 'ifo' in name.lower():
                base_colors[name] = '#3237DE'
            else:
                # pick a distinct tab10 color based on model index
                idx = model_names.index(name)
                cmap = plt.get_cmap('tab10')
                base_colors[name] = mcolors.to_hex(cmap(idx % 10))
        

        # Plot each model-quarter series with shading from darkest (Q0) to lightest (Q9)
        for name, q_dfs in processed_dfs.items():
            base = base_colors[name]
            for q in quarters_to_plot:
                q_key = f'Q{q}'
                series = q_dfs.get(q_key)
                if series is None or series.empty:
                    continue

                # generate a lighter shade for quarter q
                color = mcolors.to_rgb(base)
                white = (1, 1, 1)
                raw = q / (len(quarters_to_plot) - 1 if len(quarters_to_plot) > 1 else 1)
                factor = raw * 0.7  # Scale down to avoid too light colors
                shade = tuple(color[i] + (white[i] - color[i]) * factor for i in range(3))
                shade_hex = mcolors.to_hex(shade)

                # Shift index back by q quarters
                shifted_index = series.index.to_period('Q') - q -1
                shifted_index = shifted_index.to_timestamp(how='end')  # or 'start' if preferred

                if 'ifo' in name.lower():
                    ax.plot(
                        shifted_index, series['value'],
                        color=shade_hex, linewidth=1.5, linestyle='-',
                        alpha=0.8
                    )
                else:
                    ax.scatter(
                        shifted_index, series['value'],
                        color=shade_hex, s=30,
                        alpha=0.4
                    )
            # Add a legend for the base colors
            ax.scatter([], [], color=base_colors[name], label=name)

        

        # Customize plot
        #ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('QoQ GDP Growth', fontsize=12)
        plot_title = f'{title_prefix} - All Selected Quarters' if title_prefix else 'All Selected Quarters'
        ax.set_title(plot_title, fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        # Save figure
        if save_path and save_name_prefix:
            save_name = f"{save_name_prefix}_all_selected_quarters.png"
            fig.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches='tight')
        
        figures['all_selected_quarters'] = fig

        plt.close()
    
    return figures






# --------------------------------------------------------------------------------------------------
# Error Distribution Plots
# --------------------------------------------------------------------------------------------------

def plot_error_lines(*args, title: Optional[str] = None, figsize: tuple = (12, 8),
                       n_bars: int = 10, show: bool = False, ifocast_mode: bool = False,
                       save_path: Optional[str] = None, save_name: Optional[str] = None):
    """
    Create a plot with vertical lines for columns Q0-Q9, where each value is plotted as a point.
    
    Parameters:
    -----------
    *args : DataFrame or dict of DataFrames
        Input data containing columns Q0-Q9
    title : str, optional
        Plot title
    figsize : tuple, default (12, 8)
        Figure size (width, height)
    n_bars : int, default 10
        Number of vertical lines (should match Q0-Q9 columns)
    show : bool, default False
        Whether to display the plot
    save_path : str, optional
        Directory path to save the plot
    save_name : str, optional
        Filename for saving the plot
    """
    
    # Collect all dataframes
    dfs = []
    df_names = []
    
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            dfs.append(arg)
            if ifocast_mode:
                df_names.append(f"ifoCAST")
            else:
                df_names.append(f"ifo Forecast")
                
        elif isinstance(arg, dict):
            for name, df in arg.items():
                if isinstance(df, pd.DataFrame):
                    dfs.append(df)
                    df_names.append(name)
        else:
            raise ValueError(f"Unsupported input type: {type(arg)}. Expected DataFrame or dict of DataFrames.")
    
    if not dfs:
        raise ValueError("No valid DataFrames found in input arguments.")
    
    # Expected columns
    q_cols = [f'Q{i}' for i in range(min(n_bars, 10))]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    ## Color pallete
    cmap = plt.get_cmap("Oranges")
    colors = []
    
    # Count non-ifo entries for color scaling
    non_ifo_count = sum(1 for name in df_names if 'ifo' not in name.lower() or 'ifocast' in name.lower())
    non_ifo_idx = 0
    
    for name in df_names:
        if 'ifocast' in name.lower():
            colors.append('#FA4600')
        elif 'ifo' in name.lower():
            colors.append('#3237DE')  # dark blue for ifo
        else:
            # Sample from the dark end of orange colormap (0.7 to 1.0 range)
            color_val = 0.7 + 0.3 * non_ifo_idx / max(1, non_ifo_count - 1)
            color = cmap(color_val)
            colors.append(mcolors.to_hex(color))
            non_ifo_idx += 1
    
    # Plot data for each dataframe
    for df_idx, (df, df_name) in enumerate(zip(dfs, df_names)):
        # Check which Q columns exist in this dataframe
        available_cols = [col for col in q_cols if col in df.columns]
        
        if not available_cols:
            print(f"Warning: No Q columns found in {df_name}")
            continue
        
       # Plot each column's values
        for col_idx, col in enumerate(available_cols):
            if col in df.columns:
                # Get non-null values
                values = df[col].dropna()
                
                # X position for this column (with slight offset for multiple dataframes)
                x_pos = col_idx + (df_idx - len(dfs)/2 + 0.5) * 0.1
                
                # Add jitter to x-position to show overlapping points
                x_jitter = np.random.normal(0, 0.02, len(values))
                
                # Plot points with jitter
                ax.scatter([x_pos] * len(values) + x_jitter, values, 
                          color=colors[df_idx], alpha=0.7, s=25, 
                          edgecolors='white', linewidths=0.5,
                          label=f"{df_name}" if col_idx == 0 else "")
                

    # Draw vertical lines
    for i in range(len(q_cols)):
        ax.axvline(x=i, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Customize plot
    ax.set_xlabel('Forecast Horizon', fontsize=12)
    ax.set_ylabel('Error in p.p.', fontsize=12)
    ax.set_title(title if title else 'Distribution of Forecast Errors by Horizon', fontsize=14)

    # Set visual marker at 0
    ax.axhline(0, color='#1B263B', linewidth=1, linestyle='--', alpha=1)
    
    # Set x-axis
    ax.set_xticks(range(len(q_cols)))
    ax.set_xticklabels(q_cols)
    ax.set_xlim(-0.5, len(q_cols) - 0.5)
    
    # Add legend if multiple dataframes
    if len(dfs) > 1:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if specified
    if save_path and save_name:
        import os
        full_path = os.path.join(save_path, save_name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
    
    # Show plot if specified
    if show:
        plt.show()

    # Close to save memory
    plt.close()
    
    return fig, ax








# --------------------------------------------------------------------------------------------------
# Error Bar Plots
# --------------------------------------------------------------------------------------------------

def plot_quarterly_metrics(*args, metric_col='MSE', title=None, figsize=(12, 8), 
                           scale_by_n=True, n_bars=10, show=False,
                           save_path=None, save_name=None):
    """
    Create a bar plot comparing quarterly metrics across multiple DataFrames.
    
    Parameters:
    -----------
    *args : DataFrame or dict
        Variable number of DataFrames or dictionaries containing DataFrames
    metric_col : str, default 'MSE'
        Column name to plot (e.g., 'MSE', 'ME', 'RMSE')
    title : str, optional
        Plot title. If None, uses f'Quarterly {metric_col} Comparison'
    figsize : tuple, default (12, 8)
        Figure size (width, height)
    scale_by_n : bool, default False
        If True, scales bar width by values in column 'N' (number of observations)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Collect all DataFrames and their names
    dfs_to_plot = []
    
    for arg in args:
        if isinstance(arg, dict):
            # If it's a dictionary, add all DataFrames in it
            for name, df in arg.items():
                dfs_to_plot.append((name, df))
        elif isinstance(arg, pd.DataFrame):
            # If it's a DataFrame, its the ifo forecast
            dfs_to_plot.append((f'ifo', arg))
        else:
            raise ValueError("Arguments must be DataFrames or dictionaries containing DataFrames")

    # Enforce consistent model ordering for barplots
    dfs_to_plot = sorted(dfs_to_plot, key=lambda pair: model_barplot_sort_key(pair[0]))
    
    # Check if metric column exists in all DataFrames
    for name, df in dfs_to_plot:
        if metric_col not in df.columns:
            raise ValueError(f"Column '{metric_col}' not found in DataFrame '{name}'")
        if scale_by_n and 'N' not in df.columns:
            raise ValueError(f"Column 'N' not found in DataFrame '{name}' (required when scale_by_n=True)")
    
    # Filter to Q0-Q9 rows and extract metric values
    quarters = [f'Q{i}' for i in range(n_bars)]
    
    # If scaling by N, calculate normalized widths
    if scale_by_n:
        # Get all N values across all DataFrames for normalization
        all_n_values = []
        for name, df in dfs_to_plot:
            for q in quarters:
                if q in df.index and not pd.isna(df.loc[q, 'N']):
                    all_n_values.append(df.loc[q, 'N'])
        
        if all_n_values:
            max_n = max(all_n_values)
            min_n = min(all_n_values)
            n_range = max_n - min_n if max_n != min_n else 1
        else:
            max_n, min_n, n_range = 1, 1, 1
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up bar positions
    x_positions = np.arange(len(quarters))
    bar_width = 0.8 / len(dfs_to_plot)
    
    ## Color palette
    # Generate warm colour gradient for dictionary entries
    n_dict_entries = len(dfs_to_plot) - 1  # minus the ifo
    # Get a darker part of the orange colormap
    cmap = plt.get_cmap("Oranges")

    n_entries = len(dfs_to_plot)
    colors = []

    for i, (name, _) in enumerate(dfs_to_plot):
        if 'ifocast' in name.lower():
            colors.append('#703316')

        elif 'ifo' in name.lower():
            colors.append('#003366')  # dark blue
        else:
            # Sample from the dark end (closer to 0.8–1.0 in colormap)
            color = cmap(0.7 + 0.3 * i / max(1, n_entries - 1))  # 0.7 to 1 range
            colors.append(mcolors.to_hex(color))
    

    # Plot bars for each DataFrame
    for i, (name, df) in enumerate(dfs_to_plot):
        # Filter to Q0-Q9 rows that exist in the DataFrame
        available_quarters = [q for q in quarters if q in df.index]
        values = [df.loc[q, metric_col] if q in df.index else np.nan for q in quarters]
        
        # Create legend label based on DataFrame name
        if any(tag in name.lower() for tag in ['ar', 'sma', 'average', 'ifocast']):
            # Strip trailing underscore + number (e.g. "_2", "_10", etc.)
            legend_label = re.sub(r'_\d+$', '', name)
        elif  'ifo' in name.lower():
            legend_label = 'ifo'

        else:
            legend_label = name
        
        # Calculate bar widths
        if scale_by_n:
            # Get N values for width scaling
            n_values = [df.loc[q, 'N'] if q in df.index and not pd.isna(df.loc[q, 'N']) else min_n for q in quarters]
            # Normalize N values to bar width (0.1 to 0.8 range)
            widths = [0.1 + 0.7 * (n - min_n) / n_range for n in n_values]
        else:
            widths = [bar_width] * len(quarters)
        
        # Plot bars with potentially different widths
        for j, (quarter, value, width) in enumerate(zip(quarters, values, widths)):
            if not np.isnan(value):
                # Centered grouping: offset each model's bar within the group
                x_offset = x_positions[j] - (len(dfs_to_plot) - 1) * bar_width / 2 + i * bar_width
                
                bar = ax.bar(x_offset, value, width, 
                            label=legend_label if j == 0 else "", 
                            color=colors[i], alpha=0.8)

                ax.text(bar[0].get_x() + bar[0].get_width()/2, bar[0].get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    
    # Customize the plot
    ax.set_xlabel('Forecast Horizon (Quarters)', fontsize=12)
    ax.set_ylabel(metric_col, fontsize=12)
    ax.set_title(title if title else f'Quarterly {metric_col} Comparison', fontsize=14)
    ax.set_xticks(x_positions)
    #ax.set_xticks(x_positions + bar_width * (len(dfs_to_plot) - 1) / 2)
    ax.set_xticklabels(quarters)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Show if desired:
    if show:
        plt.show()

    # Save
    # Save figure if path and name are provided
    if save_path is not None and save_name is not None:
        fig.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches='tight')
    

    # Close to save memory
    plt.close()
    
    return fig, ax























