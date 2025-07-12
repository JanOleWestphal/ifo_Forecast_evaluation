
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

# Import built-ins
import importlib
import subprocess
import sys
import os
import glob
import re
from datetime import datetime, date
from itertools import product
from typing import Union, Dict, Optional


# Import libraries
import requests
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm










# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                   DATA PROCESSING FUNCTIONS                                      #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#



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
    
    # Collect all DataFrames and their names
    dfs_to_plot = []
    
    for arg in args:
        if isinstance(arg, dict):
            # If it's a dictionary, add all DataFrames in it
            for name, df in arg.items():
                dfs_to_plot.append((name, df))
        elif isinstance(arg, pd.DataFrame):
            # If it's a DataFrame, assume it's the ifo forecast
            dfs_to_plot.append(('ifo', arg))
        else:
            raise ValueError("Arguments must be DataFrames or dictionaries containing DataFrames")
    
    # Process each DataFrame to create Q0-Q9 time series
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
    
    # Create color palette
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
    
    figures = {}
    
    # Determine overall date range: Prefer earliest Q0 from 'ifo', fallback to global minimum
    ifo_q0_start = None
    global_q0_start = None

    for name, q_dfs in processed_dfs.items():
        if 'Q0' in q_dfs and not q_dfs['Q0'].empty:
            q0_start = q_dfs['Q0'].index.min()
            # Check for 'ifo'
            if name.lower() == 'ifo':
                ifo_q0_start = q0_start
            if global_q0_start is None or q0_start < global_q0_start:
                global_q0_start = q0_start

    # Prioritise ifo-based starting date if available
    earliest_start = ifo_q0_start if ifo_q0_start is not None else global_q0_start

    # ———— Apply the cut‑off to every Q‑series ————
    if earliest_start is not None:
        for name, q_dfs in processed_dfs.items():
            for q_key, q_df in q_dfs.items():
                # only keep dates ≥ earliest_start
                if not q_df.empty:
                    q_dfs[q_key] = q_df[q_df.index >= earliest_start]
    
    # Filter evaluation data to start from earliest Q0 date
    eval_filtered = None
    if df_eval is not None and earliest_start is not None:
        eval_filtered = df_eval[df_eval.index >= earliest_start]
    elif df_eval is not None:
        eval_filtered = df_eval
    
    # Determine which quarters to plot
    quarters_to_plot = list(range(10)) if select_quarters is None else select_quarters
    
    # Plot 1: For each input DataFrame, plot all Q0-Q9 against evaluation
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
                ax.plot(q_dfs[q_key].index, q_dfs[q_key]['value'], 
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
    
    # Plot 2: For each Q0-Q9, plot across all input DataFrames (only selected quarters)
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
                elif any(tag in name.lower() for tag in ['ar', 'sma', 'average']):
                    # Strip trailing underscore + number
                    legend_label = f"{re.sub(r'_\d+$', '', name)}" #{q_key} ahead"
                else:
                    legend_label = name
                
                ax.plot(q_dfs[q_key].index, q_dfs[q_key]['value'], 
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
    

    # Plot 3: Combined plot showing all selected quarters from all DataFrames
    if len(quarters_to_plot) > 0:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot evaluation data first (if provided)
        if eval_filtered is not None:
            ax.plot(
                eval_filtered.index, eval_filtered.iloc[:, 0],
                color='black', linewidth=3, label=evaluation_legend_name, alpha=0.9
            )
        
        # Prepare base colors for each model
        import matplotlib.colors as mcolors
        model_names = list(processed_dfs.keys())
        base_colors = {}
        for name in model_names:
            if 'ifo' in name.lower():
                base_colors[name] = '#003366'
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
                # interpolate color towards white by factor q/9
                color = mcolors.to_rgb(base)
                white = (1,1,1)
                raw = q / (len(quarters_to_plot)-1 if len(quarters_to_plot)>1 else 1)

                # Set the whiteness factor
                factor = raw * 0.7  # Scale down to avoid too light colors
                shade = tuple(color[i] + (white[i]-color[i]) * factor for i in range(3))
                shade_hex = mcolors.to_hex(shade)

                # Determine plot style
                if 'ifo' in name.lower():
                    ax.plot(
                        series.index, series['value'],
                        color=shade_hex, linewidth=1.5, linestyle='-',
                        #label=f"ifo - {q_key}", 
                        alpha=0.8
                    )
                else:
                    ax.scatter(
                        series.index, series['value'],
                        color=shade_hex, s=30,
                        #label=f"{re.sub(r'_\\d+$','',name)} - {q_key}", 
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
                       n_bars: int = 10, show: bool = False,
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
    non_ifo_count = sum(1 for name in df_names if 'ifo' not in name.lower())
    non_ifo_idx = 0
    
    for name in df_names:
        if 'ifo' in name.lower():
            colors.append('#003366')  # dark blue for ifo
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
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
    
    # Show plot if specified
    if show:
        plt.show()
    
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
        if 'ifo' in name.lower():
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
        if 'ifo' in name.lower():
            legend_label = 'ifo'
        elif any(tag in name.lower() for tag in ['ar', 'sma', 'average']):
            # Strip trailing underscore + number (e.g. "_2", "_10", etc.)
            legend_label = re.sub(r'_\d+$', '', name)
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
    
    
    return fig, ax