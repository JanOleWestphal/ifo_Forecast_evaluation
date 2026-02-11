
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Forecast Enhancement Analysis Module
#
# Author:       Jan Ole Westphal
# Date:         2025-12
#
# Description:  Subprogram to experiment on how to enhance the ifo Forecasts.
# 
#               Runs all components from Data Processing to Output Processing and Visualizations.         
# ==================================================================================================
# --------------------------------------------------------------------------------------------------


"""
- Simple baseline: combine ifo judgemental and ifoCAST / ifo judgemental and AR2 with 0.5 weights
- Run optimal weight search with two setups: 
            - train before Covid hits, evaluate after + non-diconnected full series
            - eval starting in 2021

- Plot ideal weights
- Do three-way combination exercise, plot ideal weights in 3D

Get error tables, run tests on relative performance
"""





# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        Code begins here                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

print("\n Executing the Forecast Enhancement Analysis module ... \n")


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
from typing import Union, Dict, Optional, Mapping


# Import libraries
from pathlib import Path
import requests
import pandas as pd
from pandas.tseries.offsets import QuarterBegin
from pandasgui import show  #uncomment this to allow for easier debugging

import numpy as np
from statsmodels.tsa.ar_model import AutoReg


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

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




# ==================================================================================================
# SETUP OUTOUT FOLDER STRUCTURE
# ==================================================================================================

## Result Folder Paths
result_folder = os.path.join(wd, '4_Mixed_Forecasts')


## Subfolder
table_folder = os.path.join(result_folder, '1_Tables')
error_stats_plot_folder = os.path.join(result_folder, '2_Error_Plots')
mixed_model_folder = os.path.join(result_folder, '3_Optimal_Mixing_Weighs')


## Create if needed
for folder in [result_folder, table_folder, error_stats_plot_folder, mixed_model_folder]:
    os.makedirs(folder, exist_ok=True)


## Clear Result Folders
#if settings.clear_result_folders:
#    folder_clear(folder_path)






# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                          LOAD IN DATA                                            #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

# -------------------------------------------------------------------------------------------------#
# Load realized GDP-series
# -------------------------------------------------------------------------------------------------#
eval_path = os.path.join(wd, '0_0_Data', '2_Processed_Data', '2_GDP_Evaluation_series')
qoq_path_first = os.path.join(eval_path, 'first_release_qoq_GDP.xlsx')

## First Releases
qoq_first_eval = pd.read_excel(qoq_path_first, index_col=0)
#show(qoq_first_eval)


# -------------------------------------------------------------------------------------------------#
# Load ifo qoq nowcasts
# -------------------------------------------------------------------------------------------------#
# Path
file_path_ifo_qoq = os.path.join(wd, '0_0_Data', '2_Processed_Data', '3_ifo_qoq_series',
                                  'ifo_qoq_forecasts.xlsx' )

# Load 
ifo_qoq_forecasts = pd.read_excel(file_path_ifo_qoq, index_col=0)

# Build ifo judgemental nowcasts by matching row/column on quarterly level.
def nowcast_builder(df):
    ifo_rows_quarter = pd.to_datetime(df.index).to_period('Q')
    ifo_cols_quarter = pd.to_datetime(df.columns).to_period('Q')

    dtx1_records = []
    for col, col_quarter in zip(df.columns, ifo_cols_quarter):
        matching_rows = np.where(ifo_rows_quarter == col_quarter)[0]
        if len(matching_rows) != 1:
            raise ValueError(
                f"Expected exactly one quarterly row match for column {col} ({col_quarter}), "
                f"found {len(matching_rows)}."
            )

        row_label = df.index[matching_rows[0]]
        dtx1_records.append(
            {
                'column_date': col,
                'matched_row_date': row_label,
                'ifo_judgemental_nowcast': df.loc[row_label, col]
            }
        )

    dtx1 = pd.DataFrame(dtx1_records).set_index('column_date')
    df_out = dtx1[['ifo_judgemental_nowcast']].copy()

    return df_out

ifo_judgemental_nowcasts = nowcast_builder(ifo_qoq_forecasts)
#show(ifo_judgemental_nowcasts)


# -------------------------------------------------------------------------------------------------#
# Load ifoCAST nowcasts
# -------------------------------------------------------------------------------------------------#
ifoCAST_nowcasts_full_path = os.path.join(
    wd, '0_0_Data', '0_Forecast_Inputs', '2_ifoCAST', 'ifoCAST_nowcasts_full.xlsx')

# Load 
ifoCAST_nowcast = pd.read_excel(ifoCAST_nowcasts_full_path , index_col=0)
#show(ifoCAST_nowcast)

# -------------------------------------------------------------------------------------------------#
# Load AR2-nowcasts
# -------------------------------------------------------------------------------------------------#
# Paths to the folders containing the Excel files
file_path_naive_qoq = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '1_QoQ_Forecast_Tables')

# Load all QoQ naive forecast Excel files into dictionary
naive_qoq_dfs_dict = load_excels_to_dict(file_path_naive_qoq, strip_string='naive_qoq_forecasts_')

# Get the AR2 model
matches = [k for k in naive_qoq_dfs_dict if "AR2" in k]
if not matches:
    raise KeyError("No entry containing 'AR2' found in naive_qoq_dfs_dict. Check Setting file and run Naive Forecaster Module")

df_ar2 = naive_qoq_dfs_dict[matches[0]]
#show(df_ar2)

# Get Nowcasts
AR_nowcasts = nowcast_builder(df_ar2)
#show(AR_nowcasts)







# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                          PROCESS DATA                                            #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# =================================================================================================#
#                                       Merge to joint df                                          #
# =================================================================================================#

## Call merge_quarterly_dfs_dropna() from helperfunctions
joint_nowcast_df = merge_quarterly_dfs_dropna(
    dfs= [qoq_first_eval, ifo_judgemental_nowcasts, ifoCAST_nowcast, AR_nowcasts ],
    col_names=['realized', 'judgemental', 'ifoCast', 'naiveAR2']
)

#show(joint_nowcast_df)

## Create a clean copy
joint_nowcast_base_df = joint_nowcast_df.copy()



# -------------------------------------------------------------------------------------------------#
# OPTIONAL: filter rows
# -------------------------------------------------------------------------------------------------#

"""NOTE: all rows are indexed by latest date of quarter"""

## Adjust filter if needed, boundary inclusive
joint_nowcast_df = filter_df_by_datetime_index(joint_nowcast_df, '2000-01-01', '2100-01-01')
#show(joint_nowcast_df)







# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                      OPTIMIZE ifo FORECAST                                       #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# =================================================================================================#
#                                   SIMPLE LINEAR COMBINATIONS                                     #
# =================================================================================================#

## Combination function
def combine_2_forecasts(df, col1, col2, w1, w2, new_name):
    df[new_name] = df[col1] * w1 + df[col2] *w2

    return df

def combine_3_forecasts(df, col1, col2, col3, w1, w2, w3, new_name):
    df[new_name] = df[col1] * w1 + df[col2] *w2 + df[col3] * w3

    return df


## Simple average of judgmental and IfoCAST
joint_nowcast_df = combine_2_forecasts(joint_nowcast_df, 'judgemental', 'ifoCast', 0.5, 0.5, 'judg_ifoCast')

## Simple average of judgmental and AR2
joint_nowcast_df = combine_2_forecasts(joint_nowcast_df, 'judgemental', 'naiveAR2', 0.5, 0.5, 'judg_AR')

## Simple average of ifoCAST and AR2
joint_nowcast_df = combine_2_forecasts(joint_nowcast_df, 'ifoCast', 'naiveAR2', 0.5, 0.5, 'ifoCAST_AR')

## Simple average of judgmental (0.5), ifoCAST and AR2 (0.25 each)
joint_nowcast_df = combine_3_forecasts(joint_nowcast_df, 'judgemental', 'naiveAR2', 'ifoCast', 0.5, 0.25, 0.25, 'judg_AR_ifoCast')

#show(joint_nowcast_df)


# -------------------------------------------------------------------------------------------------#
#                                       CALCULATE ERRORS                                           #
# -------------------------------------------------------------------------------------------------#

# Loop through cols and substract them from the leading realized values col
def add_error_columns(df, prefix="error"):

    first_col = df.columns[0]

    for col in df.columns[1:]:
        df[f"{prefix}_{first_col}_minus_{col}"] = df[first_col] - df[col]

    return df

# Call error function
joint_nowcast_df = add_error_columns(joint_nowcast_df)
#show(joint_nowcast_df)

## Save
joint_nowcast_df.to_excel(
    excel_writer=os.path.join(table_folder, "Nowcast_Series_full.xlsx")
)


# -------------------------------------------------------------------------------------------------#
#                                  CALCULATE ERROR STATISTICS                                      #
# -------------------------------------------------------------------------------------------------#

## Obtain error staistics
nowcast_error_summary_df = error_columns_summary_nowcast(joint_nowcast_df)

#show(nowcast_error_summary_df)

## SAVE
nowcast_error_summary_df.to_excel(
    excel_writer=os.path.join(table_folder, "Error_Statistics_Simple_Mixed_Models.xlsx")
)

"""
MAJOR RESULT: None of these improve anything
"""



# =================================================================================================#
#                                        Visualize Results                                         #
# =================================================================================================#


## Error Metric Barplots

def save_error_metric_barplots(
    stats_df: pd.DataFrame,
    error_fig_path: str,
    metrics=("ME", "MAE", "RMSE", "MSE", "SE"),
    n_col: str = "N",
    file_ext: str = "png",
    dpi: int = 200,
    rotate_xticks: int = 45,
):
    """
    For a stats DataFrame (rows = models/series, columns include error metrics),
    save one bar plot per metric to `error_fig_path`.

    Each plot:
      - x-axis: row names (stats_df.index)
      - y-axis: metric values
      - title: "{metric} (N=...)" where N is taken from stats_df[n_col]
              If N is not unique across rows, title shows "N=min-max".

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with index as row labels and columns including metrics and `n_col`.
    error_fig_path : str
        Directory where figures will be saved.
    metrics : tuple/list of str
        Metrics to plot as separate figures.
    n_col : str
        Column name containing N.
    file_ext : str
        File extension, e.g. "png" or "pdf".
    dpi : int
        DPI for raster formats.
    rotate_xticks : int
        Rotation angle for x tick labels.

    Returns
    -------
    list[str]
        List of saved file paths.
    """
    os.makedirs(error_fig_path, exist_ok=True)

    if n_col not in stats_df.columns:
        raise ValueError(f"'{n_col}' column not found in stats_df.")

    # Determine N string for titles
    n_vals = stats_df[n_col].dropna()
    if len(n_vals) == 0:
        n_str = "N=?"
    else:
        uniq = pd.unique(n_vals)
        if len(uniq) == 1:
            n_str = f"N={int(uniq[0])}"
        else:
            n_min, n_max = int(n_vals.min()), int(n_vals.max())
            n_str = f"N={n_min}-{n_max}"

    saved = []

    for metric in metrics:
        if metric not in stats_df.columns:
            continue  # skip missing metrics silently

        y = stats_df[metric]
        x = stats_df.index.astype(str)

        fig, ax = plt.subplots()
        ax.bar(x, y.values)
        ax.set_title(f"{metric} ({n_str})")
        ax.set_ylabel(metric)
        ax.set_xlabel("")

        ax.tick_params(axis="x", labelrotation=rotate_xticks)
        fig.tight_layout()

        out_path = os.path.join(error_fig_path, f"{metric}.{file_ext}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        saved.append(out_path)

    return saved


## Create Error Metrics Plots
save_error_metric_barplots(nowcast_error_summary_df, error_stats_plot_folder)





# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                   OPTIMIZED LINEAR COMBINATIONS                                  #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#



# -------------------------------------------------------------------------------------------------#
#                                            Helper                                                #
# -------------------------------------------------------------------------------------------------#

def _error_metric(errors: pd.Series, criterion: str) -> float:
    """Compute a scalar error metric from an error Series (forecast - realised)."""
    e = errors.dropna()
    if e.empty:
        return np.nan

    c = criterion.upper()
    if c == "ME":
        return float(e.mean())
    if c == "MAE":
        return float(e.abs().mean())
    if c == "MSE":
        return float((e**2).mean())
    if c == "RMSE":
        return float(np.sqrt((e**2).mean()))
    raise ValueError("criterion must be one of {'ME','MAE','MSE','RMSE'}")



# -------------------------------------------------------------------------------------------------#
#                                    One weight grid search                                        #
# -------------------------------------------------------------------------------------------------#

def optimal_weights_2_forecasts_grid(
    df: pd.DataFrame,
    *,
    realised_col: str = 'realized',
    f1_col: str,
    f2_col: str,
    criterion: str = "RMSE",
    n_grid: int = 1000,
    eps: float = 1e-6,
    plot_title: str | None = None,
    w1_explanation: str | None = None,
    show_plot: bool = False,
    save_fig: bool = True,
    fig_path: str | Path | None = None,
    print_result: bool = True,
):
    if save_fig and fig_path is None:
        raise ValueError("fig_path must be provided when save_fig=True")

    y = df[realised_col]
    f1 = df[f1_col]
    f2 = df[f2_col]

    w1_grid = np.linspace(eps, 1 - eps, n_grid)
    metric_vals = np.empty_like(w1_grid)

    for i, w1 in enumerate(w1_grid):
        combined = w1 * f1 + (1 - w1) * f2
        metric_vals[i] = _error_metric(combined - y, criterion)

    crit = criterion.upper()
    objective = np.abs(metric_vals) if crit == "ME" else metric_vals
    idx = int(np.nanargmin(objective))

    best_w1 = float(w1_grid[idx])
    best_w2 = float(1 - best_w1)
    best_metric = float(metric_vals[idx])

    grid_df = pd.DataFrame({"w1": w1_grid, "w2": 1 - w1_grid, crit: metric_vals})

    # ---- Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(grid_df["w1"], grid_df[crit], label=crit)
    ax.scatter([best_w1], [best_metric], color="crimson", s=45, zorder=3, label="optimum")
    ax.set_xlabel("w1")
    ax.set_ylabel(crit)
    if plot_title:
        ax.set_title(plot_title)
    elif w1_explanation:
        ax.set_title(w1_explanation)

    annotation = (
        f"Optimal weights:\n"
        f"w1={best_w1:.4f}\n"
        f"w2={best_w2:.4f}\n"
        f"{crit}={best_metric:.6g}"
    )
    ax.text(
        0.02,
        0.02,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
    )
    ax.legend(loc="upper right")

    if save_fig:
        fig_path = Path(fig_path)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    if print_result:
        print(
            f"Weight-column mapping: w1 -> {f1_col}, w2 -> {f2_col}\n"
            f"Optimal weights: w1={best_w1:.6f}, w2={best_w2:.6f}\n"
            f"{crit}={best_metric:.6g}"
        )

    return {
        "best_w1": best_w1,
        "best_w2": best_w2,
        "best_metric": best_metric,
        "criterion": crit,
        "grid": grid_df,
    }



# -------------------------------------------------------------------------------------------------#
#                                    Two weights grid search                                       #
# -------------------------------------------------------------------------------------------------#

def optimal_weights_3_forecasts_grid(
    df: pd.DataFrame,
    *,
    realised_col: str ='realized',
    f1_col: str,
    f2_col: str,
    f3_col: str,
    criterion: str = "RMSE",
    step: float = 0.02,
    eps: float = 1e-6,
    plot_title: str | None = None,
    weights_explanation: str | None = None,
    show_plot: bool = False,
    save_fig: bool = True,
    fig_path: str | Path | None = None,
    print_result: bool = True,
):
    if save_fig and fig_path is None:
        raise ValueError("fig_path must be provided when save_fig=True")

    y = df[realised_col]
    f1, f2, f3 = df[f1_col], df[f2_col], df[f3_col]

    rows = []
    crit = criterion.upper()

    w1_vals = np.arange(eps, 1 - 2 * eps + 1e-12, step)
    for w1 in w1_vals:
        w2_vals = np.arange(eps, 1 - w1 - eps + 1e-12, step)
        for w2 in w2_vals:
            w3 = 1 - w1 - w2
            if w3 <= eps:
                continue

            combined = w1 * f1 + w2 * f2 + w3 * f3
            m = _error_metric(combined - y, crit)
            rows.append((w1, w2, w3, m))

    grid_df = pd.DataFrame(rows, columns=["w1", "w2", "w3", crit])

    objective = np.abs(grid_df[crit]) if crit == "ME" else grid_df[crit]
    best = grid_df.loc[objective.idxmin()]

    # ---- Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(grid_df["w1"], grid_df["w2"], grid_df[crit], c=grid_df[crit], cmap="viridis", s=12)
    ax.scatter([best.w1], [best.w2], [best[crit]], color="crimson", s=80, marker="*", depthshade=False)
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_zlabel(crit)
    if plot_title:
        ax.set_title(plot_title)
    elif weights_explanation:
        ax.set_title(weights_explanation)

    annotation = (
        f"Optimal weights:\n"
        f"w1={best.w1:.4f}\n"
        f"w2={best.w2:.4f}\n"
        f"w3={best.w3:.4f}\n"
        f"{crit}={best[crit]:.6g}"
    )
    ax.text2D(
        0.98,
        0.02,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    if save_fig:
        fig_path = Path(fig_path)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    if print_result:
        print(
            f"Weight-column mapping: w1 -> {f1_col}, w2 -> {f2_col}, w3 -> {f3_col}\n"
            f"Optimal weights: "
            f"w1={best.w1:.6f}, w2={best.w2:.6f}, w3={best.w3:.6f}\n"
            f"{crit}={best[crit]:.6g}"
        )

    return {
        "best_w1": float(best.w1),
        "best_w2": float(best.w2),
        "best_w3": float(best.w3),
        "best_metric": float(best[crit]),
        "criterion": crit,
        "grid": grid_df,
    }

"""
optimal_weights_2_forecasts_grid(
    df,
    realised_col="y",
    f1_col="f_model",
    f2_col="f_judgement",
    save_fig=True,
    fig_path="figures/weight_search_2f_rmse.pdf",
)

optimal_weights_3_forecasts_grid(
    df,
    realised_col="y",
    f1_col="f1",
    f2_col="f2",
    f3_col="f3",
    save_fig=True,
    fig_path="figures/weight_search_3f_rmse.pdf",
)

"""




# =================================================================================================#
#                                   ANALYSE OPTIMAL COMBINATIONS                                   #
# =================================================================================================#

# -------------------------------------------------------------------------------------------------#
#                                    judgemental and ifoCAST                                       #
# -------------------------------------------------------------------------------------------------#
## Savepath
jc_path = os.path.join(mixed_model_folder, 'optimal_judgemental_ifoCAST_weight.png')

## Call Function
optimal_weights_2_forecasts_grid(joint_nowcast_df, f1_col='judgemental', f2_col='ifoCast', 
                                 plot_title= 'ifoCast - judgemental forecast' ,
                                 fig_path=jc_path)



# -------------------------------------------------------------------------------------------------#
#                                      judgemental and AR2                                         #
# -------------------------------------------------------------------------------------------------#
## Savepath
jar_path = os.path.join(mixed_model_folder, 'optimal_judgemental_AR2_weight.png')

## Call Function
optimal_weights_2_forecasts_grid(joint_nowcast_df, f1_col='judgemental', f2_col='naiveAR2',
                                 plot_title= 'Naive AR2 Forecast - Judgemental Forecast',
                                fig_path=jar_path)


# -------------------------------------------------------------------------------------------------#
#                                        ifoCAST and AR2                                           #
# -------------------------------------------------------------------------------------------------#
## Savepath
car_path = os.path.join(mixed_model_folder, 'optimal_ifoCAST_AR2_weight.png')

## Call Function
optimal_weights_2_forecasts_grid(joint_nowcast_df, f1_col='ifoCast', f2_col='naiveAR2', 
                                 plot_title= 'Naive AR2 Forecast - ifoCAST',
                                 fig_path=car_path)


# -------------------------------------------------------------------------------------------------#
#                                  judgemental, ifoCAST and AR2                                    #
# -------------------------------------------------------------------------------------------------#
## Savepath
jarc_path = os.path.join(mixed_model_folder, 'optimal_judgemental_ifoCAST_AR2_weight.png')

## Call Function
optimal_weights_3_forecasts_grid(joint_nowcast_df, f1_col='judgemental',f2_col='ifoCast', f3_col='naiveAR2',
                                 plot_title= 'judgemental (w1) - ifoCAST (w2) - AR2 (w3)'  ,
                                  fig_path=jarc_path)















# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#
