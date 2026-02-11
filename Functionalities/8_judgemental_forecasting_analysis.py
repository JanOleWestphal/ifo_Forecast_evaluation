
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Judgemental Forecasting Analysis Module
#
# Author:       Jan Ole Westphal
# Date:         2026-01
#
# Description:  Subprogram to run an econometric analysis on judgemental derivations in german
#               macroeconomic forecasting.
# 
#               Runs all components from Data Processing to Output Processing and Visualizations.         
# ==================================================================================================
# --------------------------------------------------------------------------------------------------



""""
Main Tasks:
- create a measure of judgemental derivations:
    - derivation from the ifoCAST
    - derivation from an AR2-benchmark
    - possibly: derivations from ifo's forecast methodology, if vintage data exists

- Create a measure of net-improvement of derivations
- Classify derivations:
    - direction of the shock
    - direction of the adjustment
    - net improvement

    --> Derivation types: r -> realized value, , b -> benchmark, j -> judgemental forecasts
        - negative shocks, r<b: 
            - r<b<j (overconfidence), 
            - r<j<b (prudent pessimism), 
            - j<r<b; |j-b|<|r-b| (mild overpessimism), 
            - j<r<b |j-b|>|r-b|(strong overpessimism)

        - positive shocks, b<r:
            - overpessimism: j<r<b
            - prudent optimism: j>b>r
            - mild overoptimism: b<r<j; |j-b|<|r-b|
            - strong overoptimism: b<r<j; |j-b|>|r


- Analyze judgement persistence through the Pedersen (2025) methodology, (autoregression of derivaitons)

VISUALIZATIONS:
- Judgemental vs Benchmark error bars by quarter
"""




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        Code begins here                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

print("\n Executing the Judgemental Derivations Analysis module ... \n")


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




# ==================================================================================================
# SETUP OUTOUT FOLDER STRUCTURE
# ==================================================================================================

## Result Folder Paths
result_folder = os.path.join(wd, '5_Judgemental_Derivations_Analysis')


## Subfolder
table_folder = os.path.join(result_folder, '1_Tables')
error_stats_plot_folder = os.path.join(result_folder, '2_Error_Plots')



## Create if needed
for folder in [result_folder, table_folder, error_stats_plot_folder]:
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
    col_names=['realized', 'judgemental', 'naiveAR2', 'ifoCast']
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





# =================================================================================================#
#                                    Create error measures                                         #
# =================================================================================================#

## for ifo judgemental, AR2 and ifoCAST nowcasts

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



# =================================================================================================#
#                                   Create derivation measures                                     #
# =================================================================================================#

## from ifoCAST
joint_nowcast_df["derivation_from_ifoCast"] = (
    joint_nowcast_df["judgemental"] - joint_nowcast_df["ifoCast"]
)

## from AR2
joint_nowcast_df["derivation_from_AR"] = (
    joint_nowcast_df["judgemental"] - joint_nowcast_df["naiveAR2"]
)




# =================================================================================================#
#                                    Obtain net improvements                                       #
# =================================================================================================#

# ------------------------------------------------------------------------------------
# Net improvement of judgemental forecast relative to baseline forecasts
#
# Definitions
# ----------
# Linear improvement  :  NI_lin  = |e_baseline| - |e_judgemental|
# Quadratic improvement: NI_quad = e_baseline^2 - e_judgemental^2
#
# Positive values  -> judgement improved the forecast
# Negative values  -> judgement worsened the forecast
#
# Required columns already present:
#   realized
#   judgemental
#   naiveAR2
#   ifoCast
#   error_realized_minus_judgemental
#   error_realized_minus_naiveAR2
#   error_realized_minus_ifoCast
#   derivation_from_ifoCast
#   derivation_from_AR
# ------------------------------------------------------------------------------------

# ---------- Judgement vs ifoCast ----------
joint_nowcast_df["net_improvement_jdg_ifoCast_lin"] = (
    joint_nowcast_df["error_realized_minus_ifoCast"].abs()
    - joint_nowcast_df["error_realized_minus_judgemental"].abs()
)

joint_nowcast_df["net_improvement_jdg_ifoCast_quad"] = (
    joint_nowcast_df["error_realized_minus_ifoCast"]**2
    - joint_nowcast_df["error_realized_minus_judgemental"]**2
)


# ---------- Judgement vs naiveAR2 ----------
joint_nowcast_df["net_improvement_jdg_AR_lin"] = (
    joint_nowcast_df["error_realized_minus_naiveAR2"].abs()
    - joint_nowcast_df["error_realized_minus_judgemental"].abs()
)

joint_nowcast_df["net_improvement_jdg_AR_quad"] = (
    joint_nowcast_df["error_realized_minus_naiveAR2"]**2
    - joint_nowcast_df["error_realized_minus_judgemental"]**2
)

#show(joint_nowcast_df)





# =================================================================================================#
#                                Classify Judgemental Derications                                  #
# =================================================================================================#

"""
r<b: True (negative shock), False; 'r_less_b'
j<b: True (negatve adjustment), False; 'j_less_b'
|j-r|<|b-r| True (judgemental improvement), False; 'jdiff_less_bdiff'
"""

# -------------------------------------------------------------------------------------------------#
# Classification builder function
# -------------------------------------------------------------------------------------------------#


def _classify_derivations(df, b_col: str, suffix: str, r_col: str = "realized", j_col: str = "judgemental"):
    r = df[r_col]
    j = df[j_col]
    b = df[b_col]

    # Ensure NA-safe comparisons: keep pd.NA where any input is missing
    valid = r.notna() & j.notna() & b.notna()

    df[f"r_less_{suffix}"] = pd.Series(np.where(valid, r < b, pd.NA), index=df.index, dtype="boolean")
    df[f"j_less_{suffix}"] = pd.Series(np.where(valid, j < b, pd.NA), index=df.index, dtype="boolean")
    df[f"j_diff_less_{suffix}_diff"] = pd.Series(
        np.where(valid, (j - r).abs() < (b - r).abs(), pd.NA),
        index=df.index,
        dtype="boolean",
    )


# -------------------------------------------------------------------------------------------------#
# Evaluation against the ifoCAST df
# -------------------------------------------------------------------------------------------------#
_classify_derivations(joint_nowcast_df, b_col="ifoCast", suffix="ifoCast")


# -------------------------------------------------------------------------------------------------#
# Evaluation against the AR2 df
# -------------------------------------------------------------------------------------------------#
_classify_derivations(joint_nowcast_df, b_col="naiveAR2", suffix="AR2")

# show(joint_nowcast_df)



# =================================================================================================#
#                                  Split dfs and save results                                      #
# =================================================================================================#

## Split

# Create copy
judgment_eval_df_joint = joint_nowcast_df.copy()

# Columns always included
base_cols = [
    "realized",
    "judgemental",
    "error_realized_minus_judgemental",
]

# ---- ifoCast subset ----
ifo_cols = [c for c in judgment_eval_df_joint .columns if "ifoCast" in c]
judgment_eval_df_ifoCast = judgment_eval_df_joint [base_cols + ifo_cols].copy()

# ---- AR subset (matches AR / naiveAR2 / derivation_from_AR etc.) ----
ar_cols = [c for c in judgment_eval_df_joint .columns if "AR" in c]
joint_nowcast_AR_df = judgment_eval_df_joint [base_cols + ar_cols].copy()


## Save
judgment_eval_df_joint.to_excel(os.path.join(table_folder, "judgemental_derivations_full.xlsx"))
judgment_eval_df_ifoCast.to_excel(os.path.join(table_folder, "judgemental_derivations_ifoCast.xlsx"))
joint_nowcast_AR_df.to_excel(os.path.join(table_folder, "judgemental_derivations_AR2.xlsx"))












# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                Judgemental derivations analysis                                  #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#



# =================================================================================================#
#                                    Obtain Summary Statistics                                     #
# =================================================================================================#



# =================================================================================================#
#                                   Analyze forecast persistence                                   #
# =================================================================================================#













# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                       Visualize Results                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# =================================================================================================#
#                          Judgemental vs Benchmark Error Bars by Quarter                          #
# =================================================================================================#

# -------------------------------------------------------------------------------------------------#
# Error Bar Series Plotter
# -------------------------------------------------------------------------------------------------#



# -------------------------------------------------------------------------------------------------#
# Evaluation against the ifoCAST df
# -------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------#
# Evaluation against the AR2 df
# -------------------------------------------------------------------------------------------------#








# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#