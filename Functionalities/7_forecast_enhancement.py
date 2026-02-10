
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

print("\n Executing the forecast Enhancement Analysis module ... \n")


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







# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                      OPTIMIZE ifo FORECAST                                       #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# =================================================================================================#
#                                   SIMPLE LINEAR COMBINATIONS                                     #
# =================================================================================================#














# =================================================================================================#
#                                   OPTIMIZED LINEAR COMBINATIONS                                  #
# =================================================================================================#






# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        Visualize Results                                         #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#



















# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#