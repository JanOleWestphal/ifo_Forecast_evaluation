
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Quarterly Evaluation
#
# Author:       Jan Ole Westphal
# Date:         2025-07
#
# Description:  Subprogram to evaluate quarterly forecasts of both ifo and the Naive Forecaster,
#               possibly other providers as well          
# ==================================================================================================
# --------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        Code begins here                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

print("\n Executing the Quarterly Evaluation Module ... \n")


# ==================================================================================================
#                                           SETUP
# ==================================================================================================

# Import built-ins
import importlib
import subprocess
import sys
import os
import glob
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

# Define the horizon of first releases which should be evaluated: available from 1995-Q3 onwards
first_release_limit_year = settings.first_release_limit_year
first_release_limit_quarter = settings.first_release_limit_quarter







# ==================================================================================================
# SETUP OUTOUT FOLDER STRUCTURE
# ==================================================================================================

## Result Folder Paths
table_folder = os.path.join(wd, '1_Result_Tables')
graph_folder = os.path.join(wd, '2_Result_Graphs')

## Parent Folder
base_path = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data')


## Create if needed
for folder in [base_path, table_folder, graph_folder]:
    os.makedirs(folder, exist_ok=True)


## Clear Result Folders
#if settings.clear_result_folders:
#    folder_clear(folder_path)












# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                   Load in evaluation data                                        #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

print("\n Loading Evaluation Data ...  \n")


# ==================================================================================================
# Load Data 
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Load Evaluation Data
# --------------------------------------------------------------------------------------------------

eval_path = os.path.join(wd, '0_0_Data', '2_Processed_Data', '2_Evaluation_series')

## First Releases
qoq_path_first = os.path.join(eval_path, 'first_release_qoq_GDP.xlsx')
qoq_first_eval = pd.read_excel(qoq_path_first, index_col=0)
#show(qoq_first_eval)


## Latest Releases
qoq_path_latest= os.path.join(eval_path, 'latest_release_qoq_GDP.xlsx')
qoq_latest_eval = pd.read_excel(qoq_path_latest, index_col=0)
#show(qoq_latest_eval)


## Revision
qoq_path_rev = os.path.join(eval_path, 'revision_qoq_GDP.xlsx')
qoq_rev = pd.read_excel(qoq_path_rev, index_col=0)



# --------------------------------------------------------------------------------------------------
# Load naive forecasts
# --------------------------------------------------------------------------------------------------

# Paths to the folders containing the Excel files
file_path_naive_qoq = os.path.join(wd, '0_0_Data', '3_Naive_Forecaster_Data', '1_QoQ_Forecast_Tables')

# Load all QoQ and YoY naive forecast Excel files into dictionaries
naive_qoq_dfs = load_excels_to_dict(file_path_naive_qoq)
#show(naive_qoq_dfs)

"""
def load_excels_to_dict(folder_path):
    excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
    dfs = {}
    for file in excel_files:
        name = os.path.splitext(os.path.basename(file))[0]
        dfs[name] = pd.read_excel(file, index_col=0)
    return dfs
"""





# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                              Build joint evaluation dataframes                                   #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# ==================================================================================================
# Necessary Functions
# ==================================================================================================

def create_qoq_evaluation_df(qoq_forecast_df, eval_vector):
    """
    Join in the evaluation time series for every col's observations by quarter-based matching,
    get a new col with the differences
    """


    return qoq_evaluation_df


def collapse_quarterly_prognosis(df):
    """
    Move all cols to the same rows, rename rows Q1-Qx: gets quarterly error measures based on 
    forecast horizons
    """

    return df




# ==================================================================================================
# ifo QoQ FORECASTS
# ==================================================================================================




# ==================================================================================================
# NAIVE QoQ FORECASTS
# ==================================================================================================







# ==================================================================================================
# Select Evaluation timeframe
# ==================================================================================================

# Apply row and col selection from helperfunctions:
for df in []:
    df = filter_first_release_limit(df)
    df = filter_evaluation_limit(df)






# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                    Analyzing Error Measures                                      #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

print("Computing error statistics ...  \n")


# ==================================================================================================
# Necessary Functions
# ==================================================================================================








# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                  Visualizing Error Measures                                      #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

print("Visualizing error statistics ...  \n")



# ==================================================================================================
# Necessary Functions
# ==================================================================================================












# --------------------------------------------------------------------------------------------------
print(f" \n Quarterly Evaluation Module complete! \n",f"Find Results in {base_path}\n")
# --------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#