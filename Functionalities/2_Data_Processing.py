
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Data Processing
#
# Description:  Creates the dataframes used in the project
#               Stores them in Folder 0_0_Data/2_Processed_Data.
#
#               Processes both the real-time and historic GDP-data inputs, turns them into a joint 
#               dataframe and creates evaluation time series.
#               New Forecasts may be included by updating the respective files in the Subfolder
#               0_0_Data\0_Forecast_Inputs
# ==================================================================================================
# --------------------------------------------------------------------------------------------------

"""
Functions:
-> Get Real-Time Bundesbank GDP-data (2005-Q2 onwards):
    -> get  QoQs, YoYs

-> build long-run GDP dataset (merged):
    -> harmonize the three sources to the standard of the realtime data.
            -> Scale 95-2005 s.t. it fits real time data, do the same for Lange Reihe (pre 1995)
    -> get YoYs

-> Get first release Evaluation data
    -> Pull this from joint df
    -> absolute as indexed in real-time GDP
    -> YoY, QoQ

-> Get latest release and revision evaluation data
"""



print(" \n Data Processing started ... \n")




# ==================================================================================================
#                                           SETUP
# ==================================================================================================

# Import built-ins
import importlib
import subprocess
import sys
import os
import re
from datetime import datetime, date


# Import libraries
import requests
import pandas as pd
from pandasgui import show  #uncomment this to allow for easier debugging
import numpy as np
# from statsmodels.tsa.ar_model import AutoReg
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

## Import custom functions
from Functionalities.helpers.helperfunctions import *


# Import settings from the settings file
import ifo_forecast_evaluation_settings as settings


def _safe_to_excel(df, path, index=True):
    """Try to save DataFrame to Excel using available engines; fallback to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df.to_excel(path, index=index)
        return
    except Exception:
        pass

    # Try openpyxl explicitly
    try:
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df.to_excel(writer, index=index)
        return
    except Exception:
        pass

    # Fallback: write CSV
    try:
        csv_path = os.path.splitext(path)[0] + '.csv'
        df.to_csv(csv_path, index=index)
        return
    except Exception as e:
        raise IOError(f"Failed to save DataFrame to {path} or fallback CSV: {e}") from e








# ==================================================================================================
#                                DEFINE OUTPUT DIRECTORIES
# ==================================================================================================


# --------------------------------------------------------------------------------------------------
# GDP and GVA
# --------------------------------------------------------------------------------------------------

## GDP
output_dir_gdp = os.path.join(wd, '0_0_Data', '2_Processed_Data', '1_rt_GDP_series')
os.makedirs(output_dir_gdp, exist_ok=True)

## GVA
output_dir_gva = os.path.join(wd, '0_0_Data', '2_Processed_Data', '1_rt_GVA_series')
os.makedirs(output_dir_gva, exist_ok=True)

## ifo qoq forecasts
ifo_qoq_output_dir = os.path.join(wd, '0_0_Data', '2_Processed_Data', '3_ifo_qoq_series')
os.makedirs(ifo_qoq_output_dir, exist_ok=True)


# --------------------------------------------------------------------------------------------------
# Component-level data
# --------------------------------------------------------------------------------------------------

## Component output directory
output_dir_ts_components = os.path.join(wd, '0_0_Data', '2_Processed_Data', '1_rt_component_series')
os.makedirs(output_dir_ts_components, exist_ok=True)

## Evaluation series directory
output_dir_eval_components = os.path.join(wd, '0_0_Data', f'2_Processed_Data', f'2_component_Evaluation_series')
os.makedirs(output_dir_eval_components, exist_ok=True)

## Directory for component time series
component_forecast_output_dir = os.path.join(wd, '0_0_Data', '2_Processed_Data', '3_gdp_component_forecast')
os.makedirs(component_forecast_output_dir, exist_ok=True)





# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                  DATA PROCESSING - Functions                                     #
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

    ## avg(Q1+Q2+Q3+Q4) YoY growth	
    # base_level_gdp = 100
    df_out = df_factor.cumprod(axis=0) #* base_level_gdp

    # take yearly arithmetic mean of GDP
    df_out = df_out.resample('YE').sum(min_count=4)

    # Calculate YoY growth rates
    df_out = df_out.pct_change(fill_method=None) * 100

    print("Calculating year over year changes (relative to previous year)... \n")

    return df_out






# ==================================================================================================
#                                    Load Real-Time data
# ==================================================================================================


def process_realtime_data(rt_foldername="1_GDP_Data", rt_filename='Bundesbank_GDP_raw.csv', 
                          api_link='https://api.statistiken.bundesbank.de/rest/download/BBKRT/Q.DE.Y.A.AG1.CA010.A.I?format=csv&lang=de',
                          data_name='GDP', 
                          output_dir_df=output_dir_gdp):
    """
    Load real-time GDP and GVA data from Bundesbank API or local file.
    Process and return the cleaned DataFrame.
    """
 
    # --------------------------------------------------------------------------------------------------
    # Import Data via Bundesbank API or locally
    # --------------------------------------------------------------------------------------------------

    # Output filename "1_GDP_Data"
    rt_filepath = os.path.join( wd, "0_0_Data", rt_foldername, rt_filename)


    # Get file
    # Optional API-pull:
    if settings.api_pull == True:
        # Bundesbank API link
        url = api_link

        # Download and save the CSV file
        response = requests.get(url)
        if response.status_code == 200:
            with open(rt_filepath, 'wb') as f:
                f.write(response.content)
            print(f"Bundesbank {data_name} data downloaded and saved to: {rt_filepath} ... \n ")
        else:
            raise FileNotFoundError(f"""Failed to download file. 
                                    Library: requests,  Status code: {response.status_code}.
                                    Try again or use local version of the file (api_pull = False)""")

    elif settings.api_pull == False:
        print('Attention: local version of the data is being used, set api_pull = True for real-time data \n')

    else:
        raise ValueError('ERROR: api_pull must be set to either True or False, check settings file \n')



    # --------------------------------------------------------------------------------------------------
    # Reformat into a usable dataframe
    # --------------------------------------------------------------------------------------------------

    # Load first line to get headers
    with open(rt_filepath, encoding="utf-8") as f:
        header = f.readline().strip().split(';')

    # Load data (stored in row 12 onwards)
    df_rt = pd.read_csv(rt_filepath, skiprows=11, names=header, sep=';', index_col=0, na_values=['', ' '])

    # Process the Real-Time GDP data
    df_rt = process_BB_GDP(df_rt)
    
    # Inspect
    # show(rt_gdp)

    print("Data cleaned ... \n")

    # --------------------------------------------------------------------------------------------------
    # Create quarterly growth rates or changes: df_qoq (Quarter over Quarter)
    # --------------------------------------------------------------------------------------------------

    df_qoq_rt = get_qoq(df_rt)

    # Inspect
    # print(df_qoq.head())


    # --------------------------------------------------------------------------------------------------
    # Create yearly growth rates or changes: df_yoy (Year over Year)
    # --------------------------------------------------------------------------------------------------

    df_yoy_rt = get_yoy(df_qoq_rt)

    # show(df_yoy_rt)



    # --------------------------------------------------------------------------------------------------
    # Store Real-Time Data
    # --------------------------------------------------------------------------------------------------

    # Define file paths
    df_path_rt = os.path.join(output_dir_df, f'absolute_rt_{data_name}_data.xlsx')
    df_qoq_path_rt = os.path.join(output_dir_df, f'qoq_rt_{data_name}_data.xlsx')
    df_yoy_path_rt = os.path.join(output_dir_df, f'yoy_rt_{data_name}_data.xlsx')

    # Save files
    df_rt.to_excel(df_path_rt, index=True)      
    df_qoq_rt.to_excel(df_qoq_path_rt, index=True)
    df_yoy_rt.to_excel(df_yoy_path_rt, index=True)





    # --------------------------------------------------------------------------------------------------
    # Return raw, quarterly and yearly Real-Time Data and the filepath
    # --------------------------------------------------------------------------------------------------

    return df_rt, df_qoq_rt, df_yoy_rt









# -------------------------------------------------------------------------------------------------#
# ==================================================================================================
#                           Functions: Create and Store Evaluation DataFrames
# ==================================================================================================
# -------------------------------------------------------------------------------------------------#


## HELPER: 

# Define a function which selects the first instance of a new GDP release
def build_first_release_series(df_input):
    """
    Build a DataFrame containing the last (most recent) value from each column of df_input,
    along with its corresponding index. If the last value's index is the same as the previous,
    skip it to avoid duplicates.
    """
    last_indices = []
    last_values = []
    prev_index = None

    for col in df_input.columns:
        # Drop NaNs to get the last valid value
        col_data = df_input[col].dropna()
        if not col_data.empty:
            idx = col_data.index[-1]
            if idx != prev_index:
                last_indices.append(idx)
                last_values.append(col_data.iloc[-1])
                prev_index = idx

    result_df = pd.DataFrame({'value': last_values}, index=last_indices)
    result_df.index.name = 'date'
    return result_df



def build_store_evaluation_timeseries(df_combined, df_qoq_combined, df_yoy_combined, output_dir_df, data_name):

    # --------------------------------------------------------------------------------------------------
    # First Release Time Series
    # --------------------------------------------------------------------------------------------------

    # Call this first_release on absolute, qoq and yoy
    first_release_df = build_first_release_series(df_combined)
    first_release_qoq_df = build_first_release_series(df_qoq_combined)
    first_release_yoy_df = build_first_release_series(df_yoy_combined)



    # --------------------------------------------------------------------------------------------------
    # Latest Release Time Series
    # --------------------------------------------------------------------------------------------------

    # Latest release: take the last (most recent) column from each DataFrame, keep its index and values
    latest_release_df = df_combined.iloc[:, -1].to_frame(name='value')
    latest_release_qoq_df = df_qoq_combined.iloc[:, -1].to_frame(name='value')
    latest_release_yoy_df = df_yoy_combined.iloc[:, -1].to_frame(name='value')

    # Truncate to only rows present in the corresponding first_release_dfs by index
    latest_release_df = latest_release_df.loc[first_release_df.index]
    latest_release_qoq_df = latest_release_qoq_df.loc[first_release_qoq_df.index]
    latest_release_yoy_df = latest_release_yoy_df.loc[first_release_yoy_df.index]





    # --------------------------------------------------------------------------------------------------
    # Revision Time Series
    # --------------------------------------------------------------------------------------------------

    # Calculate revision as first_release - latest_release
    revision_df = first_release_df['value'] - latest_release_df['value']
    revision_df = revision_df.to_frame(name='revision')

    revision_qoq_df = first_release_qoq_df['value'] - latest_release_qoq_df['value']
    revision_qoq_df = revision_qoq_df.to_frame(name='revision')

    revision_yoy_df = first_release_yoy_df['value'] - latest_release_yoy_df['value']
    revision_yoy_df = revision_yoy_df.to_frame(name='revision')






    # =================================================================================================#
    #                                          Store Data                                              #
    # =================================================================================================#

    """
    Only for GDP, as backward looking GVA is not relevant at this point
    """



    if data_name == 'GDP':

        # Ensure directory exists
        output_dir_df = os.path.join(wd, '0_0_Data', f'2_Processed_Data', f'1_combined_{data_name}_series')
        os.makedirs(output_dir_df, exist_ok=True)

        # --------------------------------------------------------------------------------------------------
        # Combined DataSet
        # --------------------------------------------------------------------------------------------------

        # Define file paths
        df_path_comb = os.path.join(output_dir_df, f'absolute_combined_{data_name}.xlsx')
        df_qoq_path_comb = os.path.join(output_dir_df, f'qoq_combined_{data_name}_data.xlsx')
        df_yoy_path_comb = os.path.join(output_dir_df, f'yoy_combined_{data_name}_data.xlsx')

        # Save files
        df_combined.to_excel(df_path_comb, index=True)      
        df_qoq_combined.to_excel(df_qoq_path_comb, index=True)
        df_yoy_combined.to_excel(df_yoy_path_comb, index=True)




    # ==================================================================================================
    #  Evaluation Time Series
    # ==================================================================================================

    # Ensure directory exists
    output_dir_ts = os.path.join(wd, '0_0_Data', f'2_Processed_Data', f'2_{data_name}_Evaluation_series')
    output_dir_ts_2 = os.path.join(wd, '0_1_Output_Data', f'1_{data_name}_Evaluation_series')

    os.makedirs(output_dir_ts, exist_ok=True)
    os.makedirs(output_dir_ts_2, exist_ok=True)

    # --------------------------------------------------------------------------------------------------
    # First Release
    # --------------------------------------------------------------------------------------------------

    # Save these to two locations:

    for output_dir_ts in [output_dir_ts, output_dir_ts_2]:
        # Define file paths
        first_release_path = os.path.join(output_dir_ts, f'first_release_absolute_{data_name}.xlsx')
        first_release_qoq_path = os.path.join(output_dir_ts, f'first_release_qoq_{data_name}.xlsx')
        first_release_yoy_path = os.path.join(output_dir_ts, f'first_release_yoy_{data_name}.xlsx')

        # Save first release files
        first_release_df.to_excel(first_release_path, index=True)
        first_release_qoq_df.to_excel(first_release_qoq_path, index=True)
        first_release_yoy_df.to_excel(first_release_yoy_path, index=True)


        # ---------------------------------------------------------------------------------------------
        # Latest Release
        # ---------------------------------------------------------------------------------------------

        # Define file paths
        latest_release_path = os.path.join(output_dir_ts, f'latest_release_absolute_{data_name}.xlsx')
        latest_release_qoq_path = os.path.join(output_dir_ts, f'latest_release_qoq_{data_name}.xlsx')
        latest_release_yoy_path = os.path.join(output_dir_ts, f'latest_release_yoy_{data_name}.xlsx')

        # Save latest release files
        latest_release_df.to_excel(latest_release_path, index=True)
        latest_release_qoq_df.to_excel(latest_release_qoq_path, index=True)
        latest_release_yoy_df.to_excel(latest_release_yoy_path, index=True)


        # ---------------------------------------------------------------------------------------------
        # Revision Data
        # ---------------------------------------------------------------------------------------------

        # Define file paths
        revision_path = os.path.join(output_dir_ts, f'revision_absolute_{data_name}.xlsx')
        revision_qoq_path = os.path.join(output_dir_ts, f'revision_qoq_{data_name}.xlsx')
        revision_yoy_path = os.path.join(output_dir_ts, f'revision_yoy_{data_name}.xlsx')

        # Save revision files
        revision_df.to_excel(revision_path, index=True)
        revision_qoq_df.to_excel(revision_qoq_path, index=True)
        revision_yoy_df.to_excel(revision_yoy_path, index=True)
















# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                      PROCESS GDP DATA                                            #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# ==================================================================================================
#                                    Process BB Real-Time Data
# ==================================================================================================

gdp_rt, gdp_qoq_rt, gdp_yoy_rt = process_realtime_data(
                          rt_foldername="1_GDP_Data", rt_filename='Bundesbank_GDP_raw.csv', 
                          api_link='https://api.statistiken.bundesbank.de/rest/download/BBKRT/Q.DE.Y.A.AG1.CA010.A.I?format=csv&lang=de',
                          data_name='GDP', 
                          output_dir_df=output_dir_gdp)





# ==================================================================================================
#                                    Process 1995-2005 data
# ==================================================================================================


# -------------------------------------------------------------------------------------------------#
# Import
# -------------------------------------------------------------------------------------------------#

gdp_filepath = os.path.join(wd, "0_0_Data", "1_GDP_Data")

# Filenames
filename_95_05 = 'GDP_1995-2005_release.xlsx'
filepath_95_05 = os.path.join(gdp_filepath, filename_95_05)

# Load data (stored in row 10 onwards)
gdp_95_05 = pd.read_excel(
    filepath_95_05,
    header=0,           # First row contains the headers
    skiprows=range(1, 9),  # Skip rows 2-9 (zero-indexed rows 1-8)
    index_col=0,
    na_values=['', ' ']
)


# Reformat
gdp_95_05 = process_BB_GDP(gdp_95_05, col_convert= True, col_subset= False)




# -------------------------------------------------------------------------------------------------#
# Rescale
# -------------------------------------------------------------------------------------------------#

# Set 2000 = 100 by using the mean of the year 2000 in the last available column
rows_2000 = gdp_95_05.index.year == 2000
last_col = gdp_95_05.columns[-1]
avg_2000 = gdp_95_05.loc[rows_2000, last_col].mean()

# Rescale the df
gdp_95_05 = 100 * gdp_95_05 / avg_2000

# show(gdp_95_05)





# ==================================================================================================
#                                    Joint long-term dataset
# ==================================================================================================


# ==================================================================================================
# Concatenate columns from 1995-2005-Q1 and 2005-Q2 onwards
# ==================================================================================================

# Ensure no overlapping columns
overlap_cols = gdp_95_05.columns.intersection(gdp_rt.columns)
gdp_95_05_no_overlap = gdp_95_05.drop(columns=overlap_cols, errors='ignore')

# Concatenate columns (axis=1) to get all vintages from 1995 to today
gdp_merged = pd.concat([gdp_95_05_no_overlap, gdp_rt], axis=1)

# Sort columns by date if needed
gdp_merged = gdp_merged.reindex(sorted(gdp_merged.columns), axis=1)

# show(gdp_merged)




# ==================================================================================================
# ADD LONG-ROW DATA
# ==================================================================================================

# -------------------------------------------------------------------------------------------------#
# Import
# -------------------------------------------------------------------------------------------------#

# Filenames
filename_lr = 'GDP_lange_reihe.xlsx'
filepath_lr = os.path.join( gdp_filepath, filename_lr)

# Load data 
gdp_lr = pd.read_excel(
    filepath_lr,
    header=0,         
    index_col=0,
    na_values=['', ' ']
)

# Reformat
gdp_lr = process_BB_GDP(gdp_lr, col_convert = False, col_subset = False)




# ==================================================================================================
# Extend the report-date cols by the available historical long-term data
# ==================================================================================================

# -------------------------------------------------------------------------------------------------#
# Select new data from the long row
# -------------------------------------------------------------------------------------------------#
# Cutoff date
match_date = pd.Timestamp('1991-02-15')
# Select gdp_lr subset up to cutoff
gdp_lr_cut = gdp_lr[gdp_lr.index <= match_date]

# Rescale the single column in gdp_lr to match each column in gdp_merged at match_date
gdp_lr_scaled = pd.DataFrame(index=gdp_lr_cut.index)


# Rescale columnwise, as latest data point changes
scaled_cols = {}
lr_col = gdp_lr.columns[0]

for col in gdp_merged.columns:
    if match_date in gdp_merged.index and match_date in gdp_lr.index:
        rt_val = gdp_merged.loc[match_date, col]
        lr_val = gdp_lr.loc[match_date, lr_col]
        scale_factor = rt_val / lr_val if lr_val != 0 else 1
        scaled_cols[col] = gdp_lr_cut[lr_col] * scale_factor
    else:
        scaled_cols[col] = gdp_lr_cut[lr_col]

# Combine
gdp_lr_scaled = pd.concat(scaled_cols, axis=1)



# -------------------------------------------------------------------------------------------------#
# Merge
# -------------------------------------------------------------------------------------------------#

# Remove match_date from gdp_lr_scaled to avoid duplicate row
gdp_lr_scaled = gdp_lr_scaled[gdp_lr_scaled.index < match_date]

# Concatenate vertically
gdp_combined = pd.concat([gdp_lr_scaled, gdp_merged])

# Sort index if necessary
gdp_combined = gdp_combined.sort_index()


#show(gdp_combined)



# ==================================================================================================
# OPTIONAL: Extend the real-time data backwards
# ==================================================================================================

if settings.extend_rt_data_backwards:

    print('Imputing real-time releases for Q1-1985 to Q2-1995 ... \n')

    # Ensure proper datetime formats
    gdp_combined = gdp_combined.copy()
    gdp_combined.index = pd.to_datetime(gdp_combined.index).to_period('Q').to_timestamp()
    gdp_combined.columns = pd.to_datetime(gdp_combined.columns)

    # Select the first datetime column
    truncated_cols = {}
    first_col = gdp_combined.iloc[:,0].dropna()
    #show(first_col)
    
    
    # Set starting point for release dates
    current_col_time = gdp_combined.columns[0]
    current_series = first_col.copy()


    # Set lower date limit for index (not col name!)
    while len(current_series) > 1 and current_col_time >= pd.Timestamp("1989-01-01"):

        # Drop the most recent value to simulate a past release
        current_series = current_series.iloc[:-1]
        current_col_time = current_col_time - pd.offsets.QuarterEnd(1)
        truncated_cols[current_col_time] = current_series.copy()

    # Create DataFrame and reindex to match gdp_combined
    truncated_df = pd.DataFrame(truncated_cols)
    truncated_df = truncated_df.reindex(gdp_combined.index)

    # Combine and sort columns
    gdp_combined = pd.concat([gdp_combined, truncated_df], axis=1)
    gdp_combined = gdp_combined.reindex(sorted(gdp_combined.columns), axis=1)

    #show(gdp_combined)





# ==================================================================================================
# Get qoq and yoy
# ==================================================================================================

gdp_qoq_combined = get_qoq(gdp_combined)
gdp_yoy_combined = get_yoy(gdp_qoq_combined)

#show(gdp_qoq_combined)




# ==================================================================================================
# STORE GDP DATA
# ==================================================================================================

build_store_evaluation_timeseries(df_combined=gdp_combined, 
                                  df_qoq_combined=gdp_qoq_combined, 
                                  df_yoy_combined=gdp_yoy_combined, 
                                  output_dir_df=output_dir_gdp, data_name='GDP')










# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                 DATA PROCESSING - Bruttowertschöpfung (gross value added)                        #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#



if settings.run_gva_evaluation:

    print(" \n Starting GVA data processing ... \n")

    # ==================================================================================================
    #                                    Process BB Real-Time Data
    # ==================================================================================================

    gva_rt, gva_qoq_rt, gva_yoy_rt= process_realtime_data(
                            rt_foldername="1_GVA_Data", rt_filename='Bundesbank_GVA_raw.csv', 
                            api_link='https://api.statistiken.bundesbank.de/rest/download/BBKRT/Q.DE.Y.A.AU1.CA010.A.I?format=csv&lang=de',
                            data_name='GVA', 
                            output_dir_df=output_dir_gva)

    


    # ==================================================================================================
    # STORE GVA DATA
    # ==================================================================================================

    build_store_evaluation_timeseries(df_combined=gva_rt, 
                                    df_qoq_combined=gva_qoq_rt, 
                                    df_yoy_combined=gva_yoy_rt, 
                                    output_dir_df=output_dir_gva, data_name='GVA')














# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                          DATA PROCESSING - quarterly ifo Forecasts                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#



# ==================================================================================================
#  PROCESSING PIPELINE
# ==================================================================================================

def process_ifo_qoq_forecasts(ifo_qoq_raw, ifo_qoq_output_path):

    ## Set first column as index and drop empty columns
    ifo_qoq_raw.set_index(ifo_qoq_raw.columns[0], inplace=True)
    ifo_qoq_raw.dropna(axis=1, how='all', inplace=True)

    #show(ifo_qoq_raw)

    # --------------------------------------------------------------------------------------------------
    # Drop all values which are not predictions
    # --------------------------------------------------------------------------------------------------

    # Loop through columns and set entries to NaN if row date is >= 2 months older than column date
    for col in ifo_qoq_raw.columns:
        too_old = ifo_qoq_raw.index <= (col - pd.DateOffset(months=2))
        ifo_qoq_raw.loc[too_old, col] = pd.NA

    # Drop rows that are all NA
    ifo_qoq = ifo_qoq_raw.dropna(how='all')

    # show(ifo_qoq)

    ## Save files
    ifo_qoq.to_excel(ifo_qoq_output_path, index=True)   




# ==================================================================================================
#  GDP FORECASTS
# ==================================================================================================

## Filepath
ifo_qoq_input_path = os.path.join(wd, '0_0_Data', '0_Forecast_Inputs')

# --------------------------------------------------------------------------------------------------
# Load in the excel dynamically
# --------------------------------------------------------------------------------------------------

# Find the evaluation Excel files in the directory
excel_files = [f for f in os.listdir(ifo_qoq_input_path) if f.endswith('.xlsx') and f.startswith('ifo_Konjunkturprognose')]

# If no matching files, raise error
if not excel_files:
    raise ValueError(f"No Excel files starting with 'ifo_Konjunkturprognose' found in the directory {ifo_qoq_input_path}.")

# Helper to parse season and year from filename
def _season_year_key(filename):
    # Get last part split by '_', e.g. 'S24' or 'F25'
    last_part = filename.rsplit('_', 1)[-1]
    # Remove extension
    last_part = last_part.replace('.xlsx', '')
    # Season: first char, Year: rest
    if len(last_part) < 3:
        return (0, 0)  # fallback for malformed
    season = last_part[0]
    year = int(last_part[1:])
    # Define season order: W < H < S < F
    season_order = {'W': 1, 'H': 2, 'S': 3, 'F': 4}
    return (year, season_order.get(season, 0))

# Sort files by (year, season) descending, pick latest
excel_files_sorted = sorted(excel_files, key=_season_year_key, reverse=True)
ifo_qoq_filename = excel_files_sorted[0]
ifo_qoq_path = os.path.join(ifo_qoq_input_path, ifo_qoq_filename)

# Set path
ifo_qoq_path = os.path.join(ifo_qoq_input_path, excel_files[0])

## Load
ifo_qoq_raw = pd.read_excel(
    ifo_qoq_path,
    sheet_name='BIP',
    skiprows=2,      # skip first two rows
    header=0         # now row 3 becomes the header (index 0 after skipping)
)

# Message 
print(f"Loaded Excel file: {excel_files[0]}")


# --------------------------------------------------------------------------------------------------
# Process and save
# --------------------------------------------------------------------------------------------------

ifo_qoq_output_path = os.path.join(ifo_qoq_output_dir, 'ifo_qoq_forecasts.xlsx')
process_ifo_qoq_forecasts(ifo_qoq_raw, ifo_qoq_output_path)



















# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                           DATA PROCESSING - COMPONENT FORECASTS                                  #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

""" 
Rescale the component-level data from excel file to match Bundesbank real-time data format, 
reapply similar processing pipeline, store results

-> output_dir_ts_components, component_forecast_output_dir

NOTE: component-level date is already in qoq-format
"""

"""
REQUIRED OUTPUTS:
-> simulated rt_component_series w/o forecasts, used in the Naive Forecaster, 
store to: wd, '0_0_Data', '2_Processed_Data', '1_rt_component_series'
-> forecast series in the desiered format, required in evaluation module,
store to: wd, '0_0_Data', '2_Processed_Data', '3_gdp_component_forecast'
-> Evaluation series mx1 for all components, used in evlaution module
store to: wd, '0_0_Data', '2_Processed_Data', '2_component_Evaluation_series'
"""

## INPUT
# Source of the component data, Excel created in 2_1 out of the ifo forecast archive
ifo_components_path = os.path.join( wd, "0_0_Data", "0_Forecast_Inputs", "1_ifo_quarterly_components",
                                    "ifo_BIP_Komponenten.xlsx")

if not os.path.exists(ifo_components_path):
    raise FileNotFoundError(f"Missing input file: {ifo_components_path}")



# =================================================================================================#
#                             EXTRACT COMPONENT_LEVEL FORECASTS                                    #
# =================================================================================================#

## HELPER: turn sheet names into tokens used for filenaming
def _safe_sheet_filename(sheet_name: str) -> str:
    # keep filenames portable
    s = re.sub(r"\s+", "_", sheet_name.strip())
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s or "sheet"


## HELPER: Convert datetimes to quarters
def _to_quarter(dt):
    """Convert a datetime to (year, quarter) tuple for quarterly comparison."""
    if pd.isna(dt):
        return None
    dt = pd.to_datetime(dt)
    quarter = (dt.month - 1) // 3 + 1
    return (dt.year, quarter)


## HELPER: Convert quarters back to mid-quarter timestamp
def _quarter_to_midquarter_timestamp(year, quarter):
    """Convert (year, quarter) tuple to mid-quarter timestamp."""
    month_map = {1: 2, 2: 5, 3: 8, 4: 11}
    month = month_map[quarter]
    dt = pd.Timestamp(year=year, month=month, day=15)
    return dt


## HELPER: Process raw component dataframe to extract real-time data, forecast series, and first release evaluation
def process_ifo_component_realtime(df_raw_component):
    """
    Process ifo component data to extract real-time data, forecast series, and first releases.
    
    Algorithm:
    1. Convert all dates to quarterly level (year, quarter tuples)
    2. Build real-time df: for each column, keep only rows where row_quarter < col_quarter
    3. Row-wise forward-fill: fill NAs by left neighbor if left neighbor is non-NA
    4. Build forecast df: opposite of real-time - set to NaN where row_quarter < col_quarter (keep forecasts)
    5. Build first release evaluation series: progression through columns, new observations each time
    
    Returns: 
        df_rt (real-time data with mid-quarter cols, original date rows)
        df_forecast (forecast data with same structure)
        first_release_eval_df (mx1 first release evaluation series)
    """
    # Set first column as index
    if df_raw_component.shape[1] >= 2:
        df_raw_component.set_index(df_raw_component.columns[0], inplace=True)
    
    # Convert index and columns to datetime first
    df_raw_component.index = pd.to_datetime(df_raw_component.index)
    df_raw_component.columns = pd.to_datetime(df_raw_component.columns)
    
    # Convert to quarters
    row_quarters = df_raw_component.index.map(_to_quarter)
    col_quarters = df_raw_component.columns.map(_to_quarter)
    
    # ============================================================================
    # 1. Build real-time data: for each column, keep only rows where row_quarter < col_quarter
    # ============================================================================
    df_rt = df_raw_component.copy()
    for i, col in enumerate(df_rt.columns):
        col_quarter = col_quarters[i]
        # Set to NaN where row_quarter >= col_quarter (keep only strictly before)
        for j, row_idx in enumerate(df_rt.index):
            row_quarter = row_quarters[j]
            if row_quarter >= col_quarter:
                df_rt.iloc[j, i] = np.nan
    
    # ============================================================================
    # 2. Row-wise forward-fill for remaining NAs (left-to-right across columns)
    # ============================================================================
    # Go through each row and fill NAs from left neighbor
    for row_idx in range(len(df_rt)):
        for col_idx in range(1, len(df_rt.columns)):
            if pd.isna(df_rt.iloc[row_idx, col_idx]) and not pd.isna(df_rt.iloc[row_idx, col_idx - 1]):
                df_rt.iloc[row_idx, col_idx] = df_rt.iloc[row_idx, col_idx - 1]
    
    # Drop completely empty rows and columns
    df_rt = df_rt.dropna(how='all').dropna(how='all', axis=1)
    
    # Convert rows (index) to mid-quarter timestamps, keep columns at original datetime
    df_rt.index = df_rt.index.map(lambda dt: _quarter_to_midquarter_timestamp(*_to_quarter(dt)))
    df_rt.index.name = "date"
    
    # ============================================================================
    # 3. Build forecast data: opposite of real-time (keep forecasts where row_quarter >= col_quarter)
    # ============================================================================
    df_forecast = df_raw_component.copy()
    for i, col in enumerate(df_forecast.columns):
        col_quarter = col_quarters[i]
        # Set to NaN where row_quarter < col_quarter (keep only where row >= col, i.e., forecasts)
        for j, row_idx in enumerate(df_forecast.index):
            row_quarter = row_quarters[j]
            if row_quarter < col_quarter:
                df_forecast.iloc[j, i] = np.nan
    
    # Drop completely empty rows and columns
    df_forecast = df_forecast.dropna(how='all').dropna(how='all', axis=1)
    
    # Convert rows (index) to mid-quarter timestamps, keep columns at original datetime
    df_forecast.index = df_forecast.index.map(lambda dt: _quarter_to_midquarter_timestamp(*_to_quarter(dt)))
    df_forecast.index.name = "date"
    
    # ============================================================================
    # 4. Build first release evaluation series
    # ============================================================================
    # Algorithm: loop over columns, taking first row initially, then all new observations
    eval_data = {}
    last_stored_row = None
    
    rt_sorted = df_rt.copy()
    rt_sorted = rt_sorted.reindex(sorted(rt_sorted.columns), axis=1)
    
    for col in rt_sorted.columns:
        col_quarter = _to_quarter(col)
        col_data = rt_sorted[col].dropna()
        
        if col_data.empty:
            continue
        
        # Get candidates: values available at this publication
        candidates = col_data.copy()
        
        # Filter to only new observations
        if last_stored_row is not None:
            candidates = candidates[candidates.index > last_stored_row]
        
        # Add to evaluation series
        if not candidates.empty:
            for idx, val in candidates.items():
                if idx not in eval_data:
                    eval_data[idx] = val
        
        # Update last stored row
        if not candidates.empty:
            last_stored_row = candidates.index.max()
    
    # Create evaluation DataFrame (mx1)
    if eval_data:
        first_release_eval_df = pd.DataFrame.from_dict(eval_data, orient='index', columns=['first_release_value'])
        first_release_eval_df.index.name = "date"
        first_release_eval_df = first_release_eval_df.sort_index()
    else:
        first_release_eval_df = pd.DataFrame(columns=['first_release_value'])
        first_release_eval_df.index.name = "date"
    
    return df_rt, df_forecast, first_release_eval_df





# Discover sheet names first
xls = pd.ExcelFile(ifo_components_path)

for sheet in xls.sheet_names:
    # Load each sheet
    # First two rows are meta, row 3 is header (publication dates), col1 is target date
    df_raw_component = pd.read_excel(
        ifo_components_path,
        sheet_name=sheet,
        skiprows=2,
        header=0,
    )
    
    # Process to get real-time data, forecast series, and first release evaluation
    df_rt, df_forecast, first_release_eval_df = process_ifo_component_realtime(df_raw_component.copy())

    # Get YoY from QoQ real-time data
    try:
        df_yoy_rt = get_yoy(df_rt)
    except Exception:
        df_yoy_rt = pd.DataFrame()

    # Get YoY from QoQ forecast data
    try:
        df_yoy_forecast = get_yoy(df_forecast)
    except Exception:
        df_yoy_forecast = pd.DataFrame()

    # Save real-time QoQ and YoY data
    safe_sheet_name = _safe_sheet_filename(sheet)
    out_path_qoq_rt = os.path.join(output_dir_ts_components, f"qoq_rt_data_{safe_sheet_name}.xlsx")
    out_path_yoy_rt = os.path.join(output_dir_ts_components, f"yoy_rt_data_{safe_sheet_name}.xlsx")

    _safe_to_excel(df_rt, out_path_qoq_rt, index=True)
    _safe_to_excel(df_yoy_rt, out_path_yoy_rt, index=True)

    # Save forecast QoQ and YoY data
    out_path_qoq_forecast = os.path.join(component_forecast_output_dir, f"qoq_forecast_data_{safe_sheet_name}.xlsx")
    out_path_yoy_forecast = os.path.join(component_forecast_output_dir, f"yoy_forecast_data_{safe_sheet_name}.xlsx")

    _safe_to_excel(df_forecast, out_path_qoq_forecast, index=True)
    _safe_to_excel(df_yoy_forecast, out_path_yoy_forecast, index=True)

    # Save evaluation QoQ table (first release)
    os.makedirs(output_dir_eval_components, exist_ok=True)
    out_eval_qoq = os.path.join(output_dir_eval_components, f"first_release_qoq_{safe_sheet_name}.xlsx")
    _safe_to_excel(first_release_eval_df, out_eval_qoq, index=True)

    # Build YoY evaluation series from the QoQ evaluation table
    try:
        first_release_eval_yoy_df = get_yoy(first_release_eval_df)
    except Exception:
        first_release_eval_yoy_df = pd.DataFrame()

    out_eval_yoy = os.path.join(output_dir_eval_components, f"first_release_yoy_{safe_sheet_name}.xlsx")
    _safe_to_excel(first_release_eval_yoy_df, out_eval_yoy, index=True)







# --------------------------------------------------------------------------------------------------
print(f" \n Data Processing complete, results are in Subfolders {output_dir_gdp} and {output_dir_gva} of working directory {wd} \n")
# --------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#
