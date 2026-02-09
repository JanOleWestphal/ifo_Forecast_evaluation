
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
#                              SIMULATE COMPONENT-LEVEL RT DATA                                    #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

""" 
Rescale the component-level data from excel file to match Bundesbank real-time data format, reapply 
the same processing pipeline, store results

-> output_dir_ts_components, component_forecast_output_dir

NOTE: component-level date is already in qoq-format
"""





# =================================================================================================#
#                           SIMULATED QOQ and YOY Realtime Data                                    #
# =================================================================================================#


component_input_dir = os.path.join(wd, "0_0_Data", "2_Processed_Data", "3_gdp_component_forecast")


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------

def fourth_token_from_filename(path: str) -> str:
    """
    Expect filenames like: ifo_qoq_forecasts_CONSTR.xlsx
    Return: CONSTR (3rd underscore-separated token).
    """
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format (need at least 4 '_' tokens): {base}")
    return parts[3]



def fill_rowwise_left_on_na_after_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row:
      - Find first non-NA (start).
      - For all subsequent columns moving right:
          if cell is NA -> replace with value from immediate left column (same row).
          else keep as is.
    """
    out = df.copy()

    for idx in out.index:
        row = out.loc[idx]
        vals = row.to_numpy()

        # first non-NA
        start_pos = None
        for j, v in enumerate(vals):
            if pd.notna(v):
                start_pos = j
                break

        if start_pos is None:
            continue

        for j in range(start_pos + 1, len(vals)):
            if pd.isna(vals[j]):
                vals[j] = vals[j - 1]

        out.loc[idx] = vals

    return out


def filter_invalid_forecast_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove observations where the row date (target quarter) is in the same quarter 
    or later than the column date (publication/forecast date).
    
    Assumes rows are sorted chronologically (target dates) and columns are sorted 
    chronologically (publication dates). Both should be datetime objects or convertible to datetime.
    
    Also drops any rows and columns that are completely empty (all NaN) after filtering.
    
    Returns a filtered DataFrame with only valid forecast observations.
    """
    if df is None or df.empty:
        return df
    
    df_filtered = df.copy()
    
    # Convert index and columns to datetime if needed
    try:
        row_dates = pd.to_datetime(df_filtered.index)
    except Exception:
        row_dates = df_filtered.index
    
    try:
        col_dates = pd.to_datetime(df_filtered.columns)
    except Exception:
        col_dates = df_filtered.columns
    
    # Create a mask for invalid cells (row_date >= col_date)
    # These are cells where the target quarter is the same or later than publication date
    invalid_mask = pd.DataFrame(False, index=df_filtered.index, columns=df_filtered.columns)
    
    for i, row_date in enumerate(row_dates):
        for j, col_date in enumerate(col_dates):
            if row_date >= col_date:
                invalid_mask.iloc[i, j] = True
    
    # Set invalid cells to NaN
    df_filtered[invalid_mask] = np.nan
    
    # Drop completely empty rows and columns (all NaN)
    df_filtered = df_filtered.dropna(how='all').dropna(how='all', axis=1)
    
    return df_filtered


def build_first_release_series_rowwise(df_input: pd.DataFrame) -> pd.DataFrame:
    if df_input is None or df_input.empty:
        out = pd.DataFrame({"value": []})
        out.index.name = "date"
        return out

    df = df_input.copy()

    # Ensure chronological order
    try:
        df = df.reindex(sorted(df.columns), axis=1)
    except Exception:
        pass
    try:
        df = df.sort_index()
    except Exception:
        pass

    cols = list(df.columns)
    if len(cols) == 0:
        out = pd.DataFrame({"value": []})
        out.index.name = "date"
        return out

    # Use np.nan-compatible dtype
    out = pd.Series(np.nan, index=df.index, dtype="float64")

    # Column 0: take all non-NAs
    m0 = df[cols[0]].notna()
    out.loc[m0] = pd.to_numeric(df.loc[m0, cols[0]], errors="coerce").to_numpy()

    # Next columns: take only the "newly released" tail
    for j in range(1, len(cols)):
        prev_col = cols[j - 1]
        cur_col = cols[j]

        prev = df[prev_col]
        na_mask = prev.isna()
        if not na_mask.any():
            continue

        # first date where prev is NA (index assumed sorted)
        start_pos = int(np.argmax(na_mask.to_numpy()))
        start_date = df.index[start_pos]

        cur_tail = df.loc[df.index >= start_date, cur_col]
        take_mask = cur_tail.notna()
        if take_mask.any():
            out.loc[cur_tail.index[take_mask]] = pd.to_numeric(cur_tail.loc[take_mask], errors="coerce").to_numpy()

    result_df = out.dropna().to_frame("value")
    result_df.index.name = "date"
    return result_df


# --------------------------------------------------------------------------------------------------
# Main: process all Excel files in directory, all sheets
# --------------------------------------------------------------------------------------------------
excel_files = [f for f in os.listdir(component_input_dir) if f.lower().endswith(".xlsx")]

for f in excel_files:
    in_path = os.path.join(component_input_dir, f)
    token = fourth_token_from_filename(in_path)

    xls = pd.ExcelFile(in_path)

    # QoQ RT output
    out_path_qoq = os.path.join(output_dir_ts_components, f"qoq_rt_data_{token}.xlsx")
    # YoY RT output
    out_path_yoy = os.path.join(output_dir_ts_components, f"yoy_rt_data_{token}.xlsx")

    # Evaluation outputs (first-release series)
    os.makedirs(output_dir_eval_components, exist_ok=True)
    out_path_eval_qoq = os.path.join(output_dir_eval_components, f"first_release_qoq_{token}.xlsx")
    out_path_eval_yoy = os.path.join(output_dir_eval_components, f"first_release_yoy_{token}.xlsx")

    # Collect combined data across sheets (one df per workbook type)
    qoq_dfs = {}
    yoy_dfs = {}

    # --- read, fill, compute yoy; store per sheet for writing + for combined ---
    for sheet in xls.sheet_names:
        df = pd.read_excel(in_path, sheet_name=sheet)

        if df.shape[1] >= 2:
            df.set_index(df.columns[0], inplace=True)

        df_filled = fill_rowwise_left_on_na_after_start(df)
        df_yoy = get_yoy(df_filled)

        qoq_dfs[sheet] = df_filled
        yoy_dfs[sheet] = df_yoy

    # --- filter invalid forecast dates for QoQ/YoY data (row_date must be later than col_date) ---
    qoq_dfs = {sheet: filter_invalid_forecast_dates(df) for sheet, df in qoq_dfs.items()}
    yoy_dfs = {sheet: filter_invalid_forecast_dates(df) for sheet, df in yoy_dfs.items()}

    # --- write QoQ workbook ---
    with pd.ExcelWriter(out_path_qoq, engine="openpyxl") as writer_qoq:
        for sheet, df_filled in qoq_dfs.items():
            df_filled.to_excel(writer_qoq, sheet_name=sheet, index=True)

    # --- write YoY workbook ---
    with pd.ExcelWriter(out_path_yoy, engine="openpyxl") as writer_yoy:
        for sheet, df_yoy in yoy_dfs.items():
            df_yoy.to_excel(writer_yoy, sheet_name=sheet, index=True)

    # --- build "combined" dfs for evaluation series ---
    # If there is only one sheet, combined == that sheet.
    # If multiple sheets, stack them with a MultiIndex (sheet, date) to avoid collisions.
    if len(qoq_dfs) == 1:
        df_qoq_combined = next(iter(qoq_dfs.values()))
        df_yoy_combined = next(iter(yoy_dfs.values()))
    else:
        df_qoq_combined = pd.concat(qoq_dfs, names=["sheet", "date"])
        df_yoy_combined = pd.concat(yoy_dfs, names=["sheet", "date"])

    # --- filter invalid forecast dates for evaluation series ---
    df_qoq_combined = filter_invalid_forecast_dates(df_qoq_combined)
    df_yoy_combined = filter_invalid_forecast_dates(df_yoy_combined)

    # --- evaluation series (first-release) ---
    first_release_qoq_df = build_first_release_series_rowwise(df_qoq_combined)
    first_release_yoy_df = build_first_release_series_rowwise(df_yoy_combined)

    # --- save evaluation series ---
    first_release_qoq_df.to_excel(out_path_eval_qoq, index=True)
    first_release_yoy_df.to_excel(out_path_eval_yoy, index=True)
























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
ifo_qoq_input_path = os.path.join(wd, '0_0_Data', '0_Forecast_Inputs', '1_ifo_quarterly_gdp')

# --------------------------------------------------------------------------------------------------
# Load in the excel dynamically
# --------------------------------------------------------------------------------------------------

# Find the first Excel file in the directory
excel_files = [f for f in os.listdir(ifo_qoq_input_path) if f.endswith('.xlsx')]

# Check that it is the latest one
if len(excel_files) != 1:
    raise ValueError("Expected exactly one Excel file in the directory. Found: " + ", ".join(excel_files))

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





# ==================================================================================================
#  COMPONENT-LEVEL FORECASTS
# ==================================================================================================

def _safe_sheet_filename(sheet_name: str) -> str:
    # keep filenames portable
    s = re.sub(r"\s+", "_", sheet_name.strip())
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s or "sheet"


def process_ifo_component_realtime(df_raw):
    """
    Process ifo component data to extract real-time data and first releases.
    
    Algorithm:
    1. For each column, extract all non-NA values where row_date < col_date (real-time data)
    2. Forward-fill: if a column's first entry is NA, take value from the column to the left
    3. For first release: take the value from one quarter before the column date
    
    Returns: df_rt (real-time data), first_release_dict (first release values)
    """
    # Set first column as index
    if df_raw.shape[1] >= 2:
        df_raw.set_index(df_raw.columns[0], inplace=True)
    
    # Convert index and columns to datetime
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw.columns = pd.to_datetime(df_raw.columns)
    
    # 1. Build real-time data: for each column, keep only rows where row_date < col_date
    df_rt = df_raw.copy()
    for col in df_rt.columns:
        col_date = col
        # Set to NaN where row_date >= col_date (keep only dates strictly before column date)
        invalid_rows = df_rt.index >= col_date
        df_rt.loc[invalid_rows, col] = np.nan

    # 2. Row-wise forward-fill (left-to-right across columns)
    #    so that if a cell is NA and the previous (left) column has a value in the same row, copy it.
    df_rt = df_rt.ffill(axis=1)

    # Drop completely empty rows and columns
    df_rt = df_rt.dropna(how='all').dropna(how='all', axis=1)

    # 3. Build first release evaluation series per user's algorithm.
    def build_evaluation_df(source_df):
        eval_df = pd.DataFrame(index=pd.DatetimeIndex([]), columns=source_df.columns)
        last_stored = None
        for col in source_df.columns:
            col_date = col
            upper = col_date - pd.DateOffset(months=3)
            if last_stored is None:
                lower = source_df.index.min() - pd.DateOffset(months=3)
            else:
                lower = last_stored - pd.DateOffset(months=3)

            mask = (source_df.index > lower) & (source_df.index <= upper)
            candidates = source_df.loc[mask, col]

            for row_date, val in candidates.dropna().items():
                if row_date not in eval_df.index:
                    eval_df = eval_df.reindex(eval_df.index.append(pd.DatetimeIndex([row_date])))
                eval_df.at[row_date, col] = val

            if len(eval_df.index) > 0:
                last_stored = eval_df.index.max()

        eval_df = eval_df.sort_index().dropna(how='all').dropna(how='all', axis=1)
        return eval_df

    first_release_eval_df = build_evaluation_df(df_raw)

    # Build a lightweight first-release dict for legacy usage: last non-NA per column
    first_release_dict = {}
    for col in df_raw.columns:
        if col in first_release_eval_df.columns:
            vals = first_release_eval_df[col].dropna()
            if len(vals) > 0:
                first_release_dict[col] = vals.iloc[-1]
                continue
        target = col - pd.DateOffset(months=3)
        if target in df_raw.index and pd.notna(df_raw.at[target, col]):
            first_release_dict[col] = df_raw.at[target, col]

    return df_rt, first_release_dict, first_release_eval_df


# --------------------------------------------------------------------------------------------------
#  COMPONENT-LEVEL FORECASTS PROCESSING
# --------------------------------------------------------------------------------------------------

ifo_components_dir = os.path.join(
    wd, "0_0_Data", "0_Forecast_Inputs", "1_ifo_quarterly_components"
)
ifo_components_path = os.path.join(ifo_components_dir, "ifo_BIP_Komponenten.xlsx")

if not os.path.exists(ifo_components_path):
    raise FileNotFoundError(f"Missing input file: {ifo_components_path}")

# Discover sheet names first
xls = pd.ExcelFile(ifo_components_path)

for sheet in xls.sheet_names:
    # Load each sheet
    # First two rows are meta, row 3 is header (publication dates), col1 is target date
    df_raw = pd.read_excel(
        ifo_components_path,
        sheet_name=sheet,
        skiprows=2,
        header=0,
    )
    
    # Process to get real-time data and first releases (including evaluation df)
    df_rt, first_release_dict, first_release_eval_df = process_ifo_component_realtime(df_raw.copy())

    # Get YoY from QoQ
    df_yoy_rt = get_yoy(df_rt)

    # Save real-time QoQ and YoY data
    safe_sheet_name = _safe_sheet_filename(sheet)
    out_path_qoq = os.path.join(output_dir_ts_components, f"qoq_rt_data_{safe_sheet_name}.xlsx")
    out_path_yoy = os.path.join(output_dir_ts_components, f"yoy_rt_data_{safe_sheet_name}.xlsx")

    _safe_to_excel(df_rt, out_path_qoq, index=True)
    _safe_to_excel(df_yoy_rt, out_path_yoy, index=True)

    # Save evaluation QoQ table built by the algorithm
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

    # Also keep legacy simple first-release series derived from the dict (value at t-1q)
    first_release_series = pd.Series(first_release_dict, name='value')
    first_release_df = first_release_series.to_frame()
    first_release_df.index.name = 'date'
    
    # Build a legacy first-release YoY series from the evaluation YoY table
    first_release_yoy_dict = {}
    if not first_release_eval_yoy_df.empty:
        for col in first_release_eval_yoy_df.columns:
            vals = first_release_eval_yoy_df[col].dropna()
            if len(vals) > 0:
                first_release_yoy_dict[col] = vals.iloc[-1]

    first_release_yoy_series = pd.Series(first_release_yoy_dict, name='value')
    first_release_yoy_df = first_release_yoy_series.to_frame()
    first_release_yoy_df.index.name = 'date'
    
    # Save evaluation series (first releases)
    os.makedirs(output_dir_eval_components, exist_ok=True)
    out_path_eval_qoq = os.path.join(output_dir_eval_components, f"first_release_qoq_{safe_sheet_name}.xlsx")
    out_path_eval_yoy = os.path.join(output_dir_eval_components, f"first_release_yoy_{safe_sheet_name}.xlsx")
    
    _safe_to_excel(first_release_df, out_path_eval_qoq, index=True)
    _safe_to_excel(first_release_yoy_df, out_path_eval_yoy, index=True)












# --------------------------------------------------------------------------------------------------
print(f" \n Data Processing complete, results are in Subfolders {output_dir_gdp} and {output_dir_gva} of working directory {wd} \n")
# --------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#
