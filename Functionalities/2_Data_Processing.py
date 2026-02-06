
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

## Directory for component time series
component_output_dir = os.path.join(wd, '0_0_Data', '2_Processed_Data', '0_gdp_component_series')
os.makedirs(component_output_dir, exist_ok=True)







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



def build_store_evaluation_timeseries(df_combined, df_qoq_combined, df_yoy_combined, output_dir_df, data_name):

    # --------------------------------------------------------------------------------------------------
    # First Release Time Series
    # --------------------------------------------------------------------------------------------------

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


    # Call this function on absolute, qoq and yoy
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

# --------------------------------------------------------------------------------------------------
#  COMPONENT-LEVEL FORECASTS
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
    # Load each sheet with the same layout assumption as before:
    # first two rows are meta, row 3 is header (publication dates), col1 is target date
    ifo_qoq_raw = pd.read_excel(
        ifo_components_path,
        sheet_name=sheet,
        skiprows=2,
        header=0,
    )

    out_name = f"ifo_qoq_{_safe_sheet_filename(sheet)}_forecasts.xlsx"
    ifo_qoq_output_path = os.path.join(component_output_dir, out_name)

    process_ifo_qoq_forecasts(ifo_qoq_raw, ifo_qoq_output_path)
    print(f"Saved: {ifo_qoq_output_path}")












# --------------------------------------------------------------------------------------------------
print(f" \n Data Processing complete, results are in Subfolders {output_dir_gdp} and {output_dir_gva} of working directory {wd} \n")
# --------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#
