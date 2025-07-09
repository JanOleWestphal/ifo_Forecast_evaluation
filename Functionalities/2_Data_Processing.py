
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Data Processing
#
# Description:  Creates the dataframes used in the project
#               Stores them in Folder 0_0_Data/2_Processed_Data.
#
#               Processes both the real-time and historic GDP-data inputs as well as the forecasts.
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




# ==================================================================================================
#                                           SETUP
# ==================================================================================================

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




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                    DATA PROCESSING - GDP                                         #
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
#                                    Bundesbank Real-Time GDP
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Import Data via Bundesbank API or locally
# --------------------------------------------------------------------------------------------------

# Output filename
rt_filename = 'Bundesbank_GDP_raw.csv'
gdp_filepath = os.path.join(wd, "0_0_Data", "1_GDP_Data")
rt_filepath = os.path.join( gdp_filepath, rt_filename)


# Get file
# Optional API-pull:
if settings.api_pull == True:
    # Bundesbank API link
    url = 'https://api.statistiken.bundesbank.de/rest/download/BBKRT/Q.DE.Y.A.AG1.CA010.A.I?format=csv&lang=de'

    # Download and save the CSV file
    response = requests.get(url)
    if response.status_code == 200:
        with open(rt_filepath, 'wb') as f:
            f.write(response.content)
        print(f"Bundesbank data downloaded and saved to: {rt_filepath} ... \n ")
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
gdp_rt = pd.read_csv(rt_filepath, skiprows=11, names=header, sep=';', index_col=0, na_values=['', ' '])

# Process the Real-Time GDP data
gdp_rt = process_BB_GDP(gdp_rt)
 

# Inspect
# show(rt_gdp)

print("Data cleaned ... \n")



# --------------------------------------------------------------------------------------------------
# Create quarterly growth rates or changes: df_qoq (Quarter over Quarter)
# --------------------------------------------------------------------------------------------------

gdp_qoq_rt = get_qoq(gdp_rt)

# Inspect
# print(df_qoq.head())
#show(gdp_qoq_rt)


# --------------------------------------------------------------------------------------------------
# Create yearly growth rates or changes: df_yoy (Year over Year)
# --------------------------------------------------------------------------------------------------

gdp_yoy_rt = get_yoy(gdp_qoq_rt)

# show(df_yoy_rt)










# -------------------------------------------------------------------------------------------------#
# ==================================================================================================
#                                    Process 1995-2005 data
# ==================================================================================================
# -------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------#
# Import
# -------------------------------------------------------------------------------------------------#

# Filenames
filename_95_05 = 'GDP_1995-2005_release.xlsx'
filepath_95_05 = os.path.join( gdp_filepath, filename_95_05)

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







# -------------------------------------------------------------------------------------------------#
# ==================================================================================================
#                                    Joint long-term dataset
# ==================================================================================================
# -------------------------------------------------------------------------------------------------#


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



# ==================================================================================================
# Get qoq and yoy
# ==================================================================================================

gdp_qoq_combined = get_qoq(gdp_combined)
gdp_yoy_combined = get_yoy(gdp_qoq_combined)

#show(gdp_qoq_combined)









# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                    Evaluation Time Series                                        #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


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
first_release_gdp = build_first_release_series(gdp_combined)
first_release_qoq_gdp = build_first_release_series(gdp_qoq_combined)
first_release_yoy_gdp = build_first_release_series(gdp_yoy_combined)




# --------------------------------------------------------------------------------------------------
# Latest Release Time Series
# --------------------------------------------------------------------------------------------------

# Latest release: take the last (most recent) column from each DataFrame, keep its index and values
latest_release_gdp = gdp_combined.iloc[:, -1].to_frame(name='value')
latest_release_qoq_gdp = gdp_qoq_combined.iloc[:, -1].to_frame(name='value')
latest_release_yoy_gdp = gdp_yoy_combined.iloc[:, -1].to_frame(name='value')

# Truncate to only rows present in the corresponding first_release_dfs by index
latest_release_gdp = latest_release_gdp.loc[first_release_gdp.index]
latest_release_qoq_gdp = latest_release_qoq_gdp.loc[first_release_qoq_gdp.index]
latest_release_yoy_gdp = latest_release_yoy_gdp.loc[first_release_yoy_gdp.index]





# --------------------------------------------------------------------------------------------------
# Revision Time Series
# --------------------------------------------------------------------------------------------------

# Calculate revision as first_release - latest_release
revision_gdp = first_release_gdp['value'] - latest_release_gdp['value']
revision_gdp = revision_gdp.to_frame(name='revision')

revision_qoq_gdp = first_release_qoq_gdp['value'] - latest_release_qoq_gdp['value']
revision_qoq_gdp = revision_qoq_gdp.to_frame(name='revision')

revision_yoy_gdp = first_release_yoy_gdp['value'] - latest_release_yoy_gdp['value']
revision_yoy_gdp = revision_yoy_gdp.to_frame(name='revision')

















# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                          DATA PROCESSING - quarterly ifo Forecasts                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

# ==================================================================================================
#  Load ifo qoq Forecasts
# ==================================================================================================

## Filepath
ifo_qoq_input_path = os.path.join(wd, '0_0_Data', '0_Forecast_Inputs', '1_ifo_quarterly')

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



# ==================================================================================================
#  Rescale, drop all non-prediction values
# ==================================================================================================

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
















# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        Store Output                                              #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# ==================================================================================================
#  GDP Data
# ==================================================================================================

# Ensure directory exists
output_dir_gdp = os.path.join(wd, '0_0_Data', '2_Processed_Data', '1_GDP_series')
os.makedirs(output_dir_gdp, exist_ok=True)


# --------------------------------------------------------------------------------------------------
# Real-time GDP
# --------------------------------------------------------------------------------------------------

# Define file paths
df_path_rt = os.path.join(output_dir_gdp, 'absolute_rt_GDP.xlsx')
df_qoq_path_rt = os.path.join(output_dir_gdp, 'qoq_rt_GDP_data.xlsx')
df_yoy_path_rt = os.path.join(output_dir_gdp, 'yoy_rt_GDP_data.xlsx')

# Save files
gdp_rt.to_excel(df_path_rt, index=True)      
gdp_qoq_rt.to_excel(df_qoq_path_rt, index=True)
gdp_yoy_rt.to_excel(df_yoy_path_rt, index=True)


# --------------------------------------------------------------------------------------------------
# Combined DataSet
# --------------------------------------------------------------------------------------------------

# Define file paths
df_path_comb = os.path.join(output_dir_gdp, 'absolute_combined_GDP.xlsx')
df_qoq_path_comb = os.path.join(output_dir_gdp, 'qoq_combined_GDP_data.xlsx')
df_yoy_path_comb = os.path.join(output_dir_gdp, 'yoy_combined_GDP_data.xlsx')

# Save files
gdp_combined.to_excel(df_path_comb, index=True)      
gdp_qoq_combined.to_excel(df_qoq_path_comb, index=True)
gdp_yoy_combined.to_excel(df_yoy_path_comb, index=True)




# ==================================================================================================
#  Evaluation Time Series
# ==================================================================================================

# Ensure directory exists
output_dir_ts = os.path.join(wd, '0_0_Data', '2_Processed_Data', '2_Evaluation_series')
output_dir_ts_2 = os.path.join(wd, '0_1_Output_Data', '1_Evaluation_series')

os.makedirs(output_dir_ts, exist_ok=True)
os.makedirs(output_dir_ts_2, exist_ok=True)

# --------------------------------------------------------------------------------------------------
# First Release
# --------------------------------------------------------------------------------------------------

# Save these to two locations:

for output_dir_ts in [output_dir_ts, output_dir_ts_2]:
    # Define file paths
    first_release_path = os.path.join(output_dir_ts, 'first_release_absolute_GDP.xlsx')
    first_release_qoq_path = os.path.join(output_dir_ts, 'first_release_qoq_GDP.xlsx')
    first_release_yoy_path = os.path.join(output_dir_ts, 'first_release_yoy_GDP.xlsx')

    # Save first release files
    first_release_gdp.to_excel(first_release_path, index=True)
    first_release_qoq_gdp.to_excel(first_release_qoq_path, index=True)
    first_release_yoy_gdp.to_excel(first_release_yoy_path, index=True)


    # ---------------------------------------------------------------------------------------------
    # Latest Release
    # ---------------------------------------------------------------------------------------------

    # Define file paths
    latest_release_path = os.path.join(output_dir_ts, 'latest_release_absolute_GDP.xlsx')
    latest_release_qoq_path = os.path.join(output_dir_ts, 'latest_release_qoq_GDP.xlsx')
    latest_release_yoy_path = os.path.join(output_dir_ts, 'latest_release_yoy_GDP.xlsx')

    # Save latest release files
    latest_release_gdp.to_excel(latest_release_path, index=True)
    latest_release_qoq_gdp.to_excel(latest_release_qoq_path, index=True)
    latest_release_yoy_gdp.to_excel(latest_release_yoy_path, index=True)


    # ---------------------------------------------------------------------------------------------
    # Revision Data
    # ---------------------------------------------------------------------------------------------

    # Define file paths
    revision_path = os.path.join(output_dir_ts, 'revision_absolute_GDP.xlsx')
    revision_qoq_path = os.path.join(output_dir_ts, 'revision_qoq_GDP.xlsx')
    revision_yoy_path = os.path.join(output_dir_ts, 'revision_yoy_GDP.xlsx')

    # Save revision files
    revision_gdp.to_excel(revision_path, index=True)
    revision_qoq_gdp.to_excel(revision_qoq_path, index=True)
    revision_yoy_gdp.to_excel(revision_yoy_path, index=True)




# ==================================================================================================
#  Quarterly ifo Forecasts
# ==================================================================================================

## Folder
ifo_qoq_output_dir = os.path.join(wd, '0_0_Data', '2_Processed_Data', '3_ifo_qoq_series')
os.makedirs(output_dir_gdp, exist_ok=True)

## Define file paths
ifo_qoq_output_path = os.path.join(ifo_qoq_output_dir, 'ifo_qoq_forecasts.xlsx')

# Save files
ifo_qoq.to_excel(ifo_qoq_output_path, index=True)   




# --------------------------------------------------------------------------------------------------
print(f" \n Data Processing complete, results are in Subfolders {output_dir_gdp} and {output_dir_ts} of working directory {wd} \n")
# --------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#
