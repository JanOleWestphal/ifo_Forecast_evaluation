

# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        ~  OPTIONS  ~                                             #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#


# ==================================================================================================
# PARAMETER Setup:
#
# wd                   Sets the current working directory
# resultfolder_name_n_forecast    Sets the folder name where results of a given run are printed, e.g. 'results',
#                      Default naming: 'Results_model_memory-horizon_forecast'
#                      OVERRIDES WITH EVERY RUN!
#                       
# api_pull             Determines whether the data is automatically pulled form the Bundesbank API 
#                      (set = True) or whether a local version is used (set = False). 
#                      ATTENTION: if False, local file must be named 'Bundesbank_GDP_raw.csv'
# 
# model                Sets the forecast model used by the agent, options: 
#                           - 'AR': Auto-regressive model of order AR_order with a constant 
#                           - 'SMA': Simple moving average
#                           - 'AVERAGE': Naive average; just predict previous quarters' average for 
#                              all future quarters
# naming_convention    Determines whether the column names for the series published in Q2 are "Q2" 
#                      ('published') or "Q1", which would correspond to the last available data point
#                      ('data')
# 
# average_horizon      Sets the amount of previous quarters averaged over for 'SMA' and 'AVERAGE'
# AR_horizon           Sets the time frame on which the AR- models are estimated in quarters. Set to 
#                      a natural number n for looking backwards up to n quaters, set 'FULL' for use 
#                      of full series
# AR_order             Number of lags in the autoregressive model, recommendation: 2
# forecast_horizon     Determines how many quaters into the future predictions are made
#
# ==================================================================================================


# --------------------------------------------------------------------------------------------------
# Folders and Data
# --------------------------------------------------------------------------------------------------

# Set wd as '/path/to/your/project' or r'\path\to\your\project'
wd = r'C:\Users\janol\OneDrive\Desktop\ifo\Konjunkturprognose Evaluierung\Python Workfolder'

# Customize Resultfolder names, e.g. 'AR2_results', suggested: 'Default'
resultfolder_name_n_forecast = 'Default'   #set 'Default' for default: 'Results_model_memory_forecast'

# Decide wether to use the API pull or a local version of the file; True (suggested) or False 
api_pull = True # True or False


# Define whether column names follow the publishing date or the last available data point 
naming_convention = 'data'  # 'published' or 'data'

# Note for debugging purposes: 
"""
-> column names of any internal dataframes are always derived from the publishing date, setting them
   to the date of latest data availability is done only in the last processing steps.
-> to make the code easier to follow, this is done quick and dirty by just shifting the col datetime
   back by one quarter. This fits all historical records, but if the Bundesbank ever puts out 
   realtime data or misses publishing the previous quarter, this will lead to an error and needs to 
   be fixed.
"""



# --------------------------------------------------------------------------------------------------
# Define the model
# --------------------------------------------------------------------------------------------------

# Set the agent's forecasting method; options: 'AR', 'AVERAGE', 'SMA'
model = 'AR'

# For AR model: set number of lags (sugested: 2); int
AR_order = 2


# For average-based models: set time frame over which the agent averages in quarters; int or 'FULL'
average_horizon = 2

# For AR model: set the memory of the agent (timeframe the model is estimated on); int or 'FULL'
AR_horizon = 50
# unstable estimates for 2020_Q3 (release date, last observation 2020_Q2) for below 48


# Set how far the agent predicts into the future; int
forecast_horizon = 20

#-------------#
#  Note: If data is released at Time T, data is available up to T-1. Accordingly, forecasting t 
#        periods into the future requires t+1 forecasts, including the so-called nowcast for the 
#        present. The above parameter does not count this nowcast and sets how many FUTURE periods, 
#        are estimated. Setting forecast_horizon = t thus outputs t+1 prediction values.
#-------------#



# Set working directory
if wd:
    try:
        os.makedirs(wd, exist_ok=True)
        os.chdir(wd)
        print(f"Working directory set to: {wd} ... \n")
    except FileNotFoundError:
        print(f"Directory not found: {wd} \n")
else:
    print("No working directory set.")









def save_processed_forecaster_output(df_qoq, qoq_forecast_df, qoq_forecast_index_df, AR_summary=None):

    # --------------------------------------------------------------------------------------------------
    #  Build combined observed-forecasted dataframe
    # --------------------------------------------------------------------------------------------------

    # Store realized-predicted series in list:
    series_list = []

    # Loop through paired columns from df_qoq and qoq_forecast_df
    for col_real, col_forecast in zip(df_qoq.columns, qoq_forecast_df.columns):

        # Select observed values to array
        real_values = df_qoq[col_real].dropna().to_numpy()

        # Get forecasts as array.
        forecast_values = qoq_forecast_df[col_forecast].to_numpy()

        # Concatenate real and forecast data.
        combined = np.concatenate([real_values, forecast_values])

        # Convert the combined array into a Series.
        s = pd.Series(combined, name=col_real)

        # Append the Series to the list.
        series_list.append(s)

    # Concatenate all Series columnwise
    df_combined_qoq = pd.concat(series_list, axis=1)

    # Dynamicaly reset index
    start_date = pd.to_datetime(df.index[0])  # Convert to datetime if not already
    n_periods = len(df_combined_qoq)

    # Generate quarterly datetime index
    df_combined_qoq.index = pd.date_range(start=start_date, periods=n_periods, freq='QE')



    # --------------------------------------------------------------------------------------------------
    #  Create yearly values
    # --------------------------------------------------------------------------------------------------

    df_combined_yoy = get_yoy(df_combined_qoq)



    # ==================================================================================================
    # SAVE DT-INDEXED RESULTS AS EXCEL: df_combined_qoq, df_combined_yoy
    # ==================================================================================================

    """
    Modify this bit
    """

    # If clause for dynamic naming of results
    if model in ['AVERAGE', 'SMA']:

        # Full Data qoq
        filename_df_combined_qoq = f'dt_full_qoq{model}_{average_horizon}_{forecast_horizon-1}.xlsx'
        df_combined_qoq.to_excel(os.path.join(file_path_dt_qoq, filename_df_combined_qoq))

        # Full Data yoy
        filename_df_combined_yoy = f'dt_full_yoy_{model}_{average_horizon}_{forecast_horizon-1}.xlsx'
        df_combined_yoy.to_excel(os.path.join(file_path_dt_yoy, filename_df_combined_yoy)) 
 

    elif model == 'AR':

        # Full Data qoq
        filename_df_combined_qoq = f'dt_full_qoq_{model}{AR_order}_{AR_horizon}_{forecast_horizon-1}.xlsx'
        df_combined_qoq.to_excel(os.path.join(file_path_dt_qoq, filename_df_combined_qoq))

        # Full Data yoy
        filename_df_combined_yoy = f'dt_full_yoy_{model}_{AR_horizon}_{forecast_horizon-1}.xlsx'
        df_combined_yoy.to_excel(os.path.join(file_path_dt_yoy, filename_df_combined_yoy)) 




    # ==================================================================================================
    #  Reset indices and column names; Set Dates to Excel-friendly format
    #
    # 4 relevant dataframes: qoq_forecast_index_df, df_combined_qoq, df_combined_yoy, AR_summary
    #
    #  Renaming: 
    #            -> Applies switch accounting for whether cols are named by publishment data or by last
    #               data point 
    #            -> indices back to YYYY-Qx format for QoQ, YYYY format for YoY
    #            -> colnames to qoq_YYYY_0x and yoy_YYYY_0x
    #            -> prediction col names to qoq_YYYY_0x_f
    #            -> qoq_forecast_index_df: first col to YYYY_0x, second to YYYY_Qx
    #
    # ==================================================================================================



    # --------------------------------------------------------------------------------------------------
    # qoq_forecast_index_df: Manually rename
    # --------------------------------------------------------------------------------------------------

    # Pre-formating to datetime:
    qoq_forecast_index_df.iloc[:, 0] = pd.to_datetime(qoq_forecast_index_df.iloc[:, 0])
    qoq_forecast_index_df.iloc[:, 1] = pd.to_datetime(qoq_forecast_index_df.iloc[:, 1])

    # Define colnames
    col0_name = qoq_forecast_index_df.columns[0]
    col1_name = qoq_forecast_index_df.columns[1]

    # First col to YYYY_0x
    qoq_forecast_index_df[col0_name] = qoq_forecast_index_df[col0_name].apply(
        lambda x: f"{x.year}_0{((x.month - 1) // 3 + 1)}"
    )

    # Second col to YYYY_Qx
    qoq_forecast_index_df[col1_name] = qoq_forecast_index_df[col1_name].apply(
        lambda x: f"{x.year}_Q{((x.month - 1) // 3 + 1)}"
    )



    # --------------------------------------------------------------------------------------------------
    # Rename combined dfs
    # --------------------------------------------------------------------------------------------------

    ## Indices
    df_combined_qoq = rename_index_qoq(df_combined_qoq)
    df_combined_yoy = rename_index_yoy(df_combined_yoy)

    if 'AR_summary' in globals():
        AR_summary = rename_index_qoq(AR_summary)


    ## Columns
    # QoQ and YoY
    if naming_convention == 'published':
        df_combined_qoq = rename_col_publish('qoq', df_combined_qoq)
        df_combined_yoy = rename_col_publish('yoy', df_combined_yoy)

    elif naming_convention == 'data':
        df_combined_qoq = rename_col_data('qoq', df_combined_qoq)
        df_combined_yoy= rename_col_data('yoy', df_combined_yoy)
    else: 
        print("This should never be printed, check whether naming_convention is still up to date")



    # ==================================================================================================
    # SAVE RESULTS AS EXCEL: df_combined_qoq, df_combined_yoy,  qoq_forecast_index_df
    # ==================================================================================================

    # If clause for better naming of results
    if model in ['AVERAGE', 'SMA']:

        # Full Data qoq
        filename_df_combined_qoq = f'Real_and_Predicted_QoQ_{model}_{average_horizon}_{forecast_horizon-1}.xlsx'
        df_combined_qoq.to_excel(os.path.join(folder_path, filename_df_combined_qoq))

        # Full Data yoy
        filename_df_combined_yoy = f'Real_and_Predicted_YoY_{model}_{average_horizon}_{forecast_horizon-1}.xlsx'
        df_combined_yoy.to_excel(os.path.join(folder_path, filename_df_combined_yoy)) 

        # Indexed Predictions df
        filename_qoq_forecast_index_df = f'Indexed_Forecasts_QoQ_{model}_{average_horizon}_{forecast_horizon-1}.xlsx'
        qoq_forecast_index_df.to_excel(os.path.join(folder_path, filename_qoq_forecast_index_df)) 


    elif model == 'AR':

        # Full Data qoq
        filename_df_combined_qoq = f'Real_and_Predicted_QoQ_{model}{AR_order}_{AR_horizon}_{forecast_horizon-1}.xlsx'
        df_combined_qoq.to_excel(os.path.join(folder_path, filename_df_combined_qoq))

        # Full Data yoy
        filename_df_combined_yoy = f'Real_and_Predicted_YoY_{model}_{AR_horizon}_{forecast_horizon-1}.xlsx'
        df_combined_yoy.to_excel(os.path.join(folder_path, filename_df_combined_yoy)) 

        # Indexed Predictions df
        filename_qoq_forecast_index_df = f'Indexed_Forecasts_QoQ_{model}{AR_order}_{AR_horizon}_{forecast_horizon-1}.xlsx'
        qoq_forecast_index_df.to_excel(os.path.join(folder_path, filename_qoq_forecast_index_df)) 


    # Faulty model selection
    else:
        print("ERROR: wrong naming of outputs, check SAVE RESULTS section")



    # ----------------------------------------#
    #    Save Model Summary if Model is AR    #
    # ----------------------------------------#

    # Check wether there is an AR_summary, save if yes
    if 'AR_summary' in globals():
        filename_AR_summary = f'AR{AR_order}_{AR_horizon}_model_statistics.xlsx'
        AR_summary.to_excel(os.path.join(folder_path, filename_AR_summary))
