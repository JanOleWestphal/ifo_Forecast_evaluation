

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













def create_qoq_evaluation_df(qoq_forecast_df, eval_vector):
    """
    For each column in qoq_forecast_df, create:
    - an evaluation column matched by quarter to eval_vector
    - a difference column (forecast - actual)
    
    Matching is done on quarterly frequency.
    """
    
    # Ensure datetime and align to quarterly frequency
    qoq_forecast_df = qoq_forecast_df.copy()
    qoq_forecast_df.index = pd.to_datetime(qoq_forecast_df.index).to_period('Q').to_timestamp()
    eval_vector = eval_vector.copy()
    eval_vector.index = pd.to_datetime(eval_vector.index).to_period('Q').to_timestamp()

    for col in qoq_forecast_df.columns:
        # Create an empty Series for evaluations
        eval_col = []

        for date in qoq_forecast_df.index:
            forecast_val = qoq_forecast_df.at[date, col]

            if pd.notna(forecast_val):
                # Try to get evaluation for the same quarter
                eval_val = eval_vector.get(date, pd.NA)
            else:
                eval_val = pd.NA

            eval_col.append(eval_val)

        # Add evaluation column
        eval_colname = f"{col}_eval"
        qoq_forecast_df[eval_colname] = eval_col

        # Add difference column
        diff_colname = f"{col}_diff"
        qoq_forecast_df[diff_colname] = qoq_forecast_df[col] - qoq_forecast_df[eval_colname]

    return qoq_forecast_df




def plot_forecast_timeseries(*args, df_eval=None, title_prefix=None, figsize=(12, 8), 
                             show=False, save_path=None, save_name_prefix=None):
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
    
    # Plot 1: For each input DataFrame, plot all Q0-Q9 against evaluation
    for name, q_dfs in processed_dfs.items():
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot evaluation data first (if provided)
        if df_eval is not None:
            ax.plot(df_eval.index, df_eval.iloc[:, 0], 
                   color='black', linewidth=2, label='Evaluation', alpha=0.8)
        
        # Plot Q0-Q9 series
        colors = get_color_palette(10)
        for i, (q_key, q_df) in enumerate(q_dfs.items()):
            if not q_df.empty:
                ax.plot(q_df.index, q_df['value'], 
                       color=colors[i], marker='o', linewidth=1.5, 
                       markersize=4, label=q_key, alpha=0.7)
        
        # Customize plot
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        plot_title = f'{title_prefix} - {name} Forecast Horizons' if title_prefix else f'{name} Forecast Horizons'
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
    
    # Plot 2: For each Q0-Q9, plot across all input DataFrames
    for q in range(10):
        q_key = f'Q{q}'
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot evaluation data first (if provided)
        if df_eval is not None:
            ax.plot(df_eval.index, df_eval.iloc[:, 0], 
                   color='black', linewidth=2, label='Evaluation', alpha=0.8)
        
        # Plot Q series from each DataFrame
        colors = get_color_palette(len(processed_dfs))
        for i, (name, q_dfs) in enumerate(processed_dfs.items()):
            if q_key in q_dfs and not q_dfs[q_key].empty:
                # Create legend label
                if 'ifo' in name.lower():
                    legend_label = 'ifo'
                elif any(tag in name.lower() for tag in ['ar', 'sma', 'average']):
                    # Strip trailing underscore + number
                    legend_label = re.sub(r'_\d+$', '', name)
                else:
                    legend_label = name
                
                ax.plot(q_dfs[q_key].index, q_dfs[q_key]['value'], 
                       color=colors[i], marker='o', linewidth=1.5, 
                       markersize=4, label=legend_label, alpha=0.7)
        
        # Customize plot
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        plot_title = f'{title_prefix} - {q_key} Comparison' if title_prefix else f'{q_key} Comparison'
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
    
    return figures

















    """
    Matches forecast_df to eval_df by quarter, creating *_eval and *_diff columns.
    Uses np.nan as the missing placeholder so that all columns end up float64.
    """
    # 1) Copy & datetime‐index
    forecast = forecast_df.copy()
    eval_     = eval_df.copy()
    forecast.index = pd.to_datetime(forecast.index)
    eval_.index    = pd.to_datetime(eval_.index)

    # 2) Extract & coerce eval series to numeric (float64 + np.nan)
    eval_ser = pd.to_numeric(eval_.iloc[:, 0], errors='coerce')

    # 3) Coerce all forecast columns to numeric float64
    forecast = forecast.apply(pd.to_numeric, errors='coerce')

    result = pd.DataFrame(index=forecast.index)

    for col in forecast.columns:
        # Determine how to shift index
        if 'Qminus1' in col:
            targets = forecast.index - QuarterBegin(startingMonth=1)
        elif 'Q1' in col:
            targets = forecast.index + QuarterBegin(startingMonth=1)
        else:  # Q0
            targets = forecast.index

        # 4) Build the matched Series with np.nan defaults
        matched = pd.Series(
            [
                float(eval_ser.loc[d]) if d in eval_ser.index and not isinstance(eval_ser.loc[d], (list, np.ndarray, pd.Series))
                else float(eval_ser.loc[d][0]) if d in eval_ser.index and isinstance(eval_ser.loc[d], (list, np.ndarray, pd.Series)) and len(eval_ser.loc[d]) > 0
                else np.nan
                for d in targets
            ],
            index=forecast.index,
            dtype=float
        )

        # 5) Assign the eval and diff columns
        result[f'{col}_eval'] = matched.values  # ensure scalar values
        result[f'{col}_diff'] = forecast[col].values - matched.values  # ensure scalar values

    return result.T



"""
def get_ifoCAST_differences(forecast_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    
    For each timestamp in forecast_df, match evaluation values from eval_df based on
    quarters (previous quarter, same quarter, next quarter), and compute
    differences between forecast values and their corresponding quarterly evaluation.

    Parameters:
    - forecast_df: DataFrame with a DatetimeIndex and one or more forecast columns.
    - eval_df: DataFrame or Series with a DatetimeIndex and a single evaluation column.

    Returns:
    - DataFrame: A copy of forecast_df with added columns:
        * Qminus1_eval, Q0_eval, Q1_eval: evaluation values for prev, current, next quarter
        * {orig_col}_Qminus1_diff, {orig_col}_Q0_diff, {orig_col}_Q1_diff for each original forecast column
    
    # Copy original forecasts
    forecast = forecast_df.copy()

    # Preserve original forecast column names
    orig_cols = list(forecast_df.columns)

    # Prepare evaluation series
    eval_series = eval_df.squeeze()
    eval_series.index = pd.to_datetime(eval_series.index)

    # Convert indices to quarterly periods
    eval_per = eval_series.copy()
    eval_per.index = eval_per.index.to_period('Q')

    forecast_idx = pd.to_datetime(forecast.index)
    forecast_per = forecast_idx.to_period('Q')

    # Align evaluation for each quarter offset
    forecast['Qminus1_eval'] = eval_per.reindex(forecast_per - 1).values
    forecast['Q0_eval']       = eval_per.reindex(forecast_per).values
    forecast['Q1_eval']       = eval_per.reindex(forecast_per + 1).values

    # Compute diffs: forecast- evalautio
    forecast["Qminus1_diff"] = forecast['Qminus1'] - forecast['Qminus1_eval'] 
    forecast["Q0_diff"]       = forecast['Q0'] -  forecast['Q0_eval']
    forecast["Q1_diff"]       = forecast['Q1'] - forecast['Q1_eval']

    return forecast
"""