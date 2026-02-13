# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        ifo Forecast Evaluation Settings File
#
# Description:  Settings file for the ifo Forecast Evaluation project. This file contains all 
#               settings, for the evaluation code and is called in all subprocesses.
#            
# ==================================================================================================
# --------------------------------------------------------------------------------------------------





# ==================================================================================================
#                          SYSTEM SETTINGS: Which Time Series should be evaluated?
# ==================================================================================================

"""
The original project was to evaluate the ifo GDP forecasts, but its component's may also be evaluated.
To do this, select the series to be evaluated below and run scripts 1,2,3,4
"""

# Deicde whether to run the main evaluation of ifo's GDP forecasts against rt data; set to False to only run the component evaluation
evaluate_quarterly_gdp_forecasts = True # True or False

# Select whether to run the component evaluation; set to False to only run the GDP evaluation
evaluate_forecast_components = True # True or False

# Select which components to include, List of strings
included_components = ['EQUIPMENT']

# ['GDP', 'PRIVCON', 'PUBCON', 'CONSTR', 'EQUIPMENT','OPA', 'INVINV', 'DOMUSE', 'TRDBAL', 'EXPORT', 'IMPORT']

# Options:
"""
'GDP' -> for GDP, naive forecsats are built on incompletely revisioned real-time data, its only kept as a sanity check; for proper evaluation run main analysis
'PRIVCON' -> private consumption, Private Konsumausgaben 
'PUBCON' -> public consumption, Öffentlicher Konsum
'CONSTR' -> construction, Bauten 
'EQUIPMENT' -> Ausrüstungen
'OPA' -> other product assets, Sonstige Anlagen 
'INVINV' -> investment inventories, Vorratsinvestitionen
'DOMUSE'-> domestic use, inländische Verwendung
'TRDBAL' -> trade balance, Außenbeitrag 
'EXPORT'
'IMPORT'
"""


# Decide whether to run filtered timeframe analysis for components (e.g., 2010-Q1 to 2017-Q1) as 
# well as outlier filtering; Set to False for faster execution
run_component_filter = False # True or False







# ==================================================================================================
#                                     EVALUATION SETTINGS
# ==================================================================================================

## Decide how many quarters to plot in the Quarterly Evaluation Module; int, should be more than 
QoQ_eval_n_bars = 7  # int in [1,10]; Remember 0-indexing: n_bars = 5 plots Q0-Q4

## Define whether to subset the naive forecaster series s.t. only forecasts available  in ifo's 
# main QoQ Forecasts are evaluated
match_ifo_naive_dates = True  # True or False; RECOMMENDATION:True



# --------------------------------------------------------------------------------------------------
#                                     OUTLIER SETTINGS
# --------------------------------------------------------------------------------------------------

# Drop Outliers: Decide to drop forecast errors exceeding a certain threshhold
drop_outliers = True  # True or False

# Set the Thresshold which drops errors exceeding sd_threshold * sd(col)
sd_threshold = 2.5  # float, e.g. 0.05 or 5; runs if drop_outliers = True



## Select an evaluation intervall, e.g. 2010-Q1 - 2019-Q4:
"""
NOTE: 
- Does not apply to YoY evaluations, which are always run for the full intervall
- This is not called in the ifoCAST evaluation, but could be changed if needed
- Full evaluation is always run as well
"""

# Define the horizon of first releases which should be evaluated from below:
first_release_lower_limit_year = 2022           # Forecasts start in 2007; set as integer
first_release_lower_limit_quarter = 1            # 1,2,3 or 4; set as integer

# Define the horizon of first releases which should be evaluated from below:
first_release_upper_limit_year = 2100          # Forecasts start in 2007; set as integer
first_release_upper_limit_quarter = 1            # 1,2,3 or 4; set as integer

# Decide whether to filter for outliers within the evaluation intervall
filter_outliers_within_eval_intervall = False  # True or False



## ifoCAST evaluation settings

# Decide whether to ADDITIONALLY evaluate the ifoCAST starting in Q1-2021
run_ifocast_2021_subset = True  # True or False; only runs if run_ifoCAST_evaluation = True

# Decide whether to ADDITIONALLY evaluate the ifoCAST starting in Q1-2022
run_ifocast_2022_subset = True  # True or False; only runs if run_ifoCAST_evaluation = True









# ==================================================================================================
#                          SYSTEM SETTINGS: Which Scripts should be run?
# ==================================================================================================
"""
This code is compartementalized, which means that every module can in principle be run autonomously.
This might make sense to save some time on execution, but be weary that the input data generated by 
previous iterations of earlier modules is what you want to analyze
"""

# 1.:  Decide whether or not to run automatic package installation
install_packages = False  # True or False

# 1_1.: Decide whether to run the GDP-Component preprocessing pipeline
run_component_preprocessing = True  # True or False; needed when running the full component evaluation

# 2.: Decide whether you want to re-run the data processing
run_data_processing = True  # True or False

# 3.: Decide whether to re-run the Naive Forecaster
run_naive_forecaster = True  # True or False

# 4.: Decide whether to run the quarterly evaluation output module
run_quarterly_evaluation = True  # True or False

# 5.: Decide whether to run the ifoCAST evaluation module
run_ifoCAST_evaluation = True  # True or False

# 6.: Decide whether to run the ifoCAST long-term evaluation module
run_ifoCAST_long_term_evaluation = True  # True or False

# 7. Decide whether to run the forecat enhancement module
run_forecast_enhancement_module = True  # True or False, requieres data from module 6

# 8. Run Judgemental Forecasting Analysis Module
run_judgemental_forecasting_analysis = True  # True or False




## Decide whether to overwrite previous output (always overwriten if same functionality is executed)
clear_result_folders = True   # True or False

## Decide whether to clear the result folders on a larger level: (clears folrders 0_1, 1 and 2)
macro_clear = True  # True or False




# ==================================================================================================
#                                    DATA SETTINGS - GDP and GVA
# ==================================================================================================

# Decide wether to use Real time (True) or a local version of the Bundesbank data (False)
api_pull = True # True or False; only set False if no internet connection

# Decide wether to extend the available real-time data by using the earliest available data release 
# Q2-1995 and imputing it backwards to Q1-1989 (True)
extend_rt_data_backwards = True # True or False

# Decide whether to run the entire exercise on Gross Value Added as well
run_gva_evaluation = True  # True or False

# Define which quarter of which year should be the earliest data point, available from Q1-1970 onwards
horizon_limit_year = 1970            # from 1970 onwards; set as integer
horizon_limit_quarter = 1            # 1,2,3 or 4; set as integer

















# ==================================================================================================
#                                   NAIVE FORECAST SETTINGS
# ==================================================================================================


# --------------------------------------------------------------------------------------------------
#                                       Define the model
# --------------------------------------------------------------------------------------------------

# Set the agent's forecasting method; options: 'AR', 'GLIDING_AVERAGE', 'AVERAGE' - where 'AVERAGE' is an SMA
models = ['AR', 'AVERAGE']

# For AR model: set number of lags (sugested: 2); list of int
AR_orders = [2]

# For AR model: set the memory of the agent (timeframe the model is estimated on); list of int or 'FULL'
AR_horizons = [50]
"""
Note:
-> unstable estimates for 2020_Q3 (release date, last observation 2020_Q2) for below 48
-> this parameter does not conduct a degree of freedom correction, i.e. there are AR_horizon - AR_order 
   free observations available.
"""

# For AR model: set the minimum number of observations required to estimate the AR model; int, should be more than AR_order
min_AR_observations = 20

# For average-based models: set time frame over which the agent averages in quarters; list of int or 'FULL'
average_horizons = [1,10]


# Set how far the agent predicts into the future; int, NO LIST ITERATION
forecast_horizon = 9  # int in [0,9]

# Note on forecast_horizon:
"""
For a data release at Time T, GDP data is available up to T-1. Accordingly, forecasting t 
periods into the future requires t+1 forecasts, including the so-called nowcast for the 
present. 
When setting the above parameter, this nowcast should not be counted. Set how many FUTURE periods 
are estimated. 
-> Setting forecast_horizon = t thus outputs t+1 prediction values.
"""


# --------------------------------------------------------------------------------------------------
#                                Format the naive forecaster output
# --------------------------------------------------------------------------------------------------

# For the Naive Forecaster, customize the results folder name; 
# set 'Default' for default: 'Results_model_memory_forecast'
resultfolder_name_n_forecast = 'Default'   

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



