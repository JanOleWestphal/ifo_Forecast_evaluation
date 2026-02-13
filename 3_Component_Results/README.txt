The subfolders in this folder are created rhtough 4_Quarterly_Evaluation.py, with relevant preprocessing Happening in all other previous scripts.
To update a subfolder, select it in the settings-file, save the updated settings-file and run the main file.

To evaluate a different time horizon, go to the Settings file and change these Parameters:

# Define the horizon of first releases which should be evaluated from below:
first_release_lower_limit_year = 2022           # from 1989 onwards; set as integer
first_release_lower_limit_quarter = 1            # 1,2,3 or 4; set as integer

# Define the horizon of first releases which should be evaluated from below:
first_release_upper_limit_year = 2100          # from 1989 onwards; set as integer
first_release_upper_limit_quarter = 1            # 1,2,3 or 4; set as integer


In Resultfolders, graph Name indicate their subsample: 
	-> filename_sd_filtered means full available sample filtered for outliers as set in the Settings file. 
	-> filename_yyyyq_yyyyq indicates results for a subsample from yyyy-Qq to yyyy-Qq, where q stands for quarter
	-> if the core filename has no Suffix (e.g. ifo_sc, jnt_sc), it's the full, unfiltered sample


Each subfolder contains the following:
-> Data Folder: 
	-> includes the raw error series of ifo forecasts, naive forecasts and subsetted error series, but not outlier-corrected ones.
-> Graphs Folder: 
	-> Includes error time series for ifo's main forecasts and for ifo and the naive forecasts jointly (in subfolders), 
	-> error scatter plots and
	-> error bar plots for different samples and measures
-> Tables Folder: 
	-> Data source for the error bar plots, hold error measures for different samples and objects



The subfolder names correspond to the following GDP demand-side components:
'GDP' -> this folder's naive forecsats are built on incompletely revisioned real-time data, its only kept as a sanity check; for proper evaluation run main analysis
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