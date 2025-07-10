

# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        ifo Forecast Evaluation
#
# Author:       Jan Ole Westphal, building on previous work by Timo Wollmershäuser and others
# Date:         2025-05
#
# Description:  Execute this file to conduct the ifo Forecast Evaluation.
#               Set all settings in the Settings file. 
#               
#               Calls all submodules in the following order:
#
#               Relies on a setup file buried in the Functionalities folder and imported in all
#               other files.
# ==================================================================================================
# --------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        Code begins here                                          #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

print('\n \nINNITIATING THE ifo FORECAST EVALUATION ... \n')

# --------------------------------------------------------------------------------------------------
#                                               Setup
# --------------------------------------------------------------------------------------------------

import sys
import os
import subprocess

# Automatically set the workfolder to the location of this script
workfolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(workfolder)

# Define Script Execution Function:
def execute_script(script_name):
    """
    Execute a Python script from the Functionalities directory as a module.
    """

    # Remove .py extension if present
    module_name = script_name.replace('.py', '')
    module_path = f"Functionalities.{module_name}"

    print(f"\nExecuting module: {module_path}\n")

    try:
        result = subprocess.run(
            [sys.executable, "-m", module_path],
            cwd=workfolder,  # workfolder should be your project root
            # capture_output=True,
            text=True,
            check=True  # Raise CalledProcessError if non-zero exit
        )

        print(f"Executed {script_name} successfully.\n \n")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to execute {script_name} (exit code {e.returncode})")
        print("----- STDERR -----")
        print(e.stderr.strip())
        print("------------------\n")

    except Exception as e:
        print(f"\n❌ Unexpected error during execution of {script_name}: {e}\n")

    # Debug output if needed
    """
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    """

    # Check for errors
    """
    try:
        result.check_returncode()  # Raises CalledProcessError if exit code != 0
    except subprocess.CalledProcessError as e:
        print(f"Error: Module {module_path} exited with code {e.returncode}")
        raise
    """





# ==================================================================================================
# Load Settings
# ==================================================================================================

import ifo_forecast_evaluation_settings as settings


# ==================================================================================================
# Install Packages
# ==================================================================================================
if settings.install_packages:
    execute_script('1_Install_Packages.py')


# ==================================================================================================
# DATA PROCESSING
# ==================================================================================================
if settings.run_data_processing:
    execute_script('2_Data_Processing.py')
else: 
    print("Warning: No data processing, might use depricated GDP-data.\n" \
    "Set run_data_processing = True for real-time data")


# ==================================================================================================
# NAIVE FORECAST GENERATION
# ==================================================================================================
if settings.run_naive_forecaster:
    execute_script('3_Naive_Forecaster.py')
else:
    print("Warning: Naive Forecaster deactivated, make sure the evaluated forecasts are what you desire\n" \
    "Set run_naive_forecaster = True to execute the current naive forecast specification \n")


# ==================================================================================================
# QUARTERLY EVALUATION and OUTPUT
# ==================================================================================================
if settings.run_quarterly_evaluation:
    execute_script('4_Quarterly_Evaluation.py')
else: 
    print("Warning: No new evalaution output has been generated, set run_evaluation if desired. \n ")


# ==================================================================================================
# PERFORMANCE EVALUATION and OUTPUT
# ==================================================================================================
"""
if settings.run_evaluation:
    execute_script('x_Evaluation_and_Output_old.py')
else: 
    print("Warning: No new evalaution output has been generated, set run_evaluation if desired. \n ")
"""

print('\nifo Forecast Evaluation completed successfully! \n')


# -------------------------------------------------------------------------------------------------#
# =================================================================================================#
#                                        End of Code                                               #
# =================================================================================================#
# -------------------------------------------------------------------------------------------------#

