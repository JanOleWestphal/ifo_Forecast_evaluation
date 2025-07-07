
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
# Title:        Install Packages
#
# Description:  Installs the required packages for the project. This script is executed at the 
#               beginning of the project and can be switched off
# ==================================================================================================
# --------------------------------------------------------------------------------------------------


# Import built-ins
import importlib
import subprocess
import sys


# Check for libraries
required_packages = [
    'requests',
    'openpyxl',
    'pandas',
    'pandasgui',
    'numpy',
    'statsmodels',
    'matplotlib',
    'seaborn'
]

# Install if needed
for pkg in required_packages:
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        print(f" {pkg} installed.")

print("All required packages are installed. \n")