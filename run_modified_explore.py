## check that the files have been generated thanks to the script/script_generate_info_inter_XXX.py
## Make sure the MATCH_CLS param is the good one ! check the configuration file.

import sys,os
import numpy as np
import pandas as pd

"""
This runs a modified version of the explorer that was usefull to compute some
usefull metrics. 
"""

#This might be called as a module, so we need to set the path here
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from utils.exploring.modified_explorer import Explorer
from config import CSRA_DSD_explore as conf
# from config import CSRA_COCOWU_explore as conf
# from config import test_explore as conf

explorer = Explorer(conf)
explorer.run()
