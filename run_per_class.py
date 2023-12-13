## check that the files have been generated thanks to the script/script_generate_info_inter_XXX.py
## Make sure the MATCH_CLS param is the good one ! check the configuration file.

import sys,os
import numpy as np
import pandas as pd

"""
This script computes the per class benchmark scores and writes them out.
It needs valid configurations with the rights paths.
If it fails, check the paths from the config files.
"""

#This might be called as a module, so we need to set the path here
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from utils.dataset.dataset import Dataset
from utils.evaluation.evaluation_NL import Eval_NL
from utils.reporting.outputs_writer import OutputsWriter
# from config import CSRA_COCOWU_per_class_config as conf
from config import google_fuzzy_config,google_intersection_config,amazon_fuzzy_config,amazon_intersection_config,microsoft_fuzzy_config,microsoft_intersection_config,gyfcc_intersection_config,gyfcc_fuzzy_config,empty_intersection_config,CSRA_cocowu
from config import CSRA_DSD_per_class_config as conf

## Handling params

model_names=["Google","Amazon","Microsoft"]

## Script functions
"""
Get all the configs for the different dataset for a single model
*model_name : the name of the model that will be used
"""
def get_configs(model_name):
    if model_name=="Google":
        confs=[google_fuzzy_config,google_intersection_config]
    if model_name=="Amazon":
        confs=[amazon_fuzzy_config,amazon_intersection_config]
    if model_name=="Microsoft":
        confs=[microsoft_fuzzy_config,microsoft_intersection_config]
    return confs

"""
Run the configuration given in a parameter
*conf : configuration for a model - dataset combo
"""
def run_conf(conf):
    eval = Eval_NL(conf)
    ow = OutputsWriter(conf)
    ##prepare data
    df_prediction = pd.read_csv(conf.PREDICTION_PATH)

    # get evaluations
    # benchmark
    classes_occ,classes_not_occ,classes_all,results_ACC= eval.compute_per_class_benchmark_scores(df_prediction)
    classes_occ_empty,classes_not_occ_empty,classes_all_empty,results_ACC_empty =eval.compute_per_class_non_empty_benchmark_scores(df_prediction)

    classes_all['occ_count']=classes_occ['count']
    classes_all['occ_ACC']=classes_occ['ACC']
    classes_all['not_occ_count']=classes_not_occ['count']
    classes_all['not_occ_ACC']=classes_not_occ['ACC']

    classes_all_empty['occ_count']=classes_occ_empty['count']
    classes_all_empty['occ_ACC']=classes_occ_empty['ACC']
    classes_all_empty['not_occ_count']=classes_not_occ_empty['count']
    classes_all_empty['not_occ_ACC']=classes_not_occ_empty['ACC']
    classes_all.to_csv(conf.CLASS_CSV_PATH)
    ow.write_per_class_report(results_ACC + results_ACC_empty)

"""
Run all the configuration for a list of model names
*model_names : list of models we want to run our benchmark on 
"""
def run_all(model_names):
    for model_name in model_names:
        confs = get_configs(model_name)
        for conf in confs:
            run_conf(conf)

## Run script

run_all(model_names)
# run_conf(conf)
