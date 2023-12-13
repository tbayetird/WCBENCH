## check that the files have been generated thanks to the script/script_generate_info_inter_XXX.py
## Make sure the MATCH_CLS param is the good one ! check the configuration file.

import sys,os
import numpy as np
import pandas as pd

"""
This script computes the benchmark scores and writes them out.
It needs valid configurations with the rights paths.
If it fails, check the paths from the config files.
"""


#This might be called as a module, so we need to set the path here
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from utils.dataset.dataset import Dataset
from utils.evaluation.evaluation_NL import Eval_NL
from utils.reporting.outputs_writer import OutputsWriter
from config import test_config as config
from config import google_fuzzy_config,google_intersection_config,amazon_fuzzy_config,amazon_intersection_config,microsoft_fuzzy_config,microsoft_intersection_config,gyfcc_intersection_config,gyfcc_fuzzy_config,empty_intersection_config,CSRA_cocowu
from config import google_three_classes,amazon_three_classes,microsoft_three_classes
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
Get all the configs for computing the benchmark on three classes
"""
def get_config_three_classes():
    confs = [google_three_classes,amazon_three_classes,microsoft_three_classes]
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
    output_df = eval.compute_evaluation(df_prediction)
    # output_df = ow.prepare_output_df(df_info,df_prediction, df_eval)
    df_prediction['HL']=output_df['HL']
    df_prediction['ACC']=output_df['ACC']
    df_prediction['ACC5']=output_df['ACC5']
    df_prediction['F1']=output_df['F1']

    ## benchmark
    results_HL, results_ACC, results_ACC5, results_F1 = eval.compute_benchmark_scores(df_prediction)
    results_HL_empty, results_ACC_empty , results_ACC5_empty, results_F1_empty= eval.compute_non_empty_benchmark_scores(df_prediction)
    ow.write_report(df_prediction,results_HL + results_HL_empty,results_ACC + results_ACC_empty, results_ACC5 + results_ACC5_empty, results_F1 + results_F1_empty)

"""
Run all the configuration for a list of model names
*model_names : list of models we want to run our benchmark on
"""
def run_all(model_names):
    for model_name in model_names:
        confs = get_configs(model_name)
        for conf in confs:
            run_conf(conf)

"""
Run the configuration for the benchmark on three classes.
"""
def run_three_classes():
    confs = get_config_three_classes()
    for conf in confs :
        run_conf(conf)

## Run script
run_all(model_names)
# run_conf(CSRA_cocowu)
# run_conf(gyfcc_fuzzy_config)
# run_conf(gyfcc_intersection_config)
# run_three_classes()
