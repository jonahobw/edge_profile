"""
Generate a table (csv format/ pandas dataframe) where the columns are the 
train/val acc1 and acc5 by the subset of data used to train the architecture
prediction model (with the count of # of features), and the rows are the type 
of architecture prediction model.
"""

import json
from pathlib import Path
import shutil
from typing import Dict, List, Union
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
import matplotlib.pyplot as plt
from sklearn.metrics import top_k_accuracy_score

# plt.style.use('ggplot')

# setting path
sys.path.append("../edge_profile")

from data_engineering import (
    filter_cols,
    shared_data,
    get_data_and_labels,
    all_data,
    add_indicator_cols_to_input,
    remove_cols,
    removeColumnsFromOther,
)
from format_profiles import parse_one_profile
from architecture_prediction import (
    get_arch_pred_model,
    ArchPredBase,
    RFArchPred,
    arch_model_names,
    arch_model_full_name
)
from model_manager import predictVictimArchs
from config import SYSTEM_SIGNALS
from utils import latest_file

PROFILE_FOLDER = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
SAVE_FOLDER = Path(__file__).parent.absolute() / "arch_pred_acc_by_data_subset"
QUADRO_VICT_PROFILE_FOLDER = Path.cwd() / "victim_profiles"

def loadReport(filename: str):
    report_path = Path(__file__).parent.absolute() / filename
    with open(report_path, "r") as f:
        report = json.load(f)
    return report


def generateTable(data_subsets: Dict[str, pd.DataFrame], victim_profile_folder: Path, model_names: List[str] = None, topk=[1], save_name: str = None):

    if save_name is None:
        save_name = "arch_pred_acc_by_data_subset.csv"
    save_path = SAVE_FOLDER / save_name

    if model_names is None:
        model_names = arch_model_names()  # all the models we want to use

    columns = ["Architecture Prediction Model Type"]
    
    num_columns = {}
    
    for data_subset in data_subsets:
        num_columns[data_subset] = len(list(data_subsets[data_subset].columns)) - 3
        for k in topk:
            columns.append(f"{data_subset} ({num_columns[data_subset]}) Top {k}")    # redundancy is for formatting
            columns.append(f"{data_subset} ({num_columns[data_subset]}) Top {k} Train")
            columns.append(f"{data_subset} ({num_columns[data_subset]}) Top {k} Val")
            columns.append(f"{data_subset} ({num_columns[data_subset]}) Top {k} Test")

    table = pd.DataFrame(columns=columns)

    for model_type in model_names:
        row_data = {"Architecture Prediction Model Type": arch_model_full_name()[model_type]}
        for data_subset in data_subsets:
            print(f"Training {model_type} on {data_subset}")
            model = get_arch_pred_model(model_type=model_type, df=data_subsets[data_subset])
            for k in topk:
                model_pred_fn = None
                if hasattr(model.model, "decision_function"):
                    model_pred_fn = model.model.decision_function
                elif hasattr(model.model, "predict_proba"):
                    model_pred_fn = model.model.predict_proba
                
                test = predictVictimArchs(model, victim_profile_folder, save=False, topk=1, verbose=False)["accuracy_k"][1]

                if model_pred_fn is not None:
                    train = top_k_accuracy_score(model.y_train, model_pred_fn(model.x_tr), k=k)
                    val = top_k_accuracy_score(model.y_test, model_pred_fn(model.x_test), k=k)
                else:
                    train = np.nan
                    val = np.nan
                    if k == 1:
                        train = model.evaluateTrain()
                        val = model.evaluateTest()

                row_data[f"{data_subset} ({num_columns[data_subset]}) Top {k}"] = "{:.3g}/{:.3g}".format(train * 100, val * 100)
                row_data[f"{data_subset} ({num_columns[data_subset]}) Top {k} Train"] = train * 100
                row_data[f"{data_subset} ({num_columns[data_subset]}) Top {k} Val"] = val * 100
                row_data[f"{data_subset} ({num_columns[data_subset]}) Top {k} Test"] = test * 100

        table = table.append(row_data, ignore_index=True)
    
    table.to_csv(save_path)
    transpose_path = SAVE_FOLDER / f"{save_path.name[:-4]}_transpose.csv"
    table.T.to_csv(transpose_path)

    printNumFeatures(data_subsets)


def semanticSubsets():
    data_subsets = {
        "All": all_data(PROFILE_FOLDER),
        "System" : all_data(PROFILE_FOLDER, system_data_only=True),
        "No System" : all_data(PROFILE_FOLDER, no_system_data=True),
        "GPU Kernel" : all_data(PROFILE_FOLDER, gpu_activities_only=True, no_system_data=True),
        "API Calls" : all_data(PROFILE_FOLDER, api_calls_only=True, no_system_data=True),
        "Indicator": all_data(PROFILE_FOLDER, indicators_only=True),
        "No Indicator": remove_cols(all_data(PROFILE_FOLDER), substrs=["indicator"]),
        "GPU Kernel, No Memory": remove_cols(all_data(PROFILE_FOLDER, gpu_activities_only=True, no_system_data=True), substrs=["mem"]),
        "GPU Kernel, Memory Only": filter_cols(all_data(PROFILE_FOLDER, gpu_activities_only=True, no_system_data=True), substrs=["mem"]),
    }
    return data_subsets


def topFeatureSubsets(feature_rank_model: str = "lr", num_features = [5]):
    feature_rank = loadReport(f"feature_rank_{feature_rank_model}.json")["feature_rank"]

    data_subsets = {}
    for feature_count in num_features:
        new_features = feature_rank[:feature_count]
        data_subsets[f"Top {feature_count} Features"] = filter_cols(all_data(PROFILE_FOLDER), substrs=new_features)
    return data_subsets


def createTable():
    subsets = semanticSubsets()
    subsets.update(topFeatureSubsets())
    generateTable(subsets, victim_profile_folder=QUADRO_VICT_PROFILE_FOLDER)

def printNumFeatures(subsets):
    for subset in subsets:
        print(f"{subset.ljust(30)}: {len(list(subsets[subset].columns)) - 3} features")

def small():
    # use for small experiments
    subsets = semanticSubsets()
    subsets =  {"GPU Kernel, No Memory": subsets["GPU Kernel, No Memory"]}

    generateTable(subsets, victim_profile_folder=QUADRO_VICT_PROFILE_FOLDER, save_name="gpu_nomem.csv")

if __name__ == '__main__':
    #createTable()
    small()
    exit(0)

    



