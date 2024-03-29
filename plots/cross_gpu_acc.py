"""
Generate a table with 2 columns:

1) accuracy of arch pred models trained on 
    Quadro RTX 8000 GPU on victim model profiles
    from Tesla T4 GPU

2) accuracy of arch pred models trained on 
    Tesla T4 GPU on victim model profiles
    from Quadro RTX 8000 GPU

The rows are the arch pred models.
The features used and dataset size are configurable.
"""

import datetime
import json
from pathlib import Path
from typing import List
import pandas as pd

import sys

# plt.style.use('ggplot')

# setting path
sys.path.append("../edge_profile")

from data_engineering import (
    filter_cols,
    all_data,
    remove_cols,
    removeColumnsFromOther,
)
from architecture_prediction import (
    arch_model_full_name,
    get_arch_pred_model,
    ArchPredBase,
    arch_model_names,
)
from experiments import predictVictimArchs
from model_manager import VictimModelManager

SAVE_FOLDER = Path(__file__).parent.absolute() / "cross_gpu_acc"
if not SAVE_FOLDER.exists():
    SAVE_FOLDER.mkdir(exist_ok=True)

QUADRO_TRAIN = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
QUADRO_TEST = Path.cwd() / "victim_profiles"
TESLA_TRAIN = Path.cwd() / "profiles" / "tesla_t4" / "colab_zero_exe_pretrained"
TESLA_TEST = Path.cwd() / "victim_profiles_tesla"

def loadFeatureRank(filename: str):
    if not filename.endswith(".json"):
        filename += ".json"
    report_path = Path(__file__).parent.absolute() / "feature_ranks" / filename
    with open(report_path, "r") as f:
        report = json.load(f)
    return report["feature_rank"]


def getDF(path: Path, to_keep_path: Path = None, df_args: dict = {}):
    df = all_data(path, **df_args)
    if to_keep_path is not None:
        keep_df = all_data(to_keep_path, **df_args)
        # remove cols of df if they aren't in keep_df
        df = removeColumnsFromOther(keep_df, df)
    return df

def getTestAcc(model: ArchPredBase, gpu_type: str, verbose: bool = False):
    """NOT USING THIS, INSTEAD LOAD THE PROFILES INTO A FOLDER 
    USING loadProfilesToFolder() from model_manager.py
    """
    vict_model_paths = VictimModelManager.getModelPaths()
    for vict_path in vict_model_paths:
        print(f"Getting profiles for {vict_path.parent.name}...")
        manager = VictimModelManager.load(vict_path)
        

def generateCrossGPUReport(quadro_train: pd.DataFrame, tesla_train: pd.DataFrame, config: dict, model_names: List[str], topk: List[int] = [1, 5], train_size: int = None):
    #TODO do we want number of experiments as measure of centrality?
    columns = ["Architecture Prediction Model Type", "Train Quadro RTX 8000, Test Tesla", "Train Quadro RTX 8000, Test Tesla Family", "Train Tesla, Test Quadro RTX 8000", "Train Tesla, Test Quadro RTX 8000 Family"]

    for k in topk:
        columns.append(f"Train Quadro RTX 8000, Test Tesla Top {k}")
        columns.append(f"Train Tesla, Test Quadro RTX 8000 Top {k}")

    table = pd.DataFrame(columns=columns)

    for model_type in model_names:
        row_data = {"Architecture Prediction Model Type": arch_model_full_name()[model_type]}
        # train on quadro
        model = get_arch_pred_model(model_type=model_type, df=quadro_train, kwargs={"train_size": train_size})

        # test on tesla
        k_acc_report = predictVictimArchs(model, TESLA_TEST, save=False, topk=max(topk), verbose=False)
        k_acc = k_acc_report["accuracy_k"]

        row_data["Train Quadro RTX 8000, Test Tesla Family"] = k_acc_report["family_accuracy"]
        
        topk_str = ""
        for k in topk:
            row_data[f"Train Quadro RTX 8000, Test Tesla Top {k}"] = k_acc[k] * 100
            topk_str += "{:.3g}/".format(k_acc[k] * 100)
        
        row_data["Train Quadro RTX 8000, Test Tesla"] = topk_str[:-1]

        #train on tesla
        model = get_arch_pred_model(model_type=model_type, df=tesla_train, kwargs={"train_size": train_size})

        # test on quadro
        k_acc_report = predictVictimArchs(model, QUADRO_TEST, save=False, topk=max(topk), verbose=False)
        k_acc = k_acc_report["accuracy_k"]
        
        row_data["Train Tesla, Test Quadro RTX 8000 Family"] = k_acc_report["family_accuracy"]

        topk_str = ""
        for k in topk:
            row_data[f"Train Tesla, Test Quadro RTX 8000 Top {k}"] = k_acc[k] * 100
            topk_str += "{:.3g}/".format(k_acc[k] * 100)
        
        row_data["Train Tesla, Test Quadro RTX 8000"] = topk_str[:-1]

        table = table.append(row_data, ignore_index=True)

    # save table and config
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    table.to_csv(SAVE_FOLDER / f"{time}.csv")

    config["topk"] = topk
    config["feature_analysis"] = featureAnalysis(quadro_train, tesla_train)
    with open(SAVE_FOLDER / f"{time}.json", 'w') as f:
        json.dump(config, f, indent=4)


def featureAnalysis(quadro_train: pd.DataFrame, tesla_train: pd.DataFrame):
    quadro_features = set(list(quadro_train.columns))
    tesla_features = set(list(tesla_train.columns))

    shared_cols = quadro_features.intersection(tesla_features)

    quadro_unique = quadro_features - tesla_features
    tesla_unique = tesla_features - quadro_features

    print("\n\nNote: these feature counts include the 3 label features")
    print("model, model_family, and file")
    print(f"Quadro:          {len(quadro_features)} features, {len(quadro_unique)} unique.")
    for feature in quadro_unique:
        print(f"\t{feature}")
    print(f"Tesla:           {len(tesla_features)} features, {len(tesla_unique)} unique.")
    for feature in tesla_unique:
        print(f"\t{feature}")
    print(f"Shared Features: {len(shared_cols)}")

    report = {
        "quadro": {
            "num_features": len(quadro_features),
            "num_unique_features": len(quadro_unique),
            "unique_features": list(quadro_unique),
            "features": list(quadro_features),
        },
        "tesla": {
            "num_features": len(tesla_features),
            "num_unique_features": len(tesla_unique),
            "unique_features": list(tesla_unique),
            "features": list(tesla_features),
        },
        "num_shared_features": len(shared_cols),
        "shared_features": list(shared_cols),
    }
    return report




if __name__ == '__main__':

    model_names = arch_model_names()  # all the models we want to use

    topk = [1, 5]

    # if not None, can be an int representing how many profiles to
    # keep per architecture in the training data
    train_size = None

    # if None, use all features. Else, this is a name of a feature ranking under
    # feature_ranks/
    feature_ranking_file = "rf_gpu_kernels_nomem.json"
    feature_num = 25    # the number of features to use

    # args to pass to load training data, if feature rank file is provided,
    # then this should be an empty dict
    df_args = {}    #{"no_system_data": True, "gpu_activities_only": True}

    # substrings to remove from the dataframe, if feature rank file is provided,
    # then this should be empty
    df_remove_substrs = []

    # ----------------------------------------------------------------------
    if feature_ranking_file is not None:
        assert len(df_args) == 0
        assert len(df_remove_substrs) == 0

    quadro_train = getDF(QUADRO_TRAIN, df_args=df_args)
    tesla_train = getDF(TESLA_TRAIN, df_args=df_args)
    
    if feature_ranking_file is not None:
        feature_ranking = loadFeatureRank(feature_ranking_file)
        relevant_features = feature_ranking[:feature_num]

        quadro_train = filter_cols(quadro_train, substrs=relevant_features)
        tesla_train = filter_cols(tesla_train, substrs=relevant_features)
    
    quadro_train = remove_cols(quadro_train, substrs=df_remove_substrs)
    tesla_train = remove_cols(tesla_train, df_remove_substrs)
    
    config = {
        "train_size": train_size,
        "feature_ranking_file": feature_ranking_file,
        "feature_num": feature_num,
        "df_args": df_args,
        "df_remove_substrs": df_remove_substrs,
        "model_names": model_names,
    }

    generateCrossGPUReport(quadro_train=quadro_train, tesla_train=tesla_train, config=config, model_names=model_names, topk=topk, train_size=train_size)


    


