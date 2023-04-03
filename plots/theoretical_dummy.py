"""
This file will take a set of GPU profiles
as input, a set of GPU kernels/features to
perturb, and a set of GPU kernels/features
on which to train an architecture prediction
model, then will plot the architecture
prediction accuracy as various levels of noise
are added to the perturbed features.

**Note - we train the architecture prediction 
models on clean profiles (no noise added) and 
then evaluate the accuracy on profiles with 
noise added. 
"""
from pathlib import Path
import sys
import json
from typing import List
import datetime

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

# setting path
sys.path.append("../edge_profile")

from arch_pred_accuracy import getDF
from data_engineering import (
    filter_cols,
)
from architecture_prediction import (
    get_arch_pred_model,
    arch_model_names,
)

rc('font',**{'family':'serif','serif':['Times'], 'size': 14})
rc('figure', **{'figsize': (5, 4)})

REPORT_FOLDER = Path(__file__).parent.absolute() / "theoretical_dummy"
REPORT_FOLDER.mkdir(exist_ok=True)
plot_folder = REPORT_FOLDER / "plots"
plot_folder.mkdir(exist_ok=True)

def generateReport(
        profile_df: pd.DataFrame, 
        noise_features: List[str], 
        num_experiments: int, 
        noise_levels: List[int], 
        arch_model_names: List[str]
    ):
    report = {
        "features": list(profile_df.columns), 
        "noise_features": noise_features, 
        "num_experiments": num_experiments, 
        "noise_levels": noise_levels,
        "arch_model_names": arch_model_names,
        "results": {},
    }
    dataset_size = len(profile_df)
    
    for noise_level in noise_levels:
        print(f"Generating results for noise level {noise_level}...")
        noise_config = {model_name: [] for model_name in arch_model_names}
        for experiment in range(num_experiments):
            df_experiment = df.copy(deep=True)
            for noise_feature in noise_features:
                # replace feature column with noise added
                noise_to_add = np.random.uniform(0, noise_level*df_experiment[noise_feature].std(), size=dataset_size)
                df_experiment[noise_feature] += noise_to_add
            for model_name in arch_model_names:
                model = get_arch_pred_model(model_name, df=df)
                noise_config[model_name].append(model.evaluateAcc(df_experiment))
        report["results"][noise_level] = noise_config

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = REPORT_FOLDER / f"{time}.json"
    with open(save_path, 'w+') as f:
        json.dump(report, f, indent=4)


def plotFromReport(report_path: Path, arch_model_names: List[str]):
    with open(report_path, "r+") as f:
        report = json.load(f)
    
    for model_name in arch_model_names:
        model_avgs = []
        model_stds = []
        for noise_lvl in report["noise_levels"]:
            model_results = report["results"][str(noise_lvl)][model_name]
            model_avgs.append(sum(model_results)/len(model_results))
            model_stds.append(np.std(model_results))

        plt.plot(report["noise_levels"], model_avgs, label=model_name)
        minus_std = []
        plus_std = []
        for mean, std in zip(model_avgs, model_stds):
            minus_std.append(mean - std)
            plus_std.append(mean + std)
        plt.fill_between(
            report["noise_levels"],
            minus_std,
            plus_std,
            alpha=0.2,
        )
    #plt.rcParams["figure.figsize"] = (10, 10)
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.xlabel("Range of Uniform Distribution Used to Add Noise\n(as a multiple of feature STD)")

    plt.xticks(report["noise_levels"])

    plt.ylabel(f"Architecture Prediction Accuracy")

    plt.title(
        f"Architecture Prediction Accuracy\nby Range of Uniform Distribution Used to Add Noise to Profiles"
    )

    plt.savefig(REPORT_FOLDER / "plots" / report_path.name.replace(".json", ".png"), dpi=500, bbox_inches="tight")
    



if __name__ == '__main__':

    GPU_PROFILES_PATH = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe_pretrained"
    FEATURE_RANK_PATH = Path.cwd() / "plots" / "feature_ranks" / "rf_gpu_kernels_nomem.json"
    NUM_FEATURES = 3 # num features to train architecture prediction models
    NOISE_FEATURES = 3  # number of features to add noise to

    NOISE_LEVELS = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    NUM_EXPERIMENTS = 10
    ARCH_MODEL_NAMES = ["rf", "knn"]   # arch_model_names()

    GENERATE_REPORT = False

    PLOT_FILENAME = REPORT_FOLDER / f"20230320-133823.json"
    PLOT = True


    #---------------------------------------------------
    if GENERATE_REPORT:
        assert NOISE_FEATURES <= NUM_FEATURES

        with open(FEATURE_RANK_PATH, "r") as f:
            report = json.load(f)["feature_rank"]
        features = report[:NUM_FEATURES]
        noise_features = report[:NOISE_FEATURES]

        df = getDF(GPU_PROFILES_PATH)
        df = filter_cols(df, substrs=features)

        generateReport(
            profile_df=df, 
            noise_features=noise_features,
            num_experiments=NUM_EXPERIMENTS,
            noise_levels=NOISE_LEVELS,
            arch_model_names=ARCH_MODEL_NAMES,
        )
    
    if PLOT:
        plotFromReport(report_path=PLOT_FILENAME, arch_model_names=ARCH_MODEL_NAMES)