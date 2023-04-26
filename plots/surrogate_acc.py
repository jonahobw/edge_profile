"""
Plots (surrogate model accuracy) / (victim model accuracy)
for each surrogate victim pair.

They should be tested on the same dataset and should be trained
on different subsets of the same dataset. Note that models
trained on different subsets also have different validation
subsets so this will need to be held constant.
"""

import datetime
from pathlib import Path
from typing import List, Tuple
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use('ggplot')

rc('font',**{'family':'serif','serif':['Times'], 'size': 14})
rc('figure', **{'figsize': (5, 4)})

import sys

# setting path
sys.path.append("../edge_profile")

from model_manager import VictimModelManager, SurrogateModelManager, getVictimSurrogateModels
from config import MODELS

SAVE_FOLDER = Path(__file__).parent.absolute() / "surrogate_acc"
SAVE_FOLDER.mkdir(exist_ok=True)

def getVictimSurrogateAccs(models: List[Tuple[Path, Path]], dataset) -> Tuple[List[str], List[float], List[float]]:
    acc_config_name = f"entire_{dataset}_val_acc"

    labels = []
    victim_accs = []
    surrogate_accs = []

    for _, surrogate_path in models:
        surrogate_manager = SurrogateModelManager.load(surrogate_path)
        labels.append(surrogate_manager.model_name)
        # get surrogate acc
        if acc_config_name not in surrogate_manager.config:
            surrogate_manager.config[acc_config_name] = (surrogate_manager.topKAcc(surrogate_manager.dataset.val_dl, topk=(1))[1] + surrogate_manager.topKAcc(surrogate_manager.victim_model.dataset.val_dl, topk=(1))[1]) / 2
            surrogate_manager.saveConfig()
        surrogate_accs.append(surrogate_manager.config[acc_config_name])
        # get victim acc
        if acc_config_name not in surrogate_manager.victim_model.config:
            surrogate_manager.victim_model.config = (surrogate_manager.victim_model.topKAcc(surrogate_manager.dataset.val_dl, topk=(1))[1] + surrogate_manager.victim_model.topKAcc(surrogate_manager.victim_model.dataset.val_dl, topk=(1))[1]) / 2
            surrogate_manager.victim_model.saveConfig()
        victim_accs.append(surrogate_manager.victim_model.config[acc_config_name])
    return labels, victim_accs, surrogate_accs


def plotCompare(labels, victim_accs, surrogate_accs, dataset):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, victim_accs, width, label='Victim Acc')
    rects2 = ax.bar(x + width/2, surrogate_accs, width, label='Surrogate Acc')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Victim/Surrogate Top1 Accuracy on CIFAR10 Val Data\nSurrogates Trained Using Knowledge Distillation')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    plt.savefig(SAVE_FOLDER / f"{dataset}_{timestamp}.png", dpi=500)
    




if __name__ == '__main__':
    dataset = "cifar10"

    models = getVictimSurrogateModels(args={"dataset": dataset})

    labels, vict_accs, surr_accs = getVictimSurrogateAccs(models, dataset)

    plotCompare(labels, vict_accs, surr_accs, dataset)