"""
Different plots:

the first 3 plots are implemented by the function
plotMetricByModelAndStrategy()

Relative metric between surrogate and victim:
    Plot (surrogate model metric) / (victim model metric). This is a 
    histogram of a single datapoint at the end of training. One color 
    will be used for each method of surrogate training, which may 
    include pure knowledge distribution, or transfer set training using 
    transfer sets with different parameters. The metric must exist in
    both the victim and surrogate's config files.
    For example, plotting with 'val_acc1' shows how much of the victim's
    validation accuracy is recouped through the surrogate model's 
    training process. The victim's validation set
    is the default validation set during surrogate model training regardless
    of if a transfer set is used. 

    usage: in plotMetricByModelAndStrategy() set absolute to false and
    include_victim to true.

Absolute metric comparision between surrogate and victim:
    Plot surrogate model metric and victim model metric side-by-side.
    This is the same plot as above but in absolute rather than relative terms.
    For example, for accuracy, this will plot the surrogate and victim accuracy
    side-by-side.

    usage: in plotMetricByModelAndStrategy() set absolute to True and
    include_victim to true.

Absolute metric for surrogate model:
    The same as the plot above but only plotting surrogate model datapoints,
    not victim model datapoints. Used to show data specific to the surrogate
    model, like agreement and transfer attack accuracy.
    Like above, one color is used for each surrogate training strategy.

    usage: in plotMetricByModelAndStrategy() set absolute to True and
    include_victim to False.

The same 3 as above, but averaged over all the models into 1 datapoint per
metric, with bars for multiple metrics and different colors for different
training strategies.

---Plots during training---
Training metrics:
    This is a time series plot of different training metrics where the x axis
    is the training epoch. The metrics are:
    train_loss, train_acc1, train_acc5, train_agreement
    val_loss, val_acc1, val_acc5, val_agreement
    l1_weight_bound,transfer_attack_success.

    Note that everything except the l1_weight_bound is between 0 and 1, so
    they can all be on the same plot, and l1_weight_bound will need to be
    on a different plot. Also, this plot has several options:
        (1) plot a single metric (e.g. train_acc1) compared across different
         surrogate model training strategies, averaged by model architecture
        (2) plot a single metric for the same surrogate model training 
         strategy, compared across model architecture
        (3) plot multiple metrics averaged by model architecture for a single
        surrogate model training strategy
"""

import datetime
from pathlib import Path
from typing import List, Dict, Union, Tuple
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
#plt.style.use('ggplot')

#rc('font',**{'family':'serif','serif':['Times'], 'size': 14})
#rc('figure', **{'figsize': (5, 4)})

import sys

# setting path
sys.path.append("../edge_profile")

from model_manager import SurrogateModelManager, VictimModelManager, getVictimSurrogateModels, getModelsFromSurrogateTrainStrategies
from config import MODELS

SAVE_FOLDER = Path(__file__).parent.absolute() / "surrogate_acc"
SAVE_FOLDER.mkdir(exist_ok=True)

"""
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
"""
    

def plotMetricByModelAndStrategy(strategies: dict, models: List[str], metric: str, absolute: bool = False, include_victim: bool = True, save: bool = True):
    """
    Note this currently assumes that there is only 1 victim model per architecture.
    
    Plot (surrogate model metric) / (victim model metric). This is a 
    histogram of a single datapoint at the end of training. One color 
    will be used for each method of surrogate training, which may 
    include pure knowledge distribution, or transfer set training using 
    transfer sets with different parameters. The names of these 
    strategies are the keys of the <strategies> dict.
    
    The metric must exist in both the victim and surrogate's config files.
    For example, plotting with 'val_acc1' shows how much of the victim's
    validation accuracy is recouped through the surrogate model's 
    training process. The victim's validation set
    is the default validation set during surrogate model training regardless
    of if a transfer set is used. 

    If absolute is True:
    Plots absolute surrogate model metric and victim model metric side-by-side.
    This is the same plot as above but in absolute rather than relative terms.
    For example, for accuracy, this will plot the surrogate and victim accuracy
    side-by-side.
    """
    if not include_victim:
        assert absolute, """Cannot plot surrogate metrics relative to victim model
        if include_victim is False.  Either set include_victim to True or set 
        absolute to True."""

    plt.cla()
    # manager_paths is a dict of {strategy name: {architecture_name: path to surrogate model}}
    manager_paths = getModelsFromSurrogateTrainStrategies(strategies=strategies, architectures=models)
    strategy_labels = list(strategies.keys())

    # will be a 2d list of [[architecture1_name, metric under strategy 1, 
    # metric under strategy 2 ...], [architecture2_name, ...], ...]
    data = []

    for model in models:
        model_data = [model]
        try:
            for strategy in strategies:
                path = manager_paths[strategy][model]
                model_data.append(SurrogateModelManager.loadConfig(path.parent)[metric])
        except Exception as e:
            print(f"Strategy: {strategy}\nPath: {path}")
            raise e
        
        if include_victim:
            # get metric of victim model, assumes only 1 victim
            victim_val_acc = SurrogateModelManager.loadVictimConfig(path.parent)[metric]
            model_data.append(victim_val_acc)
        
        data.append(model_data)

    data = sorted(data, key = lambda x: x[0])
    x_labels = [x[0] for x in data]
    x = np.arange(len(x_labels))  # the label locations
    width = 0.8  # the width of all bars for a single architecture
    bar_width = width / len(strategies) # width of a single bar

    if absolute and include_victim:
        # account for the victim model bar
        bar_width = width / (len(strategies) + 1)

    fig, ax = plt.subplots()
    for i, strategy_name in enumerate(strategy_labels):
        offset = (-1 *  width/2) + (i * bar_width) + (bar_width/2)
        strategy_data = [x[i+1]/x[-1] for x in data]
        if absolute:
            strategy_data = [x[i+1] for x in data]
        ax.bar(x - offset, strategy_data, bar_width, label=strategy_name)
    
    # if absolute, add the victim model metric for comparison
    if absolute and include_victim:
        offset = (width/2) - bar_width/2
        victim_data = [x[-1] for x in data]
        ax.bar(x - offset, victim_data, bar_width, label="victim_model")


    # Add some text for labels, title and custom x-axis tick labels, etc.
    y_label = f"{metric}" if absolute else f"Surrogate {metric}/ Victim {metric}"
    ax.set_ylabel(y_label)
    title = f"Surrogate {metric} Relative to Victim {metric}\nby Training Strategy and DNN Architecture"
    if absolute:
        title = f"{metric} by Training Strategy and DNN Architecture"
    ax.set_title(title)
    ax.set_xticks(x, x_labels)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(SAVE_FOLDER / f"{metric}_{'with' if include_victim else 'no'}_victim_{timestamp}.png", dpi=500)
    else:
        plt.show()


def plotMultipleTrainingMetrics(strategies: dict, models: List[str], metrics: List[str], y_lim: Tuple[float, float] = None, save: bool = True):
    """
    plot multiple metrics averaged by model architecture for a single
    surrogate model training strategy.

    The number of training epochs must be the same for each model.
    """
    plt.cla()
    assert len(strategies) == 1
    # manager_paths is a dict of {strategy name: {architecture_name: path to surrogate model}}
    manager_paths = getModelsFromSurrogateTrainStrategies(strategies=strategies, architectures=models)
    strategy_name = list(strategies.keys())[0]
    num_epochs = SurrogateModelManager.loadConfig(manager_paths[strategy_name][models[0]].parent)["epochs_trained"]

    # will be a dict of {metric: [metric per epoch over training, summed over the architectures]}
    data = {metric: [0]* num_epochs for metric in metrics}

    for model in models:
        path = manager_paths[strategy_name][model]
        model_train_df = SurrogateModelManager.loadTrainLog(path.parent)
        for metric in metrics:
            model_metric_data = model_train_df[metric].values.tolist()
            data[metric] = [data[metric][i] + model_metric_data[i] for i in range(num_epochs)]

    x_axis = list(range(1, num_epochs + 1))

    for metric in data:
        avg_metric = [data[metric][i]/len(models) for i in range(num_epochs)]
        plt.plot(x_axis, avg_metric, label=metric)
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel("Metric Value")
    arch_str = f"{str(models)}"[1:-1].replace("'", "") if len(models) < 4 else str(len(models))
    title = f"Surrogate Model Metrics for {strategy_name} Training\nAveraged over {arch_str} Architecture{'s' if len(models) > 1 else ''}"
    plt.xlabel("Training Epoch")
    plt.title(title)
    plt.xticks()
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.legend()
    plt.tight_layout()
    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(SAVE_FOLDER / f"{strategy_name}_{timestamp}.png", dpi=500)    
    else:
        plt.show()


if __name__ == '__main__':

    # this is a set of args to match with model manager config
    # values.
    strategies = {
        # "other_half_cifar10" : {
        #     "pretrained": False,
        #     "knockoff_transfer_set": None,
        # },
        "knockoff_cifar100" : {
            "pretrained": True,
            "knockoff_transfer_set": {
                "dataset_name": "cifar100",
                "transfer_size": 40000,
                "sample_avg": 50,
                "random_policy": False,
                "entropy": True,
            },
        },
    }

    # the set of model architectures to use.
    models = MODELS
    # models = ["resnet18"]
    # models = [models[9]]
    exclude = []
    # exclude = ['vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'squeezenet1_0', 'squeezenet1_1', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',]
    # exclude.extend(["alexnet", "resnet152", "resnext50_32x4d"])
    for ex in exclude:
        models.remove(ex)

    plotMetricByModelAndStrategy(strategies=strategies, models=models, metric = "val_acc1", absolute=True, include_victim=True)
    # plotMetricByModelAndStrategy(strategies=strategies, models=models, metric = "val_acc1", absolute=False, include_victim=True)
    # plotMultipleTrainingMetrics(strategies=strategies, models=models, metrics=["val_acc1", "train_acc1"])
    # plotMultipleTrainingMetrics(strategies=strategies, models=models, metrics=["val_loss", "train_loss"], y_lim=(0.01, 0.1), save=True)
    # plotMultipleTrainingMetrics(strategies=strategies, models=models, metrics=["l1_weight_bound"], save=True)
    # plotMultipleTrainingMetrics(strategies=strategies, models=models, metrics=["transfer_attack_success"], save=True)

