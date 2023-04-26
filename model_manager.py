"""
Generic model training from Pytorch model zoo, used for the victim model.

Assuming that the victim model architecture has already been found/predicted,
the surrogate model can be trained using labels from the victim model.
"""

import datetime
import json
from pathlib import Path
import random
import time
from typing import Callable, Dict, List, Tuple
import traceback
from abc import ABC, abstractmethod
import shutil

import torch
from torch.nn.utils import prune
import numpy as np
from tqdm import tqdm
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from get_model import (
    get_model,
    getModelParams,
    all_models,
    get_quantized_model,
    quantized_models,
    name_to_family,
)
from datasets import Dataset
from logger import CSVLogger
from online import OnlineStats
from model_metrics import correct, accuracy, both_correct
from collect_profiles import run_command, generateExeName
from utils import latest_file, latestFileFromList
from format_profiles import parse_one_profile, avgProfiles
from architecture_prediction import ArchPredBase, get_arch_pred_model
import config


def loadModel(path: Path, model: torch.nn.Module, device: torch.device = None) -> None:
    assert path.exists(), f"Model load path \n{path}\n does not exist."
    if device is not None:
        params = torch.load(path, map_location=device)
    else:
        params = torch.load(path)
    model.load_state_dict(params, strict=False)
    model.eval()


class ModelManagerBase(ABC):
    """
    Generic model manager class.
    Can train a model on a dataset and save/load a model.

    Note- inheriting classes should not modify self.config until
    after calling this constructor because this constructor will overwrite
    self.config

    This constructor leaves no footprint on the filesystem.
    """

    MODEL_FILENAME = "checkpoint.pt"

    def __init__(
        self,
        architecture: str,
        model_name: str,
        path: Path,
        dataset: str,
        data_subset_percent: float = None,
        data_idx: int = 0,
        gpu: int = -1,
        save_model: bool = True,
    ) -> None:
        """
        path: path to folder
        """
        self.architecture = architecture
        self.model_name = model_name
        self.path = path
        self.data_subset_percent = data_subset_percent
        self.data_idx = data_idx
        self.dataset = self.loadDataset(dataset)
        self.save_model = save_model
        self.model = None  # set by self.model=self.constructModel()
        self.model_path = None  # set by self.model=self.constructModel()
        self.device = torch.device("cpu")
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu}")
        self.gpu = gpu

        self.config = {
            "path": str(self.path),
            "architecture": self.architecture,
            "dataset": dataset,
            "dataset_config": self.dataset.config,
            "data_subset_percent": data_subset_percent,
            "data_idx": data_idx,
            "model_name": self.model_name,
            "device": str(self.device),
        }

        self.epochs_trained = 0  # if loading a model, this gets updated in inheriting classes' constructors
        # when victim models are loaded, they call self.loadModel, which calls
        # self.saveConfig().  If config["epochs_trained"] = 0 when this happens,
        # then the actual epochs_trained is overwritten.  To prevent this, we
        # have the check below.
        if not self.path.exists():
            self.config["epochs_trained"] = self.epochs_trained

    def constructModel(
        self, pretrained: bool = False, quantized: bool = False, kwargs=None
    ) -> torch.nn.Module:
        if kwargs is None:
            kwargs = {}
        kwargs.update({"num_classes": self.dataset.num_classes})
        print(f"Model Manager - passing {kwargs} args to construct {self.architecture}")
        if not quantized:
            model = get_model(self.architecture, pretrained=pretrained, kwargs=kwargs)
        else:
            model = get_quantized_model(self.architecture, kwargs=kwargs)
        model.to(self.device)
        return model

    def loadModel(self, path: Path) -> None:
        assert Path(path).exists(), f"Model load path \n{path}\n does not exist."
        # the epochs trained should already be done from loading the config.
        # if "_" in str(path.name):
        #     self.epochs_trained = int(str(path.name).split("_")[1].split(".")[0])
        params = torch.load(path, map_location=self.device)
        self.model.load_state_dict(params, strict=False)
        self.model.eval()
        self.model.to(self.device)
        self.model_path = path
        self.saveConfig({"model_path": str(path)})

    def saveModel(
        self, name: str = None, epoch: int = None, replace: bool = False
    ) -> None:
        """
        If epoch is passed, will append '_<epoch>' before the file extension
        in <name>.  Example: if name="checkpoint.pt" and epoch = 10, then
        will save as "checkpoint_10.pt".
        If replace is true, then if the model file already exists it will be replaced.
        If replace is false and the model file already exists, an error is raised.
        """
        if not self.save_model:
            return
        # todo add remove option
        if name is None:
            name = self.MODEL_FILENAME
        if epoch is not None:
            name = name.split(".")[0] + f"_{epoch}." + name.split(".")[1]
        model_file = self.path / name
        if model_file.exists():
            if replace:
                print(f"Replacing model {model_file}")
                model_file.unlink()
            else:
                raise FileExistsError
        torch.save(self.model.state_dict(), model_file)
        self.model_path = model_file
        self.saveConfig({"model_path": str(model_file)})

    def loadDataset(self, name: str) -> Dataset:
        return Dataset(
            name,
            data_subset_percent=self.data_subset_percent,
            idx=self.data_idx,
            resize=getModelParams(self.architecture).get("input_size", None),
        )

    @staticmethod
    @abstractmethod
    def load(self, path: Path, gpu: int = -1):
        """Abstract method"""

    def saveConfig(self, args: dict = {}) -> None:
        """
        Write parameters to a json file.  If file exists already, then will be
        appended to/overwritten.  If args are provided, they are added to the config file.
        """
        if not self.save_model:
            return
        self.config.update(args)
        # look for config file
        config_files = [x for x in self.path.glob("params_*")]
        if len(config_files) > 1:
            raise ValueError(f"Too many config files in path {self.path}")
        if len(config_files) == 1:
            with open(config_files[0], "r") as f:
                conf = json.load(f)
            conf.update(self.config)
            with open(config_files[0], "w") as f:
                json.dump(conf, f, indent=4)
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.path / f"params_{timestamp}.json"
        with open(path, "w+") as f:
            json.dump(self.config, f, indent=4)

    @staticmethod
    def loadConfig(path: Path) -> dict:
        """path is not hardcoded as self.path so that this method can be called
        by inheriting classes before self.path is set (which occurs when invoking
        this class's constructor)
        """
        config_files = [x for x in path.glob("params_*")]
        if len(config_files) != 1:
            raise ValueError(
                f"There are {len(config_files)} config files in path {path}\nThere should only be one."
            )
        with open(config_files[0], "r") as f:
            conf = json.load(f)
        return conf

    def trainModel(
        self,
        num_epochs: int,
        lr: float = None,
        debug: int = None,
        patience: int = 10,
        replace: bool = False,
    ):
        """Trains the model using dataset self.dataset.

        Args:
            num_epochs (int): number of training epochs
            lr (float): initial learning rate.  This function decreases the learning rate
                by a factor of 0.1 when the loss fails to decrease by 1e-4 for 10 iterations.
                If not passed, will default to learning rate of model from get_model.py, and
                if there is no learning rate specified there, defaults to 0.1.
            save_freq: how often to save model, default is only at the end. models are overwritten.

        Returns:
            Nothing, only sets the self.model class variable.
        """
        # todo add checkpoint freq, also make note to flush logger when checkpointing
        assert self.dataset is not None
        assert self.model is not None, "Must call constructModel() first"

        if lr is None:
            lr = getModelParams(self.architecture).get("lr", 0.1)

        self.epochs = num_epochs
        if self.save_model:
            logger = CSVLogger(self.path, self.train_metrics)

        optim = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=1e-4,
        )
        if getModelParams(self.architecture).get("optim", "") == "adam":
            optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        loss_func = torch.nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=patience
        )

        since = time.time()
        try:
            for epoch in range(1, num_epochs + 1):
                if debug is not None and epoch > debug:
                    break
                loss, acc1, acc5 = self.runEpoch(
                    train=True,
                    epoch=epoch,
                    optim=optim,
                    loss_fn=loss_func,
                    lr_scheduler=lr_scheduler,
                    debug=debug,
                )
                val_loss, val_acc1, val_acc5 = self.runEpoch(
                    train=False,
                    epoch=epoch,
                    optim=optim,
                    loss_fn=loss_func,
                    lr_scheduler=lr_scheduler,
                    debug=debug,
                )

                metrics = {
                    "train_loss": loss,
                    "train_acc1": acc1,
                    "train_acc5": acc5,
                    "val_loss": val_loss,
                    "val_acc1": val_acc1,
                    "val_acc5": val_acc5,
                    "lr": optim.param_groups[0]["lr"],
                }

                self.epochs_trained += 1
                if self.save_model:
                    # we only write to the log file once the model is saved, see below after self.saveModel()
                    logger.futureWrite(
                        {
                            "timestamp": time.time() - since,
                            "epoch": self.epochs_trained,
                            **metrics,
                        }
                    )

        except KeyboardInterrupt:
            print(f"\nInterrupted at epoch {epoch}. Tearing Down")

        self.model.eval()
        print("Training ended, saving model.")
        self.saveModel(replace=replace)  # this function already checks self.save_model
        if self.save_model:
            logger.flush()  # this saves all the futureWrite() calls
            logger.close()
        self.config["epochs_trained"] = self.epochs_trained
        self.config["initialLR"] = lr
        self.config["finalLR"] = optim.param_groups[0]["lr"]
        self.config.update(metrics)
        self.saveConfig()

    def runEpoch(
        self,
        train: bool,
        epoch: int,
        optim: torch.optim.Optimizer,
        loss_fn: Callable,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        debug: int = None,
    ) -> tuple[int]:
        """Run a single epoch."""

        self.model.eval()
        prefix = "val"
        dl = self.dataset.val_dl
        if train:
            self.model.train()
            prefix = "train"
            dl = self.dataset.train_dl

        total_loss = OnlineStats()
        acc1 = OnlineStats()
        acc5 = OnlineStats()
        step_size = OnlineStats()
        step_size.add(optim.param_groups[0]["lr"])

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(
            f"{prefix.capitalize()} Epoch {epoch if train else '1'}/{self.epochs if train else '1'}"
        )

        with torch.set_grad_enabled(train):
            for i, (x, y) in enumerate(epoch_iter, start=1):
                if debug and i > debug:
                    break
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = loss_fn(yhat, y)

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                c1, c5 = correct(yhat, y, (1, 5))
                total_loss.add(loss.item() / len(x))
                acc1.add(c1 / len(x))
                acc5.add(c5 / len(x))

                epoch_iter.set_postfix(
                    loss=total_loss.mean,
                    top1=acc1.mean,
                    top5=acc5.mean,
                    step_size=step_size.mean,
                )

        loss = total_loss.mean
        top1 = acc1.mean
        top5 = acc5.mean

        if train and debug is None:
            lr_scheduler.step(loss)
            # get actual train accuracy/loss after weights update
            top1, top5, loss = accuracy(
                model=self.model,
                dataloader=self.dataset.train_acc_dl,
                loss_func=loss_fn,
                topk=(1, 5),
            )

        return loss, top1, top5

    @property
    def train_metrics(self) -> list:
        """Generate the training metrics to be logged to a csv."""
        return [
            "epoch",
            "timestamp",
            "train_loss",
            "train_acc1",
            "train_acc5",
            "val_loss",
            "val_acc1",
            "val_acc5",
            "lr",
        ]

    def runPGD(
        self,
        x: torch.Tensor,
        eps: float,
        step_size: float,
        iterations: int,
        norm=np.inf,
    ) -> torch.Tensor:
        return projected_gradient_descent(
            self.model, x, eps=eps, eps_iter=step_size, nb_iter=iterations, norm=norm
        )

    def topKAcc(self, dataloader: torch.utils.data.DataLoader, topk=(1, 5)):
        self.model.eval()
        online_stats = {}
        for k in topk:
            online_stats[k] = OnlineStats()

        data_iter = tqdm(dataloader)
        for x, y in data_iter:
            x = x[:1]
            y = y[:1]
            x, y = x.to(self.device), y.to(self.device)
            yhat = self.model(x)
            topk_correct = correct(yhat, y, topk)
            for k, k_correct in zip(topk, topk_correct):
                online_stats[k].add(k_correct)

            for k in topk:
                data_iter.set_postfix(
                    **{str(k): online_stats[k].mean for k in online_stats}
                )

        print({k: online_stats[k].mean for k in online_stats})

    def __repr__(self) -> str:
        return str(self.path.relative_to(Path.cwd()))


class ProfiledModelManager(ModelManagerBase):
    """Extends ModelManagerBase to include support for profiling"""

    def runNVProf(
        self, use_exe: bool = True, seed: int = 47, n: int = 10, input: str = "0"
    ):
        """
        Creates a subfolder self.path/profiles, and adds a profile file profile_{pid}.csv and
        associated params_{pid}.json file to this subfolder, if the profile succeeded.
        There is support for multiple profiles.
        Note - this function does not check for collisions in pid.
        """
        assert self.gpu >= 0
        assert self.model_path is not None
        profile_folder = self.path / "profiles"
        profile_folder.mkdir(exist_ok=True)
        prefix = profile_folder / "profile_"
        executable = generateExeName(use_exe)
        print(f"Using executable {executable} for nvprof")
        command = (
            f"nvprof --csv --log-file {prefix}%p.csv --system-profiling on "
            f"--profile-child-processes {executable} -gpu {self.gpu} -load_path {self.model_path}"
            f" -seed {seed} -n {n} -input {input}"
        )

        print(f"\nCommand being run:\n{command}\n\n")

        success, file = run_command(profile_folder, command)
        retries = 0
        print(f"{'Success' if success else 'Failure'} on file {file}")
        while not success:
            print("\nNvprof retrying ... \n")
            time.sleep(10)
            profile_file = latest_file(profile_folder, "profile_")
            if profile_file is not None and profile_file.exists():
                profile_file.unlink()
            success, file = run_command(profile_folder, command)
            retries += 1
            if retries > 5:
                print("Reached 5 retries, exiting...")
                break
        if not success:
            latest_file(profile_folder, "profile_").unlink()
            raise RuntimeError("Nvprof failed 5 times in a row.")
        profile_num = str(file.name).split("_")[1].split(".")[0]
        params = {
            "file": str(file),
            "profile_number": profile_num,
            "use_exe": use_exe,
            "seed": seed,
            "n": n,
            "input": input,
            "success": success,
            "gpu": self.gpu,
            "gpu_type": torch.cuda.get_device_name(0).lower().replace(" ", "_"),
        }
        with open(profile_folder / f"params_{profile_num}.json", "w") as f:
            json.dump(params, f, indent=4)
        assert self.isProfiled()

    def isProfiled(self) -> bool:
        """
        Checks if the model has been profiled. Returns True if there
        is a subfolder self.path/profiles with at least one profile_{pid}.csv
        and associated params_{pid}.csv.
        """
        profile_folder = self.path / "profiles"
        profile_config = [x for x in profile_folder.glob("params_*")]
        if len(profile_config) == 0:
            return False
        with open(profile_config[0], "r") as f:
            conf = json.load(f)
        # instead of taking the path directly from the config file, use the name
        # this allows the model to be profiled on one machine and then downloaded
        # to another machine, and this method will still return true.
        profile_path = self.path / "profiles" / Path(conf["file"]).name
        return profile_path.exists()

    def getProfile(self, filters: dict = None) -> Tuple[Path, Dict]:
        """
        Return a tuple of (path to profile_{pid}.csv,
        dictionary obtained from reading params_{pid}.json).
        filters: a dict and each argument in the dict must match
            the argument from the config file associated with a profile.
            to get a profile by name, can specify {"profile_number": "2181935"}
        If there are multiple profiles which fit the filters, return the latest one.
        """
        if filters is None:
            filters = {}
        profile_folder = self.path / "profiles"
        # get config files
        profile_config = [x for x in profile_folder.glob("params_*")]
        assert len(profile_config) > 0

        fit_filters = {}

        for config_path in profile_config:
            with open(config_path, "r") as f:
                conf = json.load(f)
            matched_filter = True
            for arg in filters:
                if arg not in conf or filters[arg] != conf[arg]:
                    matched_filter = False
                    break
            if matched_filter:
                prof_num = conf["profile_number"]
                profile_path = profile_folder / f"profile_{prof_num}.csv"
                assert profile_path.exists()
                fit_filters[profile_path] = conf

        if len(fit_filters) == 0:
            raise ValueError(
                f"No profiles with filters {filters} found in {profile_folder}"
            )
        latest_valid_path = latestFileFromList(list(fit_filters.keys()))
        conf = fit_filters[latest_valid_path]
        return latest_valid_path, conf

    def getAllProfiles(self, filters: dict = None) -> List[Tuple[Path, Dict]]:
        """
        Returns a list of tuples (path to profile_{pid}.csv,
        dictionary obtained from reading params_{pid}.json) for
        every profile in self.path/profiles
        filters: a dict and each argument in the dict must match
            the argument from the config file associated with a profile.
            to get a profile by name, can specify {"profile_number": "2181935"}
        """
        if filters is None:
            filters = {}
        result = []
        profile_folder = self.path / "profiles"
        profile_configs = [x for x in profile_folder.glob("params_*")]
        for config_file in profile_configs:
            with open(config_file, "r") as f:
                conf = json.load(f)
            matched_filters = True
            for arg in filters:
                if filters[arg] != conf[arg]:
                    matched_filters = False
                    break
            if matched_filters:
                prof_num = conf["profile_number"]
                profile_path = profile_folder / f"profile_{prof_num}.csv"
                if profile_path.exists():
                    result.append((profile_path, conf))
        return result

    def predictVictimArch(
        self, arch_pred_model: ArchPredBase, average: bool = False, filters: dict = None
    ) -> Tuple[str, float]:
        """
        Given an architecture prediction model, use it to predict the architecture of the model associated
        with this model manager.
        average: if true, will average the features from all of the profiles on this model and then pass the
            features to the architecture prediction model.
        filters: a dict and each argument in the dict must match
            the argument from the config file associated with a profile.
            to get a profile by name, can specify {"profile_number": "2181935"}
        """
        assert self.isProfiled()

        profile_csv, config = self.getProfile(filters=filters)
        profile_features = parse_one_profile(profile_csv, gpu=config["gpu"])
        if average:
            all_profiles = self.getAllProfiles(filters=filters)
            gpu = all_profiles[0][1]["gpu"]
            for _, config in all_profiles:
                assert config["gpu"] == gpu
            profile_features = avgProfiles(
                profile_paths=[x[0] for x in all_profiles], gpu=gpu
            )

        arch, conf = arch_pred_model.predict(profile_features)
        print(
            f"Predicted surrogate model architecture for victim model\n{self.path}\n is {arch} with {conf * 100}% confidence."
        )
        # format is to store results in self.config as such:
        # {
        #    "pred_arch": {
        #        "nn (model type)": {
        #            "profile_4832947.csv: [
        #                {"pred_arch": "resnet18", "conf": 0.834593048}
        #            ]
        #        }
        #    }
        # }
        #
        # this way there is support for multiple predictions from the same model type on the same
        # profile
        prof_name = str(profile_csv.name)
        arch_conf = {"pred_arch": arch, "conf": conf}
        results = {prof_name: [arch_conf]}
        if "pred_arch" not in self.config:
            self.config["pred_arch"] = {arch_pred_model.name: results}
        else:
            if arch_pred_model.name not in self.config["pred_arch"]:
                self.config["pred_arch"][arch_pred_model.name] = results
            else:
                if prof_name not in self.config["pred_arch"][arch_pred_model.name]:
                    self.config["pred_arch"][arch_pred_model.name][prof_name] = [
                        arch_conf
                    ]
                else:
                    self.config["pred_arch"][arch_pred_model.name][prof_name].append(
                        arch_conf
                    )
        self.saveConfig()
        return arch, conf, arch_pred_model


class VictimModelManager(ProfiledModelManager):
    def __init__(
        self,
        architecture: str,
        dataset: str,
        model_name: str,
        load: str = None,
        gpu: int = -1,
        data_subset_percent: float = 0.5,
        pretrained: bool = False,
        save_model: bool = True,
    ):
        """
        Models files are stored in a folder
        ./models/{model_architecture}/{self.name}_{date_time}/

        This includes the model file, a csv documenting training, and a config file.

        Args:
            architecture (str): the exact string representation of the model architecture.
                See get_model.py.
            dataset (str): the name of the dataset all lowercase.
            model_name (str): The name of the model, can be anything except don't use underscores.
            load (str, optional): If provided, should be the absolute path to the model folder,
                {cwd}/models/{model_architecture}/{self.name}{date_time}.  This will load the model
                stored there.
            data_subset_percent (float, optional): If provided, should be the fraction of the dataset
                to use.  This will be generated determinisitcally.  Uses torch.utils.data.random_split
                (see datasets.py)
            idx (int): the index into the subset of the dataset.  0 for victim model and 1 for surrogate.
        """
        path = self.generateFolder(load, architecture, model_name)
        super().__init__(
            architecture=architecture,
            model_name=model_name,
            path=path,
            dataset=dataset,
            data_subset_percent=data_subset_percent,
            gpu=gpu,
            save_model=save_model,
        )

        self.pretrained = pretrained
        self.config["pretrained"] = pretrained

        if load is None:
            # creating the object for the first time
            self.model = self.constructModel(pretrained=pretrained)
            self.trained = False
            if save_model:
                assert not path.exists()
                path.mkdir(parents=True)
        else:
            # load from previous run
            self.model = self.constructModel(pretrained=False)
            self.trained = True
            self.loadModel(load)
            # this causes stored parameters to overwrite new ones
            self.config.update(self.loadConfig(self.path))
            # update with the new device
            self.config["device"] = str(self.device)
            self.epochs_trained = self.config["epochs_trained"]

        self.saveConfig()

    @staticmethod
    def load(model_path: Path, gpu: int = -1):
        """Create a ModelManager Object from a path to a model file."""
        folder_path = Path(model_path).parent
        conf = ModelManagerBase.loadConfig(folder_path)
        print(
            f"Loading {conf['architecture']} trained on {conf['dataset']} from path {model_path}"
        )
        model_manager = VictimModelManager(
            architecture=conf["architecture"],
            dataset=conf["dataset"],
            model_name=conf["model_name"],
            load=Path(model_path),
            gpu=gpu,
            data_subset_percent=conf["data_subset_percent"],
            pretrained=conf["pretrained"],
        )
        model_manager.config = conf
        return model_manager

    def generateFolder(self, load: Path, architecture: str, model_name: str) -> str:
        """
        Generates the model folder as ./models/model_architecture/{self.name}_{date_time}/
        """
        if load:
            return Path(load).parent
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_folder = Path.cwd() / "models" / architecture / f"{model_name}_{time}"
        return model_folder

    @staticmethod
    def getModelPaths(prefix: str = None) -> List[Path]:
        """
        Return a list of paths to all victim models
        in directory "./<prefix>".  This directory must be organized
        by model architecture folders whose subfolders are victim model
        folders and contain a model stored in 'checkpoint.pt'.
        Default prefix is ./models/
        """
        if prefix is None:
            prefix = "models"

        models_folder = Path.cwd() / prefix
        arch_folders = [i for i in models_folder.glob("*") if i.is_dir()]
        model_paths = []
        for arch in arch_folders:
            for model_folder in [i for i in arch.glob("*") if i.is_dir()]:
                victim_path = (
                    models_folder
                    / arch
                    / model_folder
                    / VictimModelManager.MODEL_FILENAME
                )
                if victim_path.exists():
                    model_paths.append(victim_path)
                else:
                    print(f"Warning, no model found {victim_path}")
        return model_paths

    def getSurrogateModelPaths(self) -> List[Path]:
        """Returns a list of paths to surrogate models of this victim model."""
        res = []
        surrogate_paths = [x for x in self.path.glob(f"surrogate*")]
        for path in surrogate_paths:
            res.append(path / ModelManagerBase.MODEL_FILENAME)
        return res

    def generateKnockoffTransferSet(
        self,
        dataset_name: str,
        transfer_size: int,
        sample_avg: int = 10,
        random_policy: bool = False,
    ) -> None:
        """
        TODO add labels to the stored indices, the current query budget is a bit of a hack.
        Uses adaptations of the Knockoff Nets paper https://arxiv.org/pdf/1812.02766.pdf
        to generate a transfer set for this victim model.  A transfer set is a subset of
        a dataset not used to train the victim model. For example, if the victim is
        trained on CIFAR10, the transfer set could be a subset of TinyImageNet. The
        transfer set can then be used to train a surrogate model.

        Two strategies are considered: random takes a random subset, and adaptive.
        Adaptive is not a faithful implementation of the paper.  Instead, the victim
        model's output on samples of each class is averaged over some number of samples,
        then the entropy of this average is taken to get a measure of how influential
        these samples are.  Low entropy means that the samples are influential, and
        vice versa.  There is one entropy value per class. We want to sample more
        from influential classes, so we make a vector of (1-entropy) for each class,
        normalize it, and then use it as a multinomial distribution from which to
        sample elements for the transfer set.

        The transfer size is also the query budget.

        The resulting dataset will be stored in a json format under
        self.path/transfer_sets/<dataset>_<datetime>.  This file will include the
        parameters passed to this algorithm as well as the indices of the dataset
        which are included in the transfer set.

        dataset_name: name of the dataset from which to generate the transfer set (should
            not be the dataset on which the victim model was trained)
        transfer_size: the size of the transfer set
        sample_avg: this is the number of samples to take per class before averaging,
            higher number means better entropy estimation.  Only used if random_policy=0
        random_policy: if true, generates the transfer set randomly. If false, uses the
            adaptive method.
        """
        config = {
            "dataset_name": dataset_name,
            "transfer_size": transfer_size,
            "sample_avg": sample_avg,
            "random_policy": random_policy,
        }
        assert (
            dataset_name != self.dataset.name
        ), "Don't use the same dataset for training and transfer set"
        # dataset is a torchvision.datasets.ImageFolder object
        dataset = Dataset(
            dataset_name,
            resize=getModelParams(self.architecture).get("input_size", None),
        ).train_data
        num_classes = len(dataset.classes)

        assert transfer_size <= len(
            dataset
        ), f"Requested transfer set size of {transfer_size} but {dataset_name} dataset has only {len(dataset)} samples."
        assert sample_avg * num_classes < len(
            dataset
        ), f"Requested {sample_avg} samples per class, with {num_classes} classes this is {sample_avg * num_classes} samples but {dataset_name} dataset has only {len(dataset)} samples."

        assert sample_avg * num_classes <= transfer_size, f"Requested {sample_avg} samples per class, with {num_classes} classes this is {sample_avg * num_classes} samples, but the transfer size (budget) is only {transfer_size}.  Either decrease the sample_avg or increase the transfer budget"

        print(
            f"Generating a transfer dataset for victim model {self} with configuration\n{json.dumps(config, indent=4)}\n"
        )

        sample_indices = None
        if random_policy:
            sample_indices = random.sample(range(len(dataset)), transfer_size)
        else:
            # adaptive policy

            # THE FOLLOWING DOESN'T WORK BECAUSE THE SAMPLES ARE NOT
            # SORTED IN ALL DATASETS, FOR EXAMPLE CIFAR100
            # if the dataset samples are sorted, we need to find
            # the start and end indices of each class and store as
            # [(start index for class 0, end index for class 0),
            # (start index for class 1, end index for class 1), ...]
            # this implementation does not assume a balanced dataset
            # use binary search
            # transitions = []
            # start = 0
            # for class_idx in range(num_classes - 1):
            #     # look for point in dataset.targets that changes from
            #     # class_idx to class_idx + 1, this is the end index
            #     hi = len(dataset) - 1
            #     lo = start
            #     while hi >= lo:
            #         mid = (hi + lo) // 2
            #         if dataset.targets[mid] == class_idx:
            #             lo = mid + 1
            #         else:
            #             if mid > start and dataset.targets[mid - 1] == class_idx:
            #                 # found
            #                 transitions.append((start, mid - 1))
            #                 start = mid
            #                 break
            #             hi = mid - 1
            # # need to account for last class
            # transitions.append((start, len(dataset) - 1))
            #
            # assert len(transitions) == num_classes
            #
            # def sampleClass(n, class_idx) -> List[int]:
            #     # returns n samples from class <class_idx> without replacement
            #     # need to generate n random numbers between the start and end
            #     # indices of this class, inclusive
            #     return random.sample(
            #         range(transitions[class_idx][0], transitions[class_idx][1] + 1), n
            #     )

            # first collect <sample_avg> samples of each class, store as a
            # list [[index of sample 1 of class 1, index of sample 2 of class 1, ...],
            # [index of sample 1 of class 2, index of sample 2 of class 2, ...], ...]
            # the samples of dataset are not necessarily sorted by class
            print(f"Generating a mapping from indices to labels ...")
            samples = [[] for _ in range(num_classes)]
            for idx in range(len(dataset)):
                samples[dataset.targets[idx]].append(idx)
            # check that there are enough samples per class for the sample average
            for class_name in range(num_classes):
                assert len(samples[class_name]) >= sample_avg, f"Could only find {len(samples[class_name])} samples for class {class_idx}, but {sample_avg} samples were requested"
            

            # get <sample_avg> samples per class
            class_samples = [random.sample(samples[class_idx], sample_avg) for class_idx in range(num_classes)]
            # now, for each class, get the average of the victim model's output on the samples
            # then compute (1-entropy of average output)
            print(f"Calculating class entropies ...")
            class_influence = []
            for class_idx in range(num_classes):
                sample_indices = class_samples[class_idx]
                x = torch.stack([dataset[i][0] for i in sample_indices]).to(self.device)
                y = self.model(x).cpu()
                y_prob = torch.softmax(y, dim=1)
                avg_y = torch.mean(y_prob, dim=0)
                avg_y = avg_y / torch.sum(avg_y)
                log_avg = torch.log(avg_y) / torch.log(torch.tensor(num_classes))
                entropy = -1 * torch.dot(avg_y, log_avg)
                class_influence.append(1 - entropy.item())

            # now, normalize the samples array to a multinomial distribution and use that to
            # sample from the dataset. note that we are going to use the class_samples variable as 
            # the starting point, so each class starts with sample_avg samples.
            samples_sum = sum(class_influence)
            multinomial_dist = [x / samples_sum for x in class_influence]
            config["class_importance"] = multinomial_dist
            samples_per_class = np.random.multinomial(transfer_size - (sample_avg * num_classes), multinomial_dist).tolist()
            samples_per_class = [samples_per_class[i] + class_samples[i] for i in range(num_classes)]
            # check to see if we sampled some classes more than we have data for
            overflow_samples = 0
            unsaturated_classes = [0 for _ in range(num_classes)]
            for class_idx in range(num_classes):
                if len(samples[class_idx]) < samples_per_class[class_idx]:
                    diff = samples_per_class[class_idx] - len(samples[class_idx])
                    samples_per_class[class_idx] -= diff
                    overflow_samples += diff
                else:
                    unsaturated_classes[class_idx] += len(samples[class_idx]) - samples_per_class[class_idx]
            #sample overflows randomly
            for x in range(overflow_samples):
                # normalize unsaturated classes
                unsaturated_classes_norm = [x/sum(unsaturated_classes) for x in unsaturated_classes]
                # choose from unsaturated classes randomly
                class_idx = np.random.multinomial(1, unsaturated_classes_norm)[0]
                unsaturated_classes[class_idx] -= 1
                samples_per_class[class_idx] += 1
            config["samples_per_class"] = samples_per_class

            # now generate samples per class and add them to a main list
            print(f"Sampling classes and writing to file ...")
            sample_indices = []
            for class_idx, num_samples in enumerate(samples_per_class):
                sample_indices.extend(random.sample(samples[class_idx], num_samples))

        config["sample_indices"] = sample_indices

        transfer_folder = self.path / "transfer_sets"
        transfer_folder.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = transfer_folder / f"{dataset_name}_{timestamp}.json"
        with open(save_path, "w+") as f:
            json.dump(config, f, indent=4)
        print(f"Completed generating the transfer set.")
        

    def loadKnockoffTransferSet(
        self,
        dataset_name: str,
        transfer_size: int,
        sample_avg: int = 10,
        random_policy: bool = False,
    ) -> Tuple[Path, Dataset]:
        """
        Loads a dataset from a json file produced by self.generateKnockoffTransferSet().
        This json file will be stored in self.path/transfer_sets/

        This function will search all of the json files in this folder,
        looking for a transfer set that fits the provided arguments.
        If none exists, raises a filenotfound error.

        Return value is a tuple of (path to transfer set json file, Dataset object).
        """
        config = {
            "dataset_name": dataset_name,
            "transfer_size": transfer_size,
            "sample_avg": sample_avg,
            "random_policy": random_policy,
        }
        transfer_folder = self.path / "transfer_sets"
        for file in transfer_folder.glob("*.json"):
            with open(file, "r+") as f:
                transfer_set_args = json.load(f)
            valid = True
            for arg in config:
                if config[arg] != transfer_set_args[arg]:
                    valid = False
                    break
            if valid:
                break
        if not valid:
            raise FileNotFoundError(
                f"No transfer sets found in {transfer_folder} matching configuration\n{json.dumps(config, indent=4)}\nCall generateKnockoffTransferSet to make one."
            )

        # only need to add indices for the training data
        transfer_set = Dataset(
            dataset=transfer_set_args["dataset_name"],
            indices=(transfer_set_args["sample_indices"], []),
        )
        return file, transfer_set


class QuantizedModelManager(ProfiledModelManager):
    FOLDER_NAME = "quantize"
    MODEL_FILENAME = "quantized.pt"

    def __init__(
        self,
        victim_model_path: Path,
        backend: str = "fbgemm",
        load_path: Path = None,
        gpu: int = -1,
        save_model: bool = True,
    ) -> None:
        self.victim_manager = VictimModelManager.load(victim_model_path)
        assert self.victim_manager.config["epochs_trained"] > 0
        assert self.victim_manager.architecture in quantized_models
        path = self.victim_manager.path / self.FOLDER_NAME
        super().__init__(
            architecture=self.victim_manager.architecture,
            model_name=f"quantized_{self.victim_manager.architecture}",
            path=path,
            dataset=self.victim_manager.dataset.name,
            data_subset_percent=self.victim_manager.data_subset_percent,
            data_idx=self.victim_manager.data_idx,
            gpu=gpu,
            save_model=save_model,
        )
        self.backend = backend
        self.model = self.constructModel(quantized=True)
        if load_path is None:
            # construct this object for the first time
            self.quantize()
            if save_model:
                path.mkdir()
                self.saveModel(self.MODEL_FILENAME)
        else:
            self.loadQuantizedModel(load_path)
            # this causes stored parameters to overwrite new ones
            self.config.update(self.loadConfig(self.path))
            # update with the new device
            self.config["device"] = str(self.device)
            self.epochs_trained = self.config["epochs_trained"]

        self.saveConfig(
            {"victim_model_path": str(victim_model_path), "backend": backend}
        )

    def prepare_for_quantization(self) -> None:
        torch.backends.quantized.engine = self.backend
        self.model.eval()
        # Make sure that weight qconfig matches that of the serialized models
        if self.backend == "fbgemm":
            self.model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_per_channel_weight_observer,
            )
        elif self.backend == "qnnpack":
            self.model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_weight_observer,
            )

        self.model.fuse_model()
        torch.quantization.prepare(self.model, inplace=True)

    def quantize(self) -> None:
        self.prepare_for_quantization()
        # calibrate model
        dl_iter = tqdm(self.victim_manager.dataset.train_acc_dl)
        dl_iter.set_description("Calibrating model for quantization")
        for i, (x, y) in enumerate(dl_iter):
            self.model(x)
        torch.quantization.convert(self.model, inplace=True)

    def loadQuantizedModel(self, path: Path):
        self.prepare_for_quantization()
        torch.quantization.convert(self.model, inplace=True)
        self.loadModel(path)

    @staticmethod
    def load(model_path: Path, gpu: int = -1):
        """model_path is a path to the quantized model checkpoint"""
        quantized_folder = Path(model_path).parent
        victim_model_path = quantized_folder.parent / VictimModelManager.MODEL_FILENAME
        conf = ModelManagerBase.loadConfig(quantized_folder)
        return QuantizedModelManager(
            victim_model_path=victim_model_path,
            backend=conf["backend"],
            load_path=model_path,
            gpu=gpu,
        )


class PruneModelManager(ProfiledModelManager):
    FOLDER_NAME = "prune"
    MODEL_FILENAME = "pruned.pt"

    def __init__(
        self,
        victim_model_path: Path,
        ratio: float = 0.5,
        finetune_epochs: int = 20,
        gpu: int = -1,
        load_path: Path = None,
        save_model: bool = True,
        debug: int = None,
    ) -> None:
        self.victim_manager = VictimModelManager.load(victim_model_path, gpu=gpu)
        assert self.victim_manager.config["epochs_trained"] > 0
        path = self.victim_manager.path / self.FOLDER_NAME
        super().__init__(
            architecture=self.victim_manager.architecture,
            model_name=f"pruned_{self.victim_manager.architecture}",
            path=path,
            dataset=self.victim_manager.dataset.name,
            data_subset_percent=self.victim_manager.data_subset_percent,
            data_idx=self.victim_manager.data_idx,
            gpu=gpu,
            save_model=save_model,
        )
        self.ratio = ratio
        self.finetune_epochs = finetune_epochs
        self.model = self.constructModel()
        self.pruned_modules = self.paramsToPrune()

        if load_path is None:
            if save_model:
                # this must be called before self.prune() because
                # self.prune() calls updateConfigSparsity() which
                # assumes the path exists.
                path.mkdir()
            # construct this object for the first time
            self.prune()  # modifies self.model
            # finetune
            self.trainModel(finetune_epochs, debug=debug)
        else:
            self.loadModel(load_path)
            # this causes stored parameters to overwrite new ones
            self.config.update(self.loadConfig(self.path))
            # update with the new device
            self.config["device"] = str(self.device)
            self.epochs_trained = self.config["epochs_trained"]

        self.saveConfig(
            {
                "victim_model_path": str(victim_model_path),
                "ratio": ratio,
                "finetune_epochs": finetune_epochs,
            }
        )

    def prune(self) -> None:
        # modifies self.model (through self.pruned_params)
        prune.global_unstructured(
            self.pruned_modules,
            pruning_method=prune.L1Unstructured,
            amount=self.ratio,
        )
        self.updateConfigSparsity()

    def paramsToPrune(self) -> List[Tuple[torch.nn.Module, str]]:
        res = []
        for name, module in self.model.named_modules():
            if name.startswith("classifier"):
                continue
            if name.startswith("fc"):
                continue
            if hasattr(module, "weight"):
                res.append((module, "weight"))
        self.pruned_modules = res
        return res

    def updateConfigSparsity(self) -> None:
        sparsity = {"module_sparsity": {}}

        pruned_mods_reformatted = {}
        for mod, name in self.pruned_modules:
            if mod not in pruned_mods_reformatted:
                pruned_mods_reformatted[mod] = [name]
            else:
                pruned_mods_reformatted[mod].append(name)

        zero_params_count = 0
        for name, module in self.model.named_modules():
            if module in pruned_mods_reformatted:
                total_params = sum(p.numel() for p in module.parameters())
                zero_params = np.sum(
                    [
                        getattr(module, x).detach().cpu().numpy() == 0.0
                        for x in pruned_mods_reformatted[module]
                    ]
                )
                zero_params_count += zero_params
                sparsity["module_sparsity"][name] = (
                    100.0 * float(zero_params) / float(total_params)
                )
            else:
                sparsity["module_sparsity"][name] = 0.0

        # get total sparsity
        # getting total number of parameters from
        # https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model/62764464#62764464
        total_params = sum(
            dict((p.data_ptr(), p.numel()) for p in self.model.parameters()).values()
        )
        sparsity["total_parameters"] = int(total_params)
        sparsity["zero_parameters"] = int(zero_params_count)
        # the percentage of zero params
        sparsity["total_sparsity"] = (
            100 * float(zero_params_count) / float(total_params)
        )
        self.config["sparsity"] = sparsity
        self.saveConfig()

    @staticmethod
    def load(model_path: Path, gpu: int = -1):
        """model_path is a path to the pruned model checkpoint"""

        load_folder = Path(model_path).parent  # the prune folder
        victim_path = load_folder.parent / VictimModelManager.MODEL_FILENAME
        conf = ModelManagerBase.loadConfig(load_folder)
        return PruneModelManager(
            victim_model_path=victim_path,
            ratio=conf["ratio"],
            finetune_epochs=conf["finetune_epochs"],
            gpu=gpu,
            load_path=model_path,
        )


class SurrogateModelManager(ModelManagerBase):
    """
    Constructs the surrogate model with a paired victim model, trains using from the labels from victim
    model.
    """

    FOLDER_NAME = "surrogate"

    def __init__(
        self,
        victim_model_path: Path,
        architecture: str,
        arch_conf: float,
        arch_pred_model_name: str,
        pretrained: bool = False,
        load_path: Path = None,
        gpu: int = -1,
        save_model: bool = True,
    ):
        """
        If load_path is not none, it should be a path to model.
        See SurrogateModelManager.load().

        Note - the self.config only accounts for keeping track of one history of
        victim model architecture prediction.
        Assumes victim model has been profiled/predicted.
        """
        self.victim_model = self.loadVictim(victim_model_path, gpu=gpu)
        if isinstance(self.victim_model, VictimModelManager):
            assert self.victim_model.config["epochs_trained"] > 0
        self.arch_pred_model_name = arch_pred_model_name
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.victim_model.path / f"{self.FOLDER_NAME}_{time}"
        self.pretrained = pretrained
        if load_path is None:
            # creating this object for the first time
            if not self.victim_model.isProfiled():
                print(
                    f"Warning, victim model {self.victim_model.path} has not been profiled and a surrogate model is being created."
                )
            self.arch_confidence = arch_conf
            if save_model:
                assert not path.exists()
                path.mkdir(parents=True)
        else:
            # load from a previous run
            folder = load_path.parent
            config = self.loadConfig(folder)
            self.arch_confidence = config["arch_confidence"]
            # TODO the following line is a hack, should be addressed.  The name of
            # the surrogate model including the timestamp is not included in the config
            path = self.victim_model.path / Path(config["path"]).name
        super().__init__(
            architecture=architecture,
            model_name=f"surrogate_{self.victim_model.model_name}_{architecture}",
            path=path,
            dataset=self.victim_model.dataset.name,
            data_subset_percent=self.victim_model.data_subset_percent,
            data_idx=1,
            gpu=gpu,
            save_model=save_model,
        )
        self.model = self.constructModel(pretrained=self.pretrained)
        if load_path is not None:
            self.loadModel(load_path)
            # this causes stored parameters to overwrite new ones
            self.config.update(self.loadConfig(self.path))
            # update with the new device
            self.config["device"] = str(self.device)
            self.epochs_trained = self.config["epochs_trained"]
            self.pretrained = self.config["pretrained"]
        self.saveConfig(
            {
                "victim_model_path": str(victim_model_path),
                "arch_pred_model_name": arch_pred_model_name,
                "arch_confidence": self.arch_confidence,
                "pretrained": self.pretrained,
            }
        )

    @staticmethod
    def load(model_path: str, gpu: int = -1):
        """
        model_path is a path to a surrogate models checkpoint,
        they are stored under {victim_model_path}/surrogate_{time}/checkpoint.pt
        """
        vict_model_path = (
            Path(model_path).parent.parent / VictimModelManager.MODEL_FILENAME
        )
        load_folder = Path(model_path).parent
        conf = ModelManagerBase.loadConfig(load_folder)
        surrogate_manager = SurrogateModelManager(
            victim_model_path=vict_model_path,
            architecture=conf["architecture"],
            arch_conf=conf,
            arch_pred_model_name=conf["arch_pred_model_name"],
            pretrained=False,
            load_path=model_path,
            gpu=gpu,
        )
        print(f"Loaded surrogate model\n{model_path}\n")
        return surrogate_manager

    def runEpoch(
        self,
        train: bool,
        epoch: int,
        optim: torch.optim.Optimizer,
        loss_fn: Callable,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        debug: int = None,
    ) -> tuple[int]:
        """
        Run a single epoch.
        Uses L1 loss between vitim model predictions and surrogate model predictions.
        Accuracy is still computed on the original validation set.
        """

        self.model.eval()
        prefix = "val"
        dl = self.dataset.val_dl
        if train:
            self.model.train()
            prefix = "train"
            dl = self.dataset.train_dl
        train_loss = torch.nn.L1Loss()

        total_loss = OnlineStats()
        acc1 = OnlineStats()
        acc5 = OnlineStats()
        step_size = OnlineStats()
        step_size.add(optim.param_groups[0]["lr"])

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(
            f"{prefix.capitalize()} Epoch {epoch if train else '1'}/{self.epochs if train else '1'}"
        )

        with torch.set_grad_enabled(train):
            for i, (x, y) in enumerate(epoch_iter, start=1):
                if debug and i > debug:
                    break
                x, y = x.to(self.device), y.to(self.device)
                victim_yhat = self.victim_model.model(x)
                victim_yhat = torch.autograd.Variable(victim_yhat, requires_grad=False)
                yhat = self.model(x)
                if train:
                    loss = train_loss(yhat, victim_yhat)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                else:
                    loss = loss_fn(yhat, y)

                c1, c5 = correct(yhat, y, (1, 5))
                total_loss.add(loss.item() / len(x))
                acc1.add(c1 / len(x))
                acc5.add(c5 / len(x))

                epoch_iter.set_postfix(
                    loss=total_loss.mean,
                    top1=acc1.mean,
                    top5=acc5.mean,
                    step_size=step_size.mean,
                )

        loss = total_loss.mean
        top1 = acc1.mean
        top5 = acc5.mean

        if train and debug is None:
            lr_scheduler.step(loss)

        # commented out because we want the training loss to be the loss between
        # victim and surrogate model predictions.
        # note that top1 and top5 accuracy don't mean much for the training epochs.
        #     # get actual train accuracy/loss after weights update
        #     top1, top5, loss = accuracy(
        #         model=self.model,
        #         dataloader=self.dataset.train_acc_dl,
        #         loss_func=loss_fn,
        #         topk=(1, 5),
        #     )

        return loss, top1, top5

    def transferAttackPGD(
        self,
        eps: float,
        step_size: float,
        iterations: int,
        norm=np.inf,
        train_data: bool = False,
        debug: int = None,
    ):
        """
        Run a transfer attack, generating adversarial inputs on surrogate model and applying them
        to the victim model.
        Code adapted from cleverhans tutorial
        https://github.com/cleverhans-lab/cleverhans/blob/master/tutorials/torch/cifar10_tutorial.py
        """
        topk = (1, 5)

        since = time.time()
        self.model.eval()
        self.model.to(self.device)
        self.victim_model.model.to(self.device)
        self.victim_model.model.eval()

        data = "train"
        dl = self.dataset.train_dl
        if not train_data:
            data = "val"
            dl = self.dataset.val_dl

        results = {
            "inputs_tested": 0,
            "both_correct1": 0,
            "both_correct5": 0,
            "surrogate_correct1": 0,
            "surrogate_correct5": 0,
            "victim_correct1": 0,
            "victim_correct5": 0,
        }

        surrogate_acc1 = OnlineStats()
        surrogate_acc5 = OnlineStats()
        victim_acc1 = OnlineStats()
        victim_acc5 = OnlineStats()

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(f"PGD Transfer attack on {data} dataset")

        for i, (x, y) in enumerate(epoch_iter, start=1):
            x, y = x.to(self.device), y.to(self.device)

            # generate adversarial examples using the surrogate model
            x_adv_surrogate = self.runPGD(x, eps, step_size, iterations, norm)
            # get predictions from surrogate and victim model
            y_pred_surrogate = self.model(x_adv_surrogate)
            y_pred_victim = self.victim_model.model(x_adv_surrogate)

            results["inputs_tested"] += y.size(0)

            adv_c1_surrogate, adv_c5_surrogate = correct(y_pred_surrogate, y, topk)
            results["surrogate_correct1"] += adv_c1_surrogate
            results["surrogate_correct5"] += adv_c5_surrogate
            surrogate_acc1.add(adv_c1_surrogate / dl.batch_size)
            surrogate_acc5.add(adv_c5_surrogate / dl.batch_size)

            adv_c1_victim, adv_c5_victim = correct(y_pred_victim, y, topk)
            results["victim_correct1"] += adv_c1_victim
            results["victim_correct5"] += adv_c5_victim
            victim_acc1.add(adv_c1_victim / dl.batch_size)
            victim_acc5.add(adv_c5_victim / dl.batch_size)

            both_correct1, both_correct5 = both_correct(
                y_pred_surrogate, y_pred_victim, y, topk
            )
            results["both_correct1"] += both_correct1
            results["both_correct5"] += both_correct5

            epoch_iter.set_postfix(
                surrogate_acc1=surrogate_acc1.mean,
                surrogate_acc5=surrogate_acc5.mean,
                victim_acc1=victim_acc1.mean,
                victim_acc5=victim_acc5.mean,
            )

            if debug is not None and i == debug:
                break

        results["both_correct1"] = results["both_correct1"] / results["inputs_tested"]
        results["both_correct5"] = results["both_correct5"] / results["inputs_tested"]

        results["surrogate_correct1"] = (
            results["surrogate_correct1"] / results["inputs_tested"]
        )
        results["surrogate_correct5"] = (
            results["surrogate_correct5"] / results["inputs_tested"]
        )

        results["victim_correct1"] = (
            results["victim_correct1"] / results["inputs_tested"]
        )
        results["victim_correct5"] = (
            results["victim_correct5"] / results["inputs_tested"]
        )

        results["transfer_runtime"] = time.time() - since
        results["parameters"] = {
            "eps": eps,
            "step_size": step_size,
            "iterations": iterations,
            "data": data,
        }

        print(json.dumps(results, indent=4))
        if debug is None:
            if "transfer_results" not in self.config:
                self.config["transfer_results"] = {f"{data}_results": results}
            else:
                self.config["transfer_results"][f"{data}_results"] = results
            self.saveConfig()
        return

    def loadVictim(self, victim_model_path: str, gpu: int):
        victim_folder = Path(victim_model_path).parent
        if victim_folder.name == PruneModelManager.FOLDER_NAME:
            return PruneModelManager.load(model_path=victim_model_path, gpu=gpu)
        if victim_folder.name == QuantizedModelManager.FOLDER_NAME:
            return QuantizedModelManager.load(model_path=victim_model_path, gpu=gpu)
        return VictimModelManager.load(model_path=victim_model_path, gpu=gpu)


def trainOneVictim(
    model_arch, epochs=150, gpu: int = -1, debug: int = None, save_model: bool = True
) -> VictimModelManager:
    a = VictimModelManager(
        architecture=model_arch,
        dataset="cifar10",
        model_name=model_arch,
        gpu=gpu,
        save_model=save_model,
    )
    a.trainModel(num_epochs=epochs, debug=debug)
    return a


def continueVictimTrain(
    vict_path: Path, epochs: int = 1, gpu: int = -1, debug: int = None
):
    manager = VictimModelManager.load(model_path=vict_path, gpu=gpu)
    manager.trainModel(num_epochs=epochs, debug=debug, replace=True)


def trainVictimModels(
    epochs=150,
    gpu: int = -1,
    reverse: bool = False,
    debug: int = None,
    repeat: bool = False,
    models: List[str] = None,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = Path.cwd() / f"train_progress_{timestamp}.txt"
    f = open(file_path, "w")

    if models is None:
        models = all_models
    if reverse:
        models.reverse()
    start = time.time()

    for i, model in enumerate(models):
        iter_start = time.time()
        if debug and i > debug:
            break
        model_arch_folder = Path.cwd() / "models" / model
        if not repeat and model_arch_folder.exists():
            continue
        try:
            manager = trainOneVictim(model, epochs=epochs, gpu=gpu, debug=debug)
            f.write(f"{model} success\n")
            config.EMAIL.email_update(
                start=start,
                iter_start=iter_start,
                iter=i,
                total_iters=len(models),
                subject=f"Victim {model} Finished Training",
                params=manager.config,
            )
        except Exception as e:
            print(e)
            f.write(
                f"\n\n{model} failed, error\n{e}\ntraceback:\n{traceback.format_exc()}\n\n"
            )
            config.EMAIL.email(
                f"Failed While Training {model}", f"{traceback.format_exc()}"
            )
    f.close()


def profileAllVictimModels(
    gpu: int = 0,
    prefix: str = None,
    nvprof_args: dict = {},
    count: int = 1,
    add: bool = False,
):
    """Victim models must be trained already."""
    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        vict_manager = VictimModelManager.load(model_path=vict_path, gpu=gpu)
        assert vict_manager.config["epochs_trained"] > 0
        if vict_manager.isProfiled() and not add:
            print(f"{vict_manager.model_name} is already profiled, skipping...")
            continue
        for i in range(count):
            vict_manager.runNVProf(**nvprof_args)


def trainSurrogateModels(
    predict: bool = True,
    arch_pred_model_type: str = "nn",
    model_paths: List[str] = None,
    epochs=50,
    gpu=0,
    reverse=False,
    debug=None,
    save_model: bool = True,
    patience: int = 5,
    df=None,
    average_profiles: bool = False,
    filters: dict = None,
):
    """Victim models must be trained and profiled already."""
    if model_paths is None:
        model_paths = VictimModelManager.getModelPaths()
    if reverse:
        model_paths.reverse()
    start = time.time()

    for i, victim_path in enumerate(model_paths):
        iter_start = time.time()
        vict_manager = VictimModelManager.load(victim_path)
        vict_name = Path(victim_path).parent.name
        if predict:
            arch, conf, model = vict_manager.predictVictimArch(
                model=get_arch_pred_model(model_type=arch_pred_model_type, df=df),
                average=average_profiles,
                filters=filters,
            )
        else:
            print(
                f"Warning, predict is False, not predicting for model {vict_manager.path}"
            )
            arch = vict_manager.architecture
            arch_pred_model_name = None
            conf = 0.0
        try:
            surrogate_model = SurrogateModelManager(
                victim_model_path=victim_path,
                architecture=arch,
                arch_conf=conf,
                arch_pred_model_name=arch_pred_model_name,
                gpu=gpu,
                save_model=save_model,
            )
            surrogate_model.trainModel(
                num_epochs=epochs, patience=patience, debug=debug
            )
            config.EMAIL.email_update(
                start=start,
                iter_start=iter_start,
                iter=i,
                total_iters=len(model_paths),
                subject=f"Surrogate Model for Victim {vict_name} Finished Training",
                params=surrogate_model.config,
            )
        except Exception as e:
            print(e)
            config.EMAIL.email(
                f"Failed Training Surrogate model for victim Model {vict_name}",
                f"{traceback.format_exc()}",
            )


def runTransferSurrogateModels(
    prefix: str = None,
    gpu=0,
    eps=0.031372549,
    step_size=0.0078431,
    iterations=10,
    train_data: bool = True,
    debug: int = None,
):
    """Both surrogate and victim models must be trained already."""
    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        surrogate_manager = SurrogateModelManager.load(model_path=vict_path, gpu=gpu)
        surrogate_manager.transferAttackPGD(
            eps=eps,
            step_size=step_size,
            iterations=iterations,
            train_data=train_data,
            debug=debug,
        )


def quantizeVictimModels(save: bool = True, prefix: str = None):
    """Victim models must be trained already"""
    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        arch = vict_path.parent.parent.name
        if arch in quantized_models:
            print(f"Quantizing {arch}...")
            QuantizedModelManager(victim_model_path=vict_path, save_model=save)


def profileAllQuantizedModels(
    gpu: int = 0,
    prefix: str = None,
    nvprof_args: dict = {},
    count: int = 1,
    add: bool = False,
):
    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        quant_path = (
            vict_path.parent
            / QuantizedModelManager.FOLDER_NAME
            / QuantizedModelManager.MODEL_FILENAME
        )
        if quant_path.exists():
            quant_manager = QuantizedModelManager.load(model_path=quant_path, gpu=gpu)
            if quant_manager.isProfiled() and not add:
                print(f"{quant_manager.model_name} is already profiled, skipping...")
                continue
            for i in range(count):
                quant_manager.runNVProf(**nvprof_args)


def pruneOneVictim(
    vict_path: Path,
    ratio: float = 0.5,
    finetune_epochs: int = 20,
    gpu: int = -1,
    save: bool = True,
):
    """Victim models must be trained already"""
    prune_manager = PruneModelManager(
        victim_model_path=vict_path,
        ratio=ratio,
        finetune_epochs=finetune_epochs,
        gpu=gpu,
        save_model=save,
    )


def pruneVictimModels(
    prefix: str = None,
    ratio: float = 0.5,
    finetune_epochs: int = 20,
    gpu: int = -1,
    save: bool = True,
):
    """Victim models must be trained already"""
    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        pruneOneVictim(
            vict_path=vict_path,
            ratio=ratio,
            finetune_epochs=finetune_epochs,
            gpu=gpu,
            save=save,
        )


def profileAllPrunedModels(
    gpu: int = 0,
    prefix: str = None,
    nvprof_args: dict = {},
    count: int = 1,
    add: bool = False,
):
    """Pruned models must be trained already."""
    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        prune_path = (
            vict_path.parent
            / PruneModelManager.FOLDER_NAME
            / PruneModelManager.MODEL_FILENAME
        )
        if prune_path.exists():
            prune_manager = PruneModelManager.load(model_path=prune_path, gpu=gpu)
            assert prune_manager.config["epochs_trained"] > 0
            if prune_manager.isProfiled() and not add:
                print(f"{prune_manager.model_name} is already profiled, skipping...")
                continue
            for i in range(count):
                prune_manager.runNVProf(**nvprof_args)


def loadProfilesToFolder(
    prefix: str = "models",
    folder_name: str = "victim_profiles",
    replace: bool = False,
    filters: dict = None,
    all: bool = True,
):
    """
    For every victim model, loads all the profiles into cwd/prefix/name/
    which is organized by model folder
    folder_name: results will be stored to cwd/folder_name
    Additionally creates a config json file where the keys are the paths to the profiles
    and the values are dicts of information about the profile such as path to actual profile,
    actual model architecture and architecture family, and model name.
    filters: a dict and each argument in the dict must match
        the argument from the config file associated with a profile.
        to get a profile by name, can specify {"profile_number": "2181935"}
    all: if true, loads all the profiles, else loads one per victim model
    """
    config_name = "config.json"
    all_config = {}

    folder = Path.cwd() / folder_name
    if folder.exists():
        if not replace:
            print(
                f"loadProfilesToFolder: folder already exists and replace is false, returning"
            )
            return
            # raise FileExistsError
        shutil.rmtree(folder)
    folder.mkdir(exist_ok=True, parents=True)

    file_count = 0

    vict_model_paths = VictimModelManager.getModelPaths(prefix=prefix)
    print(f"All model paths: {vict_model_paths}")
    for vict_path in vict_model_paths:
        print(f"Getting profiles for {vict_path.parent.name}...")
        manager = VictimModelManager.load(vict_path)
        profiles = manager.getAllProfiles(filters=filters)
        if not all:
            profiles = [manager.getProfile(filters=filters)]
        for profile_path, config in profiles:
            config["model"] = manager.architecture
            config["model_path"] = str(manager.path)
            config["manager_name"] = manager.model_name
            config["model_family"] = name_to_family[manager.architecture]
            new_name = f"profile_{manager.architecture}_{file_count}.csv"
            new_path = folder / new_name
            shutil.copy(profile_path, new_path)
            file_count += 1
            all_config[str(new_name)] = config
            print(f"\tSaved Profile {profile_path.name} to {new_path}")

    # save config file
    config_path = folder / config_name
    with open(config_path, "w") as f:
        json.dump(all_config, f, indent=4)


def loadPrunedProfilesToFolder(
    prefix: str = "models",
    folder_name: str = "victim_profiles_pruned",
    replace: bool = False,
    filters: dict = None,
    all: bool = True,
):
    """
    Same as loadPrunedProfilesToFolder, but for pruned models
    """
    config_name = "config.json"
    all_config = {}

    folder = Path.cwd() / folder_name
    if folder.exists():
        if not replace:
            print(
                f"loadProfilesToFolder: folder already exists and replace is false, returning"
            )
            return
            # raise FileExistsError
        shutil.rmtree(folder)
    folder.mkdir(exist_ok=True, parents=True)

    file_count = 0

    for vict_path in VictimModelManager.getModelPaths(prefix=prefix):
        prune_path = (
            vict_path.parent
            / PruneModelManager.FOLDER_NAME
            / PruneModelManager.MODEL_FILENAME
        )
        if prune_path.exists():
            manager = PruneModelManager.load(model_path=prune_path)
            assert manager.config["epochs_trained"] > 0

            profiles = manager.getAllProfiles(filters=filters)
            if not all:
                profiles = [manager.getProfile(filters=filters)]
            for profile_path, config in profiles:
                config["model"] = manager.architecture
                config["model_path"] = str(manager.path)
                config["manager_name"] = manager.model_name
                config["model_family"] = name_to_family[manager.architecture]
                new_name = f"profile_{manager.architecture}_{file_count}.csv"
                new_path = folder / new_name
                shutil.copy(profile_path, new_path)
                file_count += 1
                all_config[str(new_name)] = config
                print(f"\tSaved Profile {profile_path.name} to {new_path}")

    # save config file
    config_path = folder / config_name
    with open(config_path, "w") as f:
        json.dump(all_config, f, indent=4)


def predictVictimArchs(
    model: ArchPredBase,
    folder: Path,
    name: str = "predictions",
    save: bool = True,
    topk=5,
    verbose: bool = True,
):
    """Iterates through the profiles in <folder> which was generated
    by loadProfilesToFolder(), the architecture of each, and storing
    a report in a json file called <name>
    """
    assert folder.exists()

    predictions = {}

    config_path = latest_file(folder, pattern="*.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    total_tested = 0
    total_correctk = {k: 0 for k in range(1, topk + 1)}
    family_correct = 0
    for profile_name in config:
        profile_path = folder / profile_name
        profile_features = parse_one_profile(
            profile_path, gpu=config[profile_name]["gpu"]
        )
        true_arch = config[profile_name]["model"]
        true_family = config[profile_name]["model_family"]

        preds = model.topKConf(profile_features, k=topk)
        total_tested += 1
        for k in range(1, topk + 1):
            top_k_preds = preds[:k]
            correct = true_arch in [x[0] for x in top_k_preds]
            if correct:
                total_correctk[k] += 1
            if k == 1:
                pred_arch, conf = top_k_preds[0]
                # print(
                #     f"Predicted architecture for victim model {config[profile_name]['manager_name']} is {arch} with {conf * 100}% confidence."
                # )
                predictions[profile_name] = {
                    "pred_arch": pred_arch,
                    "conf": conf,
                    "true_arch": true_arch,
                    "true_family": true_family,
                }
                predictions[profile_name]["correct"] = correct
                predictions[profile_name]["family_correct"] = False
                if name_to_family[pred_arch] == true_family:
                    predictions[profile_name]["family_correct"] = True
                    family_correct += 1

        predictions[profile_name]["topk_labels"] = [x[0] for x in top_k_preds]
        predictions[profile_name]["topk_conf"] = [x[1] for x in top_k_preds]

    predictions["total_tested"] = total_tested
    predictions["total_correctk"] = {k: total_correctk[k] for k in total_correctk}
    predictions["family_correct"] = family_correct
    predictions["accuracy_k"] = {
        k: total_correctk[k] / total_tested for k in total_correctk
    }
    predictions["family_accuracy"] = family_correct / total_tested

    if verbose:
        print(json.dumps(predictions, indent=4))

    if save:
        report_path = folder / f"{name}.json"
        with open(report_path, "w") as f:
            json.dump(predictions, f, indent=4)
    return predictions


def getVictimSurrogateModels(args: dict = {}) -> List[Tuple[VictimModelManager, SurrogateModelManager]]:
    
    def validManager(victim_path: Path, args: dict = {}) -> List[Tuple[VictimModelManager, SurrogateModelManager]]:
        """
        Given a path to a victim model manger object, determine if 
        its configuration matches the provided args and if it has a surrogate
        model that matches the provided args.

        Returns [(path to victim, path to surrogate)] if config matches
        and [] if not.
        """
        manager = VictimModelManager.load(victim_path)
        # check victim
        for arg in args:
            if manager.config[arg] != args[arg]:
                return []
        # check surrogate
        surrogate_paths = manager.getSurrogateModelPaths()
        for surrogate_path in surrogate_paths:
            surrogate_manager = SurrogateModelManager.load(surrogate_path)
            for arg in args:
                if surrogate_manager.config[arg] != args[arg]:
                    break
                return [(victim_path, surrogate_path)]
        return []

    victim_paths = VictimModelManager.getModelPaths()
    result = []
    for vict_path in victim_paths:
        result += validManager(victim_path=vict_path, args=args)
    return result


if __name__ == "__main__":
    ans = input(
        "You are running the model manager file.  Enter yes to continue, anything else to exit."
    )
    if not ans.lower() == "yes":
        exit(0)

    profileAllPrunedModels()
    # arch = "alexnet"
    # vict_paths = VictimModelManager.getModelPaths()
    # arch_path = [x for x in vict_paths if x.find(arch) >= 0][0]
    # manager = QuantizedModelManager(arch_path)

    # quantizeVictimModels()
    # pruneVictimModels(gpu=0)
    # trainAllVictimModels(1, debug=2, reverse=True)
    # profileAllQuantizedModels()
    # profileAllVictimModels()
    # trainSurrogateModels(reverse=False, gpu=-1)
    # runTransferSurrogateModels(gpu=-1)
    # trainOneVictim("alexnet")
    # trainVictimModels(
    #     gpu=0,
    #     models = ['squeezenet1_0', 'squeezenet1_1']
    # )
    # time.sleep(100)
    # profileAllVictimModels(add=True)

    # trainOneVictim(model_arch="mobilenet_v2", epochs=1, debug=1, save_model=False)

    # path = [x for x in VictimModelManager.getModelPaths() if str(x).find("mobilenet_v2") >= 0][0]
    # quant_manager = QuantizedModelManager(victim_model_path=path, save_model=False)

    # continueVictimTrain(path, debug=1)
    # surrogate_manager = SurrogateModelManager(victim_model_path=path, save_model=True)
    # surrogate_manager.trainModel(num_epochs=1, debug=1)
    # surrogate_manager = SurrogateModelManager.load(path)
    # surrogate_manager.trainModel(num_epochs=2, debug=2, replace=True)

    # surrogate_manager.transferAttackPGD(eps=8/255, step_size=2/255, iterations=10, debug=1)
    # loadProfilesToFolder(all=False, replace=True)
    # loadProfilesToFolder(folder_name="victim_profiles_tesla", filters={"gpu_type": "tesla_t4"}, all=False)
    # predictVictimArchs()
    # trainSurrogateModels(predict=False)
    exit(0)
