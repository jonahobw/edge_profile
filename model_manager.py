"""
Generic model training from Pytorch model zoo, used for the victim model.

Assuming that the victim model architecture has already been found/predicted,
the surrogate model can be trained using labels from the victim model.
"""
import datetime
import json
from pathlib import Path
import time
from typing import Callable, Dict, List, Tuple
import traceback

import torch
import numpy as np
from tqdm import tqdm
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from get_model import get_model, model_params, all_models
from datasets import Dataset
from logger import CSVLogger
from online import OnlineStats
from model_metrics import correct, accuracy, both_correct
from collect_profiles import run_command, generateExeName
from utils import latest_file
from format_profiles import parse_one_profile
from architecture_prediction import NNArchPred
import config

def loadModel(path: Path, model: torch.nn.Module, device: torch.device = None) -> None:
    assert path.exists(), f"Model load path \n{path}\n does not exist."
    if device is not None:
        params = torch.load(path, map_location=device)
    else:
        params = torch.load(path)
    model.load_state_dict(params, strict=False)
    model.eval()


class ModelManager:
    """
    Generic model manager class.
    Can train a model on a dataset and save/load a model.
    Functionality for passing data to a model and getting predictions.
    """

    def __init__(
        self,
        architecture: str,
        dataset: str,
        model_name: str,
        load: str = None,
        gpu: int = None,
        data_subset_percent: float = 0.5,
        idx: int = 0,
        pretrained: bool = False,
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
        # todo add option to not load dataset and use this option when profiling
        self.architecture = architecture
        self.data_subset_percent = data_subset_percent
        self.dataset = Dataset(
            dataset,
            data_subset_percent=data_subset_percent,
            idx=idx,
            resize=model_params.get(architecture, {}).get("input_size", None),
        )
        self.model_name = model_name
        self.device = torch.device("cpu")
        if gpu is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu}")
        self.gpu = -1 if gpu is None else gpu
        print(
            f"Using device {self.device}, cuda available: {torch.cuda.is_available()}"
        )
        # if loading a model from a file, don't need to load pretrained weights
        self.model = self.constructModel(pretrained=(False if load else pretrained))
        self.pretrained = pretrained
        self.trained = False
        if load:
            self.trained = True
            self.path = load
            self.loadModel()
        else:
            self.path = self.generateFolder()
        self.config = {
            "path": str(self.path),
            "architecture": self.architecture,
            "dataset": self.dataset.name,
            "model_name": self.model_name,
            "device": str(self.device),
            "data_subset_percent": data_subset_percent,
            "pretrained": pretrained
        }
        self.epochs = 0

    @staticmethod
    def load(model_path: Path, gpu=None):
        """Create a ModelManager Object from a path to a model file."""
        folder_path = Path(model_path).parent
        config = folder_path.glob("params_*")
        with open(next(config), "r") as f:
            conf = json.load(f)
        print(f"Loading {conf['architecture']} trained on {conf['dataset']}")
        model_manager = ModelManager(
            conf["architecture"],
            conf["dataset"],
            conf["model_name"],
            load=folder_path,
            gpu=gpu,
            data_subset_percent=conf["data_subset_percent"],
            pretrained=conf["pretrained"]
        )
        model_manager.config = conf
        return model_manager

    def loadModel(self, name: str = None, device: torch.device = None) -> None:
        """
        Models are stored under
        self.path/checkpoint.pt
        """
        if name is None:
            name = "checkpoint.pt"
        model_file = self.path / name
        loadModel(model_file, self.model, self.device)

    def saveModel(self) -> None:
        model_file = self.path / "checkpoint.pt"
        assert not model_file.exists()
        torch.save(self.model.state_dict(), model_file)

    def constructModel(self, pretrained: bool) -> torch.nn.Module:
        model = get_model(
            self.architecture, pretrained=pretrained, kwargs={"num_classes": self.dataset.num_classes}
        )
        model.to(self.device)
        return model

    def trainModel(self, num_epochs: int, lr: float = None, debug: int = None, patience: int = 10):
        """Trains the model using dataset self.dataset.

        Args:
            num_epochs (int): number of training epochs
            lr (float): initial learning rate.  This function decreases the learning rate
                by a factor of 0.1 when the loss fails to decrease by 1e-4 for 10 iterations.
                If not passed, will default to learning rate of model from get_model.py, and
                if there is no learning rate specified there, defaults to 0.1.

        Returns:
            Nothing, only sets the self.model class variable.
        """
        # todo add checkpoint freq
        if self.trained:
            raise ValueError

        if lr is None:
            lr = model_params.get(self.architecture, {}).get("lr", None)
            if lr is None:
                lr = 0.1

        self.epochs = num_epochs
        logger = CSVLogger(self.path, self.train_metrics)

        optim = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
        if model_params.get(self.architecture, {}).get("optim", "") == "adam":
            optim = torch.optim.Adam(self.model.parameters, lr=lr, weight_decay=1e-4)
        
        loss_func = torch.nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=patience)

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

                logger.set(timestamp=time.time() - since, epoch=epoch, **metrics)
                logger.update()

        except KeyboardInterrupt:
            print(f"\nInterrupted at epoch {epoch}. Tearing Down")

        self.model.eval()
        print("Training ended, saving model.")
        self.saveModel()
        self.config["epochs"] = num_epochs
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

    def saveConfig(self, args: dict = {}):
        """
        Write parameters to a json file.  If file exists already, then will be
        appended to/overwritten.  If args are provided, they are added to the config file.
        """
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
        with open(path, "w") as f:
            json.dump(self.config, f, indent=4)

    def generateFolder(self) -> str:
        """
        Generates the model folder as ./models/model_architecture/{self.name}_{date_time}/
        """
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_folder = (
            Path.cwd() / "models" / self.architecture / f"{self.model_name}_{time}"
        )
        assert not model_folder.exists()
        model_folder.mkdir(parents=True)
        return model_folder

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
        profile_folder = self.path / "profiles"
        profile_folder.mkdir(exist_ok=True)
        prefix = profile_folder / "profile_"
        executable = generateExeName(use_exe)
        print(f"Using executable {executable} for nvprof")
        command = (
            f"nvprof --csv --log-file {prefix}%p.csv --system-profiling on "
            f"--profile-child-processes {executable} -gpu {self.gpu} -load_path {self.path/'checkpoint.pt'}"
            f" -seed {seed} -n {n} -input {input}"
        )

        print(f"\nCommand being run:\n{command}\n\n")

        success, file = run_command(profile_folder, command)
        retries = 0
        print(f"{'Success' if success else 'Failure'} on file {file}")
        while not success:
            print("\nNvprof retrying ... \n")
            time.sleep(10)
            latest_file(profile_folder).unlink()
            success, file = run_command(profile_folder, command)
            retries += 1
            if retries > 5:
                print("Reached 5 retries, exiting...")
                break
        if not success:
            latest_file(profile_folder).unlink()
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
        profile_path = Path(conf["file"])
        return profile_path.exists()

    def getProfile(self) -> Tuple[Path, Dict]:
        """
        Return a tuple of (path to profile_{pid}.csv,
        dictionary obtained from reading params_{pid}.json).
        Note - currently uses the first profile it finds,
        filters could be implemented.
        """
        profile_folder = self.path / "profiles"
        profile_config = [x for x in profile_folder.glob("params_*")]
        assert len(profile_config) > 0
        with open(profile_config[0], "r") as f:
            conf = json.load(f)
        prof_num = conf["profile_number"]
        profile_path = profile_folder / f"profile_{prof_num}.csv"
        assert profile_path.exists()
        return profile_path, conf

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
        arch_folders = [i for i in models_folder.glob("*")]
        model_paths = []
        for arch in arch_folders:
            for model_folder in [i for i in arch.glob("*")]:
                victim_path = models_folder / arch / model_folder / "checkpoint.pt"
                if victim_path.exists():
                    model_paths.append(victim_path)
                else:
                    print(f"Warning, no model found {victim_path}")
        return model_paths


class QuantizedModelManager:
    def __init__(self, victim_model_path: Path, backend: str = 'fbgemm', load_path: Path = None, save_model: bool = True) -> None:
        self.full_prec_manager = ModelManager.load(victim_model_path)
        folder = self.full_prec_manager.path / "quantize"
        folder.mkdir(exist_ok=True, parents=True)
        self.path = folder
        self.backend = backend
        self.config = {
            "victim_model_path": str(victim_model_path),
            "backend": backend
        }
        self.model = None
        if load_path is not None:
            self.model = self.load_quantized_model(load_path)
        else:
            self.model = self.quantize()
            if save_model:
                self.save_model()
    
    def prepare_for_quantization(self, model):
        torch.backends.quantized.engine = self.backend
        model.eval()
        # Make sure that weight qconfig matches that of the serialized models
        if self.backend == 'fbgemm':
            model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_per_channel_weight_observer)
        elif self.backend == 'qnnpack':
            model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_weight_observer)

        model.fuse_model()
        torch.quantization.prepare(model, inplace=True)
        return model
    
    def quantize(self) -> torch.nn.Module:
        prepped_model = self.prepare_for_quantization(self.full_prec_manager.model)
        # calibrate model
        dl_iter = tqdm(self.full_prec_manager.dataset.train_acc_dl)
        dl_iter.set_description("Calibrating model for quantization")
        for i, (x, y) in enumerate(dl_iter):
            prepped_model(x)
        torch.quantization.convert(prepped_model, inplace=True)
        return prepped_model
    
    def save_model(self) -> None:
        save_path = self.path / f"quantized.pt"

        torch.save(
            {
                "model_state_dict": self.quantized_model.state_dict(),
            },
            save_path / f"quantized.pt",
        )
        self.saveConfig()
        return None
    
    def saveConfig(self) -> None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.path / f"params_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(self.config, f, indent=4)

    def load_quantized_model(self, path: Path) -> torch.nn.Module:
        model = self.full_prec_manager.model
        model.eval()

        torch.backends.quantized.engine = self.backend
        # Make sure that weight qconfig matches that of the serialized models
        if self.backend == 'fbgemm':
            model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_per_channel_weight_observer)
        elif self.backend == 'qnnpack':
            model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_weight_observer)

        model.fuse_model()
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)

        resume = Path(path)
        assert resume.exists(), f"Quantized model path does not exist\n{resume}"
        previous = torch.load(resume, map_location=torch.device('cpu'))
        model.load_state_dict(previous["model_state_dict"], strict=False)
        return model

    @staticmethod
    def load(quantize_folder: Path):
        vict_model_path = quantize_folder.parent / "checkpoint.pt"
        config = quantize_folder.glob("params_*")
        with open(next(config), "r") as f:
            conf = json.load(f)
        load_path = quantize_folder / "quantized.pt"
        return QuantizedModelManager(victim_model_path=vict_model_path, backend=conf["backend"], load_path=load_path)

class SurrogateModelManager(ModelManager):
    """
    Constructs the surrogate model with a paired victim model, trains using from the labels from victim
    model.
    """

    arch_model = {"nn": NNArchPred}

    def __init__(
        self,
        victim_model_path: str,
        gpu: int = None,
        arch_model: str = "nn",
        load: dict = {},
        nvprof_args: dict = {},
        pretrained: bool = False,
    ):
        """
        If load is not none, it should be a dictionary containing the model architecture,
        architecture prediction model type, architecture confidence, and path to model.
        See SurrogateModelManager.load()
        """
        self.victim_model = ModelManager.load(victim_model_path, gpu=gpu)
        self.nvprof_args = nvprof_args
        load_path = None
        if load:
            self.arch_pred_model = load["arch_pred_model"]
            architecture = load["architecture"]
            self.arch_confidence = load["arch_confidence"]
            load_path = Path(load["path"])
        else:
            self.arch_pred_model = None # will get set in self.predictVictimArch
            architecture, conf = self.predictVictimArch(arch_model)
            self.arch_confidence = conf
        super().__init__(
            architecture=architecture,
            dataset=self.victim_model.dataset.name,
            model_name=f"surrogate_{self.victim_model.model_name}_{architecture}",
            gpu=gpu,
            load=load_path,
            data_subset_percent=self.victim_model.data_subset_percent,
            idx=1,
            pretrained=pretrained,
        )
        if not load:
            self.config.update(
                {
                    "arch_pred_model": arch_model,
                    "arch_confidence": conf,
                    "nvprof_args": nvprof_args,
                }
            )

    def predictVictimArch(self, model_type: str):
        if not self.victim_model.isProfiled():
            self.victim_model.runNVProf(**self.nvprof_args)
        profile_csv, config = self.victim_model.getProfile()
        profile_features = parse_one_profile(profile_csv, gpu=config["gpu"])
        print(f"Training architecture prediction model {model_type}")
        self.arch_pred_model = self.arch_model[model_type]()
        arch, conf = self.arch_pred_model.predict(profile_features)
        print(
            f"Predicted surrogate model architecture for victim model\n{self.victim_model.path}\n is {arch} with {conf * 100}% confidence."
        )
        return arch, conf

    @staticmethod
    def load(model_path: str, gpu=None):
        """
        Given a path to a surrogate model, load into a SurrogateModelManager obj.
        Surrogate models are stored in {victim_model_path}/surrogate/checkpoint.pt
        """
        surrogate_folder = Path(model_path).parent
        victim_model_path = surrogate_folder.parent / "checkpoint.pt"
        surrogate_config = surrogate_folder.glob("params_*")
        with open(next(surrogate_config), "r") as f:
            conf = json.load(f)
        surrogate_manager = SurrogateModelManager(victim_model_path, gpu, load=conf)
        surrogate_manager.config.update(conf)
        print(f"Loaded surrogate model\n{model_path}\n")
        return surrogate_manager

    def generateFolder(self) -> str:
        folder = self.victim_model.path / "surrogate"
        folder.mkdir()
        return folder

    def trainSaveAll(self, num_epochs: int, lr: float = 1e-1, debug: int = None):
        """Wrapper around trainModel to add some more config data."""
        self.trainModel(num_epochs, lr, debug)
        self.config["victim_config"] = self.victim_model.config
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
                    loss.backward()
                    optim.step()
                    optim.zero_grad()
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

            if self.debug is not None and i == self.debug:
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
        if "transfer_results" not in self.config:
            self.config["transfer_results"] = {f"{data}_results": results}
        else:
            self.config["transfer_results"][f"{data}_results"] = results
        self.saveConfig()
        return


def trainOneVictim(model_arch, epochs=150, gpu=None, debug=None) -> ModelManager:
    a = ModelManager(model_arch, "cifar10", model_arch, gpu=gpu)
    a.trainModel(num_epochs=epochs, debug=debug)
    return a


def trainVictimModels(
    epochs=150,
    gpu=None,
    reverse=False,
    debug=None,
    repeat=False,
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


def profileAllVictimModels(gpu=0):
    models_folder = Path.cwd() / "models"
    arch_folders = [i for i in models_folder.glob("*")]
    for arch in arch_folders:
        for model_folder in [i for i in arch.glob("*")]:
            path = models_folder / arch / model_folder / "checkpoint.pt"
            model_manager = ModelManager.load(path, gpu=gpu)
            model_manager.runNVProf(False)


def trainSurrogateModels(
    nvprof_args: dict,
    model_paths: List[str] = None,
    epochs=150,
    gpu=0,
    reverse=False,
    debug=None,
):
    if model_paths is None:
        model_paths = ModelManager.getModelPaths()
    if reverse:
        model_paths.reverse()
    start = time.time()

    for i, victim_path in enumerate(model_paths):
        vict_name = Path(victim_path).parent.name
        iter_start = time.time()
        try:
            surrogate_model = SurrogateModelManager(
                victim_path, gpu=gpu, nvprof_args=nvprof_args
            )
            surrogate_model.trainSaveAll(epochs, debug=debug)
            config.EMAIL.email_update(
                start=start,
                iter_start=iter_start,
                iter=i,
                total_iters=len(model_paths),
                subject=f"Surrogate Model for Victim {vict_name} Finished Training",
                params=surrogate_model.config,
            )
        except Exception as e:
            vict_name = Path(victim_path).parent.name
            print(e)
            config.EMAIL.email(
                f"Failed Training Surrogate model for victim Model {vict_name}",
                f"{traceback.format_exc()}",
            )


def runTransferSurrogateModels(
    gpu=0, eps=0.031372549, step_size=0.0078431, iterations=10, train_data: bool = True
):
    models_folder = Path.cwd() / "models"
    arch_folders = [i for i in models_folder.glob("*")]
    for arch in arch_folders:
        for model_folder in [i for i in arch.glob("*")]:
            surrogate_path = (
                models_folder / arch / model_folder / "surrogate" / "checkpoint.pt"
            )
            if surrogate_path.exists():
                surrogate_model = SurrogateModelManager.load(surrogate_path, gpu=gpu)
                surrogate_model.transferAttackPGD(
                    eps=eps,
                    step_size=step_size,
                    iterations=iterations,
                    train_data=train_data,
                )


if __name__ == "__main__":
    # trainAllVictimModels(1, debug=2, reverse=True)
    # profileAllVictimModels()
    # trainSurrogateModels(reverse=False, gpu=-1)
    # runTransferSurrogateModels(gpu=-1)
    # trainOneVictim("alexnet")
    trainVictimModels(
        gpu=0,
        models = ['alexnet', 'resnext50_32x4d',
          'resnext101_32x8d', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
          'squeezenet1_0', 'squeezenet1_1', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3'
          ]
    )
