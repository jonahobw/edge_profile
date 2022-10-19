"""
Generic model training from Pytorch model zoo, used for the victim model.

Assuming that the victim model architecture has already been found/predicted,
the surrogate model can be trained using labels from the victim model.
"""
from ast import Raise
import datetime
import json
from operator import mod
from pathlib import Path
import time
from typing import Callable, Dict, Tuple

import torch
from tqdm import tqdm

from get_model import get_model, all_models
from datasets import Dataset
from logger import CSVLogger
from online import OnlineStats
from accuracy import correct, accuracy
from collect_profiles import run_command, generateExeName, latest_file
from format_profiles import parse_one_profile
from architecture_prediction import NNArchPred


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
        """
        self.architecture = architecture
        self.dataset = Dataset(dataset)
        self.model_name = model_name
        self.device = torch.device("cpu")
        if gpu is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu}")
        self.gpu = -1 if gpu is None else gpu
        print(
            f"Using device {self.device}, cuda available: {torch.cuda.is_available()}"
        )
        self.model = self.constructModel()
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
        )
        model_manager.config = conf
        return model_manager

    def loadModel(self) -> None:
        """
        Models are stored under
        self.path/checkpoint.pt
        """
        model_file = self.path / "checkpoint.pt"
        assert model_file.exists(), f"Model load path \n{model_file}\n does not exist."
        params = torch.load(model_file, map_location=self.device)
        self.model.load_state_dict(params, strict=False)
        self.model.eval()

    def saveModel(self) -> None:
        model_file = self.path / "checkpoint.pt"
        assert not model_file.exists()
        torch.save(self.model.state_dict(), model_file)

    def constructModel(self) -> torch.nn.Module:
        model = get_model(
            self.architecture, model_kwargs={"num_classes": self.dataset.num_classes}
        )  # todo num_classes
        model.to(self.device)
        return model

    def trainModel(self, num_epochs: int, lr: float = 1e-1, debug: int = None):
        """Trains the model using dataset self.dataset.

        Args:
            num_epochs (int): number of training epochs
            lr (float): initial learning rate.  This function decreases the learning rate
                by a factor of 0.1 when the loss fails to decrease by 1e-4 for 10 iterations

        Returns:
            Nothing, only sets the self.model class variable.
        """
        if self.trained:
            raise ValueError

        self.epochs = num_epochs
        logger = CSVLogger(self.path, self.train_metrics)

        optim = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9, nesterov=True
        )
        loss_func = torch.nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)

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
                    loss.backward()
                    optim.step()
                    optim.zero_grad()

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
        nvprof_args: dict={}
    ):
        """
        If load is not none, it should be a dictionary containing the model architecture,
        architecture prediction model type, architecture confidence, and path to model.
        """
        self.victim_model = ModelManager.load(victim_model_path, gpu=gpu)
        self.nvprof_args = nvprof_args
        load_path = None
        if load:
            self.arch_pred_model = load["arch_pred_model"]
            architecture = load["architecture"]
            self.arch_confidence = load["arch_confidence"]
            load_path = Path(load["path"]) / "checkpoint.pt"
        else:
            self.arch_pred_model = None
            architecture, conf = self.predictVictimArch(arch_model)
            self.arch_confidence = conf
        super().__init__(
            architecture=architecture,
            dataset=self.victim_model.dataset.name,
            model_name=f"surrogate_{self.victim_model.model_name}_{architecture}",
            gpu=gpu,
            load=load_path,
        )
        self.config.update({"arch_pred_model": arch_model, "arch_confidence": conf, "nvprof_args": nvprof_args})

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
        return surrogate_manager

    def generateFolder(self) -> str:
        folder = self.victim_model.path / "surrogate"
        folder.mkdir()
        return folder

    def trainSaveAll(self, num_epochs: int, lr: float = 1e-3, debug: int = None):
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

        # commented out because we want the training loss to be the loss between
        # victim and surrogate model predictions.
        # note that top1 and top5 accuracy don't mean much for the training epochs.

        # if train and debug is None:
        #     lr_scheduler.step(loss)
        #     # get actual train accuracy/loss after weights update
        #     top1, top5, loss = accuracy(
        #         model=self.model,
        #         dataloader=self.dataset.train_acc_dl,
        #         loss_func=loss_fn,
        #         topk=(1, 5),
        #     )

        return loss, top1, top5


def trainAllVictimModels(epochs=150, gpu=None, reverse=False, debug=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = Path.cwd() / f"train_progress_{timestamp}.txt"
    f = open(file_path, "w")

    models = all_models
    if reverse:
        models.reverse()

    for i, model in enumerate(models):
        if debug and i > debug:
            break
        try:
            a = ModelManager(model, "cifar10", model, gpu=gpu)
            a.trainModel(num_epochs=epochs, debug=debug)
            f.write(f"{model} success\n")
        except Exception as e:
            print(e)
            f.write(f"\n\n{model} failed, error\n{e}\n\n")
    f.close()


def profileAllVictimModels(gpu=0):
    models_folder = Path.cwd() / "models"
    arch_folders = [i for i in models_folder.glob("*")]
    for arch in arch_folders:
        for model_folder in [i for i in arch.glob("*")]:
            path = models_folder / arch / model_folder / "checkpoint.pt"
            model_manager = ModelManager.load(path, gpu=gpu)
            model_manager.runNVProf(False)


def trainSurrogateModels(epochs=150, gpu=0, reverse=False, debug=None):
    nvprof_args = {"use_exe": False}
    models_folder = Path.cwd() / "models"
    arch_folders = [i for i in models_folder.glob("*")]
    if reverse:
        arch_folders.reverse()
    for arch in arch_folders:
        for model_folder in [i for i in arch.glob("*")]:
            victim_path = models_folder / arch / model_folder / "checkpoint.pt"
            surrogate_model = SurrogateModelManager(victim_path, gpu=gpu, nvprof_args=nvprof_args)
            surrogate_model.trainSaveAll(epochs, debug=debug)


if __name__ == "__main__":
    # trainAllVictimModels(1, debug=2, reverse=True)
    # profileAllVictimModels()
    trainSurrogateModels(reverse=False, gpu=0)
