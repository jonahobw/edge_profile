"""
Generic model training from Pytorch model zoo, used for the victim model.

Assuming that the victim model architecture has already been found/predicted,
the surrogate model can be trained using labels from the victim model.
"""
import datetime
import json
from operator import mod
from pathlib import Path
import time
from typing import Callable

import torch
from tqdm import tqdm

from get_model import get_model, all_models
from datasets import Dataset
from logger import CSVLogger
from online import OnlineStats
from accuracy import correct, accuracy
from collect_profiles import run_command, generateExeName


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
        ./models/model_architecture/{self.name}_{date_time}/

        This includes the model file, a csv documenting training, and a config file.

        Args:
            architecture (str): the exact string representation of the model architecture.
                See get_model.py.
            dataset (str): the name of the dataset all lowercase.
            model_name (str): The name of the model (don't use underscores).
            load (str, optional): If provided, should be the absolute path to the model folder,
                {cwd}/models/model_architecture/{self.name}{date_time}.  This will load the model
                stored there.
        """
        self.architecture = architecture
        self.dataset = Dataset(dataset)
        self.model_name = model_name
        self.device = torch.device("cpu")
        if gpu is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu}")
        self.gpu = -1 if not gpu else gpu
        print(f"Using device {self.device}, cuda available: {torch.cuda.is_available()}")
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
        folder_path = model_path.parent
        config = folder_path.glob("params_*")
        with open(next(config), "r") as f:
            conf = json.load(f)
        model_manager = ModelManager(conf["architecture"], conf["dataset"], conf["model_name"], load=folder_path, gpu=gpu)
        model_manager.loadModel()
        return model_manager

    def loadModel(self) -> None:
        """
        Models are stored under
        ./models/model_architecture/{self.name}{unique_string}/checkpoint.pt
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
            self.architecture, model_kwargs = {"num_classes": self.dataset.num_classes}
        )  # todo num_classes
        model.to(self.device)
        return model

    def trainModel(self, num_epochs: int, lr: float=1e-3, debug: int = None):
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

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, nesterov=True)
        loss_func = torch.nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)

        since = time.time()
        try:
            for epoch in range(1, num_epochs + 1):
                loss, acc1, acc5 = self.runEpoch(train=True, epoch=epoch, optim=optim, loss_fn=loss_func, lr_scheduler=lr_scheduler, debug=debug)
                val_loss, val_acc1, val_acc5 = self.runEpoch(train=False, epoch=epoch, optim=optim, loss_fn=loss_func, lr_scheduler=lr_scheduler, debug=debug)

                metrics = {
                    "train_loss": loss,
                    "train_acc1": acc1,
                    "train_acc5": acc5,
                    "val_loss": val_loss,
                    "val_acc1": val_acc1,
                    "val_acc5": val_acc5,
                    "lr": optim.param_groups[0]["lr"]
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
    
    def runEpoch(self, train: bool, epoch: int, optim: torch.optim.Optimizer, loss_fn: Callable, lr_scheduler: torch.optim.lr_scheduler._LRScheduler, debug:int = None) -> tuple[int]:
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

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(f"{prefix.capitalize()} Epoch {epoch if train else '1'}/{self.epochs if train else '1'}")

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
            top1, top5, loss = accuracy(model=self.model, dataloader=self.dataset.train_acc_dl, loss_func=loss_fn, topk=(1, 5))

        return loss, top1, top5

    def saveConfig(self):
        """
        Write parameters to a json file.
        """
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
    
    def runNVProf(self, use_exe: bool=True, seed: int=47, n: int=10, input: str="0"):
        profile_folder = self.path / "profiles"
        profile_folder.mkdir()
        executable = generateExeName(use_exe)
        command = f"nvprof --csv --log-file {profile_folder}profile.csv --system-profiling on " \
            f"--profile-child-processes {executable} -gpu {self.gpu} -load_path {self.path/'checkpoint.pt'}"\
            f" -seed {seed} -n {n} -input {input}"
        
        success, file = run_command(profile_folder, command)
        retries = 0
        while not success:
            print("\nNvprof failed, retrying ... \n")
            time.sleep(10)
            latest_file(model_folder).unlink()
            success, file = run_command(profile_folder, command)
            retries += 1
            if retries > 5:
                print("Reached 5 retries, exiting...")
                break
        params = {"use_exe": use_exe, "seed": seed, "n": n, "input": input, "success": success}
        with open(profile_folder / "params.json", "w") as f:
            json.dump(params, f)
        if not success:
            raise RuntimeError("Nvprof failed 5 times in a row.")

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

    def __init__(self, victim_model_arch: str, surrogate_model_predicted_arch: str):
        self.victim_model = get_model(victim_model_arch, pretrained=True)
        self.surrogate_model = get_model(
            surrogate_model_predicted_arch, pretrained=False
        )

    def normalizeInput(self, input):
        pass


# def getPreds(inputs, model, grad=False):
#     """
#     Returns predictions from the model on the inputs.
#     """
#     normalized_inputs = normalize(x)
#     with torch.set_grad_enabled(grad):
#         output = self(x_normalized)
#     return torch.squeeze(output)


# def surrogateTrainingLoss(inputs, surrogate_preds, victim_model: torch.nn.Module):
#     """
#     Given a list of predictions from the surrogate model, return the loss
#     as if the labels were the predictions from the victim model.
#     """
#     victim_preds = getPreds(inputs, victim_model)


def trainAllVictimModels(epochs=150, gpu = None, reverse=False, debug=None):
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
            a.trainModel(num_epochs = epochs, debug=debug)
            f.write(f"{model} success\n")
        except Exception as e:
            print(e)
            f.write(f"\n\n{model} failed, error\n{e}\n\n")
    f.close()

if __name__ == '__main__':
    # trainAllVictimModels(1, debug=2, reverse=True)
    p = Path.cwd() / "models" / "resnet18"
    path = next(p.glob("*")) / "checkpoint.pt"
    print(path)
    a = ModelManager.load(path)
    a.runNVProf()