"""
Generic model training from Pytorch model zoo, used for the victim model.

Assuming that the victim model architecture has already been found/predicted,
the surrogate model can be trained using labels from the victim model.
"""
from typing import Tuple
import torch
from get_model import get_model

class ModelManager:
    """
    Generic model manager class.  
    Can train a model on a dataset and save/load a model.
    Functionality for passing data to a model and getting predictions.
    """
    def __init__(self, architecture: str, dataset: str, model_name: str, load: bool=False):
        """
        Models files are stored in a folder
        ./models/model_architecture/{self.name}{date_time}/

        This includes the model file and a configuration file.

        Args:
            architecture (str): the exact string representation of the model architecture.
                See get_model.py.
            dataset (str): the exact string representation of the dataset.  See dataset.py
            model_name (str): The name of the model (don't use underscores)
            load (bool, optional): _description_. Defaults to False.
        """
        self.architecture = architecture
        self.dataset = dataset
        self.model_name = model_name
        self.model = None
        self.load = load
        self.trained = not load
        self.path = self.generateFolder()
    
    def loadModel(self):
        """
        Models are stored under 
        ./models/model_architecture/{self.name}{unique_string}/checkpoint{training_epochs}.pt
        """
    
    def trainModel(self, num_epochs: int, lr: float):
        """Trains the model using dataset self.dataset.

        Args:
            num_epochs (int): _description_
            lr (float): _description_

        Returns:
            Nothing, only sets the self.model class variable.
        """


class SurrogateModelManager(ModelManager):
    """
    Constructs the surrogate model with a paired victim model, trains using from the labels from victim
    model.
    """
    def __init__(self, victim_model_arch: str, surrogate_model_predicted_arch: str):
        self.victim_model = get_model(victim_model_arch, pretrained=True)
        self.surrogate_model = get_model(surrogate_model_predicted_arch, pretrained=False)

    def normalizeInput(self, input)

def getPreds(inputs, model, grad=False):
    """
    Returns predictions from the model on the inputs.
    """
    normalized_inputs = normalize(x)
    with torch.set_grad_enabled(grad):
        output = self(x_normalized)
    return torch.squeeze(output)

def surrogateTrainingLoss(inputs, surrogate_preds, victim_model: torch.nn.Module):
    """
    Given a list of predictions from the surrogate model, return the loss
    as if the labels were the predictions from the victim model.
    """
    victim_preds = getPreds(inputs, victim_model)

