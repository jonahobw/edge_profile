import config
import torch

VALID_INPUTS = {
        "random": torch.randn,
        "0": torch.zeros,
        "1": torch.ones
    }

def construct_input(type, number, seed=None):
    if type not in VALID_INPUTS:
        raise ValueError(f"Provided input argument {type} but valid options are {list(VALID_INPUTS.keys())}.")

    if seed:
        torch.manual_seed(seed)

    inputs = {
        "random": torch.randn,
        "0": torch.zeros,
        "1": torch.ones
    }

    return inputs[type](number, config.CHANNELS, config.INPUT_SIZE, config.INPUT_SIZE)