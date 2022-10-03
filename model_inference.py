"""
Runs N inferences on a model A (maybe pretrained) on GPU K with inputs X.
This file is turned into an executable and profiling is enabled while running the executable.
"""
import argparse
import torch

from construct_input import construct_input
from get_model import get_model

parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, default="resnet", required=False)
parser.add_argument("-n", type=int, default=10, required=False,
                    help="number of inferences")
parser.add_argument("-gpu", type=int, default=-1, required=False,
                    help="-1 for cpu, else number of gpu")
parser.add_argument("-input", type=str, default="random",
                    help="Input type to pass to model. See construct_inputs.py")
parser.add_argument("-seed", type=int, default=42, help="Random seed for random inputs.")
parser.add_argument("-pretrained", action='store_true', help="Use a pretrained model")

args = parser.parse_args()

model = get_model(args.model, pretrained=args.pretrained)

device = torch.device("cpu")
dev_name = "cpu"
if args.gpu >=0:
    device = torch.device(f"cuda:{args.gpu}")
    dev_name = f"gpu{args.gpu}"

inputs = construct_input(args.input, args.n, seed=args.seed)

model = get_model(args.model)
model.eval()
model.to(device)
inputs = inputs.to(device)

print(f"Running {args.n} inferences on {args.model} on {dev_name}...")

model(inputs)

print("Completed.")


