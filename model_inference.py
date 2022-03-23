import argparse

import torch
import torchvision.models as models

def get_model(model_arch:str):
    model_arch = model_arch.lower()
    if model_arch == 'resnet':
        return models.resnet18()
    if model_arch == 'googlenet':
        return models.googlenet()
    if model_arch == 'mobilenetv3':
        return models.mobilenet_v3_small()
    if model_arch == 'vgg':
        return models.vgg11_bn()
    raise ValueError("Model not supported")

parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, default="resnet", required=False)
parser.add_argument("-n", type=int, default=10, required=False,
                    help="number of inferences")
parser.add_argument("-gpu", type=int, default=-1, required=False,
                    help="-1 for cpu, else number of gpu")
parser.add_argument("-input", type=str, default="random",
                    help="Input type to pass to model.  Options are 'random', '0', or '1'.")
parser.add_argument("-seed", type=int, default=42, help="Random seed for random inputs.")

args = parser.parse_args()

if args.input not in ["random", "0", "1"]:
    raise ValueError(f"Provided -input argument {args.input} but valid options are 'random', '0', or '1'.")

device = torch.device("cpu")
dev_name = "cpu"
if args.gpu >=0:
    device = torch.device(f"cuda:{args.gpu}")
    dev_name = f"gpu{args.gpu}"

model = get_model(args.model)
model.eval()
model.to(device)

print(f"Running {args.n} inferences on {args.model} on {dev_name}...")

if args.input == "random":
    torch.manual_seed(args.seed)
    inputs = torch.randn(args.n, 3, 224, 224)
if args.input == "0":
    inputs = torch.zeros(args.n, 3, 224, 224)
if args.input == "1":
    inputs = torch.ones(args.n, 3, 224, 224)
inputs = inputs.to(device)

model(inputs)

print("Completed.")


