import sys
import subprocess
from pathlib import Path
import argparse
import shlex
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=10, required=False,
                    help="number of inferences per profile. default 10")
parser.add_argument("-i", type=int, default=100, required=False,
                    help="number of profiles to run per model, default 100")
parser.add_argument("-gpu", type=int, default=0, required=False,
                    help="-1 for cpu, else number of gpu, default 0")
parser.add_argument("-sleep", type=int, default=10, required=False,
                    help="how long to sleep in between models in seconds, default 10")
parser.add_argument("-input", type=str, help="Input type to pass to model.  Options are 'random', '0', or '1'.")
parser.add_argument("-seed", type=int, default=-1,
                    help="If random inputs are specified and this seed is given: "
                         "will generate the same inputs for every profile. "
                         "Example: profile1: modelA: inputsX, modelB: inputsX, "
                         "profile2: modelA: inputsX, modelB: inputsX. "
                         "If random inputs are specified and this seed is not given: "
                         "will generate different inputs for each profile of the same model, but different "
                         "models have the same inputs.  Example: profile1: modelA: inputsX, modelB: inputsX, "
                         "profile2: modelA: inputsY, modelB: inputsY."
                    )
parser.add_argument("-folder", type=str, default=None,
                    help="Name of subfolder under cwd/profiles/ to save these profiles to.  Default is the date and time.")

args = parser.parse_args()

if args.input not in ["random", "0", "1"]:
    raise ValueError(f"Provided -input argument {args.input} but valid options are 'random', '0', or '1'.")

subfolder = args.folder
if not subfolder:
    subfolder = time.strftime("%m-%d-%y", time.gmtime())

profile_folder = Path.cwd() / "profiles" / subfolder
profile_folder.mkdir(parents=True, exist_ok=True)

# weighted moving average
alpha = 0.8
beta = 0.2

i_seeds = [random.randint(0, 999999) for i in range(args.i)]

for model in ['resnet', 'googlenet', 'mobilenetv3', 'vgg']:
    model_folder = profile_folder / model
    model_folder.mkdir(parents=True, exist_ok=True)
    log_file_prefix = model_folder / model
    # weighted moving average
    wma = 0

    for i in range(args.i):
        start = time.time()
        print(f"Profiling {model} iteration {i+1}/{args.i}")

        if args.seed < 0:
            # Use the seed corresponding to profile i
            # different inputs for each profile i, although each model gets the same inputs at profile i
            seed = i_seeds[i]
        else:
            # each profile uses the same inputs
            seed = args.seed

        command = f"nvprof --csv --log-file {log_file_prefix}%p.csv --system-profiling on " \
                  f"--profile-child-processes exe/linux_inference -gpu {args.gpu} -model {model} -seed {seed}"
        output = subprocess.run(shlex.split(command), stdout=sys.stdout)
        wma = alpha * wma + beta * (time.time() - start)
        est_time = ((args.i - i) * wma) / 60 # in minutes
        print(f"Average {wma}s per profile on {model}, estimated time left {est_time} mins")

    print("Allowing GPUs to cool between models ...")
    time.sleep(args.sleep)
