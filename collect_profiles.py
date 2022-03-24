import json
import os
import shutil
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
                    help="Name of subfolder under cwd/profiles/ to save these profiles to.  "
                         "Default is the date and time.")
parser.add_argument("-noexe", action='store_true', help="If provided, will run the inference using the python file"
                                                        " rather than the executable file. This is faster but "
                                                        " is not the type of attack vector considered, so it "
                                                        " should only be used for debugging.")
parser.add_argument("-nosave", action='store_true', help="do not save any traces, just debug.")

args = parser.parse_args()

if args.input not in ["random", "0", "1"]:
    raise ValueError(f"Provided -input argument {args.input} but valid options are 'random', '0', or '1'.")

# create folder for these profiles
subfolder = args.folder
if not subfolder:
    subfolder = time.strftime("%I%M%p_%m-%d-%y", time.gmtime())

profile_folder = Path.cwd() / "profiles" / subfolder
profile_folder.mkdir(parents=True, exist_ok=True)

# weighted moving average
alpha = 0.8
beta = 0.2

# random seeds
i_seeds = [random.randint(0, 999999) for i in range(args.i)]

# file to execute
if os.name != 'nt':
    system = "linux"
else:
    system = "windows"
executable = f"exe/{system}/{system}_inference.exe"
if args.noexe:
    # use python file instead
    executable = "model_inference.py"

# save arguments to json file
file = profile_folder / "arguments.json"
save_args = vars(args)
save_args["executable"] = executable
save_args["random_seed"] = i_seeds
save_args["system"] = system
save_args["folder"] = str(profile_folder)
with open(file, 'w') as f:
    json.dump(save_args, f, indent=4)

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
                  f"--profile-child-processes {executable} -gpu {args.gpu} -model {model} -seed {seed} " \
                  f"-n {args.n} -input {args.input}"
        output = subprocess.run(shlex.split(command), stdout=sys.stdout)
        wma = alpha * wma + beta * (time.time() - start)
        est_time = ((args.i - i) * wma) / 60 # in minutes
        print(f"Average {wma}s per profile on {model}, estimated time left {est_time} mins")

    print("Allowing GPUs to cool between models ...")
    time.sleep(args.sleep)


if args.nosave:
    shutil.rmtree(profile_folder)