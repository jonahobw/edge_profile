"""
Runs an executable to generate and save profiles.

Some parameters come from command line and some from config.py.  TODO - make everything come from config.
"""

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
import traceback

import torch

import config
from utils import timer, getSystem, latest_file
from email_sender import EmailSender


def check_profile(profile_csv):
    """
    Loose check to see if nvprof failed, returns a boolean.

    Check 1: nvprof failed, will only be 2 lines in the file.
    Check 2: nvprof warnings, will be more than 3 lines at the beginning starting with '=='
    """

    with open(profile_csv, 'r') as f:
        equal_line_count = 0
        for i, line in enumerate(f):
            if line.startswith("=="):
                equal_line_count += 1
                if equal_line_count > 3:
                    return False    # check 2
            if i >= 5:
                return True
    return False    # check 1


def run_command(folder, command):
    """Runs a command which is assumed to add a new profile to <folder>.  Then validate the profile."""
    # should be a blocking call, so the latest file is valid.
    output = subprocess.run(shlex.split(command), stdout=sys.stdout)
    profile_file = latest_file(folder)
    return check_profile(profile_file), profile_file


def run_command_popen(folder, command, model_type):
    """
    DOESN'T WORK, USE run_command() INSTEAD. Uses subprocess.Popen() instead of subprocess.run() to get the process id.

    The reason this does not work is because the command run is nvprof, and nvprof starts another process
    which is the actual executable with another process id.
    """
    process = subprocess.Popen(shlex.split(command))
    process.wait()

    process_id = process.pid
    profile_file = folder / f"{model_type}{process_id}.csv"
    if not profile_file.exists():
        raise FileNotFoundError(f"File {profile_file} does not exist and cannot be validated.")
    return check_profile(profile_file), profile_file


def generateExeName(use_exe: bool) -> str:
    system = getSystem()
    executable = f"exe/{system}/{system}_inference.exe"
    if not use_exe:
        # use python file instead
        executable = "python model_inference.py"
    return executable


def email_update(start: float, model_num: int, last_finish_time: float, params: dict = {}) -> None:
    model_name = config.MODELS[model_num]
    num_models = len(config.MODELS)
    subject = f"Profiles Collected for {model_name}"
    left = num_models - model_num - 1
    done_percent = "{:.0f}".format((model_num + 1) / num_models * 100)
    mean_time = (time.time() - start) / (model_num + 1)
    estimated_time_remaining = timer(left * mean_time)
    content = (f"{left} Experiments Left, {done_percent}% Completed"
                    f"Time of last experiment: {timer(time.time() - last_finish_time)}\n"
                    f"Estimated time remaining ({left} experiments left and "
                    f"{timer(mean_time)} per experiment): "
                    f"{estimated_time_remaining}\n\n"
                    f"{json.dumps(params, indent=4)}\n")
    config.EMAIL.email(subject, content)


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1, required=False,
                        help="number of inferences per profile. default 1")
    parser.add_argument("-i", type=int, default=50, required=False,
                        help="number of profiles to run per model, default 50")
    parser.add_argument("-gpu", type=int, default=0, required=False,
                        help="-1 for cpu, else number of gpu, default 0")
    parser.add_argument("-sleep", type=int, default=10, required=False,
                        help="how long to sleep in between models in seconds, default 10")
    parser.add_argument("-input", type=str, help="Input type to pass to model. See construct_inputs.py")
    parser.add_argument("-pretrained", action='store_true', help="Use a pretrained model")
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

    gpu_name = torch.cuda.get_device_name(0).lower().replace(" ", "_")

    # create folder for these profiles
    subfolder = args.folder
    if not subfolder:
        subfolder = time.strftime("%I%M%p_%m-%d-%y", time.gmtime())

    profile_folder = Path.cwd() / "profiles" / gpu_name / subfolder
    profile_folder.mkdir(parents=True, exist_ok=True)

    # random seeds
    i_seeds = [random.randint(0, 999999) for i in range(args.i)]

    # file to execute
    executable = generateExeName(not args.noexe)

    # save arguments to json file
    file = profile_folder / "arguments.json"
    save_args = vars(args)
    save_args["executable"] = executable
    save_args["random_seed"] = i_seeds
    save_args["system"] = getSystem()
    save_args["folder"] = str(profile_folder)
    save_args["gpu_name"] = gpu_name
    with open(file, 'w') as f:
        json.dump(save_args, f, indent=4)

    try:
        for model_num, model in enumerate(config.MODELS):
            model_folder = profile_folder / model
            model_folder.mkdir(parents=True, exist_ok=True)
            log_file_prefix = model_folder / model
            start = time.time()

            for i in range(args.i):
                print(f"Profiling {model} iteration {i+1}/{args.i}")
                last_finish_time = time.time()  #used for email below

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

                if args.pretrained:
                    command += " -pretrained"

                # sometimes nvprof fails, keep trying until it succeeds.
                success, file = run_command(model_folder, command)
                retries = 0
                while not success:
                    print("\nNvprof failed, retrying ... \n")
                    time.sleep(10)
                    latest_file(model_folder).unlink()
                    success, file = run_command(model_folder, command)
                    retries += 1
                    if retries > 5:
                        print("Reached 5 retries, exiting...")
                        if args.nosave:
                            shutil.rmtree(profile_folder)
                        raise RuntimeError("Nvprof failed 5 times in a row.")

                elapsed_model_time = (time.time() - start) / 60 # in minutes
                avg_prof_time = elapsed_model_time / (i+1)
                est_time = (args.i - i + 1) * avg_prof_time
                print(
                    f"Average {str(avg_prof_time)[:4]}mins per profile on {model}, "
                    f"estimated time left {str(est_time)[:4]} mins"
                )

            
            email_update(start, model_num, last_finish_time)
            print("Allowing GPUs to cool between models ...")
            time.sleep(args.sleep)


        if args.nosave:
            shutil.rmtree(profile_folder)
    except Exception as e:
        tb = traceback.format_exc()
        config.EMAIL.email("PROGRAM CRASHED", f"{tb}\n\n{json.dumps(save_args, indent=4)}")