Code for
# EZClone: Improving DNN Model Extraction Attack via Shape Distillation from GPU Execution Profiles

Jonah O'Brien Weiss, Tiago Alves, and Sandip Kundu

jobrienweiss@umass.edu, tiago@ime.uerj.br, kundu@umass.edu

This repo contains code to train Deep Neural Networks (DNNs), collect GPU profiles on DNNs, and train an architecture prediction model on the GPU profiles. 

## Installation and Requirements

Software Requirements:
* Linux OS
* Python 3.9
* CUDA at least v10.2 with nvprof installed (use ```nvprof --help``` to check)

Hardware Requirements:
* Nvidia GPU.  This was tested with the Quadro GTX 8000 and Tesla T4 GPUs.

Install requirements with 
```
pip3 install -r requirements.txt
```

## Directory Structure

Notable files are outlined below.

```yaml
architecture_prediction/:
  architecture_prediction.py  # training a model on a set of profiles
plots/:                       # code for generating plots
profiles/:                  
  debug_profiles/:            # example profiles
  collect_profiles.py         # automate profile collection
                              # to make a dataset for training
  construct_input.py          # the input to DNN when profiling
  data_engineering.py         # validating and massaging data
  format_profiles.py          # parser for nvprof output
  whitebox_pyprof.py          # using PyTorch's builtin profiler
exe/:
  create_exe.py               # create an executable for profiling
  model_inference.py          # target file for the executable,
                              # this is what will be profiled
datasets/:
  datasets.py                 # manages image classification datasets
  download_dataset.py         # downloads a dataset from datasets.py
dnn/:
  get_model.py                # wrapper for getting DNN models
  model_manager.py            # training, profiling, and attacking
                              # victim DNNs
  model_metrics.py            # accuracy calculations
  neural_network.py           # custom model for architecture
                              # prediction
test/:                        # test code
tensorflow/:
  create_exe.py               # creates an exe for tensorflow profiling
  tensorflow_inference.py     # target file for tensorflow executables
utils/:
  config.py                   # global parameter configuration
  email_sender.py             # configure email notification for
                              # long-running experiments
  logger.py                   # utility for logging during training
  online.py                   # calculate stats while training DNNs

# supporting files
```

## Collecting a Dataset of Profiles

1. Create an executable for your device. Run ```python create_exe.py```
2. Collect profiles.  Run ```python collect_profiles.py```
3. Validate and parse profiles.  Run ```python format_profiles.py```.  This will validate profiles and class balance and ask you for permission to remove extra or faulty profiles.  If an error occurs, you will need to run this again.

## Train an Architecture Prediction Model

The architecture prediction model maps a profile to the architecture of the DNN that generated the profile.

The following code is how to train an architecture prediction model:

```
from data_engineering import all_data
from architecture_prediction import get_arch_pred_model

data = all_data(<Path_to_profile_folder>)
arch_pred_model = get_arch_pred_model("rf", df=data)  # "rf" = random forest, "lr" = linear regression, "nn" = neural net ...
print(arch_pred_model.evaluateTest())
print(arch_pred_model.evaluateTrain())
```
