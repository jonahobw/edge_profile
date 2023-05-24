"""
Aggregates nvprof profile data into one csv file.
Also has capability to validate nvprof success and class balance.
"""

import json
from typing import Dict, List, Mapping, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from get_model import name_to_family
import argparse


def check_profile(profile_csv):
    """
    Loose check to see if nvprof failed, returns a boolean.

    Check 1: nvprof failed, will only be 2 lines in the file.
    Check 2: nvprof warnings, will be more than 3 lines at the beginning starting with '=='
    """

    with open(profile_csv, "r") as f:
        equal_line_count = 0
        for i, line in enumerate(f):
            if line.startswith("=="):
                equal_line_count += 1
                if equal_line_count > 3:
                    print(f"nvprof failed for profile {profile_csv}: 3 beginning lines start with ==")
                    return False  # check 2
            if i >= 5:
                return True
        print(f"nvprof failed for profile {profile_csv}, not enough lines in the file.")
    return False  # check 1


def check_for_nans(profile_csv, gpu=0) -> list[str]:
    """Return a list of columns with NaNs in the supplied profile."""

    # aggregate gpu data first:
    skiprows = 3
    with open(profile_csv) as f:
        for i, line in enumerate(f):
            if line == "\n":
                break
    nrows = i - skiprows - 1

    df = pd.read_csv(profile_csv, header=0, skiprows=skiprows, nrows=nrows)
    df = df.drop(0)
    null_cols = df.columns[df.isna().any()].tolist()

    # system data
    skiprows = i + 2
    df = pd.read_csv(profile_csv, header=0, skiprows=skiprows, nrows=5 * (gpu + 1))
    # filter out rows with '=='
    df = df[df["Unnamed: 0"].str.contains("==") == False]
    null_system_cols = df.columns[df.isna().any()].tolist()

    null_cols.extend(null_system_cols)

    if len(null_cols) > 0:
        print(f"nvprof failed for profile {profile_csv}, null values in columns {null_cols}")

    return null_cols


def validProfile(profile_csv, gpu=0) -> bool:
    return check_profile(profile_csv) and len(check_for_nans(profile_csv, gpu)) == 0


def add_model_family(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'model_family' column to the dataframe based on the 'model' column.

    Example:
    wide_resnet50   -> resnet
    vgg13_bn        -> vgg
    densenet201     -> densenet

    :param df: the dataframe to add the column to.  Must have a column called 'model'
    :return: the original dataframe with the new column.
    """

    def label_family(row):
        return name_to_family[row["model"]]

    df["model_family"] = df.apply(label_family, axis=1)
    return df


def parse_one_aggregate_profile(csv_file=None, example=False, nrows=None, skiprows=3, gpu_activities_only: bool = False, api_calls_only: bool = False):
    """
    Takes a csv generated by nvprof with aggregate mode on, and returns a pandas series where the
    axis labels are the gpu activity/api call name and the metrics associated with it (aggregate time,
    percent time, # of calls, avg time, min time, max time).

    :param csv_file: the csv file
    :param example: boolean to use an example profile.  If true, csv_file is not needed.
    :param nrows: number of rows in the csv, for aggregate profiles.
    :param skiprows: number of rows to skip at the beginning of the csv, for aggregate profiles, default 3
    :return: a pandas series
    """

    if nrows is None:
        try:
            with open(csv_file) as f:
                for i, line in enumerate(f):
                    if line == "\n":
                        break
        except TimeoutError:
            raise TimeoutError(f"TimeoutError on file {csv_file}")
        nrows = i - skiprows

    if example:
        csv_file = Path.cwd() / "debug_profiles" / "resnet" / "resnet750691.csv"
    elif not csv_file:
        raise ValueError("csv_file must be provided if example is false.")

    if not csv_file.exists():
        raise ValueError(f"File {csv_file} does not exist")

    gpu_columns = {
        "Type": "type",
        "Time": "time_ms",
        "Time(%)": "time_percent",
        "Calls": "num_calls",
        "Avg": "avg_us",
        "Min": "min_us",
        "Max": "max_ms",
        "Name": "name",
    }

    gpu_prof = pd.read_csv(csv_file, header=0, skiprows=skiprows, nrows=nrows)
    gpu_prof = gpu_prof.rename(columns=gpu_columns)
    units_row = gpu_prof.iloc[0]
    gpu_prof = gpu_prof.drop(0, axis=0) # drop the units row
    # gpu_prof = gpu_prof.dropna(axis=0)  # drop rows with NaN
    # fix the units!!!!!!
    for col in ["time_ms", "avg_us", "min_us", "max_ms"]:
        unit = col.split("_")[1]
        if units_row[col] != unit:
            assert units_row[col] in ["ms", "us"], f"Profile {csv_file} column {col} has unit {units_row[col]}"
            # unit is wrong, since we only have us or ms, convert to the other
            if units_row[col] == "ms":
                # convert to us, multiply by 1000
                gpu_prof[col] = pd.to_numeric(gpu_prof[col]) * 1000
            else:
                # unit is in us, convert to ms, divide by 1000
                gpu_prof[col] = pd.to_numeric(gpu_prof[col]) / 1000
    
    assert not (gpu_activities_only and api_calls_only)

    if gpu_activities_only:
        gpu_prof = gpu_prof[gpu_prof["type"] == "GPU activities"]
    if api_calls_only:
        gpu_prof = gpu_prof[gpu_prof["type"] == "API calls"]

    attribute_cols = [
        "time_percent",
        "time_ms",
        "num_calls",
        "avg_us",
        "min_us",
        "max_ms",
    ]

    result = gpu_prof.apply(
        lambda row: retrieve_row_attrs(
            row, name_col="name", attribute_cols=attribute_cols
        ),
        axis=1,
    )  # results in sparse dataframe
    result = result.backfill()  # put all of the information in the first row

    return result.iloc[0]


def retrieve_row_attrs(row, name_col, attribute_cols):
    """
    Takes 1 row of the gpu attributes such as

    type            time_percent    time_ms     num_calls   avg_us  min_us  max_ms      name
    GPU activities	88.005407	    38.058423	125	        304.467	0.864	13.759156	[CUDA memcpy HtoD]

    and returns a new Series with columns corresponding to the name.

    Example: calling this function on the row above with

    attribute_cols = ["time_percent", "time_ms", "num_calls", "avg_us", "min_us", "max_ms"]
    and
    name_col = "name"
    yeilds

    time_percent_[CUDA memcpy HtoD]     time_ms_[CUDA memcpy HtoD]  ... max_ms_[CUDA memcpy HtoD]
    88.005407                           38.058423                   ... 13.759156
    """

    return pd.Series(
        {
            f"{attribute}_{row[name_col]}": float(row[attribute])
            for attribute in attribute_cols
        }
    )


def parse_one_system_profile(
    csv_file=None, example=False, nrows=5, skiprows=None, gpu=0
):

    """
    Takes a csv generated by nvprof with aggregate mode on, and returns a pandas series where the
    axis labels are the system signals (clock, memory clock, temp, power, and fan) and the metrics
    associated with it (avg, min, max).

    :param csv_file: the csv file
    :param example: boolean to use an example profile.  If true, csv_file is not needed.
    :param nrows: number of rows in the csv, for aggregate profiles, default is 5
    :param skiprows: number of rows to skip at the beginning of the csv, for aggregate profiles, default 61
                    because the gpu profile activity comes in the first 61 rows
    :param gpu: the number of the gpu used for profiling (nvprof automatically collects system information
                on all gpus in the system)
    :return: a pandas series
    """
    if skiprows is None:
        with open(csv_file) as f:
            for i, line in enumerate(f):
                if line == "\n":
                    break
        skiprows = i + 2  # one blank line and one line with ==System profile result

    if example:
        csv_file = Path.cwd() / "debug_profiles" / "resnet" / "resnet750691.csv"
    elif not csv_file:
        raise ValueError("csv_file must be provided if example is false.")

    if not csv_file.exists():
        raise ValueError(f"File {csv_file} does not exist")

    system_columns = {
        "Device": "device",
        "Count": "count",
        "Avg": "avg",
        "Min": "min",
        "Max": "max",
        "Unnamed: 0": "signal",
    }

    system_prof = pd.read_csv(
        csv_file, header=0, skiprows=skiprows, nrows=nrows * (gpu + 1)
    )
    system_prof = system_prof.rename(columns=system_columns)
    system_prof["signal"] = system_prof["signal"].apply(
        lambda x: x.lower().replace(" ", "_")
    )  # format signal names

    if gpu > 0:
        # drop rows for other gpus
        system_prof = system_prof.drop(list(range(gpu * 5)))

    attribute_cols = ["avg", "min", "max"]

    result = system_prof.apply(
        lambda row: retrieve_row_attrs(
            row, name_col="signal", attribute_cols=attribute_cols
        ),
        axis=1,
    )  # results in sparse dataframe
    result = result.backfill()  # put all of the information in the first row
    return result.iloc[0]


def parse_one_profile(csv_file=None, example=False, gpu=0, remove_nans=True, gpu_activities_only: bool = False, api_calls_only: bool = False):
    """
    Parse the gpu attributes and system attributes from a csv file from nvprof and return a pandas Series.

    :param csv_file: the csv filename.
    :param example: boolean indicating whether or not to use an example profile.  If true, csv_file is ignored.
    :param gpu: the gpu that the profile was run on.
    :return: a pandas Series
    """
    csv_file = Path(csv_file)
    gpu_prof = parse_one_aggregate_profile(csv_file, example=example, gpu_activities_only=gpu_activities_only, api_calls_only=api_calls_only)
    system_prof = parse_one_system_profile(csv_file, example=example, gpu=gpu)
    # return gpu_prof.append(system_prof)
    df = pd.concat((gpu_prof, system_prof))
    if remove_nans:
        df.dropna(inplace=True)
    return df


def avgProfiles(profile_paths: List[Path], gpu=0) -> pd.Series:
    """Given a list of profile paths, parse them all and take the average."""
    combined = pd.DataFrame()
    for path in profile_paths:
        features = parse_one_profile(csv_file=path, gpu=gpu)
        features = features.to_frame().T
        combined = pd.concat((combined, features), ignore_index=True, axis=0)

    return np.mean(combined, axis=0)

def minProfiles(profile_paths: List[Path], gpu=0) -> pd.Series:
    """Given a list of profile paths, parse them all and take the minimum of each feature."""
    combined = pd.DataFrame()
    for path in profile_paths:
        features = parse_one_profile(csv_file=path, gpu=gpu)
        features = features.to_frame().T
        combined = pd.concat((combined, features), ignore_index=True, axis=0)

    return np.min(combined, axis=0)


def parse_all_profiles(
    folder: Union[Path, str], save_filename=None, gpu=0, verbose=True, gpu_activities_only = False, api_calls_only = False
) -> None:
    """
    Parses all of the profiles under the folder into one dataframe saved as a csv in the folder.

    The folder, under cwd/profiles, is organized by subfolder according to model architecture.
    Model architecture and the filename are added as columns to the csv.

    :param folder: the folder containing subfolders by model architecture, which contain profiles,
                        such as ./profiles/<folder>/resnet/resnet12345.csv.  This can either be a
                        Path (if the folder is not a direct child of the directory ./profiles) or
                        a str (if the folder is in the ./profiles/ directory).
    :param save_filename: the filename of the combined csv to save, default is aggregated.csv.
    :param gpu: the gpu that the profile was run on.
    :param verbose: print messages.
    :return: None, just saves a csv file.
    """

    # validate that no profiles are corrupt and that there is a class balance
    validate_all(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    combined = pd.DataFrame()

    for subdir in [x for x in folder.iterdir() if x.is_dir()]:
        model = subdir.name
        if verbose:
            print(f"Parsing profiles for {model}")
        for csv_profile in [x for x in subdir.iterdir()]:
            file = csv_profile.name
            if verbose:
                print(f"\t{file}")
            prof_first = pd.Series({"file": file, "model": model})
            prof_second = parse_one_profile(csv_file=csv_profile, gpu=gpu, gpu_activities_only=gpu_activities_only, api_calls_only=api_calls_only)
            prof = pd.concat((prof_first, prof_second)).to_frame().T
            combined = pd.concat((combined, prof), ignore_index=True, axis=0)

    if save_filename is None:
        save_filename = "aggregated.csv"
    if gpu_activities_only:
        assert not api_calls_only
        save_filename = "aggregated_gpu_only.csv"
    if api_calls_only:
        save_filename = "aggregated_api_only.csv"

    save_path = folder / save_filename

    combined = add_model_family(combined)
    combined.to_csv(save_path, index=False)
    return


def validate_all(folder: Path) -> None:
    """
    Validates 3 things:

    (1) that nvprof did not fail on any profile.
    (2) that there are no NaNs in the profiles.
    (3) that there is the same number of profiles per class.

    If any check fails, an error is raised. Also, the user will have the option to remove profiles
    based on a response to a question in the console.

    :param folder: the root folder which has subfolders organized by class (model architecture)
    :return: None
    """

    # check that all profiles are valid
    valid, _ = validate_nvprof(folder, remove=False)
    if not valid:
        response = input(
            "\n\n\nThere are invalid profiles.  Enter 'yes' to delete them, anything "
            "else to keep them.  An error will be raised either way.  This error will "
            "continue occuring until they are moved or deleted."
        )
        if response.lower() == "yes":
            _ = validate_nvprof(folder, remove=True)
        raise ValueError("Invalid profiles, fix before aggregating.")

    no_nans, _ = validate_nans(folder, remove=False)
    if not no_nans:
        response = input(
            "\n\n\nThere are profiles with NaNs.  Enter 'yes' to delete them, anything "
            "else to keep them.  An error will be raised either way.  This error will "
            "continue occuring until they are fixed or deleted."
        )
        if response.lower() == "yes":
            _ = validate_nans(folder, remove=True)
        raise ValueError("Profiles have NaNs, fix before aggregating.")

    # check that classes are balanced
    balanced = validate_class_balance(folder, remove=False)
    if not balanced:
        response = input(
            "\n\n\nThere is a class imbalance. Enter 'yes' to delete extra profiles, "
            "enter anything else to keep them.  An error will be raised either way. "
            "This error will continue occuring until the classes are balanced."
        )
        if response.lower() == "yes":
            _ = validate_class_balance(folder, remove=True)
        raise ValueError("Class imbalance, fix before aggregating.")


def validate_nvprof(
    folder: Path, remove: bool = False
) -> Tuple[bool, Mapping[str, Mapping[str, Union[int, list[str]]]]]:
    """
    Checks all the profiles under ./profiles/<folder> to see if nvprof failed and lists them, optionally removing them.

    :param folder: the folder containing subfolders by model architecture, which contain profiles,
                        such as ./profiles/<folder>/resnet/resnet12345.csv
    :param remove: boolean whether or not to remove the files
    :return: a tuple of (boolean indicating whether there were any invalid profiles,
                    a dictionary of how many invalid profiles there are by model, with the file names)
    """

    print("Checking profile validity ... ")

    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    all_valid = True
    invalid_profiles = {}

    for subdir in [x for x in folder.iterdir() if x.is_dir()]:
        model = subdir.name
        invalid_profiles[model] = {"num_invalid": 0, "invalid_profiles": []}
        print(f"Parsing profiles for {model}")
        for csv_profile in [x for x in subdir.iterdir()]:
            file = csv_profile.name
            valid = check_profile(csv_profile)
            if not valid:
                all_valid = False
                print(f"\t{file} is invalid!")
                invalid_profiles[model]["num_invalid"] += 1
                invalid_profiles[model]["invalid_profiles"].append(str(csv_profile))
                if remove:
                    csv_profile.unlink()

    if all_valid:
        print("All profiles valid!\n\n")
    else:
        print("Invalid profiles!")
        print(json.dumps(invalid_profiles, indent=4))
    return all_valid, invalid_profiles


def validate_class_balance(folder: Path, remove: bool = False) -> bool:
    """
    Checks all the profiles under ./profiles/<folder> to see if there is a class balance, optionally removing extras.

    :param folder: the folder containing subfolders by model architecture, which contain profiles,
                        such as ./profiles/<folder>/resnet/resnet12345.csv
    :param remove: boolean whether or not to remove the files
    :return: boolean indicating whether there is a class balance
    """

    print("Checking class balance ... ")

    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    profiles = {}

    for subdir in [x for x in folder.iterdir() if x.is_dir()]:
        model = subdir.name
        profiles[model] = {"num": 0, "profiles": []}
        print(f"Parsing profiles for {model}")
        for csv_profile in [x for x in subdir.iterdir()]:
            profiles[model]["num"] += 1
            profiles[model]["profiles"].append(csv_profile)

    model_counts = [profiles[model]["num"] for model in profiles]
    balance = len(model_counts) == model_counts.count(model_counts[0])

    if balance:
        print("Classes are balanced!\n\n")
    else:
        print("Classes are imbalanced!")
        print(
            json.dumps(
                {model: f"{profiles[model]['num']} profiles" for model in profiles},
                indent=4,
            )
        )

    if remove:
        keep = min(model_counts)
        for model in profiles:
            count = profiles[model]["num"]
            need_to_remove = count - keep
            if need_to_remove > 0:
                for i in range(need_to_remove):
                    file = profiles[model]["profiles"][i]
                    print(f"Removing {file}")
                    file.unlink()
    return balance


def validate_nans(
    folder: Path, remove: bool = False
) -> Tuple[bool, Mapping[str, Mapping[str, Union[int, list[str]]]]]:
    """
    Checks all the profiles under ./profiles/<folder> to see if they include NaNs and lists them.

    :param folder: the folder containing subfolders by model architecture, which contain profiles,
                        such as ./profiles/<folder>/resnet/resnet12345.csv
    :param remove: boolean whether or not to remove the files with NaNs
    :return: a tuple of (boolean indicating whether there were any profiles with NaNs,
                    a dictionary of profiles with NaNs by model, with the file names)
    """

    print("Checking profiles for NaNs ... ")

    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    no_nans = True
    profiles_with_nan = {}

    for subdir in [x for x in folder.iterdir() if x.is_dir()]:
        model = subdir.name
        profiles_with_nan[model] = {"num_with_nan": 0, "profiles": []}
        print(f"Parsing profiles for {model}")
        for csv_profile in [x for x in subdir.iterdir()]:
            file = csv_profile.name
            cols = check_for_nans(csv_profile)
            if len(cols) > 0:
                no_nans = False
                print(f"\t{file} has NaNs in columns {cols}")
                profiles_with_nan[model]["num_with_nan"] += 1
                profiles_with_nan[model]["profiles"].append(str(csv_profile))
                if remove:
                    csv_profile.unlink()

    if no_nans:
        print("No NaNs in any profiles!\n\n")
    else:
        print("NaNs found in profiles!")
        print(json.dumps(profiles_with_nan, indent=4))
    return no_nans, profiles_with_nan


def read_csv(folder: Path = None, gpu: int = 0, gpu_activities_only = False, api_calls_only = False) -> pd.DataFrame:
    """
    Reads the aggregated csv data from the folder.  If the aggregated csv does not exist, creates it.

    :param folder: the folder where the profiles are stored.
    :return: a pandas dataframe
    """
    if not folder:
        folder = Path.cwd() / "profiles" / "debug_profiles"

    filename = "aggregated.csv"
    if gpu_activities_only:
        assert not api_calls_only
        filename = "aggregated_gpu_only.csv"
    if api_calls_only:
        filename = "aggregated_api_only.csv"

    aggregated_csv_file = folder / filename
    if not aggregated_csv_file.exists():
        parse_all_profiles(folder, gpu=gpu, gpu_activities_only=gpu_activities_only, api_calls_only=api_calls_only)
    return pd.read_csv(aggregated_csv_file, index_col=False)


def findProfiles(folder: Path) -> Dict[str, List[Path]]:
    """
    Given a path to a profile folder, whose subfolders contain profiles of
    different DNN architectures and the name of the subfolder is the architecture,
    return a dictionary of {DNN architecture name: [list of paths to profiles for
    this architecture]}
    """
    result = {}
    for subdir in [x for x in folder.iterdir() if x.is_dir()]:
        architecture = subdir.name
        model_profiles = list(subdir.glob("*.csv"))
        result[architecture] = model_profiles
    return result

if __name__ == "__main__":
    # a = parse_all_profiles("debug_2")
    # validate_nans("zero_noexe")
    # parse_all_profiles("zero_noexe_lots_models")
    # validate_nvprof("zero_noexe_lots_models")
    # validate_class_balance("zero_noexe_lots_models")
    a = argparse.ArgumentParser()
    a.add_argument("-folder", type=str, required=True, help="folder with profiles")
    a.parse_args()
    read_csv(a.folder)
    exit(0)
