from pathlib import Path
import pandas as pd
from format_profiles import read_csv
import config
import json


def missing_data(aggregated_csv_folder):
    """
    Returns the columns and number of missing datapoints by model.  Missing data are denoted by NaN.

    :param aggregated_csv_folder: path to the folder under ./profiles/ which contains the aggregated.csv file.
    :return: A dict keyed by model with the columns with missing data and the number of missing datapoints.
    """
    df = read_csv(aggregated_csv_folder)
    model_nans = {}

    for model in df["model"].unique():
        model_df = df.loc[df["model"] == model]
        n_model_profiles = len(model_df.index)
        model_nans[model] = model_df.isna().sum() / n_model_profiles

    # model_nans is now a dict of {model_type: Pandas series of {column_name: % of that column that is empty}}

    return model_nans


def mutually_exclusive_data(aggregated_csv_folder):
    """
    Returns information about the data as a dict with elements described below:

    mutually_exclusive_attributes: the attributes which are exclusive to each model.  Only include an attribute
        if each model profile includes this attribute and none of the others include it.  Organized by model

    partial_attributes: attributes which models have for some profiles but not others, organized by model.

    no_data_attributes: list of attributes for which there is no data from any model.

    complete_attributes: A list of attributes which each model has completely.

    partially_exclusive_attributes: A mapping from attributes to models for attributes which some but
        not all models have

    :param aggregated_csv_folder: path to the folder under ./profiles/ which contains the aggregated.csv file.
    """

    model_nans = missing_data(aggregated_csv_folder)
    num_models = len(list(model_nans.keys()))

    # dict of model: exclusive attributes to that model (this model has it completely and no other model has any)
    mutually_exclusive = {model: [] for model in model_nans}

    # dict of model: attributes which that model has partially (some missing values)
    partial_attribute = {model: [] for model in model_nans}

    # array of attributes on which no model has data
    attributes_with_no_data = []

    # array of attributes on which all models have complete data
    complete_attributes = []

    # dict of attribute: models with that attribute completely.
    # Only includes attributes that at least 1 model doesnt have and at least 1 model has.
    partially_exclusive = {}

    for attribute in model_nans[next(iter(model_nans))].axes[0]:    # gets the first element
        models_with_complete_attribute = []
        models_with_partial_attribute = []
        models_without_attribute = []
        for model in model_nans:
            percent_empty = model_nans[model][attribute]
            if percent_empty == 0.0:
                models_with_complete_attribute.append(model)
            if 0.0 < percent_empty < 1.0:
                models_with_partial_attribute.append(model)
                # this model has this attribute partially
                partial_attribute[model].append(attribute)
            if percent_empty == 1.0:
                models_without_attribute.append(model)
        if len(models_with_complete_attribute) == 1 and len(models_with_partial_attribute) == 0:
            # only one model has this attribute
            mutually_exclusive[models_with_complete_attribute[0]].append(attribute)
        if len(models_without_attribute) == num_models:
            # no model has this attribute
            attributes_with_no_data.append(attribute)
        if len(models_with_complete_attribute) == num_models:
            # all models have this attribute completly
            complete_attributes.append(attribute)
        if 0 < len(models_with_complete_attribute) < num_models and len(models_without_attribute) > 0:
            # at least 1 model has this and at least 1 model doesn't have this attribute
            partially_exclusive[attribute] = models_with_complete_attribute

    res = {
        "mutually_exclusive_attributes": mutually_exclusive,
        "partial_attributes": partial_attribute,
        "no_data_attributes": attributes_with_no_data,
        "complete_attributes": complete_attributes,
        "partially_exclusive_attributes": partially_exclusive
    }

    return res


def shared_data(agg_csv_folder, system_data_only=False, no_system_data=False):
    """
    Return a dataframe containing only complete features that can be used for machine learning.

    :param agg_csv_folder: the folder under ./profiles/ where the aggregated csv lives.
    :param system_data_only: If true, only return system data (clock, temp, power, fan).
    :param no_system_data: if true, excludes system data.
    :return: a dataframe
    """
    if system_data_only and no_system_data:
        raise ValueError("system_data_only and no_system_data cannot both be true.")

    df = read_csv(agg_csv_folder)
    complete_attributes = mutually_exclusive_data(agg_csv_folder)["complete_attributes"]
    df = df[complete_attributes]    # only consider complete data

    if not system_data_only and not no_system_data:
        return df

    def system_column(col):
        for sys_signal in config.SYSTEM_SIGNALS:
            if col.endswith(sys_signal):
                return True
        return False

    system_cols = [col_name for col_name in complete_attributes if system_column(col_name)]

    if system_data_only:
        system_cols.append('model')
        system_cols.append('model_family')
        system_cols.append('file')
        return df[system_cols]

    # else no_system_data is true
    return df.drop(system_cols, axis=1)


def train_test_split(df, ratio=0.8):
    """Splits the df into a train and test set with equal representation over all the model classes."""
    test_df = pd.DataFrame()
    train_df = pd.DataFrame()
    for model in config.MODELS:
        model_rows = df[df["model"] == model]
        num_rows = len(model_rows.index)

        train_rows = int(num_rows * ratio)
        train_df = pd.concat([train_df, model_rows.head(train_rows)], ignore_index=True)

        test_rows = num_rows - train_rows
        test_df = pd.concat([test_df, model_rows.tail(test_rows)], ignore_index=True)

    return train_df, test_df


def get_data_and_labels(df, shuffle=True, label=None):
    """Splits a dataframe into data points and their associated labels (model)."""
    if shuffle:
        df = df.sample(frac=1)

    if not label:
        y = df["model"]
    else:
        y = df[label]

    x = df.drop("file", axis=1)
    x = x.drop("model_family", axis=1)
    x = x.drop("model", axis=1)

    return x, y


if __name__ == '__main__':
    # test = shared_data("zero_noexe", system_data_only=True)
    # print(test)
    test = mutually_exclusive_data("zero_noexe_lots_models")
    print(json.dumps(test, indent=4))
    exit(0)
