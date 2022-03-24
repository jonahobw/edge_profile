from pathlib import Path
import pandas as pd


def read_csv(folder=None):
    """
    Reads the aggregated csv data from the folder.  If the aggregated csv does not exist, creates it.

    :param folder: the
    :return:
    """
    if not folder:
        folder = "debug_profiles"