from pathlib import Path
import pandas as pd
from format_profiles import read_csv



if __name__ == '__main__':
    test = read_csv("zero_noexe")

    print(test.head())