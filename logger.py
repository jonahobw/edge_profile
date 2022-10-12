"""
CSV Logger from https://github.com/jonahobw/shrinkbench/blob/master/util/csvlogger.py
"""
import csv
from pathlib import Path
import torch


class CSVLogger:

    def __init__(self, folder, columns):
        """General purpose CSV Logger
        Initialized with a set of columns, it then has two operations
          - set(**kwargs) - to add entries into the current row
          - update - flush a row to file
        Arguments:
            folder {str} -- Path to folder where file will be created
            columns {List[str]} -- List of keys that CSV is going to log
        """
        file = folder / "logs.csv"
        print(f"Logging results to {file}")
        self.file = open(file, 'w')
        self.columns = columns
        self.values = {}

        self.writer = csv.writer(self.file)
        self.writer.writerow(self.columns)
        self.file.flush()
        self.line = 0

    def set(self, **kwargs):
        """Set value for current row
        [description]
        Arguments:
            **kwargs {[type]} -- [description]
        Raises:
            ValueError -- [description]
        """
        for k, v in kwargs.items():
            if k in self.columns:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.values[k] = v
            else:
                raise ValueError(f"{k} not in columns {self.columns}")

    def update(self):
        """Take current values and write a row in the CSV
        """
        row = [self.values.get(c, "") for c in self.columns]
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        """Close the file descriptor for the CSV
        """
        self.file.close()


def buildLogger(metrics, path: Path) -> CSVLogger:
    printc(f"Logging results to {path}", color="MAGENTA")
    path.mkdir(exist_ok=True, parents=True)
    csvlogger = CSVLogger(path / "logs.csv", metrics)