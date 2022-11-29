import os
from pathlib import Path

def timer(time_in_s: float):
    hours, rem = divmod(time_in_s, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours),int(minutes),int(seconds))


def getSystem() -> str:
    if os.name != 'nt':
        system = "linux"
    else:
        system = "windows"
    return system


def latest_file(path: Path, pattern: str = "*"):
    """Return the latest file in the folder <path>"""
    # source https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder
    files = path.glob(pattern)
    return max(files, key=lambda x: x.stat().st_ctime)
