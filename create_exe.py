import sys
import shutil
import os
import subprocess
import shlex
from pathlib import Path

def create_exe():
    command = "pyinstaller --onefile model_inference.py"
    output = subprocess.run(shlex.split(command), stdout=sys.stdout)
    exe_file = Path.cwd() / "dist" / "model_inference.exe"
    if os.name != "nt":
        # linux
        destination_folder = Path.cwd() / "exe" / "linux"
        if not destination_folder.exists():
            destination_folder.mkdir(exist_ok=True)
        destination = destination_folder / "linux_inference.exe"
        exe_file = Path.cwd() / "dist" / "model_inference"
    else:
        # windows
        destination_folder = Path.cwd() / "exe" / "windows"
        if not destination_folder.exists():
            destination_folder.mkdir(exist_ok=True)
        destination = destination_folder / "windows_inference.exe"

    shutil.copy(exe_file, destination)

def cleanup():
    dist_folder = Path.cwd() / "dist"
    if dist_folder.exists():
        shutil.rmtree(dist_folder)

    build_folder = Path.cwd() / "build"
    if build_folder.exists():
        shutil.rmtree(build_folder)

    spec_file = Path.cwd() / "model_inference.spec"
    spec_file.unlink(missing_ok=True)


create_exe()
cleanup()
