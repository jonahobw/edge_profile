"""Downloads the CIFAR10 dataset to <current working dir>/datasets/CIFAR10"""

from pathlib import Path
from torchvision import datasets

download_path = Path.cwd() / "datasets" / "CIFAR10"

if not download_path.exists():
    download_path.mkdir(parents=True, exist_ok=True)

cifar10 = datasets.CIFAR10(root=download_path, download=True)