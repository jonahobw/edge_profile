"""Downloads a dataset to <current working dir>/datasets/<name>"""

from argparse import ArgumentParser

import requests, zipfile, io, shutil
from pathlib import Path
from torchvision import datasets
from torchvision.datasets import ImageFolder

torchvision_datasets = {
    "MNIST": datasets.MNIST,
    "CIFAR10": datasets.CIFAR10,
    "CIFAR100": datasets.CIFAR100,
    "ImageNet": datasets.ImageNet,
}

supported_datasets = ["tiny-imagenet-200"] + list(torchvision_datasets.keys())

download_path = Path.cwd() / "datasets"

if not download_path.exists():
    download_path.mkdir(parents=True, exist_ok=True)

def downloadDataset(dataset_name):
    if dataset_name not in supported_datasets:
        raise ValueError(f"{dataset_name} not supported.  Supported datasets are {supported_datasets}.")
    dataset_path = download_path / dataset_name
    print(f"Downloading {dataset_name} ...")
    if dataset_name in torchvision_datasets:
        torchvision_datasets[dataset_name](root=dataset_path, download=True)
    if dataset_name == "tiny-imagenet-200":
        download = requests.get("http://cs231n.stanford.edu/tiny-imagenet-200.zip", stream=True)
        if download.status_code != 200:  # http 200 means success
            raise RuntimeError(f"Download failed with status code {download.status_code}.")
        z = zipfile.ZipFile(io.BytesIO(download.content))
        z.extractall(dataset_path.parent)

        # format the image folder from /class/images/n01443537_0.JPEG
        # to get rid of images folder, resulting in /class/n01443537_0.JPEG
        for class_folder in (dataset_path / "train").glob("*"):
            if not class_folder.is_dir():
                class_folder.unlink()
            # text_files = [x for x in class_folder.glob("*.txt")]
            # for text_file in text_files:
            #     text_file.unlink()
            img_folder = class_folder / "images"
            if img_folder.exists():
                for img in img_folder.glob("*"):
                    img.rename(dataset_path/ "train" / class_folder.name / img.name)
                shutil.rmtree(img_folder)
        
        # format the validation folder
        val_folder = dataset_path / "val"
        annotation_file = val_folder / "val_annotations.txt"
        file_to_label = {}
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()
        for line in annotations:
            if line.find("JPEG") >= 0:
                words = line.split("\t")
                file_to_label[words[0]] = words[1]
            
        # Create subfolders (if not present) for validation images based on label,
        # and move images into the respective folders
        for img, folder in file_to_label.items():
            img_file = val_folder / "images" / img
            newpath = val_folder / folder
            newpath.mkdir(exist_ok=True)
            if img_file.exists():
                img_file.rename(newpath / img)
        shutil.rmtree(val_folder / "images")
        
    print(f"Completed.")

if __name__ == '__main__':
    a = ArgumentParser()
    a.add_argument("-name", required=True, type=str, help=f"The name of the dataset, currently supports {supported_datasets}")
    args = a.parse_args()

    downloadDataset(args.name)