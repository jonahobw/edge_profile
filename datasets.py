"""
From https://github.com/jonahobw/shrinkbench/blob/master/datasets/datasets.py
"""
import pathlib
import os

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

_constructors = {
    "MNIST": datasets.MNIST,
    "CIFAR10": datasets.CIFAR10,
    "CIFAR100": datasets.CIFAR100,
    "ImageNet": datasets.ImageNet
}


def dataset_path(dataset, path=None):
    """Get the path to a specified dataset

    Arguments:
        dataset {str} -- One of MNIST, CIFAR10, CIFAR100, ImageNet

    Keyword Arguments:
        path {str} -- Semicolon separated list of paths to look for dataset folders (default: {None})

    Returns:
        dataset_path -- pathlib.Path for the first match

    Raises:
        ValueError -- If no path is provided and DATAPATH is not set
        LookupError -- If the given dataset cannot be found
    """
    p = pathlib.Path.cwd() / "datasets" / dataset
    
    if p.exists():
        return p
    print(f"Path does not exist:\n{p}\n")

    paths = [pathlib.Path(p) for p in path.split(":")]

    for p in paths:
        p = (p / dataset).resolve()
        if p.exists():
            # print(f"Found {dataset} under {p}")
            return p
    raise LookupError(f"Could not find {dataset}")


def dataset_builder(dataset, train=True, normalize=None, preproc=None, path=None):
    """Build a torch.utils.Dataset with proper preprocessing

    Arguments:
        dataset {str} -- One of MNIST, CIFAR10, CIFAR100, ImageNet, Places365

    Keyword Arguments:
        train {bool} -- Whether to return train or validation set (default: {True})
        normalize {torchvision.Transform} -- Transform to normalize data channel wise (default: {None})
        preproc {list(torchvision.Transform)} -- List of preprocessing operations (default: {None})
        path {str} -- Semicolon separated list of paths to look for dataset folders (default: {None})

    Returns:
        torch.utils.data.Dataset -- Dataset object with transforms and normalization
    """
    if preproc is not None:
        preproc += [transforms.ToTensor()]
        if normalize is not None:
            preproc += [normalize]
        preproc = transforms.Compose(preproc)

    kwargs = {"transform": preproc}
    if dataset == "ImageNet":
        kwargs["split"] = "train" if train else "val"
    else:
        kwargs["train"] = train

    path = dataset_path(dataset, path)

    return _constructors[dataset](path, **kwargs)


def MNIST(train=True, path=None):
    """Thin wrapper around torchvision.datasets.CIFAR10"""
    mean, std = 0.1307, 0.3081
    normalize = transforms.Normalize(mean=(mean,), std=(std,))
    dataset = dataset_builder("MNIST", train, normalize, [], path)
    dataset.shape = (1, 28, 28)
    return dataset


def CIFAR10(train=True, path=None, deterministic=False):
    """Thin wrapper around torchvision.datasets.CIFAR10"""
    mean, std = [0.491, 0.482, 0.447], [0.247, 0.243, 0.262]
    normalize = transforms.Normalize(mean=mean, std=std)
    if train and not deterministic:
        preproc = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
    else:
        preproc = []
    dataset = dataset_builder("CIFAR10", train, normalize, preproc, path)
    dataset.shape = (3, 32, 32)
    return dataset


def CIFAR100(train=True, path=None):
    """Thin wrapper around torchvision.datasets.CIFAR100"""
    mean, std = [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        preproc = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
    else:
        preproc = []
    dataset = dataset_builder("CIFAR100", train, normalize, preproc, path)
    dataset.shape = (3, 32, 32)
    return dataset


def ImageNet(train=True, path=None):
    """Thin wrapper around torchvision.datasets.ImageNet"""
    # ImageNet loading from files can produce benign EXIF errors
    import warnings

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        preproc = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    else:
        preproc = [transforms.Resize(256), transforms.CenterCrop(224)]
    dataset = dataset_builder("ImageNet", train, normalize, preproc, path)
    dataset.shape = (3, 224, 224)
    return dataset

class Dataset:

    name_mapping = {
        "mnist": MNIST,
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "imagenet": ImageNet
    }
    num_classes_map = {
        "mnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "imagenet": 1000
    }
    
    def __init__(self, dataset: str, batch_size=128, workers=4) -> None:
        self.name = dataset.lower()
        self.num_classes = self.num_classes_map[self.name]
        self.train_data = self.name_mapping[self.name]()
        self.train_dl = DataLoader(self.train_data, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
        self.val_data = self.name_mapping[self.name](train=False)
        self.val_dl = DataLoader(self.val_data, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=4)

        self.train_acc_data = self.train_data
        if self.name == "cifar10":
            self.train_acc_data = self.name_mapping[self.name](deterministic=True)
        self.train_acc_dl = DataLoader(self.train_acc_data, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=4)
