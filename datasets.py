"""
adapted from https://github.com/jonahobw/shrinkbench/blob/master/datasets/datasets.py
"""
import json
import pathlib
from typing import Dict, List, Tuple, Union
from collections import Counter

from torchvision import transforms, datasets
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, random_split, Subset
from torch import Generator

_constructors = {
    "MNIST": datasets.MNIST,
    "CIFAR10": datasets.CIFAR10,
    "CIFAR100": datasets.CIFAR100,
    "ImageNet": datasets.ImageNet,
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


def dataset_builder(
    dataset, train=True, normalize=None, preproc=None, path=None, resize: int = None
):
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
        if resize is not None:
            preproc += [transforms.Resize(resize)]
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


def MNIST(train=True, path=None, resize=None, normalize: Tuple[List[float], List[float]] = None):
    """Thin wrapper around torchvision.datasets.CIFAR10"""
    mean, std = 0.1307, 0.3081
    normalize = transforms.Normalize(mean=(mean,), std=(std,))
    dataset = dataset_builder("MNIST", train, normalize, [], path, resize=resize)
    dataset.shape = (1, 28, 28)
    return dataset


def CIFAR10(train=True, path=None, deterministic=False, resize=None, normalize: Tuple[List[float], List[float]] = None):
    """Thin wrapper around torchvision.datasets.CIFAR10"""
    mean, std = [0.491, 0.482, 0.447], [0.247, 0.243, 0.262]
    if normalize is not None:
        mean, std = normalize
    normalize = transforms.Normalize(mean=mean, std=std)
    if train and not deterministic:
        preproc = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
    else:
        preproc = []
    dataset = dataset_builder("CIFAR10", train, normalize, preproc, path, resize=resize)
    dataset.shape = (3, 32, 32)
    return dataset


def CIFAR100(train=True, path=None, resize=None, normalize: Tuple[List[float], List[float]] = None):
    """Thin wrapper around torchvision.datasets.CIFAR100"""
    mean, std = [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
    if normalize is not None:
        mean, std = normalize
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        preproc = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
    else:
        preproc = []
    dataset = dataset_builder(
        "CIFAR100", train, normalize, preproc, path, resize=resize
    )
    dataset.shape = (3, 32, 32)
    return dataset


def ImageNet(train=True, path=None, resize=None, normalize: Tuple[List[float], List[float]] = None):
    """Thin wrapper around torchvision.datasets.ImageNet"""
    # ImageNet loading from files can produce benign EXIF errors
    import warnings

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if normalize is not None:
        mean, std = normalize
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        preproc = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    else:
        preproc = [transforms.Resize(256), transforms.CenterCrop(224)]
    dataset = dataset_builder(
        "ImageNet", train, normalize, preproc, path, resize=resize
    )
    dataset.shape = (3, 224, 224)
    return dataset


class Dataset:
    #todo implement lazy loading of datasets

    name_mapping = {
        "mnist": MNIST,
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "imagenet": ImageNet,
    }
    num_classes_map = {"mnist": 10, "cifar10": 10, "cifar100": 100, "imagenet": 1000}

    def __init__(
        self,
        dataset: str,
        batch_size=128,
        workers=8,
        data_subset_percent: float = None,
        seed: int = 42,
        idx: int = 0,
        resize: int = None,
        normalize: Tuple[List[float], List[float]] = None,
    ) -> None:
        """
        data_subset_percent will divide the dataset into 2 pieces, and the first will have <data_subset_percent>% of the data.
            which portion of the data is allocated depends on <idx> which is either 0 or 1
        resize will resize the data when retrieving it.  Some models need this, most do not.  Resizing significantly slows training.
        """
        self.name = dataset.lower()
        self.num_classes = self.num_classes_map[self.name]
        self.data_subset_percent = data_subset_percent

        self.train_data = self.name_mapping[self.name](resize=resize, normalize=normalize)
        if data_subset_percent is not None:
            first_amount = int(len(self.train_data) * data_subset_percent)
            second_amount = len(self.train_data) - first_amount
            self.train_data = random_split(
                self.train_data,
                [first_amount, second_amount],
                generator=Generator().manual_seed(seed),
            )[idx]
        self.train_dl = DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=workers,
        )

        self.val_data = self.name_mapping[self.name](train=False, resize=resize, normalize=normalize)
        if data_subset_percent is not None:
            first_amount = int(len(self.val_data) * data_subset_percent)
            second_amount = len(self.val_data) - first_amount
            self.val_data = random_split(
                self.val_data,
                [first_amount, second_amount],
                generator=Generator().manual_seed(seed),
            )[idx]
        self.val_dl = DataLoader(
            self.val_data,
            shuffle=False,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=workers,
        )

        self.train_acc_data = self.train_data
        if self.name == "cifar10":
            self.train_acc_data = self.name_mapping[self.name](
                deterministic=True, resize=resize, normalize=normalize
            )
            if data_subset_percent is not None:
                first_amount = int(len(self.train_acc_data) * data_subset_percent)
                second_amount = len(self.train_acc_data) - first_amount
                self.train_acc_data = random_split(
                    self.train_acc_data,
                    [first_amount, second_amount],
                    generator=Generator().manual_seed(seed),
                )[idx]
        self.train_acc_dl = DataLoader(
            self.train_acc_data,
            shuffle=False,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=workers,
        )

        self.config = {
            "dataset": self.name,
            "batch_size": batch_size,
            "workers": workers,
            "data_subset_percent": data_subset_percent,
            "seed": seed,
            "idx": idx,
            "resize": resize,
            "normalize": normalize
        }
    
    def classBalance(self, dataset: Union[VisionDataset, Subset], show=True) -> Dict[int, int]:
        if isinstance(dataset, Subset):
            indices = dataset.indices
            result = dict(Counter(dataset.dataset.targets))
        else:
            result = dict(Counter(dataset.targets))
        if show:
            print(json.dumps(result, indent=4))
        return result


def datasetPartition(
    dataset: str,
    batch_size=128,
    workers=8,
    data_subset_percent: float = None,
    seed: int = 42,
    resize: int = None,
) -> List[Dataset]:
    first_dataset = Dataset(
        dataset,
        batch_size=batch_size,
        workers=workers,
        data_subset_percent=data_subset_percent,
        seed=seed,
        idx=0,
        resize=resize,
    )
    second_dataset = Dataset(
        dataset,
        batch_size=batch_size,
        workers=workers,
        data_subset_percent=data_subset_percent,
        seed=seed,
        idx=1,
        resize=resize,
    )
    return [first_dataset, second_dataset]
