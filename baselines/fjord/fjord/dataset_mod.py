"""Dataset for CIFAR10."""

import random
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from fjord.common import create_lda_partitions

_PARTITIONS_CACHE = None

# def get_cached_partitions(dataset, num_clients, concentration):
#     global _PARTITIONS_CACHE
#     if _PARTITIONS_CACHE is None:
#         # Seed *everything* once for reproducibility
#         rng = np.random.default_rng(1234)

#         x, y = dataset.data, np.asarray(dataset.targets)
#         x, y = shuffle(x, y, random_state=rng)   # scikit-learn shuffle supports rng
#         x, y = sort_by_label(x, y)

#         _PARTITIONS_CACHE, _ = create_lda_partitions(
#             (x, y),
#             num_partitions=num_clients,
#             concentration=concentration,
#             accept_imbalanced=True,
#             dirichlet_dist=None,        # let it draw once, with rng already seeded
#             seed=rng,                   # passes the same bit-generator forward
#         )
#     return _PARTITIONS_CACHE

CIFAR_NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

class FLCifar10Client(Dataset):
    """Class implementing the partitioned CIFAR10 dataset."""

    def __init__(self, fl_dataset: Dataset, client_id: Optional[int] = None) -> None:
        """Ctor.

        Args:
        :param fl_dataset: The CIFAR10 dataset.
        :param client_id: The client id to be used.
        """
        self.fl_dataset = fl_dataset
        self.partitions = self.partition(
            fl_dataset, fl_dataset.num_clients, fl_dataset.concentration
        )
        self.set_client(client_id)

    def set_client(self, index: Optional[int] = None) -> None:
        """Set the client to the given index. If index is None, use the whole dataset.

        Args:
        :param index: Index of the client to be used.
        """
        partitions = self.partitions
        if index is None:
            self.client_id = None
            self.length = sum(len(p[0]) for p in partitions)
            self.data = np.concatenate([p[0] for p in partitions])
            self.targets = np.concatenate([p[1] for p in partitions])
        else:
            if index < 0 or index >= len(partitions):
                raise ValueError("Number of clients is out of bounds.")
            self.client_id = index
            partition = partitions[self.client_id]
            self.length = len(partition[0])
            self.data = partition[0]
            self.targets = partition[1]

    def partition(self, dataset, num_clients: int, concentration: float) -> list:
        """Create non-iid partitions.

        The partitions uses a LDA distribution based on concentration.
        """
        print(
            f">>> [Dataset] {num_clients} clients, non-iid concentration {concentration}..."
        )

        x_y = [dataset.data, np.asanyarray(dataset.targets)]
        partitions, _ = create_lda_partitions(
            x_y,
            num_partitions=num_clients,
            concentration=concentration,
            accept_imbalanced= True,
            seed=1234,
        )
        return partitions

    def __getitem__(self, index: int):
        """Return the item at the given index.

        :param index: Index of the item to be returned.
        :return: The item at the given index.
        """
        fl = self.fl_dataset
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if fl.transform is not None:
            img = fl.transform(img)

        if fl.target_transform is not None:
            target = fl.target_transform(target)

        return img, target

    def __len__(self):
        """Return the length of the dataset."""
        return self.length


class FLCifar10(CIFAR10):
    """CIFAR10 Federated Dataset."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        root: str,
        train: Optional[bool] = True,
        transform: Optional[Module] = None,
        target_transform: Optional[Module] = None,
        download: Optional[bool] = False,
        concentration: float = 0.1,
        num_clients: int = 100,
    ) -> None:
        """Ctor.

        :param root: Root directory of dataset
        :param train: If True, creates dataset from training set
        :param transform: A function/transform that takes in an PIL image and returns a
            transformed version.
        :param target_transform: A function/transform that takes in the target and
            transforms it.
        :param download: If true, downloads the dataset from the internet.
        """
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.num_clients = num_clients
        self.concentration = concentration


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Get the transforms for the CIFAR10 dataset.

    :return: The transforms for the CIFAR10 dataset.
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
        ]
    )

    return transform_train, transform_test


def load_data(
    path: str, cid: int, train_bs: int, seed: int, eval_bs: int = 1024, num_clients: int = 100, concentration: float = 0.1
) -> Tuple[DataLoader, DataLoader]:
    """Load the CIFAR10 dataset.

    :param path: The path to the dataset.
    :param cid: The client ID.
    :param train_bs: The batch size for training.
    :param seed: The seed to use for the random number generator.
    :param eval_bs: The batch size for evaluation.
    :return: The training and test sets.
    """

    def seed_worker(worker_id):  # pylint: disable=unused-argument
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    transform_train, transform_test = get_transforms()

    fl_dataset = FLCifar10(
        root=path, train=True, download=True, transform=transform_train, num_clients=num_clients, concentration=concentration
    )

    trainset = FLCifar10Client(fl_dataset, client_id=cid)
    train_sample = trainset[0]
    testset = CIFAR10(root=path, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(
        trainset,
        batch_size=train_bs,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(testset, batch_size=eval_bs)

    return train_loader, test_loader

if __name__ == "__main__":
    # Example usage
    train_loader, test_loader = load_data(
        path="./data",
        cid=0,
        train_bs=32,
        seed=42,
        eval_bs=1024,
        num_clients=100,
        concentration=0.1
    )
    print(f"Train loader: {len(train_loader)} batches, Test loader: {len(test_loader)} batches")