from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Optional, Tuple
from fjord.common import create_lda_partitions
import numpy as np
from PIL import Image
import torch

CIFAR_NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))



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


class FLCifar10Client(Dataset):
    def __init__(self, dataset, transform_train):
        self.dataset = [(dataset[0][i], dataset[1][i]) for i in range(len(dataset[1]))]
        self.transform_train = transform_train
    def __getitem__(self, index):
        img, target = self.dataset[index]
        img = Image.fromarray(img)
        img = self.transform_train(img)
        return img, target
    def __len__(self):
        return len(self.dataset)


def load_data(partitions, cid, train_bs, eval_bs, path):

    g = torch.Generator().manual_seed(1234 + cid)


    transform_train, transform_test = get_transforms()


    trainset = FLCifar10Client(partitions[cid], transform_train)

    train_loader = DataLoader(
        trainset,
        batch_size=train_bs,
        shuffle=True,
    )

    testset = CIFAR10(root=path, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=eval_bs)


    return train_loader, test_loader


def partition(dataset, num_clients: int, concentration: float) -> list:
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
        # concentration=concentration * num_classes,
        concentration=concentration,
        accept_imbalanced= True,
        seed=1234,
    )
    return partitions
