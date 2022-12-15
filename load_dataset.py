from torch.utils.data import Dataset as TorchDataset
from torch_geometric.loader import DataLoader
import torch
from torch.utils.data import random_split, Subset
import pandas as pd

class InputDataset(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index].input


def get_dataset(dataset_dir):
    input_dataset = pd.read_pickle(dataset_dir)
    return InputDataset(input_dataset)


def get_dataloader(dataset, data_args):
    """
    Args:
        dataset:
        batch_size: int
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly

    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    data_split_ratio = data_args.data_split_ratio
    seed = data_args.seed
    batch_size = data_args.batch_size

    num_train = int(data_split_ratio[0] * len(dataset))
    num_eval = int(data_split_ratio[1] * len(dataset))
    num_test = len(dataset) - num_train - num_eval

    train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                     generator=torch.Generator().manual_seed(seed))

    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['eval'] = DataLoader(eval, batch_size=batch_size, shuffle=False)
    dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=False)
    return dataloader