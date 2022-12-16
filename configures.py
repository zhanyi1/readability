import os
import torch
from tap import Tap
from typing import List



class DataParser(Tap):
    dataset_dir: str = 'Data/input.pkl'
    dataset_name = 'code'
    num_node_features: int = 840
    num_classes: int = 3
    avg_num_nodes: int = 1000
    data_split_ratio: List = [0.6, 0.2, 0.2]   # the ratio of training, validation and testing set for random split
    batch_size: int = 10
    seed: int = 1
    k_fold: int = 5


class TrainParser(Tap):
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    max_epochs: int = 255
    save_epoch: int = 10
    early_stopping: int = 100


class ModelParser(Tap):
    device: str = "cpu"
    checkpoint: str = 'checkpoint'
    hidden_channels: int = 128
    mlp_hidden: int = 128
    model_name = 'dmon'


    def process_args(self) -> None:
        self.device = torch.device('cpu')



data_args = DataParser().parse_args(known_only=True)
train_args = TrainParser().parse_args(known_only=True)
model_args = ModelParser().parse_args(known_only=True)

