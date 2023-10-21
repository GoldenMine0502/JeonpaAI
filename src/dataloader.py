import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def create_dataloader(configs, train):
    def train_collate_fn(batch):
        return batch

    def test_collate_fn(batch):
        return batch

    if train:
        return DataLoader(dataset=JeonpaDataset(configs, True),
                          batch_size=configs.train.batch_size,
                          shuffle=True,
                          num_workers=configs.train.num_workers,
                          collate_fn=train_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)
    else:
        return DataLoader(dataset=JeonpaDataset(configs, False),
                          collate_fn=test_collate_fn,
                          batch_size=1, shuffle=False, num_workers=0)


class JeonpaDataset(Dataset):
    def __init__(self, configs, train):
        self.configs = configs
        self.train = train
        self.train_file = configs.data.trainset
        self.test_file = configs.data.trainset


    def __len__(self):
        return len(self.target_wav_list)

    def __getitem__(self, idx):
        if self.train:
            return None
        else:
            return None
