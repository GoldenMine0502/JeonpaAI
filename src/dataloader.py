import os
import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from interpolation_ways import *
import torch.nn.functional as F


def create_dataloader(configs, train, root_dir=None):
    def train_collate_fn(batch):
        train_seq_list = list()
        train_pred_list = list()

        for train_seq, train_pred in batch:
            train_seq_list.append(torch.from_numpy(train_seq).float())
            train_pred_list.append(torch.from_numpy(train_pred).float())

        train_seq_list = torch.stack(train_seq_list, dim=0)
        train_pred_list = torch.stack(train_pred_list, dim=0)
        return train_seq_list, train_pred_list

    # def validation_collate_fn(batch):
    #     return train_collate_fn(batch)

    if train:
        return DataLoader(dataset=JeonpaDataset(configs, True, root_dir=root_dir),
                          batch_size=configs.train.batch_size,
                          shuffle=True,
                          num_workers=configs.train.num_workers,
                          collate_fn=train_collate_fn,
                          # pin_memory=True,
                          # drop_last=True,
                          # sampler=None
                          )
    else:
        return DataLoader(dataset=JeonpaDataset(configs, False, root_dir=root_dir),
                          collate_fn=train_collate_fn,
                          batch_size=1, shuffle=False, num_workers=0)


def create_testloader(configs, root_dir=None):
    def test_collate_fn(batch):
        train_seq_list = list()

        for train_seq in batch:
            train_seq_list.append(torch.from_numpy(train_seq).float())

        train_seq_list = torch.stack(train_seq_list, dim=0)
        return train_seq_list

    return DataLoader(dataset=JeonpaTestDataset(configs, root_dir=root_dir),
                      collate_fn=test_collate_fn,
                      batch_size=1, shuffle=False, num_workers=0)

def get_data_from_path(configs, file_path, test=False, root_dir=None):
    if root_dir is None:
        root_dir = Path(os.getcwd()).parent.absolute()
    # print(f"root directory: {root_dir}")

    dataset = pd.read_csv(f'{root_dir}/{file_path}')
    # print(dataset)

    date = np.array(dataset['date']).copy()
    flux = np.array(dataset['flux']).copy()
    print('len before interpolation:', len(flux))

    interpolation_model = InterpolationRemoveLongMissingValue(configs)
    # interpolation_model = InterpolationPoly(configs)
    # interpolation_model = InterpolationKNN(configs)

    flux = interpolation_model.get_dataset(flux, test)
    print('len after interpolation:', len(flux))

    return date, flux

class JeonpaDataset(Dataset):
    def __init__(self, configs, train, root_dir=None):
        self.configs = configs
        self.train = train
        self.train_file = configs.data.trainset
        self.test_file = configs.data.trainset

        self.split_rate = configs.data.split_rate
        self.seq_len = configs.model.seq_len
        self.pred_len = configs.model.pred_len

        # 학습데이터와 테스트데이터를 나눔.
        date, flux = get_data_from_path(configs, self.configs.data.trainset, root_dir=root_dir)
        split_index = int(len(flux) * self.split_rate)
        # print(len(flux))
        self.train = flux[:split_index]
        self.test = flux[split_index:]

    def __len__(self):
        if self.train:
            return len(self.train)
        else:
            return len(self.test)

    def __getitem__(self, idx):
        if self.train:
            # dim = 1
            # train_seq = self.train_flux[idx:idx + self.seq_len][:, np.newaxis]  # 10~70
            # train_pred = self.train_flux[idx + self.seq_len:idx + self.seq_len + self.pred_len][:, np.newaxis]  # 70~100

            # shape = (3, 2)
            # shape[:, np.newaxis, :] # 3, 1, 2
            # shape.unsqueeze(1) (3, 1, 2)
            # flatten -> 차원 밀어서 -> 1차원
            # squeeze((3, 1, 1, 1, 1, 2)) -> (3, 2)
            # sequeeze((3, 1, 2, 1), dim=1) -> (3, 2, 1)
            return self.train[idx]
        else:
            # validation_seq = self.test_flux[idx:idx + self.seq_len][:, np.newaxis]
            # validation_pred = self.test_flux[idx + self.seq_len:idx + self.seq_len + self.pred_len][:, np.newaxis]
            # return validation_seq, validation_pred
            return self.test[idx]

class JeonpaTestDataset(Dataset):
    def __init__(self, configs, root_dir=None):
        self.configs = configs
        self.seq_len = configs.model.seq_len
        self.date, self.flux = get_data_from_path(configs, self.configs.data.testset, test=True, root_dir=root_dir)
        # print(self.flux)

    def __len__(self):
        # 입력, 출력 길이에 따라 사용할 수 있는 데이터의 양이 달라진다.
        # raw 데이터 길이가 100이고, seq_len = 60, pred_len = 30인 경우 -> 90개
        # 11개의 학습 데이터를 뽑을 수 있음
        # 0-89, 1-90, 2-91, 3-92, 4-93,
        # 5-94, 6-95, 7-96, 8-97, 9-98, 10-99
        # total_train_len = dataset_len - self.seq_len - self.pred_len + 1
        # minus = self.seq_len - 1
        # return len(self.date) - minus  # or test_flux
        return len(self.flux)
    def __getitem__(self, idx):
        # train_seq = self.flux[idx:idx + self.seq_len][:, np.newaxis]
        return self.flux[idx]
