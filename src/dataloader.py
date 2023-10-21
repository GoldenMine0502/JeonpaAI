import os
import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def create_dataloader(configs, train, root_dir=None):
    def train_collate_fn(batch):
        print(batch)
        return batch

    def test_collate_fn(batch):
        return batch

    if train:
        return DataLoader(dataset=JeonpaDataset(configs, True, root_dir=root_dir),
                          batch_size=configs.train.batch_size,
                          shuffle=True,
                          # num_workers=configs.train.num_workers,
                          collate_fn=train_collate_fn,
                          # pin_memory=True,
                          # drop_last=True,
                          # sampler=None
                          )
    else:
        return DataLoader(dataset=JeonpaDataset(configs, False, root_dir=root_dir),
                          collate_fn=test_collate_fn,
                          batch_size=1, shuffle=False, num_workers=0)


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
        # raw 데이터 길이가 100이고, seq_len = 60, pred_len = 30인 경우 -> 90개
        # 10개의 학습 데이터를 뽑을 수 있음
        # 0-89, 1-90, 2-91, 3-92, 4-93,
        # 5-94, 6-95, 7-96, 8-97, 9-98, 10-99
        # total_train_len = dataset_len - self.seq_len - self.pred_len + 1
        date, flux = self.get_data_from_path(self.configs.data.trainset, root_dir=root_dir)
        split_index = int(len(flux) * self.split_rate)
        self.train_date = date[:split_index]
        self.train_flux = flux[:split_index]
        self.test_date = date[split_index:]
        self.test_flux = flux[split_index:]

    def get_data_from_path(self, file_path, root_dir=None):
        if root_dir is None:
            root_dir = Path(os.getcwd()).parent.absolute()
        # print(f"root directory: {root_dir}")

        dataset = pd.read_csv(f'{root_dir}/{file_path}')
        # print(trainset)

        date = np.array(dataset['date'])
        flux = np.array(dataset['flux'])
        mean_flux = np.nanmean(flux)
        # print(f'flux: {flux}, mean: {mean_flux}')

        flux[np.isnan(flux)] = mean_flux
        # print(f'모든 결측치를 평균으로 설정: {flux}')

        return date, flux

    def __len__(self):
        # 입력, 출력 길이에 따라 사용할 수 있는 데이터의 양이 달라진다.
        minus = self.seq_len + self.pred_len - 1
        if self.train:
            return len(self.train_date) - minus  # or train_flux
        else:
            return len(self.test_date) - minus  # or test_flux

    def __getitem__(self, idx):
        if self.train:
            train_seq = self.train_flux[idx:idx + self.seq_len]
            train_pred = self.train_flux[idx + self.seq_len:idx + self.seq_len + self.pred_len]
            return train_seq, train_pred
        else:
            test_seq = self.test_flux[idx:idx + self.seq_len]
            test_pred = self.test_flux[idx + self.seq_len:idx + self.seq_len + self.pred_len]
            return test_seq, test_pred
