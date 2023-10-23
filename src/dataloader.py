import datetime
import os
import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from interpolation_ways import *
import torch.nn.functional as F
import torch.distributed as dist


def create_dataloader(configs, train, root_dir=None):
    def train_collate_fn(batch):
        train_seq_date_list = list()
        train_seq_flux_list = list()
        train_pred_date_list = list()
        train_pred_flux_list = list()

        # for train_seq, train_pred in batch:
        for (date_seq, date_pred), (train_seq, train_pred) in batch:
            train_seq_date_list.append(torch.from_numpy(date_seq).float())
            train_seq_flux_list.append(torch.from_numpy(train_seq).float())
            train_pred_date_list.append(torch.from_numpy(date_pred).float())
            train_pred_flux_list.append(torch.from_numpy(train_pred).float())
            # print(train_seq_date_list[0].shape)

        train_seq_date_list = torch.stack(train_seq_date_list, dim=0)
        train_seq_flux_list = torch.stack(train_seq_flux_list, dim=0)
        train_pred_date_list = torch.stack(train_pred_date_list, dim=0)
        train_pred_flux_list = torch.stack(train_pred_flux_list, dim=0)
        return train_seq_date_list, train_seq_flux_list, train_pred_date_list, train_pred_flux_list
        # return train_seq_flux_list, train_pred_flux_list

    # def validation_collate_fn(batch):
    #     return train_collate_fn(batch)

    if train:
        dataset = JeonpaDataset(configs, True, root_dir=root_dir)
        # train_sampler = DistributedSampler(dataset)

        return DataLoader(dataset=dataset,
                          batch_size=configs.train.batch_size,
                          # shuffle=True,
                          num_workers=configs.train.num_workers,
                          collate_fn=train_collate_fn,
                          # sampler=train_sampler,
                          # pin_memory=True,
                          # drop_last=True,
                          # sampler=None
                          )
    else:
        dataset = JeonpaDataset(configs, False, root_dir=root_dir)
        # vali_sampler = DistributedSampler(dataset)

        return DataLoader(dataset=dataset,
                          collate_fn=train_collate_fn,

                          batch_size=1,
                          shuffle=False,
                          num_workers=configs.train.num_workers,
                          # sampler=vali_sampler
                          )


def create_testloader(configs, root_dir=None):
    def test_collate_fn(batch):
        train_seq_date_list = list()
        train_seq_flux_list = list()
        date_list = list()

        for train_date_seq, train_flux_seq, date in batch:
            train_seq_date_list.append(torch.from_numpy(train_date_seq).float())
            train_seq_flux_list.append(torch.from_numpy(train_flux_seq).float())
            date_list.append(torch.from_numpy(date).float())
            print(train_seq_date_list[0].shape)

        train_seq_date_list = torch.stack(train_seq_date_list, dim=0)
        train_seq_flux_list = torch.stack(train_seq_flux_list, dim=0)
        date_list = torch.stack(date_list, dim=0)

        return train_seq_date_list, train_seq_flux_list, date_list

    return DataLoader(dataset=JeonpaTestDataset(configs, root_dir=root_dir),
                      collate_fn=test_collate_fn,
                      batch_size=1, shuffle=False, num_workers=0)


def get_data_from_path(configs, file_path, test=False, root_dir=None):
    if root_dir is None:
        root_dir = Path(os.getcwd()).parent.absolute()
    # print(f"root directory: {root_dir}")

    dataset = pd.read_csv(f'{root_dir}/{file_path}')
    # print(dataset)

    flux = np.array(dataset['flux'])
    print('len before interpolation:', len(flux))

    def to_datetime(str):
        if type(str) is int:
            # 그냥 2023년 10/11월이라 하자.
            # print(str)
            if str <= 31:
                res = datetime.datetime(2023, 10, str)
            else:
                res = datetime.datetime(2023, 11, str - 31)
        else:
            split = str.split('/')
            res = datetime.datetime(2000 + int(split[2]), int(split[0]), int(split[1]))
        return res

    # print(dataset)
    df_stamp = dataset['date']
    # print(df_stamp)

    df_stamp = df_stamp.apply(to_datetime)
    date = np.concatenate((np.array(df_stamp.apply(lambda row: row.month, 1))[:, np.newaxis],
                           np.array(df_stamp.apply(lambda row: row.day, 1))[:, np.newaxis],
                           np.array(df_stamp.apply(lambda row: row.weekday(), 1))[:, np.newaxis],
                           # np.array(df_stamp.apply(lambda row: row.hour, 1))[:, np.newaxis],
                           # np.array(df_stamp.apply(lambda row: row.minute, 1))[:, np.newaxis]
                           ),
                          axis=1)  #.astype(dtype=np.int64)
    # print(date)

    interpolation_model = InterpolationRemoveLongMissingValue(configs)
    # interpolation_model = InterpolationPoly(configs)
    # interpolation_model = InterpolationKNN(configs)

    date, flux = interpolation_model.get_dataset(date, flux, test)
    print('len after interpolation:', len(flux))


    # print(df_stamp)
    # print(date[0])

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
        self.train_date = date[:split_index]
        self.train_flux = flux[:split_index]
        self.test_date = date[split_index:]
        self.test_flux = flux[split_index:]

    def __len__(self):
        if self.train:
            return len(self.train_flux)
        else:
            return len(self.test_flux)

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
            return self.train_date[idx], self.train_flux[idx]
        else:
            # validation_seq = self.test_flux[idx:idx + self.seq_len][:, np.newaxis]
            # validation_pred = self.test_flux[idx + self.seq_len:idx + self.seq_len + self.pred_len][:, np.newaxis]
            # return validation_seq, validation_pred
            return self.test_date[idx], self.test_flux[idx]


class JeonpaTestDataset(Dataset):
    def __init__(self, configs, root_dir=None):
        self.configs = configs
        self.seq_len = configs.model.seq_len
        self.date, self.flux = get_data_from_path(configs, self.configs.data.testset, test=True, root_dir=root_dir)
        # print(self.date[0].shxape)
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

        # date = np.concatenate((np.array(df_stamp.apply(lambda row: row.month, 1))[:, np.newaxis],
        #                        np.array(df_stamp.apply(lambda row: row.day, 1))[:, np.newaxis],
        #                        np.array(df_stamp.apply(lambda row: row.weekday(), 1))[:, np.newaxis],
        #                        # np.array(df_stamp.apply(lambda row: row.hour, 1))[:, np.newaxis],
        #                        # np.array(df_stamp.apply(lambda row: row.minute, 1))[:, np.newaxis]
        #                        ),
        months = []
        days = []
        weekdays = []
        for i in range(60):  # 예측 30개
            if i <= 29:
                res = datetime.datetime(2023, 11, i + 1)
            else:
                # print(i)
                res = datetime.datetime(2023, 12, i - 29)

            months.append(res.month)
            days.append(res.day)
            weekdays.append(res.weekday())

        date = np.concatenate((np.array(months)[:, np.newaxis],
                               np.array(days)[:, np.newaxis],
                               np.array(weekdays)[:, np.newaxis],
                               # np.array(df_stamp.apply(lambda row: row.hour, 1))[:, np.newaxis],
                               # np.array(df_stamp.apply(lambda row: row.minute, 1))[:, np.newaxis]
                               ),
                              axis=1)
        # print(date)

        return self.date[idx], self.flux[idx], date
