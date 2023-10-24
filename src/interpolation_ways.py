import random

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline
from sklearn.impute import KNNImputer


class InterpolationAllAverage:
    def __init__(self, configs):
        self.config = configs
        self.seq_len = configs.model.seq_len
        self.pred_len = configs.model.pred_len

    def get_dataset(self, date, flux, test=False):
        mean_flux = np.nanmean(flux)
        flux[np.isnan(flux)] = mean_flux

        dataset = []

        if test:
            for idx in range(len(flux) - self.seq_len + 1):
                test_seq = flux[idx:idx + self.seq_len][:, np.newaxis]

                dataset.append(test_seq)
        else:
            # 입력, 출력 길이에 따라 사용할 수 있는 데이터의 양이 달라진다.
            # raw 데이터 길이가 100이고, seq_len = 60, pred_len = 30인 경우 -> 90개
            # 11개의 학습 데이터를 뽑을 수 있음
            # 0-89, 1-90, 2-91, 3-92, 4-93,
            # 5-94, 6-95, 7-96, 8-97, 9-98, 10-99
            # total_train_len = dataset_len - self.seq_len - self.pred_len + 1
            for idx in range(len(flux) - self.seq_len - self.pred_len + 1):
                train_seq = flux[idx:idx + self.seq_len][:, np.newaxis]  # 10~70
                train_pred = flux[idx + self.seq_len:idx + self.seq_len + self.pred_len][:, np.newaxis]  # 70~100

                dataset.append((train_seq, train_pred))

        return dataset


# 모든 Nan을 전체 평균으로 대치

class InterpolationRemoveLongMissingValue:
    def __init__(self, configs, pass_count=60):
        self.config = configs
        self.seq_len = configs.model.seq_len
        self.label_len = configs.model.label_len
        self.pred_len = configs.model.pred_len
        self.pass_count = pass_count

    def replace_nan_to_mean(self, flux, mean_flux):
        flux[np.isnan(flux)] = mean_flux

    def get_dataset(self, date, flux, test=False):
        random.seed(1234)
        if test:
            dataset_date = []
            dataset_flux = []
            for idx in range(len(flux) - self.seq_len + 1):
                test_flux_seq = flux[idx:idx + self.seq_len][:, np.newaxis]
                test_date_seq = date[idx:idx + self.seq_len]
                # s_begin = idx
                # s_end = s_begin + self.seq_len
                #
                # test_flux_seq = flux[s_begin:s_end][:, np.newaxis]
                # # print(train_seq)
                # test_date_seq = date[s_begin:s_end]

                dataset_flux.append(test_flux_seq)
                dataset_date.append(test_date_seq)

            return dataset_date, dataset_flux
        else:
            fancy = []
            for idx in range(len(flux) - self.seq_len - self.pred_len + 1):
                train_seq = flux[idx:idx + self.seq_len][:, np.newaxis]  # 10~70
                # print(train_seq)
                train_pred = flux[idx + self.seq_len:idx + self.seq_len + self.pred_len][:, np.newaxis]  # 70~100
                train_date_seq = date[idx:idx + self.seq_len]
                train_date_pred = date[idx + self.seq_len:idx + self.seq_len + self.pred_len]
                # s_begin = idx
                # s_end = s_begin + self.seq_len
                # r_begin = s_end - self.label_len
                # r_end = r_begin + self.label_len + self.pred_len
                #
                # train_seq = flux[s_begin:s_end][:, np.newaxis]  # 10~70
                # # print(train_seq)
                # train_pred = flux[r_begin:r_end][:, np.newaxis]  # 70~100

                # 데이터가 연속으로 결측치면 제거
                count = 0
                to_add = True
                for seq_value in np.concatenate((train_seq, train_pred)):
                    # print(seq_value)
                    if np.isnan(seq_value):
                        count += 1
                    # else:
                    #     count = 0

                    if count >= self.pass_count:
                        to_add = False
                        break

                # print(to_add)

                # if to_add:
                fancy.append(to_add)

            # linear_interpolation(flux)
            flux = interpolate_knn(flux)
            # flux = interpolate_cubic_spline(flux)

            dataset_flux = []
            dataset_date = []
            for idx in range(len(flux) - self.seq_len - self.pred_len + 1):
                train_seq = flux[idx:idx + self.seq_len][:, np.newaxis]  # 10~70
                # print(train_seq)
                train_pred = flux[idx + self.seq_len:idx + self.seq_len + self.pred_len][:, np.newaxis]  # 70~100
                train_date_seq = date[idx:idx + self.seq_len]
                train_date_pred = date[idx + self.seq_len:idx + self.seq_len + self.pred_len]
                # s_begin = idx
                # s_end = s_begin + self.seq_len
                # r_begin = s_end - self.label_len
                # r_end = r_begin + self.label_len + self.pred_len
                #
                # train_seq = flux[s_begin:s_end][:, np.newaxis]  # 10~70
                # # print(train_seq)
                # train_pred = flux[r_begin:r_end][:, np.newaxis]  # 70~100
                # train_date_seq = date[s_begin:s_end]  # 10~70
                # # print(train_seq)
                # train_date_pred = date[r_begin:r_end]  # 70~100


                if fancy[idx]:
                    if len(dataset_flux) == 0:
                        dataset_flux.append((train_seq, train_pred))
                        dataset_date.append((train_date_seq, train_date_pred))
                    else:
                        idx_random = random.randint(0, len(dataset_flux))
                        dataset_flux.insert(idx_random, (train_seq, train_pred))
                        dataset_date.insert(idx_random, (train_date_seq, train_date_pred))

            # mean_flux = np.nanmean(flux)
            # flux[np.isnan(flux)] = mean_flux

            # for train_seq, train_pred in dataset:
            #     # batch_mean_seq = np.nanmean(train_seq)
            #     # batch_mean_pred = np.nanmean(train_pred)
            #     batch_mean = np.nanmean(np.concatenate([train_seq, train_pred]))
            #     self.replace_nan_to_mean(train_seq, batch_mean)
            #     self.replace_nan_to_mean(train_pred, batch_mean)
            #     # self.replace_nan_to_mean(train_seq, mean_flux)
            #     # self.replace_nan_to_mean(train_pred, mean_flux)

            return dataset_date, dataset_flux


def linear_interpolation(flux):
    last_nan = None
    for i, data in enumerate(flux):
        if np.isnan(data):
            if last_nan is None:
                last_nan = i
        else:
            if last_nan is not None:
                first_index = last_nan - 1
                last_index = i

                first_value = flux[last_nan - 1].copy()
                last_value = flux[i].copy()

                for j in range(last_nan - 1, i + 1):
                    flux[j] = first_value + (last_value - first_value) / (last_index - first_index + 1) * (j - first_index + 1)
                last_nan = None

                # print(last_value, flux[i], (last_index - first_index + 1), (i - first_index + 1))

def interpolate_knn(flux):
    imputer = KNNImputer(n_neighbors=135)
    # x = np.arange(len(flux)).copy().reshape(-1, 1)
    # y = flux.copy().reshape(1, -1)
    # print(x.shape)
    # print(y.shape)
    x = []
    y = []
    # nans = np.isnan(flux)

    # for i in range(len(flux)):
    #     if not nans[i]:
    #         x.append(i)
    #         y.append(flux[i])
    for i in range(len(flux)):
        # if not nans[i]:
        x.append(i)
        y.append(flux[i])

    flux = imputer.fit_transform(pd.DataFrame({'x': x, 'y': y}))
    # print(type(flux))
    return flux[:, 1]

def interpolate_cubic_spline(flux):
    x = []
    y = []
    nans = np.isnan(flux)

    for i in range(len(flux)):
        if not nans[i]:
            x.append(i)
            y.append(flux[i])

    f = CubicSpline(x, y, bc_type='natural')

    flux = f(np.arange(len(flux)))

    return flux

class InterpolationPoly:
    def __init__(self, configs):
        self.config = configs
        self.seq_len = configs.model.seq_len
        self.pred_len = configs.model.pred_len

    def get_dataset(self, date, flux, test):
        # 다항식 보간 함수
        poly_interpolator = interp1d(np.arange(len(flux)), flux, kind='cubic')
        # 결측값 보간
        flux = poly_interpolator(np.arange(len(flux)))

        dataset = []

        if test:
            for idx in range(len(flux) - self.seq_len + 1):
                test_seq = flux[idx:idx + self.seq_len][:, np.newaxis]

                dataset.append(test_seq)
        else:
            for idx in range(len(flux) - self.seq_len - self.pred_len + 1):
                train_seq = flux[idx:idx + self.seq_len][:, np.newaxis]  # 10~70
                train_pred = flux[idx + self.seq_len:idx + self.seq_len + self.pred_len][:, np.newaxis]  # 70~100

                dataset.append((train_seq, train_pred))

        return dataset

class InterpolationKNN:
    def __init__(self, configs):
        self.config = configs
        self.seq_len = configs.model.seq_len
        self.pred_len = configs.model.pred_len

    def get_dataset(self, flux, test):
        imputer = KNNImputer(n_neighbors=45)
        # x = np.arange(len(flux)).copy().reshape(-1, 1)
        # y = flux.copy().reshape(1, -1)
        dataframe = pd.DataFrame({'y': flux})
        # print(x.shape)
        # print(y.shape)
        flux = imputer.fit_transform(dataframe).copy().reshape(-1)

        dataset = []

        if test:
            for idx in range(len(flux) - self.seq_len + 1):
                test_seq = flux[idx:idx + self.seq_len][:, np.newaxis]

                dataset.append(test_seq)
        else:
            for idx in range(len(flux) - self.seq_len - self.pred_len + 1):
                train_seq = flux[idx:idx + self.seq_len][:, np.newaxis]  # 10~70
                train_pred = flux[idx + self.seq_len:idx + self.seq_len + self.pred_len][:, np.newaxis]  # 70~100

                dataset.append((train_seq, train_pred))

        return dataset
