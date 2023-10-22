import numpy as np
from scipy.interpolate import interp1d

class InterpolationAllAverage:
    def __init__(self, configs):
        self.config = configs
        self.seq_len = configs.model.seq_len
        self.pred_len = configs.model.pred_len
    def get_dataset(self, flux, test=False):
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
    def __init__(self, configs, pass_count=10):
        self.config = configs
        self.seq_len = configs.model.seq_len
        self.pred_len = configs.model.pred_len
        self.pass_count = pass_count

    def replace_nan_to_mean(self, flux, mean_flux):
        flux[np.isnan(flux)] = mean_flux

    def get_dataset(self, flux, test=False):
        dataset = []

        mean_flux = np.nanmean(flux)

        if test:
            for idx in range(len(flux) - self.seq_len + 1):
                test_seq = flux[idx:idx + self.seq_len][:, np.newaxis]

                dataset.append(test_seq)
        else:
            for idx in range(len(flux) - self.seq_len - self.pred_len + 1):
                train_seq = flux[idx:idx + self.seq_len][:, np.newaxis]  # 10~70
                train_pred = flux[idx + self.seq_len:idx + self.seq_len + self.pred_len][:, np.newaxis]  # 70~100

                # 데이터가 연속으로 결측치면 제거
                count = 0
                to_add = True
                for seq_value in np.concatenate((train_seq, train_pred)):
                    if np.isnan(seq_value):
                        count += 1
                    else:
                        count = 0

                    if count >= self.pass_count:
                        to_add = False
                        break

                if to_add:
                    self.replace_nan_to_mean(train_seq, mean_flux)
                    self.replace_nan_to_mean(train_pred, mean_flux)
                    dataset.append((train_seq, train_pred))

        return dataset


class InterpolationPoly:
    def __init__(self, configs, pass_count=10):
        self.config = configs
        self.seq_len = configs.model.seq_len
        self.pred_len = configs.model.pred_len

    def get_dataset(self, flux, test):
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