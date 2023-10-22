import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from configs import Config
from sklearn.impute import KNNImputer
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline, BSpline, CubicSpline

root_dir = Path(os.getcwd()).parent.absolute()
print(f"root directory: {root_dir}")

config_path = f'{root_dir}/config.yaml'
config = Config(config_path)

datalist = [(f'{root_dir}/{config.data.trainset}', 'r'), ]


def interpolate_cubic_spline(flux):
    x = []
    y = []
    nans = np.isnan(flux)

    for i in range(len(flux)):
        if not nans[i]:
            x.append(i)
            y.append(flux[i])

    f = CubicSpline(x, y, bc_type='clamped')

    flux = f(np.arange(len(flux)))

    return flux

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

for data, color in datalist:
    flux = np.array(pd.read_csv(data)['flux'])

    # flux = interpolate_cubic_spline(flux)
    flux = interpolate_knn(flux)
    # imputer = KNNImputer(n_neighbors=90)
    # x = np.arange(len(flux)).copy().reshape(-1, 1)
    # y = flux.copy().reshape(1, -1)
    # dataframe = pd.DataFrame({'y': flux})
    # print(x.shape)
    # print(y.shape)
    # result = imputer.fit_transform(dataframe)
    # flux = result
    # print(imputer.fit_transform(x, y))

    x = np.arange(len(flux))

    # linear_interpolation(flux)
    # mean_flux = np.nanmean(flux)
    # flux[np.isnan(flux)] = mean_flux

    # poly_interpolator = interp1d(np.arange(len(flux)), flux, kind='cubic')
    # # 결측값 보간
    # flux = poly_interpolator(np.arange(len(flux)))

    # print(flux)

    plt.plot(x, flux, color=color, linestyle='-', linewidth=1.5)

plt.show()

