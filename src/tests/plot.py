import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from configs import Config
from scipy.interpolate import interp1d
# from scipy.interpolate import make_interp_spline, BSpline

root_dir = Path(os.getcwd()).parent.absolute()
print(f"root directory: {root_dir}")

config_path = f'{root_dir}/config.yaml'
config = Config(config_path)

datalist = [(f'{root_dir}/{config.data.trainset}', 'r'), ]

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

for data, color in datalist:
    flux = np.array(pd.read_csv(data)['flux'])
    x = np.arange(len(flux))

    linear_interpolation(flux)
    # mean_flux = np.nanmean(flux)
    # flux[np.isnan(flux)] = mean_flux

    # poly_interpolator = interp1d(np.arange(len(flux)), flux, kind='cubic')
    # # 결측값 보간
    # flux = poly_interpolator(np.arange(len(flux)))

    print(flux)

    plt.plot(x, flux, color=color, linestyle='-', linewidth=1.5)

plt.show()

