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

for data, color in datalist:
    flux = pd.read_csv(data)['flux']
    x = np.arange(len(flux))

    mean_flux = np.nanmean(flux)
    flux[np.isnan(flux)] = mean_flux

    # poly_interpolator = interp1d(np.arange(len(flux)), flux, kind='cubic')
    # # 결측값 보간
    # flux = poly_interpolator(np.arange(len(flux)))

    print(flux)

    plt.plot(x, flux, color=color, linestyle='-', linewidth=1.5)

plt.show()

