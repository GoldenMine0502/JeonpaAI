import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from configs import Config
# from scipy.interpolate import make_interp_spline, BSpline

root_dir = Path(os.getcwd()).absolute()
print(f"root directory: {root_dir}")

config_path = f'{root_dir}/config.yaml'
config = Config(config_path)

datalist = [(f'{root_dir}/{config.data.trainset}', 'r'), ]

for data, color in datalist:
    data = pd.read_csv(data)['flux']
    x = np.arange(len(data))

    plt.plot(x, data, color=color, linestyle='-', linewidth=1.5)

plt.show()

