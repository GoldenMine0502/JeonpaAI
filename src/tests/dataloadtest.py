import numpy as np
import os
from pathlib import Path
from configs import Config
import pandas as pd

root_dir = Path(os.getcwd()).parent.parent.absolute()
print(f"root directory: {root_dir}")

config_path = f'{root_dir}/config.yaml'
config = Config(config_path)

trainset_filepath = f'{root_dir}/{config.data.trainset}'
print(f'trainset path: {trainset_filepath}')

trainset = pd.read_csv(f'{root_dir}/{config.data.trainset}')
print(trainset)

date = np.array(trainset['date'])
flux = np.array(trainset['flux'])
mean_flux = np.nanmean(flux)

print(f'flux: {flux}, mean: {mean_flux}')

flux[np.isnan(flux)] = mean_flux
print(f'모든 결측치를 평균으로 설정: {flux}')

