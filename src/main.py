import os
import math
import torch
import torch.nn as nn
from model import Model
from pathlib import Path
from configs import Config
from dataloader import create_dataloader
from train import Train

# device
device = torch.device('cpu')

# config
root_dir = Path(os.getcwd()).absolute()
print(f"root directory: {root_dir}")

config_path = f'{root_dir}/config.yaml'
config = Config(config_path)

# hp_str
with open(config_path, 'r') as f:
    hp_str = ''.join(f.readlines())

# train
train = Train(config, hp_str, root_dir=root_dir)
train.train()
