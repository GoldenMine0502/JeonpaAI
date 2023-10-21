import numpy as np
import os
from pathlib import Path
from configs import Config
import pandas as pd
from dataloader import create_dataloader

root_dir = Path(os.getcwd()).parent.parent.absolute()
print(f"root directory: {root_dir}")

config_path = f'{root_dir}/config.yaml'
config = Config(config_path)
train_dataloader = create_dataloader(config, True, root_dir)

for train in train_dataloader:
    print(train)