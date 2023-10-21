from configs import Config
from model import Model

config_path = '../config.yaml'
config = Config(config_path)

model = Model(config)