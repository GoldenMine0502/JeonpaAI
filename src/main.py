import os
from pathlib import Path
from configs import Config
from train import Train
from tensorboardX import SummaryWriter

# config
root_dir = Path(os.getcwd()).absolute()
print(f"root directory: {root_dir}")

config_path = f'{root_dir}/config.yaml'
config = Config(config_path)

# hp_str
with open(config_path, 'r') as f:
    hp_str = ''.join(f.readlines())


# writer
class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.hp = hp

    def write_train(self, step, loss):
        self.add_scalar('loss', loss, step)

    def write_val(self, step, val_loss, val_rmse_loss):
        self.add_scalar('val_loss', val_loss, step)
        self.add_scalar('val_rmse_loss', val_rmse_loss, step)


log_dir = os.path.join(root_dir, config.log.log_dir)
os.makedirs(log_dir, exist_ok=True)
writer = MyWriter(config, log_dir)

# train
train = Train(config, hp_str, writer, root_dir=root_dir)
train.train()
