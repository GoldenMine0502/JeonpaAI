import os
import torch
from pathlib import Path
from configs import Config
from train import Train
from tensorboardX import SummaryWriter

# import torch.distributed as dist
# dist.init_process_group(backend="gloo")

# config
root_dir = Path(os.getcwd()).absolute()
print(f"root directory: {root_dir}")

config_path = f'{root_dir}/config.yaml'
config = Config(config_path)

# hp_str
with open(config_path, 'r') as f:
    hp_str = ''.join(f.readlines())

# torch.set_num_interop_threads(8)  # Inter-op parallelism
# torch.set_num_threads(8)  # Intra-op parallelism

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
# torch.Size([1, 60, 1]) torch.Size([1, 60, 1]) torch.Size([1, 60, 3]) torch.Size([1, 30, 3])
# torch.Size([256, 60, 1]) torch.Size([256, 60, 1]) torch.Size([256, 60, 3]) torch.Size([256, 60, 3])
train = Train(config, hp_str, writer, root_dir=root_dir)
# train.test(0)
train.train()
# torchrun
#     --standalone
#     --nnodes=1
#     --nproc_per_node=$NUM_TRAINERS
#     YOUR_TRAINING_SCRIPT.py