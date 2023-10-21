import os
import math
import torch
import torch.nn as nn
from model import Model
from pathlib import Path
from configs import Config
from dataloader import create_dataloader

# device
device = torch.device('cpu')

# config
root_dir = Path(os.getcwd()).absolute()
print(f"root directory: {root_dir}")

config_path = f'{root_dir}/config.yaml'
config = Config(config_path)

# datasets
trainloader = create_dataloader(config, True, root_dir)

# model
model = Model(config)
model.to(device)

# routes
chkpt_path = f'{root_dir}/{config.log.chkpt_dir}'

# train
with open(config_path, 'r') as f:
    hp_str = ''.join(f.readlines())

optimizer = torch.optim.Adam(model.parameters(), lr=config.train.adam)

step = 0
# try:
criterion = nn.HuberLoss()
while True:
    model.train()
    for train_seq, train_pred in trainloader:  # 요게 다 돌면 에포크
        # print("train_seq:", train_seq)
        # print("train_pred:", train_pred)
        # print(train_seq.shape, train_pred.shape)

        result = model(train_seq)
        loss = criterion(result, train_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()

        if loss > 1e8 or math.isnan(loss):
            print("Loss exploded to %.02f at step %d!" % (loss, step))
            raise Exception("Loss exploded")

    step += 1

    # write loss to tensorboard
    if step % config.train.summary_interval == 0:
        # writer.log_training(loss, step)
        print("Wrote summary at step %d, loss: %f" % (step, loss))
    # 1. save checkpoint file to resume training
    # 2. evaluate and save sample to tensorboard
    if step % config.train.checkpoint_interval == 0:
        save_path = os.path.join(chkpt_path, 'chkpt_%d.pt' % step)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'hp_str': hp_str,
        }, save_path)
        print("Saved checkpoint to: %s" % save_path)
        # validate(hp, audio, model, testloader, writer, step)

    if step == config.train.step_limit:
        raise Exception(f"step {step}")

# except Exception as e:
#     print("Exiting due to exception: %s" % e)
#     traceback.print_exc()
