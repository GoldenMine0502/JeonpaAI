import numpy as np
import math
import torch
import torch.nn as nn
import os

import dataloader
from model import Model
from dataloader import create_dataloader, create_testloader


class Train:

    def __init__(self, config, hp_str, root_dir=None):
        self.config = config
        self.hp_str = hp_str
        self.root_dir = root_dir
        self.trainloader = create_dataloader(config, True, root_dir=root_dir)
        self.validationloader = create_dataloader(config, False, root_dir=root_dir)
        self.testloader = create_testloader(config, root_dir=root_dir)

        self.model = self.get_model()
        self.optimizer = self.get_optimizer(self.model)
        self.criterion = self.get_criterion()

    def get_model(self):
        device = torch.device('cpu')
        model = Model(self.config)
        model.to(device)

        return model

    def get_criterion(self):
        # Berhu_loss
        def berhu_loss(y_pred, y_true):
            delta = 0.2  # default 0.2
            abs_error = torch.abs(y_pred - y_true)
            c = delta * torch.max(abs_error).detach()
            return torch.mean(torch.where(abs_error <= c, abs_error, (abs_error ** 2 + c ** 2 / (2 * c))))

        criterion = nn.HuberLoss(delta=1)
        # criterion = berhu_loss
        # criterion = nn.L1Loss()
        # criterion = nn.MSELoss()

        return criterion

    def get_optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.train.adam)
        # optimizer = torch.optim.SGD(model.parameters(), lr=config.train.adam)

        return optimizer

    def train(self):
        step = 0
        # try:
        while True:
            self.model.train()
            losses = []
            for train_seq, train_pred in self.trainloader:  # 요게 다 돌면 에포크
                # print("train_seq:", train_seq)
                # print("train_pred:", train_pred)
                # print(train_seq.shape, train_pred.shape)

                result = self.model(train_seq)
                # RMSE = torch.sqrt(criterion(x, y))
                # loss = torch.sqrt(criterion(result, train_pred))
                loss = self.criterion(result, train_pred)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss = loss.item()
                losses.append(loss)

                if loss > 1e8 or math.isnan(loss):
                    print("Loss exploded to %.02f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")

            loss = np.mean(losses)

            step += 1

            # write loss to tensorboard
            if step % self.config.train.summary_interval == 0:
                # writer.log_training(loss, step)
                val_loss = self.validate()
                print("Wrote summary at step %d, loss: %f, val_loss: %f" % (step, loss, val_loss))
            # 1. save checkpoint file to resume training
            # 2. evaluate and save sample to tensorboard
            if step % self.config.train.checkpoint_interval == 0:
                chkpt_path = f'{self.root_dir}/{self.config.log.chkpt_dir}'
                save_path = os.path.join(chkpt_path, 'chkpt_%d.pt' % step)
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'step': step,
                    'hp_str': self.hp_str,
                }, save_path)
                print("Saved checkpoint to: %s" % save_path)
                # self.test()

            if step == self.config.train.step_limit:
                print(f"Quit step {step}")
                break

        # except Exception as e:
        #     print("Exiting due to exception: %s" % e)
        #     traceback.print_exc()

    def validate(self):
        with torch.no_grad():
            losses = []

            for validation_seq, validation_pred in self.validationloader:
                result = self.model(validation_seq)
                # RMSE = torch.sqrt(criterion(x, y))
                # loss = torch.sqrt(criterion(result, train_pred))
                loss = self.criterion(result, validation_pred)
                loss = loss.item()
                losses.append(loss)

            loss = np.mean(losses)
            # print(f"validation loss: {loss}")

        return loss

    def test(self):
        with torch.no_grad():
            for test_seq in self.testloader:
                result = self.model(test_seq)
                print(result)
