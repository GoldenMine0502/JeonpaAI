import numpy as np
import math
import torch
import torch.nn as nn
import os
import xlsxwriter

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

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Model(self.config)
        self.model.to(self.device)
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()

    def get_criterion(self):
        # Berhu_loss
        def berhu_loss(y_pred, y_true):
            delta = 0.1  # default 0.2
            abs_error = torch.abs(y_pred - y_true)
            c = delta * torch.max(abs_error).detach()
            return torch.mean(torch.where(abs_error <= c, abs_error, (torch.sqrt(abs_error ** 2 + c ** 2) / (2 * c))))

        # criterion = nn.HuberLoss(delta=1)
        # criterion = berhu_loss
        # criterion = nn.L1Loss()
        criterion = nn.MSELoss()

        return criterion

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.train.adam)
        # optimizer = torch.optim.SGD(model.parameters(), lr=config.train.adam)

        return optimizer

    def train(self):
        step = 0
        # try:
        while True:
            self.model.train()
            losses = []
            for train_seq, train_pred in self.trainloader:  # 요게 다 돌면 에포크
                train_seq = train_seq.to(self.device)
                train_pred = train_pred.to(self.device)
                # print("train_seq:", train_seq)
                # print("train_pred:", train_pred)
                # print(train_seq.shape, train_pred.shape)

                result = self.model(train_seq)
                # RMSE = torch.sqrt(criterion(x, y))
                # loss = torch.sqrt(self.criterion(result, train_pred))
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
                val_loss_rmse = self.validate(rmse=True)
                print("Wrote summary at step %d, loss: %f, val_loss: %f, val_rmse_loss: %f" % (step, loss, val_loss, val_loss_rmse))
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
                self.test(step)

            if step == self.config.train.step_limit:
                print(f"Quit step {step}")
                break

        # except Exception as e:
        #     print("Exiting due to exception: %s" % e)
        #     traceback.print_exc()

    def validate(self, rmse=False):
        with torch.no_grad():
            losses = []


            criterion = nn.MSELoss() if rmse else self.criterion

            for validation_seq, validation_pred in self.validationloader:
                validation_seq = validation_seq.to(self.device)
                validation_pred = validation_pred.to(self.device)

                result = self.model(validation_seq)
                # RMSE = torch.sqrt(criterion(x, y))
                # loss = torch.sqrt(criterion(result, train_pred))
                if rmse:
                    loss = torch.sqrt(criterion(result, validation_pred))
                else:
                    loss = criterion(result, validation_pred)
                loss = loss.item()
                losses.append(loss)

            loss = np.mean(losses)
            # print(f"validation loss: {loss}")

        return loss

    def test(self, step):
        with torch.no_grad():
            for test_seq in self.testloader:
                test_seq = test_seq.to(self.device)
                result = self.model(test_seq)
                self.write_csv(result, step)

    def write_csv(self, result, step):
        result = result[0]
        # date,flux,,,
        # 1,,,,
        # 2,,,,
        # 3,,,,
        # with open(f'{self.root_dir}/result.csv', 'wt') as file:
        #     file.write("date,flux,,,\n")
        #
        #     length = len(result)
        #
        #     for date in range(length):
        #         file.write(f'{date + 1},{result[date][0]},,,\n')
        folder = f'{self.root_dir}/results'
        filepath = f'{self.root_dir}/results/result_{step}.xlsx'

        if not os.path.exists(folder):
            os.mkdir(f'{self.root_dir}/results')
        if os.path.exists(filepath):
            os.remove(filepath)

        workbook = xlsxwriter.Workbook(filepath)
        worksheet = workbook.add_worksheet()

        worksheet.write(0, 0, 'date')
        worksheet.write(0, 1, 'flux')
        for i, date in enumerate(result):
            worksheet.write(i + 1, 0, i + 1)
            worksheet.write(i + 1, 1, date[0])

        workbook.close()

        print(f'Saved test to {filepath}')