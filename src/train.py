import numpy as np
import math
import os
import xlsxwriter

from models.linears import *
from models.dcrnn import DCRNNModel
from models.crnn import CRNN
from models.autoformer import AutoFormer
from dataloader import create_dataloader, create_testloader


class Train:

    def __init__(self, config, hp_str, writer, root_dir=None):
        self.config = config
        self.hp_str = hp_str
        self.root_dir = root_dir
        self.writer = writer
        self.trainloader = create_dataloader(config, True, root_dir=root_dir)
        self.validationloader = create_dataloader(config, False, root_dir=root_dir)
        self.testloader = create_testloader(config, root_dir=root_dir)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        # self.model = DLinear(self.config)

        # self.model = DCRNNModel(
        #     adj_mat=None,
        #     batch_size=config.train.batch_size,
        #     enc_input_dim=2,
        #     dec_input_dim=1,
        #     max_diffusion_step=2,
        #     num_nodes=207,
        #     num_rnn_layers=2,
        #     rnn_units=64,
        #     seq_len=12,
        #     output_dim=1,
        #     filter_type=None
        # )
        # self.model = CRNN(self.config.model.seq_len, self.config.model.pred_len)

        # python -u run.py \
        #   --is_training 1 \
        #   --root_path ./dataset/electricity/ \
        #   --data_path electricity.csv \
        #   --model_id ECL_96_96 \
        #   --model Autoformer \
        #   --data custom \
        #   --features M \
        #   --seq_len 96 \
        #   --label_len 48 \
        #   --pred_len 96 \
        #   --e_layers 2 \
        #   --d_layers 1 \
        #   --factor 3 \
        #   --enc_in 321 \
        #   --dec_in 321 \
        #   --c_out 321 \
        #   --des 'Exp' \
        #   --itr 1
        self.model = AutoFormer(config)
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()

        # local_rank = int(os.environ["LOCAL_RANK"])
        # print("local rank:", local_rank)
        #
        # self.model = torch.nn.parallel.DistributedDataParallel(
        #     self.model,
        #     find_unused_parameters=True,
        #     # device_ids=[local_rank],
        #     # output_device=local_rank,
        # )

        self.model.to(self.device)

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

    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.config.model.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.config.model.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        def _run_model():
            # print(batch_x.shape, batch_x_mark.shape, dec_inp.shape, batch_y_mark.shape, batch_y.shape)
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.config.model.output_attention:
                outputs = outputs[0]
            return outputs

        # if self.args.use_amp:
        #     with torch.cuda.amp.autocast():
        #         outputs = _run_model()
        # else:
        outputs = _run_model()

        f_dim = -1 if self.config.model.features == 'MS' else 0
        outputs = outputs[:, -self.config.model.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.config.model.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y

    def train(self):
        step = 0
        # try:
        while True:
            step += 1

            self.model.train()
            losses = []
            # self.trainloader.sampler.set_epoch(step)

            for train_date_seq, train_flux_seq, train_date_pred, train_flux_pred in self.trainloader:  # 요게 다 돌면 에포크
            # for train_flux_seq, train_flux_pred in self.trainloader:
                # CRNN
                train_date_seq = train_date_seq.to(self.device)
                train_flux_seq = train_flux_seq.to(self.device)
                train_date_pred = train_date_pred.to(self.device)
                train_flux_pred = train_flux_pred.to(self.device)
                # train_seq = train_seq.squeeze(2).to(self.device)
                # train_pred = train_pred.squeeze(2).to(self.device)
                # print("train_seq:", train_seq)
                # print("train_pred:", train_pred)
                # print(train_date_seq.shape, train_date_pred.shape)

                # result = self.model(train_flux_seq)
                result, batch_y = self._predict(train_flux_seq, train_flux_pred, train_date_seq, train_date_pred)
                # RMSE = torch.sqrt(criterion(x, y))
                # loss = torch.sqrt(self.criterion(result, train_pred))
                loss = self.criterion(result, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss = loss.item()
                losses.append(loss)

                if loss > 1e8 or math.isnan(loss):
                    print("Loss exploded to %.02f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")

            loss = np.mean(losses)

            self.writer.write_train(step, loss)
            # write loss to tensorboard
            if step % self.config.train.summary_interval == 0:
                val_loss = self.validate()
                val_loss_rmse = self.validate(rmse=True)
                self.writer.write_val(step, val_loss, val_loss_rmse)

                print("Wrote summary at step %d, loss: %f, val_loss: %f, val_rmse_loss: %f"
                      % (step, loss, val_loss, val_loss_rmse))

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

            for train_date_seq, train_flux_seq, train_date_pred, train_flux_pred in self.validationloader:  # 요게 다 돌면 에포크
            # for train_flux_seq, train_flux_pred in self.trainloader:
                # CRNN
                train_date_seq = train_date_seq.to(self.device)
                train_flux_seq = train_flux_seq.to(self.device)
                train_date_pred = train_date_pred.to(self.device)
                train_flux_pred = train_flux_pred.to(self.device)
                # train_seq = train_seq.squeeze(2).to(self.device)
                # train_pred = train_pred.squeeze(2).to(self.device)
                # print("train_seq:", train_seq)
                # print("train_pred:", train_pred)
                # print(train_seq.shape, train_pred.shape)

                # result = self.model(train_flux_seq)
                result, batch_y = self._predict(train_flux_seq, train_flux_pred, train_date_seq, train_date_pred)
            # for validation_seq, validation_pred in self.validationloader:
            #     validation_seq = validation_seq.to(self.device)
            #     validation_pred = validation_pred.to(self.device)
            #     # validation_seq = validation_seq.squeeze(2).to(self.device)
            #     # validation_pred = validation_pred.squeeze(2).to(self.device)
            #
            #     result = self.model(validation_seq)
                # RMSE = torch.sqrt(criterion(x, y))
                # loss = torch.sqrt(criterion(result, train_pred))
                if rmse:
                    loss = torch.sqrt(criterion(result, batch_y))
                else:
                    loss = criterion(result, batch_y)
                loss = loss.item()
                losses.append(loss)

            loss = np.mean(losses)
            # print(f"validation loss: {loss}")

        return loss

    def test(self, step):
        with torch.no_grad():
            for test_date_seq, test_flux_seq, date in self.testloader:
                test_date_seq = test_date_seq.to(self.device)
                test_flux_seq = test_flux_seq.to(self.device)

                # test_seq = test_seq.squeeze(2).to(self.device)
                test_flux_pred = torch.zeros((test_date_seq.size(dim=0), self.model.pred_len, self.config.model.channels))

                result, batch_y = self._predict(test_flux_seq, test_flux_pred, test_date_seq, date)
                # result, batch_y = self._predict(train_flux_seq, train_flux_pred, train_date_seq, train_date_pred)
                self.write_csv(result, step)

    def write_csv(self, result, step):
        # print(result.shape)
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
            worksheet.write(i + 1, 1, date)

        workbook.close()

        print(f'Saved test to {filepath}')
