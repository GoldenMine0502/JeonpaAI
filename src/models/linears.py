import torch
from torch import nn


# 시계열 데이터를 분해하는 부분
class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # 시계열 데이터의 앞뒤에 맨 앞값과 맨 앞뒤를 여러번 배열하면서 패딩
        # [batch_size, sequence, value]
        # [[1, 2, 3, 4]
        # [1, 2, 3, 4]
        # [1, 2, 3, 4]
        # [1, 2, 3, 4]
        # [1, 2, 3, 4]
        # [1, 2, 3, 4]
        # [1, 2, 3, 4]]

        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# 1-layer linear network 구현 부분
class DLinear(nn.Module):
    """
    DLinear
    """

    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.seq_len = configs.model.seq_len  # 60일간 데이터로
        self.pred_len = configs.model.pred_len  # 30일 미래예측
        self.individual = configs.model.individual
        self.channels = configs.model.channels

        # Decompsition Kernel Size
        kernel_size = 15
        self.decompsition = SeriesDecomp(kernel_size)

        mid = 120

        self.linears = [self.seq_len, mid, mid, mid, mid, self.pred_len]

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Decoder.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = list()
            self.Linear_Seasonal_Dropout = list()
            self.Linear_Trend = list()
            self.Linear_Trend_Dropout = list()
            self.Linear_Decoder = list()

            for i in range(len(self.linears) - 1):
                start = self.linears[i]
                end = self.linears[i + 1]
                Linear_Seasonal = nn.Linear(start, end)
                Linear_Trend = nn.Linear(start, end)
                Linear_Decoder = nn.Linear(start, end)
                Linear_Seasonal.weight = nn.Parameter((1 / start) * torch.ones([end, start]))
                Linear_Trend.weight = nn.Parameter((1 / start) * torch.ones([end, start]))

                self.add_module("Linear_Seasonal{}".format(i), Linear_Seasonal)
                self.add_module("Linear_Trend{}".format(i), Linear_Trend)
                self.add_module("Linear_Decoder{}".format(i), Linear_Decoder)

                self.Linear_Seasonal.append(Linear_Seasonal)
                self.Linear_Trend.append(Linear_Trend)
                self.Linear_Decoder.append(Linear_Decoder)


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)

            # print(seasonal_init.shape, seasonal_output.shape, trend_init.shape, trend_output.shape)

            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = seasonal_init
            trend_output = trend_init

            for i in range(0, len(self.linears) - 1):
                seasonal_output = self.Linear_Seasonal[i](seasonal_output)
                trend_output = self.Linear_Trend[i](trend_output)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]


class NLinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs):
        super(NLinear, self).__init__()
        self.seq_len = configs.model.seq_len  # 60일간 데이터로
        self.pred_len = configs.model.pred_len  # 30일 미래예측
        self.individual = configs.model.individual
        self.channels = configs.model.channels
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]