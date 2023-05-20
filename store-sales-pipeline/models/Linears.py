import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self, seq_len, pred_len):
        super(LinearModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return torch.clip(x, min=0)  # [Batch, Output length, Channel]


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DlinearModel(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, seq_len, pred_len):
        super(DlinearModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = 5
        self.decompsition = series_decomp(kernel_size)

        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        self.weight_init = 1.0 / self.seq_len
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.constant_(self.Linear_Seasonal.weight, self.weight_init)
        nn.init.constant_(self.Linear_Seasonal.bias, 0)
        nn.init.constant_(self.Linear_Trend.weight, self.weight_init)
        nn.init.constant_(self.Linear_Trend.bias, 0)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return torch.abs(x.permute(0, 2, 1))


class DlinearModelMulti(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, seq_len, hidden_size, pred_len):
        super(DlinearModelMulti, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size

        # Decompsition Kernel Size
        kernel_size = 5
        self.decompsition = series_decomp(kernel_size)

        self.Linear_Seasonal1 = nn.Linear(self.seq_len, self.hidden_size)
        self.Relu_Seasonal = nn.ReLU()
        self.Linear_Seasonal2 = nn.Linear(hidden_size, self.pred_len)

        self.Linear_Trend1 = nn.Linear(self.seq_len, self.hidden_size)
        self.Relu_Trend = nn.ReLU()
        self.Linear_Trend2 = nn.Linear(self.hidden_size, self.pred_len)

        self.initialize_weights()

    def initialize_weights(self):
        weight_init1 = 1.0 / self.seq_len
        weight_init2 = 1.0 / self.hidden_size
        nn.init.constant_(self.Linear_Seasonal1.weight, weight_init1)
        nn.init.constant_(self.Linear_Seasonal1.bias, 0)
        nn.init.constant_(self.Linear_Seasonal2.weight, weight_init2)
        nn.init.constant_(self.Linear_Seasonal2.bias, 0)

        nn.init.constant_(self.Linear_Trend1.weight, weight_init1)
        nn.init.constant_(self.Linear_Trend1.bias, 0)
        nn.init.constant_(self.Linear_Trend2.weight, weight_init2)
        nn.init.constant_(self.Linear_Trend2.bias, 0)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal2(self.Relu_Seasonal(self.Linear_Seasonal1(seasonal_init)))
        trend_output = self.Linear_Trend2(self.Relu_Trend(self.Linear_Trend1(trend_init)))
        x = seasonal_output + trend_output
        return torch.abs(x.permute(0, 2, 1))


class NlinearModel(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, seq_len, pred_len):
        super(NlinearModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return torch.clip(x, min=0)  # [Batch, Output length, Channel]
