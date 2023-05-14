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

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        b, i, c = x.size()
        res = torch.zeros(b, self.pred_len, c)
        for i in range(33):
            z = x[:, :, i * 54:(i + 1) * 54]
            seasonal_init, trend_init = self.decompsition(z)
            seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

            z = seasonal_output + trend_output
            res[:, :, i * 54:(i + 1) * 54] = torch.clip(z.permute(0, 2, 1), min=0)  # [Batch, Output length, Channel]
        return res


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
        b, i, c = x.size()
        res = torch.zeros(b, self.pred_len, c)
        for i in range(33):
            z = x[:, :, i*54:(i+1)*54]
            seq_last = z[:, -1:, :].detach()
            z = z - seq_last
            z = self.Linear(z.permute(0, 2, 1)).permute(0, 2, 1)
            z = z + seq_last
            res[:, :, i*54:(i+1)*54] = torch.relu(z)  # [Batch, Output length, Channel]
        return res
