import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class LazyScaler:
    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

    def fit(self, x):
        pass

    def fit_transform(self, x):
        return x


class MinMaxCustomScaler:
    def __init__(self, feature_range=(0, 1)):
        self.minmax_scaler = MinMaxScaler(feature_range)

    def transform(self, x):
        x = self.minmax_scaler.transform(x)
        return x

    def inverse_transform(self, x):
        x = self.minmax_scaler.inverse_transform(x)
        return x

    def fit(self, x):
        self.minmax_scaler.fit(x)

    def fit_transform(self, x):
        x = self.minmax_scaler.fit_transform(x)
        return x


class MinMaxLogScaler:
    def __init__(self, feature_range=(0, 1)):
        self.minmax_scaler = MinMaxScaler(feature_range)

    def transform(self, x):
        x = self.minmax_scaler.transform(x)
        x = np.log(x + 1)
        return x

    def inverse_transform(self, x):
        x = np.exp(x) - 1
        x = self.minmax_scaler.inverse_transform(x)
        return x

    def fit(self, x):
        self.minmax_scaler.fit(x)

    def fit_transform(self, x):
        x = self.minmax_scaler.fit_transform(x)
        x = np.log(x + 1)
        return x


class MinMaxLogFamilyScaler:
    def __init__(self, feature_range=(0, 1)):
        self.minmax_family_scalers = [MinMaxScaler(feature_range) for i in range(33)]

    def transform(self, x):
        x = torch.transpose(x, 0, 1)
        shape = x.size()
        seq_len = shape[1]
        x = x.reshape(-1, 1)
        res = np.zeros_like(x)
        for i in range(33):
            z = x[i * 54 * seq_len: (i + 1) * 54 * seq_len]
            z = self.minmax_family_scalers[i].transform(z)
            z = np.log(z + 1)
            res[i * 54 * seq_len: (i + 1) * 54 * seq_len] = z
        res = torch.tensor(res, dtype=torch.float32)
        res = res.reshape(*shape)
        res = torch.transpose(res, 0, 1)
        return res

    def inverse_transform(self, x):
        x = torch.transpose(x, 0, 1)
        shape = x.size()
        seq_len = shape[1]
        x = x.reshape(-1, 1)
        res = np.zeros_like(x)
        for i in range(33):
            z = x[i * 54 * seq_len: (i + 1) * 54 * seq_len]
            z = np.exp(z) - 1
            z = self.minmax_family_scalers[i].inverse_transform(z)
            res[i * 54 * seq_len: (i + 1) * 54 * seq_len] = z
        res = torch.tensor(res, dtype=torch.float32)
        res = res.reshape(*shape)
        res = torch.transpose(res, 0, 1)
        return res

    def fit(self, x):
        x = torch.transpose(x, 0, 1)
        shape = x.size()
        seq_len = shape[1]
        x = x.reshape(-1, 1)
        for i in range(33):
            self.minmax_family_scalers[i].fit(x[i * 54 * seq_len: (i + 1) * 54 * seq_len])

    def fit_transform(self, x):
        x = torch.transpose(x, 0, 1)
        shape = x.size()
        seq_len = shape[1]
        x = x.reshape(-1, 1)
        res = np.zeros_like(x)
        for i in range(33):
            z = x[i * 54 * seq_len: (i + 1) * 54 * seq_len]
            z = self.minmax_family_scalers[i].fit_transform(z)
            z = np.log(z + 1)
            res[i * 54 * seq_len: (i + 1) * 54 * seq_len] = z
        res = torch.tensor(res, dtype=torch.float32)
        res = res.reshape(*shape)
        res = torch.transpose(res, 0, 1)
        return res
