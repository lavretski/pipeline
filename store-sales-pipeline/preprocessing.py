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


@torch.no_grad()
def inverse_transform(data, scaler):
    shape = data.size()
    data = data.detach().cpu().numpy()
    data = torch.tensor(scaler.inverse_transform(data.reshape(-1, 1)), dtype=torch.float32)
    return data.reshape(*shape)
