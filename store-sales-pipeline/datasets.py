import torch.utils.data as data
import torch


class TimeSeriesDataset(data.Dataset):
    def __init__(self, data, seq_len, pred_len, scaler, val=False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = scaler
        self.data = self.scaler.fit_transform(data)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        data = self.data[index:index + self.seq_len]
        target = self.data[index + self.seq_len:index + self.seq_len + self.pred_len]
        return data, target
