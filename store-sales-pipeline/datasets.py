import torch.utils.data as data
import torch


class TimeSeriesDataset(data.Dataset):
    def __init__(self, data, seq_len, pred_len, scaler):
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


class TimeSeriesDatasetWithCovariates(data.Dataset):
    def __init__(self, sales, past_covariates, future_covariates, sales_len, past_cov_len, fut_cov, pred_len, scaler):
        self.sales_len = sales_len
        self.past_cov_len = past_cov_len
        self.fut_cov = fut_cov
        self.pred_len = pred_len
        self.scaler = scaler
        self.sales = self.scaler.fit_transform(sales)
        self.past_covariates = past_covariates
        self.future_covariates = future_covariates

    def __len__(self):
        return len(self.sales) - self.sales_len - self.pred_len + 1

    def __getitem__(self, index):
        past_covariates = self.past_covariates[index + self.sales_len - self.past_cov_len:index + self.sales_len].permute(0, 2, 1).reshape(-1, 54*33)  # (past_cov_len * 5, 54*33)
        future_covariates = self.future_covariates[index + self.sales_len - self.fut_cov[0]:index + self.sales_len + self.fut_cov[1]].permute(0, 2, 1).reshape(-1, 54*33)  # (fut_cov_len * 20, 54*33)
        sales = self.sales[index:index + self.sales_len]
        data = torch.cat([past_covariates, sales, future_covariates], dim=0)
        target = self.sales[index + self.sales_len:index + self.sales_len + self.pred_len]
        return data, target
