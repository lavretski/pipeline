import torch
import torch.nn as nn

class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        y_pred: predicted values, torch.Tensor (B, pred_len, channels)
        y_true: true values, torch.Tensor (B, pred_len, channels)
        """
        log_diff = torch.log1p(y_pred) - torch.log1p(y_true)
        return torch.sqrt(torch.mean(log_diff ** 2))


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        y_pred: predicted values, torch.Tensor
        y_true: true values, torch.Tensor
        """
        diff = y_pred - y_true
        return torch.sqrt(torch.mean(diff ** 2))


class RMSLELossZero(nn.Module):
    def __init__(self):
        super(RMSLELossZero, self).__init__()

    def forward(self, y_pred, y_true):
        """
        y_pred: predicted values, torch.Tensor (B, pred_len, channels)
        y_true: true values, torch.Tensor (B, pred_len, channels)
        """
        log_diff = torch.log1p(y_pred) - torch.log1p(y_true)
        B, P, C = y_pred.size()
        y_pred = (y_pred.reshape(B, C, P) * torch.where(torch.sum(y_true, dim = 1) == 0, torch.tensor(0), torch.tensor(1))[:, :, None]).view(B, P, C)
        return torch.sqrt(torch.mean(log_diff ** 2))