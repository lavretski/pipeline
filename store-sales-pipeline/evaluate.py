import torch

@torch.no_grad()
def evaluate_horizons(actuals: dict, horizons, criterion):
    return criterion(actuals, horizons)