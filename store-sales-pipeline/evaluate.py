import torch

@torch.no_grad()
def evaluate_horizons(actuals, horizons, criterion):
    return criterion(actuals, horizons)