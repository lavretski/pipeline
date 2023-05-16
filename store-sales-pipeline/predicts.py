import torch

@torch.no_grad()
def get_horizons(model, data, scaler):
    model.eval()
    x = data
    x = scaler.transform(x)
    x = x[None, ...]
    predict = model(x)[0]
    predict = scaler.inverse_transform(predict)
    return predict