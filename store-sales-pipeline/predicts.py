import torch

@torch.no_grad()
def inverse_transform(data, scaler):
    shape = data.size()
    data = data.detach().cpu().numpy()
    data = torch.tensor(scaler.inverse_transform(data.reshape(-1, 1)), dtype=torch.float32)
    return data.reshape(*shape)


@torch.no_grad()
def get_horizons(model, data, scaler, device):
    model.eval()
    x = data
    shape = x.size()
    x = torch.tensor(scaler.transform(x.reshape(-1, 1)), dtype=torch.float32)
    x = x.reshape(*shape)
    x = x[None, ...]
    x = x.to(device)
    predict = model(x)
    predict = inverse_transform(predict, scaler)
    return predict[0]

# @torch.no_grad()
# def zero_forecasting(actuals: dict, horizons_family):
#     for horizon, actual in zip(horizons_family, actuals.values()):
#         for i in range(54):
#             if torch.all(actual[-16:, i] == 0):
#                 horizon[:, i] = 0
#     return horizons_family