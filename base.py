from evaluate import evaluate_horizons
from get_data import get_sales, get_covariates
from fit_predict import fit_predict
from submission import horizons_to_submission
import torch
from losses import RMSELoss, RMSLELoss
from models.Linears import DlinearModel, NlinearModel, DlinearModelMulti, LinearModel, DlinearModelCovariates
from models.Convolutions import TimeSeriesCNN
from preprocessing import MinMaxLogScaler, MinMaxLogFamilyScaler
from datasets import TimeSeriesDataset, TimeSeriesDatasetWithCovariates
from loaders import TimeSeriesDataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sales = get_sales()
# past_covariates, future_covariates = get_covariates()


def fold(i):
    sales_len = 150
    past_cov_len = 22
    fut_cov = [16, 16]
    lr = 0.001
    epochs = 15
    feature_range = (0, 100)
    batch_size = 3
    t_0 = 16
    eta_min = lr / 100

    # seq_len = past_cov_len * 5 + (fut_cov[0] + fut_cov[1]) * 20 + sales_len
    seq_len = sales_len

    model = DlinearModel(seq_len=seq_len, pred_len=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, eta_min=eta_min)
    scaler = MinMaxLogFamilyScaler(feature_range=feature_range)
    # train_dataset = TimeSeriesDatasetWithCovariates(sales=sales[:-i * 16] if i != 0 else sales,
    #                                                past_covariates=past_covariates[:-i * 16] if i != 0 else past_covariates,
    #                                                future_covariates=future_covariates[:-i * 16] if i != 0 else future_covariates,
    #                                                sales_len=sales_len, past_cov_len=past_cov_len,
    #                                                fut_cov=fut_cov, pred_len=16, scaler=scaler)
    train_dataset = TimeSeriesDataset(sales[:-i * 16] if i != 0 else sales, seq_len, 16, scaler)
    train_loader = TimeSeriesDataLoader(train_dataset, batch_size=batch_size)
    # val_data = None
    # if i != 0:
    #     val_data
    #     val_predictions
    #     past_cov = past_covariates[-past_cov_len-(i+1) * 16 : -(i+1)*16].permute(0, 2, 1).reshape(-1, 54 * 33)
    #     future_cov = future_covariates[-fut_cov_len-(i+1)*16 : -(i+1)*16].permute(0, 2, 1).reshape(-1, 54 * 33)
    #     data_for_prediction = torch.cat([past_cov, scaler.transform(sales[-sales_len - (i+1) * 16:-(i+1) * 16]), future_cov], dim=0)

    # past_cov = past_covariates[-past_cov_len - i * 16:-i * 16] if i != 0 else past_covariates[-past_cov_len:]
    # future_cov = future_covariates[-fut_cov[0] - (i+1) * 16:-(i+1) * 16 + fut_cov[1]] if (-(i+1) * 16 + fut_cov[1]) != 0 else future_covariates[-fut_cov[0] - 16:]
    # past_cov = past_cov.permute(0, 2, 1).reshape(-1, 54 * 33)
    # future_cov = future_cov.permute(0, 2, 1).reshape(-1, 54 * 33)
    s = sales[-sales_len - i * 16:-i * 16] if i != 0 else sales[-sales_len:]
    data_for_prediction = scaler.transform(s)
    # data_for_prediction = torch.cat([past_cov, future_cov, scaler.transform(s)], dim=0)

    return fit_predict(model=model,
                       train_loader=train_loader,
                       # val_data=val_data,
                       # val_predictions=val_predictions,
                       data_for_prediction=data_for_prediction.to(device),
                       optimizer=optimizer,
                       scheduler=scheduler,
                       epochs=epochs,
                       scaler=scaler,
                       device=device)


print('cross validation')
criterion = RMSLELoss()
horizon1 = fold(3)
val_loss1 = evaluate_horizons(sales[-3 * 16:-2 * 16], horizon1, criterion)
print(f'val_loss1 = {val_loss1}')
horizon2 = fold(2)
val_loss2 = evaluate_horizons(sales[-2 * 16:-16], horizon2, criterion)
print(f'val_loss2 = {val_loss2}')
horizon3 = fold(1)
val_loss3 = evaluate_horizons(sales[-16:], horizon3, criterion)
print(f'val_loss3 = {val_loss3}')
print('cross validation loss', (val_loss1 + val_loss2 + val_loss3) / 3)

horizons = fold(0)
print('making submission')
horizons_to_submission(horizons)
