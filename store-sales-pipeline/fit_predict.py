import os
import torch
from losses import RMSELoss, RMSLELoss
from models.Linears import DlinearModel, NlinearModel
from preprocessing import MinMaxLogScaler
from datasets import TimeSeriesDataset
from loaders import TimeSeriesDataLoader
from predicts import get_horizons
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from early_stopping import EarlyStopping
from pathes import path_to_temp
from evaluate import evaluate_horizons


def fit_predict(sales, device):
    seq_len = 150
    lr = 0.001
    epochs = 15
    feature_range = (0, 100)
    batch_size = 3
    factor = 0.5
    patience_scheduler = 3
    patience_early_stopping = 10
    best_model_path = os.path.join(path_to_temp, 'DLinear.pth')

    criterion = RMSELoss()
    early_stopping = EarlyStopping(path=best_model_path, patience=patience_early_stopping, verbose=True)
    model = DlinearModel(seq_len=seq_len, pred_len=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience_scheduler, factor=factor)
    scaler = MinMaxLogScaler(feature_range=feature_range)
    train_dataset = TimeSeriesDataset(data=sales[:-16], seq_len=seq_len, pred_len=16, scaler=scaler)
    train_loader = TimeSeriesDataLoader(train_dataset, batch_size=batch_size)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step(running_loss / len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.6f}")

        criterion = RMSLELoss()
        horizons = get_horizons(model, sales[-seq_len - 16:-16], scaler=scaler, device=device)
        vali_loss = evaluate_horizons(sales[-16:], horizons, criterion)
        early_stopping(vali_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(best_model_path))
    return get_horizons(model, sales[-seq_len:], scaler=scaler, device=device)
