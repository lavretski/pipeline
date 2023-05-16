import torch
from losses import RMSELoss, RMSLELoss
from models.Linears import DlinearModel, NlinearModel
from preprocessing import MinMaxLogScaler, MinMaxLogFamilyScaler
from datasets import TimeSeriesDataset
from loaders import TimeSeriesDataLoader
from predicts import get_horizons
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def fit_predict(sales, device):
    seq_len = 150
    lr = 0.0001
    epochs = 15
    feature_range = (0, 100)
    batch_size = 3
    factor = 0.5
    patience_scheduler = 2

    criterion = RMSELoss()
    model = DlinearModel(seq_len=seq_len, pred_len=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience_scheduler, factor=factor)
    scaler = MinMaxLogFamilyScaler(feature_range=feature_range)
    train_dataset = TimeSeriesDataset(data=sales, seq_len=seq_len, pred_len=16, scaler=scaler)
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

    return get_horizons(model, sales[-seq_len:], scaler=scaler)
