import torch
from losses import RMSELoss, RMSLELoss
from models.Linears import DlinearModel, NlinearModel, DlinearModelMulti
from preprocessing import MinMaxLogScaler, MinMaxLogFamilyScaler
from datasets import TimeSeriesDataset
from loaders import TimeSeriesDataLoader
from predicts import get_horizons
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from evaluate import evaluate_horizons
import matplotlib.pyplot as plt


def fit_predict(train_data, device, val_data=None):
    seq_len = 150
    lr = 0.001
    epochs = 15
    feature_range = (0, 100)
    batch_size = 3
    t_0 = 16
    eta_min = lr / 100

    criterion_train = RMSELoss()
    criterion_val = RMSLELoss()
    model = DlinearModel(seq_len=seq_len, pred_len=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, eta_min=eta_min, verbose=True)
    scaler = MinMaxLogFamilyScaler(feature_range=feature_range)
    train_dataset = TimeSeriesDataset(data=train_data, seq_len=seq_len, pred_len=16, scaler=scaler)
    train_loader = TimeSeriesDataLoader(train_dataset, batch_size=batch_size)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        scheduler.step()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion_train(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        val_loss = 0
        if val_data is not None:
            val_loss = evaluate_horizons(val_data, get_horizons(model, train_data[-seq_len:], scaler=scaler), criterion_val)
            val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    plt.plot(range(len(train_losses)), train_losses, label='train_loss')
    if val_data is not None:
        plt.plot(range(len(train_losses)), val_losses, label='val_loss')
    plt.legend()
    plt.show()
    return get_horizons(model, train_data[-seq_len:], scaler=scaler)
