from losses import RMSELoss, RMSLELoss
from predicts import get_horizons
from evaluate import evaluate_horizons
import matplotlib.pyplot as plt


def fit_predict(model, train_loader, data_for_prediction, optimizer, scheduler, epochs, scaler, device):
    criterion_train = RMSELoss()
    criterion_val = RMSLELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
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
        # if val_data is not None:
        #     horizons = get_horizons(model, val_data, scaler)
        #     val_loss = evaluate_horizons(val_predictions, horizons, criterion_val)
        #     val_losses.append(val_loss)
        #     print(1)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    plt.plot(range(len(train_losses)), train_losses, label='train_loss')
    # if val_data is not None:
    #     plt.plot(range(len(train_losses)), val_losses, label='val_loss')
    plt.legend()
    plt.show()
    return get_horizons(model, data_for_prediction, scaler)
