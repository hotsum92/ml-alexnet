from matplotlib import pyplot as plt
import torch


def evaluate(data_loader, model, loss_func):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    lossess = [] # バッチごとの損失
    correct_preds = 0 # 正解数
    total_samples = 0 # 処理されたデータ数

    for x, y in data_loader:

        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            loss = loss_func(preds, y)

            lossess.append(loss.item())

            # 予測されたクラスのインデックスを取得
            _, predicted = torch.max(preds, 1)

            # 正解数をカウント
            correct_preds += (predicted == y).sum().item()

            # 処理されたデータ数をカウント
            total_samples += y.size(0)

    # 全体の損失は、バッチごとの損失の平均
    average_loss = sum(lossess) / len(lossess)

    # 精度は、正確な予測の数を全体のデータ数で割ったもの
    accuracy = correct_preds / total_samples

    return average_loss, accuracy

def train_eval(model, num_epochs, train_loader, test_loader, loss_func, optimizer):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        total_accuracy = 0.0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            preds = model(x)
            loss = loss_func(preds, y)

            accuracy = (preds.argmax(dim=1) == y).float().mean()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_accuracy / len(train_loader)

        val_loss, val_accuracy = evaluate(test_loader, model, loss_func)

        print(
            f"Epoch: {epoch + 1}/{num_epochs}, "
            f"  Train: Loss {avg_train_loss:.3f}, Accuracy: {avg_train_accuracy:.3f}, "
            f"  Validation: Loss {val_loss:.3f}, Accuracy: {val_accuracy:.3f}"
        )

        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()

    plt.savefig("loss_accuracy.png")

