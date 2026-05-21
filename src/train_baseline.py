"""
src/train_baseline.py
---------------------
Обучение baseline-модели (SimpleCNN с нуля) от начала до конца:
загрузка данных -> обучение -> графики -> оценка на тесте -> сохранение результатов.

Запуск:  python src/train_baseline.py
"""

import os
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loading import get_dataloaders
from model import SimpleCNN
from engine import get_device, fit, evaluate, predict_all, plot_history

# --- Гиперпараметры (их меняем при экспериментах) ---
EPOCHS = 10
LR = 1e-3            # скорость обучения для Adam
SEED = 42


def main():
    os.makedirs("results", exist_ok=True)
    torch.manual_seed(SEED)            # воспроизводимость

    device = get_device()
    print("Устройство:", device)

    # 1) Данные
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print("Классы:", classes)

    # 2) Модель, функция потерь, оптимизатор
    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()             # для многоклассовой классификации
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 3) Обучение
    history = fit(model, train_loader, val_loader, criterion, optimizer, device, EPOCHS)
    plot_history(history, os.path.join("results", "baseline_curves.png"))

    # 4) Оценка на ТЕСТЕ (его не трогали при обучении/выборе модели)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest: loss {test_loss:.3f} | accuracy {test_acc:.3f}")

    # 5) Подробные метрики по классам + матрица ошибок
    preds, labels = predict_all(model, test_loader, device)
    report = classification_report(labels, preds, target_names=classes, digits=3)
    print("\n" + report)

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Baseline confusion matrix")
    for i in range(len(classes)):                       # подписываем числа в клетках
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join("results", "baseline_confusion.png"), dpi=120)

    # 6) Сохраняем веса и текстовые метрики
    torch.save(model.state_dict(), "baseline_cnn.pth")
    with open(os.path.join("results", "baseline_metrics.txt"), "w") as f:
        f.write(f"Test loss: {test_loss:.3f}\nTest accuracy: {test_acc:.3f}\n\n{report}")
    print("\nГотово. Результаты в папке results/.")


if __name__ == "__main__":
    main()
