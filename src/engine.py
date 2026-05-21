"""
src/engine.py
-------------
Функции обучения и оценки модели (общие для baseline и transfer learning).

Содержит:
- get_device()      : выбрать GPU, если есть, иначе CPU
- train_one_epoch() : один проход по обучающим данным
- evaluate()        : оценка на val/test (loss + accuracy)
- predict_all()     : собрать все предсказания и метки (для confusion matrix)
- fit()             : полный цикл обучения на несколько эпох (+ история)
- plot_history()    : графики loss и accuracy
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_device():
    """GPU (cuda), если доступен, иначе CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Один проход по train: считаем градиенты и обновляем веса."""
    model.train()                      # режим обучения (включает Dropout и т.п.)
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)  # данные на GPU/CPU

        optimizer.zero_grad()          # обнулить градиенты с прошлого шага
        outputs = model(images)        # прямой проход -> логиты
        loss = criterion(outputs, labels)  # ошибка (cross-entropy)
        loss.backward()                # обратное распространение: считаем градиенты
        optimizer.step()               # шаг оптимизатора: обновляем веса

        total_loss += loss.item() * images.size(0)        # суммарная ошибка
        preds = outputs.argmax(dim=1)                      # класс = индекс макс. логита
        correct += (preds == labels).sum().item()          # сколько угадали
        total += labels.size(0)

    return total_loss / total, correct / total             # средние loss и accuracy


@torch.no_grad()  # отключаем подсчёт градиентов — оценка идёт быстрее и без обучения
def evaluate(model, loader, criterion, device):
    """Оценка на val/test: средние loss и accuracy."""
    model.eval()                       # режим оценки (Dropout выключен)
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def predict_all(model, loader, device):
    """Возвращает все предсказанные и истинные метки (для confusion matrix / отчёта)."""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def fit(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    """Полный цикл обучения. Возвращает историю метрик по эпохам."""
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"Эпоха {epoch:2d}/{epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}")

    return history


def plot_history(history, out_path):
    """Рисует две кривые: loss и accuracy (train vs val) и сохраняет в файл."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Графики сохранены в: {out_path}")
