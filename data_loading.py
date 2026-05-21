"""
src/data_loading.py
-------------------
Загрузка датасета Intel Image Classification и базовый анализ данных (EDA).

Что делает этот файл:
1) задаёт преобразования (transforms) для изображений;
2) строит наборы данных (Dataset) и загрузчики (DataLoader) для train/val/test;
3) считает количество картинок по классам и сохраняет сетку примеров в results/.

Запуск:  python src/data_loading.py
"""

import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")  # без графического окна — просто сохраняем картинки в файл
import matplotlib.pyplot as plt

# --- Пути к данным -----------------------------------------------------------
# Структура после распаковки Kaggle-архива (см. data/README.md):
#   data/seg_train/seg_train/<класс>/*.jpg
#   data/seg_test/seg_test/<класс>/*.jpg
TRAIN_DIR = os.path.join("data", "seg_train", "seg_train")
TEST_DIR = os.path.join("data", "seg_test", "seg_test")

IMG_SIZE = 150          # изображения приведём к 150x150 (исходный размер датасета)
BATCH_SIZE = 32         # сколько картинок в одной "пачке" при обучении
VAL_RATIO = 0.2         # 20% обучающих данных отдаём под валидацию


# --- Преобразования (transforms) --------------------------------------------
# Для валидации/теста НЕ добавляем случайность: только привести к нужному
# размеру и перевести в тензор. Так оценка будет честной и воспроизводимой.
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # одинаковый размер для всех картинок
    transforms.ToTensor(),                    # PIL-изображение -> тензор [0,1], форма (C,H,W)
])

# Базовый train-трансформ (БЕЗ аугментации) — пригодится для сравнения на Неделе 3.
train_transform_plain = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def get_datasets():
    """Возвращает (train, val, test) как объекты ImageFolder/Subset.

    ImageFolder сам понимает классы по именам подпапок и присваивает им метки.
    """
    # Полный обучающий набор; разобьём его на train и val.
    full_train = datasets.ImageFolder(TRAIN_DIR, transform=train_transform_plain)
    test_set = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

    # Считаем размеры частей и делаем воспроизводимое разбиение (фиксируем seed).
    val_size = int(len(full_train) * VAL_RATIO)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(42)  # seed -> одинаковое разбиение каждый раз
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    return train_set, val_set, test_set, full_train.classes


def get_dataloaders():
    """Оборачивает наборы данных в DataLoader (выдаёт данные пачками)."""
    train_set, val_set, test_set, classes = get_datasets()
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)   # перемешиваем train
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader, classes


# --- EDA: анализ данных ------------------------------------------------------
def run_eda():
    """Считает картинки по классам и сохраняет сетку примеров в results/."""
    os.makedirs("results", exist_ok=True)

    # Загружаем train БЕЗ преобразований, чтобы видеть оригинальные картинки.
    raw = datasets.ImageFolder(TRAIN_DIR)
    classes = raw.classes
    print("Классы:", classes)

    # raw.targets — список меток-классов для каждой картинки. Counter их посчитает.
    counts = Counter(raw.targets)
    print("\nКоличество изображений по классам (train):")
    for idx, name in enumerate(classes):
        print(f"  {name:10s}: {counts[idx]}")

    # Сохраняем по одному примеру каждого класса в одну картинку-сетку.
    fig, axes = plt.subplots(1, len(classes), figsize=(3 * len(classes), 3))
    for idx, name in enumerate(classes):
        # находим первую картинку, у которой метка == idx
        sample_pos = raw.targets.index(idx)
        image, _ = raw[sample_pos]
        axes[idx].imshow(image)
        axes[idx].set_title(name)
        axes[idx].axis("off")
    fig.tight_layout()
    out_path = os.path.join("results", "class_samples.png")
    fig.savefig(out_path, dpi=120)
    print(f"\nСетка примеров сохранена в: {out_path}")


if __name__ == "__main__":
    run_eda()
