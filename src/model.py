"""
src/model.py
------------
Baseline-модель: свёрточная нейросеть (CNN), написанная с нуля.

Идея: 3 свёрточных блока вытягивают признаки из картинки (края -> текстуры ->
части объектов), затем полносвязные слои по этим признакам предсказывают класс.
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """Простая CNN для классификации изображений на num_classes классов."""

    def __init__(self, num_classes: int = 6):
        super().__init__()  # обязательный вызов конструктора nn.Module

        # --- Свёрточная часть (feature extractor) ---
        # Каждый блок: Conv -> ReLU -> MaxPool.
        # Conv2d учит фильтры, ReLU добавляет нелинейность, MaxPool уменьшает
        # размер картинки вдвое (и оставляет самые сильные отклики).
        self.features = nn.Sequential(
            # Блок 1: вход 3 канала (RGB) -> 32 карты признаков
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # размер картинки / 2

            # Блок 2: 32 -> 64 карты признаков
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),               # ещё / 2

            # Блок 3: 64 -> 128 карт признаков
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),               # ещё / 2
        )

        # AdaptiveAvgPool приводит карту признаков к фиксированному размеру 4x4
        # независимо от размера входной картинки. Поэтому дальше всегда 128*4*4.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # --- Классификатор (полносвязная часть) ---
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # (B,128,4,4) -> (B, 128*4*4=2048)
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),               # отключаем половину нейронов -> меньше переобучение
            nn.Linear(256, num_classes),   # 6 выходов = 6 классов (логиты)
        )

    def forward(self, x):
        """Прямой проход: картинка -> признаки -> логиты по классам."""
        x = self.features(x)   # свёртки
        x = self.pool(x)       # фиксируем пространственный размер
        x = self.classifier(x) # предсказание
        return x               # логиты (без softmax: его применит CrossEntropyLoss)
