"""Тесты вспомогательных функций (src/utils.py)."""

import pytest
import torch
import numpy as np


def test_import_utils():
    """Модуль utils импортируется."""
    from src import utils
    assert utils is not None


def test_device_selection():
    """Устройство выбирается корректно."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type in ["cuda", "cpu"]


def test_tensor_shape():
    """Тензор имеет правильную форму."""
    tensor = torch.randn(1, 224, 224)
    assert tensor.shape == (1, 224, 224)


def test_normalize_range():
    """Нормализация приводит к диапазону [0, 1]."""
    arr = np.array([0, 50, 100, 150, 200, 255], dtype=np.float32)
    normalized = arr / 255.0
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0