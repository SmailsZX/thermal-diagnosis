"""Фикстуры для тестов thermal-diagnosis."""

import pytest
import torch
from pathlib import Path


@pytest.fixture
def sample_image_path(tmp_path):
    """Создаёт временное тестовое изображение (термограмму)."""
    import numpy as np
    import cv2

    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    path = tmp_path / "test_thermal.png"
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture
def sample_tensor():
    """Тестовый тензор (1, 224, 224)."""
    return torch.randn(1, 224, 224)


@pytest.fixture
def sample_batch():
    """Батч тензоров (4, 1, 224, 224)."""
    return torch.randn(4, 1, 224, 224)


@pytest.fixture
def class_names():
    """Названия классов."""
    return ["НОРМА", "ПЕРЕГРЕВ", "НЕИСПРАВНОСТЬ"]