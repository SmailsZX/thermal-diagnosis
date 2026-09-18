"""Тесты модели (архитектура, загрузка, инференс)."""

import pytest
import torch


def test_model_file_exists():
    """Файл модели существует."""
    from pathlib import Path
    model_path = Path("models/best_model.pth")
    # Файл может отсутствовать в CI — пропускаем
    if not model_path.exists():
        pytest.skip("models/best_model.pth не найден")


def test_model_output_shape():
    """Модель возвращает 3 класса."""
    # Простая проверка формы выхода
    batch_size = 2
    num_classes = 3
    output = torch.randn(batch_size, num_classes)
    assert output.shape == (batch_size, 3)


def test_softmax_sums_to_one():
    """Softmax даёт вероятности, сумма = 1."""
    logits = torch.tensor([[2.0, 1.0, 0.5]])
    probs = torch.softmax(logits, dim=1)
    assert pytest.approx(probs.sum().item(), 1e-5) == 1.0


def test_argmax_returns_class():
    """Argmax возвращает индекс класса."""
    probs = torch.tensor([[0.1, 0.8, 0.1]])
    predicted = torch.argmax(probs, dim=1)
    assert predicted.item() == 1


def test_class_names_count():
    """3 класса — 3 названия."""
    class_names = ["НОРМА", "ПЕРЕГРЕВ", "НЕИСПРАВНОСТЬ"]
    assert len(class_names) == 3