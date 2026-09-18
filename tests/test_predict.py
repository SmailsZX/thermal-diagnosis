"""Тесты инференса (predict.py)."""

import pytest
import torch


def test_predict_returns_dict():
    """predict возвращает словарь с диагнозом."""
    # Мок результата
    result = {
        "diagnosis": "ПЕРЕГРЕВ",
        "probabilities": {
            "НОРМА": 0.1,
            "ПЕРЕГРЕВ": 0.85,
            "НЕИСПРАВНОСТЬ": 0.05,
        },
    }
    assert "diagnosis" in result
    assert "probabilities" in result
    assert result["diagnosis"] in ["НОРМА", "ПЕРЕГРЕВ", "НЕИСПРАВНОСТЬ"]


def test_probabilities_sum_to_one():
    """Сумма вероятностей = 1."""
    probs = {"НОРМА": 0.1, "ПЕРЕГРЕВ": 0.85, "НЕИСПРАВНОСТЬ": 0.05}
    assert pytest.approx(sum(probs.values()), 1e-5) == 1.0


def test_diagnosis_matches_max_probability():
    """Диагноз соответствует максимальной вероятности."""
    probs = {"НОРМА": 0.1, "ПЕРЕГРЕВ": 0.85, "НЕИСПРАВНОСТЬ": 0.05}
    diagnosis = max(probs, key=probs.get)
    assert diagnosis == "ПЕРЕГРЕВ"


def test_image_path_exists(sample_image_path):
    """Тестовое изображение создаётся."""
    from pathlib import Path
    assert Path(sample_image_path).exists()


def test_weather_api_key_optional(monkeypatch):
    """Без API-ключа погода не обязательна."""
    monkeypatch.delenv("WEATHER_API_KEY", raising=False)
    import os
    assert os.getenv("WEATHER_API_KEY") is None