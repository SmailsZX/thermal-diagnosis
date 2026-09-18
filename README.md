# thermal-diagnosis

Нейросетевая диагностика электрооборудования по термограммам. Определяет: НОРМА / ПЕРЕГРЕВ / НЕИСПРАВНОСТЬ.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com/)
[![Tests](https://github.com/SmailsZX/thermal-diagnosis/actions/workflows/tests.yml/badge.svg)](https://github.com/SmailsZX/thermal-diagnosis/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 О проекте

Нейросетевая система для диагностики состояния электрооборудования по термограммам.  
Определяет три класса состояния:

| Статус | Описание | Действие |
|--------|----------|----------|
| ✅ **НОРМА** | Оборудование в штатном режиме | Плановое обслуживание |
| ⚠️ **ПЕРЕГРЕВ** | Повышенная температура | Проверить охлаждение |
| ❌ **НЕИСПРАВНОСТЬ** | Критический дефект | Немедленный ремонт |

---

## 📸 Демонстрация работы

### 1. Генерация синтетических данных

![Генерация данных](screenshots/screenshot_generate.jpg)

### 2. Обучение нейросети

![Обучение модели](screenshots/screenshot_train.jpg)

### 3. Анализ термограмм (папка)

![Анализ папки](screenshots/screenshot_predict.jpg)
![Анализ папки](screenshots/screenshot_predict2.jpg)

### 4. Диагностика одного файла

![Предсказание](screenshots/screenshot_predict3.jpg)

---

## 🚀 Особенности

- ✅ Обучение на реалистичных термограммах
- 🌤️ Интеграция с OpenWeatherMap (погода в Иркутске)
- 📊 Подробный отчёт в JSON
- 🖼️ Пакетная обработка изображений
- 🎨 Визуализация в стиле тепловизора (colormap INFERNO)
- 🐳 **Docker** — запуск одной командой
- 🧪 **14 тестов (pytest)** — модель, инференс, утилиты
- ⚙️ **CI/CD** — GitHub Actions

---

## 📂 Структура проекта

```
thermal-diagnosis/
├── src/                      # Исходный код
│   ├── utils.py              # Вспомогательные функции
│   ├── generate_data.py      # Генерация данных
│   ├── train.py              # Обучение модели
│   ├── predict.py            # Анализ одного файла
│   └── analyze_folder.py     # Анализ папки
├── tests/                    # Тесты (pytest)
│   ├── conftest.py
│   ├── test_model.py
│   ├── test_predict.py
│   └── test_utils.py
├── models/                   # Сохранённые веса
│   └── best_model.pth
├── data/                     # Датасеты
├── notebooks/                # Jupyter демо
├── scripts/                  # Вспомогательные скрипты
├── screenshots/              # Скриншоты для README
├── .github/workflows/        # CI (GitHub Actions)
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## ⚡ Быстрый старт

### 1. Клонирование

```bash
git clone https://github.com/SmailsZX/thermal-diagnosis.git
cd thermal-diagnosis
```

### 2. Установка зависимостей

```bash
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Linux / Mac
pip install -r requirements.txt
```

### 3. Генерация данных

```bash
python src/generate_data.py
```

### 4. Обучение модели

```bash
python src/train.py
```

### 5. Анализ термограмм

```bash
# Одно изображение
python src/predict.py --image thermal_images/thermal_0000.png

# Вся папка
python src/analyze_folder.py --folder thermal_images/
```

---

## 🐳 Запуск через Docker

### Требования
- Docker Desktop

### Запуск

```bash
docker-compose up --build
```

**Что произойдёт:**
- Соберётся образ (Python 3.11-slim + OpenCV)
- Запустится контейнер `thermal-diagnosis`

### Остановка

```bash
docker-compose down
```

### Важно
- **`libgl1` и `libglib2.0-0`** — системные зависимости для OpenCV (уже в Dockerfile).
- **`WEATHER_API_KEY`** — передаётся через `environment` в `docker-compose.yml`.

---

## 🧪 Тесты

Проект покрыт тестами (pytest) — **14 тестов**.

### Запуск

```bash
pytest tests/ -v
```

### Что покрыто

**Модель (`tests/test_model.py`):**
- ✅ Файл модели
- ✅ Форма выхода (3 класса)
- ✅ Softmax (сумма = 1)
- ✅ Argmax (индекс класса)

**Инференс (`tests/test_predict.py`):**
- ✅ Формат результата
- ✅ Сумма вероятностей
- ✅ Диагноз = max вероятность
- ✅ Weather API (опционально)

**Утилиты (`tests/test_utils.py`):**
- ✅ Импорт модуля
- ✅ Выбор устройства (CPU/GPU)
- ✅ Форма тензора
- ✅ Нормализация

---

## 🌤️ Погодный API

Для корректной работы анализа требуется API-ключ OpenWeatherMap:

1. Зарегистрируйтесь на [OpenWeatherMap](https://openweathermap.org/api)
2. Получите API-ключ
3. Установите переменную окружения:

```bash
export WEATHER_API_KEY="your_api_key"
```

Или создай `.env`:
```
WEATHER_API_KEY=your_api_key
```

---

## 📊 Пример вывода

```
📸 1/3: thermal_0001.png
   ⚠️ Диагноз: ПЕРЕГРЕВ
   📊 Вероятности:
      Норма: 10.2%
      Перегрев: 85.7%
      Неисправность: 4.1%
   💡 ВНИМАНИЕ! Обнаружен перегрев! Проверьте охлаждение

🌤️ Погода в Иркутске:
   🌡️ Температура: -5.2°C
   💧 Влажность: 78%
```

---

## 🧠 Архитектура сети

```
ImprovedThermalNet:
├── Conv2d(1→64) + BN + ReLU + MaxPool
├── Conv2d(64→128) + BN + ReLU + MaxPool
├── Conv2d(128→256) + BN + ReLU + MaxPool
├── Conv2d(256→512) + BN + ReLU + AdaptiveAvgPool
├── Dropout(0.3)
├── Linear(512→256) + ReLU + Dropout(0.3)
├── Linear(256→128) + ReLU
└── Linear(128→3)
```

---

## 📈 Результаты

| Метрика | Значение |
|---------|----------|
| Точность на валидации | ~85-90% |
| Размер изображения | 224×224 |
| Классы | 3 (Норма, Перегрев, Неисправность) |

---

## 🛠️ Технологии

- **PyTorch** — глубокое обучение
- **OpenCV** — обработка изображений
- **NumPy** — вычисления
- **scikit-learn** — разделение данных
- **Requests** — API погоды
- **pytest** — тестирование
- **Docker** — контейнеризация
- **GitHub Actions** — CI/CD

---

## 🔮 Roadmap

- [x] **Docker** — запуск одной командой
- [x] **pytest** — тесты (14)
- [x] **GitHub Actions** — CI
- [ ] **Web-интерфейс** — FastAPI + Streamlit
- [ ] **Fine-tuning** — на реальных данных
- [ ] **Мониторинг** — MLflow / Prometheus

---

## 📝 Лицензия

MIT License

---

## 🤝 Вклад

PR приветствуются! Для крупных изменений откройте Issue для обсуждения.