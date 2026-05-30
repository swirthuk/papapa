# Digit Recognition Service

REST-API для автоматического распознавания рукописных цифр (0–9).

**Стек:** Python · scikit-learn (SVM) · FastAPI · Pillow · joblib

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/swirthuk/digit-recognition.git
cd digit-recognition
```

### 2. Создать виртуальное окружение и установить зависимости

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Обучить модель

```bash
python train.py
```

После выполнения появится файл `models/model_pipeline.pkl` и график `plots/confusion_matrix.png`.

### 4. Запустить сервер

```bash
uvicorn app:app --reload
```

Сервер будет доступен по адресу: **http://127.0.0.1:8000**

---

## API

### `GET /health`
Проверка работоспособности сервиса.

```json
{
  "status": "ok",
  "model_loaded": true,
  "metrics": {"accuracy": 0.9778, "f1_macro": 0.9776}
}
```

### `GET /docs`
Интерактивная Swagger-документация.

---

### `POST /predict`
Предсказание по вектору из 64 пикселей (формат JSON).

**Тело запроса:**
```json
{
  "pixels": [0, 0, 5, 13, 9, 1, 0, 0, ...]
}
```

**Ответ:**
```json
{
  "digit": 3,
  "confidence": 0.9821,
  "needs_manual_review": false
}
```

---

### `POST /predict/upload`
Предсказание по загруженному изображению (PNG / JPEG / BMP).

Изображение автоматически масштабируется до 8×8 пикселей.

**Пример через curl:**
```bash
curl -X POST http://127.0.0.1:8000/predict/upload \
     -F "file=@digit.png"
```

---

## Структура проекта

```
digit-recognition/
├── app.py              # FastAPI приложение
├── train.py            # Обучение и сохранение модели
├── requirements.txt    # Зависимости
├── README.md
├── .gitignore
└── models/
    └── .gitkeep        # Папка для модели (pkl не в git)
```

---

## Метрики модели (SVM, тестовая выборка)

| Метрика    | Значение |
|------------|----------|
| Accuracy   | 0.9778   |
| F1-macro   | 0.9776   |

Модель обучена на встроенном датасете `sklearn.datasets.load_digits` (1 797 изображений, 8×8 пикселей, 10 классов).
