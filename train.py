"""
train.py — Обучение и сохранение модели распознавания рукописных цифр.

Запуск:
    python train.py

После выполнения создаётся файл models/model_pipeline.pkl
"""

import os
import warnings
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# ── Конфигурация ──────────────────────────────────────────────────────────────
CONFIG = {
    "RANDOM_STATE": 42,
    "TEST_SIZE": 0.2,
    "CV_FOLDS": 5,
}

np.random.seed(CONFIG["RANDOM_STATE"])
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── Загрузка данных ───────────────────────────────────────────────────────────
print("Загрузка датасета...")
digits = load_digits()
X, y = digits.data, digits.target

print(f"  Объектов:  {X.shape[0]}")
print(f"  Признаков: {X.shape[1]} (8×8 пикселей)")
print(f"  Классов:   {len(np.unique(y))} (цифры 0–9)")

# ── Проверка целостности ──────────────────────────────────────────────────────
import pandas as pd
df = pd.DataFrame(X)
assert df.isnull().sum().sum() == 0, "Найдены пропуски!"
print(f"  Пропуски: 0  |  Дубликаты: {df.duplicated().sum()}")

# ── Train / Test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=CONFIG["TEST_SIZE"],
    random_state=CONFIG["RANDOM_STATE"],
    stratify=y,
)
print(f"\nТрейн: {len(X_train)}  |  Тест: {len(X_test)}")

# ── Подбор гиперпараметров (GridSearchCV) ─────────────────────────────────────
print("\nПоиск лучших гиперпараметров (GridSearchCV)...")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(probability=True, random_state=CONFIG["RANDOM_STATE"])),
])

param_grid = {
    "clf__C":     [0.1, 1, 10],
    "clf__gamma": ["scale", "auto"],
}

cv = StratifiedKFold(n_splits=CONFIG["CV_FOLDS"], shuffle=True, random_state=CONFIG["RANDOM_STATE"])

gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

best_model = gs.best_estimator_
print(f"  Лучшие параметры: {gs.best_params_}")
print(f"  CV Accuracy:      {gs.best_score_:.4f}")

# ── Оценка на тестовой выборке ────────────────────────────────────────────────
y_pred = best_model.predict(X_test)
acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred, average="macro")

print(f"\nТест Accuracy: {acc:.4f}  |  F1-macro: {f1:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# ── Confusion matrix ──────────────────────────────────────────────────────────
plt.figure(figsize=(7, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix сохранена в plots/confusion_matrix.png")

# ── Сохранение модели ─────────────────────────────────────────────────────────
artifacts = {
    "model": best_model,
    "metrics": {"accuracy": round(acc, 4), "f1_macro": round(f1, 4)},
}
joblib.dump(artifacts, "models/model_pipeline.pkl")
print("\nМодель сохранена: models/model_pipeline.pkl")
print("Обучение завершено.")
