"""Единая конфигурация рантайма: пути, наборы колонок, пороги чистки, гиперпараметры.

Все остальные модули (loader, cleaner, builder, bot) импортируют значения отсюда,
чтобы не было рассинхронизации между этапами конвейера.
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = BASE_DIR / "resources"
LOGS_DIR = BASE_DIR / "logs"

RAW_CSV = RESOURCES_DIR / "vehicles.csv"
CLEAN_CSV = RESOURCES_DIR / "vehicles_clean.csv"
MODEL_PKL = RESOURCES_DIR / "car_price_model.pkl"

KAGGLE_DATASET = "austinreese/craigslist-carstrucks-data"

TARGET = "price"
NUMERIC_FEATURES = ["year", "odometer"]
CATEGORICAL_FEATURES = [
    "manufacturer", "condition", "cylinders", "fuel",
    "title_status", "transmission", "drive", "size",
    "type", "paint_color",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RAW_USECOLS = [TARGET] + FEATURE_COLUMNS

PRICE_MIN, PRICE_MAX = 500, 100_000
YEAR_MIN, YEAR_MAX = 1990, 2022
ODOMETER_MIN, ODOMETER_MAX = 0, 400_000

SAMPLE_SIZE = 60_000
TEST_SIZE = 0.2
RANDOM_STATE = 42
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


def ensure_resources_dir() -> None:
    """Создаёт папку resources/, если её нет."""
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
