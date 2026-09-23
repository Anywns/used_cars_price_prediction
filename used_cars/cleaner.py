"""Этап 2 цепочки — cleaner.

Задача: гарантировать, что очищенный датасет `resources/vehicles_clean.csv`
существует. Если его нет — берёт сырьё у loader, чистит и
сохраняет. Реализует чистку, которая в ноутбуке была только списком.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config, loader

log = logging.getLogger(__name__)


def ensure_clean_csv() -> Path:
    """Вернуть путь к очищенному CSV, построив его при отсутствии.

    Returns:
        Путь к `resources/vehicles_clean.csv`.
    """
    config.ensure_resources_dir()

    if config.CLEAN_CSV.exists():
        log.info("Очищенные данные на месте: %s", config.CLEAN_CSV)
        return config.CLEAN_CSV

    raw_path = loader.ensure_raw_csv()
    log.info("Очищаю данные из %s", raw_path)

    df = pd.read_csv(raw_path, usecols=config.RAW_USECOLS, low_memory=False)
    clean = _clean(df)

    clean.to_csv(config.CLEAN_CSV, index=False)
    log.info("Сохранено %d строк → %s", len(clean), config.CLEAN_CSV)
    return config.CLEAN_CSV


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Чистка по правилам notebook 1 / слайда «Предобработка».

    1) удаляем строки с пропусками в числовых (price, year, odometer);
    2) фильтруем выбросы по цене / году / пробегу;
    3) категориальные пропуски заполняем 'unknown'.
    """
    start = len(df)

    df = df.dropna(subset=[config.TARGET, *config.NUMERIC_FEATURES])

    df = df[
        df[config.TARGET].between(config.PRICE_MIN, config.PRICE_MAX)
        & df["year"].between(config.YEAR_MIN, config.YEAR_MAX)
        & df["odometer"].between(config.ODOMETER_MIN, config.ODOMETER_MAX)
    ]

    df = df.copy()
    df[config.CATEGORICAL_FEATURES] = (
        df[config.CATEGORICAL_FEATURES].fillna("unknown").astype(str)
    )

    df["year"] = df["year"].astype(int)
    df["odometer"] = df["odometer"].astype(int)

    log.info("Чистка: %d → %d строк (отброшено %d)", start, len(df), start - len(df))
    return df[[config.TARGET, *config.FEATURE_COLUMNS]].reset_index(drop=True)
