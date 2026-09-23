"""Этап 3 цепочки — builder.

Задача: гарантировать, что обученная модель `resources/car_price_model.pkl`
существует. Если её нет — берёт очищенные данные у cleaner, обучает выбранную
в research/ модель (RandomForest в sklearn-pipeline) и сохраняет.

Выбор лучшей модели — задача ноутбуков в research/. Здесь воспроизводится
уже выбранная модель, а не сравниваются варианты.
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import cleaner, config

log = logging.getLogger(__name__)


def ensure_model(force_retrain: bool = False) -> Pipeline:
    """Вернуть обученный pipeline, обучив его при отсутствии.

    Args:
        force_retrain: переобучить, даже если .pkl уже существует.

    Returns:
        Обученный sklearn ``Pipeline`` (предобработка + RandomForest).
    """
    config.ensure_resources_dir()

    if config.MODEL_PKL.exists() and not force_retrain:
        log.info("Модель на месте, загружаю: %s", config.MODEL_PKL)
        return joblib.load(config.MODEL_PKL)

    clean_path = cleaner.ensure_clean_csv()
    log.info("Обучаю модель на %s", clean_path)
    df = pd.read_csv(clean_path)

    model = _train(df)

    joblib.dump(model, config.MODEL_PKL)
    log.info("Модель сохранена: %s", config.MODEL_PKL)
    return model


def _build_pipeline() -> Pipeline:
    """Собрать pipeline: предобработка внутри → нет утечки данных при CV."""
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), config.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("prep", preprocess),
        ("model", RandomForestRegressor(**config.RF_PARAMS)),
    ])


def _train(df: pd.DataFrame) -> Pipeline:
    """Обучить и оценить модель; вернуть pipeline, дообученный на всех данных.

    Цель — log1p(price) (устойчивость к выбросам). Метрики считаем на отложенной
    выборке в реальных долларах, затем переобучаем финальную модель на всех
    строках и её возвращаем.
    """
    df = df.dropna(subset=[config.TARGET])
    if config.SAMPLE_SIZE and len(df) > config.SAMPLE_SIZE:
        df = df.sample(config.SAMPLE_SIZE, random_state=config.RANDOM_STATE)
        log.info("Подвыборка для обучения: %d строк", len(df))

    X = df[config.FEATURE_COLUMNS]
    y = np.log1p(df[config.TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)
    _log_metrics(pipe, X_test, y_test)

    final = _build_pipeline()
    final.fit(X, y)
    return final


def _log_metrics(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Залогировать MAE / RMSE / R² на отложенной выборке (в долларах)."""
    y_true = np.expm1(y_test)
    y_pred = np.expm1(pipe.predict(X_test))
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    log.info(
        "Качество на holdout: MAE=$%s  RMSE=$%s  R2=%.3f",
        f"{mae:,.0f}", f"{rmse:,.0f}", r2,
    )
