"""Оркестратор цепочки получения модели: loader → cleaner → builder.

Точка входа (app.py) вызывает :func:`provide_model` и получает готовый
обученный pipeline. Цепочка ленивая: каждый этап пропускается, если его
артефакт уже лежит в resources/.

    builder.ensure_model()        # нет .pkl  ↓
      └─ cleaner.ensure_clean_csv()   # нет clean.csv ↓
           └─ loader.ensure_raw_csv() # нет raw.csv → качаем с Kaggle

Если .pkl уже есть — ни данные, ни Kaggle не трогаются.
"""
from __future__ import annotations

import logging

from sklearn.pipeline import Pipeline

from . import builder

log = logging.getLogger(__name__)


def provide_model() -> Pipeline:
    """Обеспечить и вернуть обученную модель оценки цены."""
    log.info("Получение модели: запускаю цепочку loader → cleaner → builder")
    model = builder.ensure_model()
    log.info("Модель готова (%d признаков на вход)", len(model.feature_names_in_))
    return model
