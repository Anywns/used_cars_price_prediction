"""Настройка логирования: вывод в консоль и в файл `logs/<дата>.log`."""
from __future__ import annotations

import logging
from datetime import date

from . import config


def setup_logging(level: int = logging.INFO) -> None:
    """Логировать в консоль и в файл `logs/YYYY-MM-DD.log` (один файл на дату)."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_DIR / f"{date.today():%Y-%m-%d}.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
