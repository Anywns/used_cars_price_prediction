"""Telegram-бот «Оценщик цены подержанного авто»."""
from __future__ import annotations

import asyncio
import logging

import numpy as np
import pandas as pd
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Application, CommandHandler, ContextTypes,
    ConversationHandler, MessageHandler, filters,
)

from . import config
from .comment_agent import detect_market_segment, explain_price, is_unsupported_luxury

log = logging.getLogger(__name__)

ASK = 0
RANGES = {
    "year": (config.YEAR_MIN, config.YEAR_MAX),
    "odometer": (config.ODOMETER_MIN, config.ODOMETER_MAX),
}

QUESTIONS = [
    ("manufacturer", "Марка авто? (латиницей: ford, toyota, bmw …)"),
    ("year", f"Год выпуска? Допустимо {config.YEAR_MIN}–{config.YEAR_MAX}:"),
    ("odometer", f"Пробег в милях? Допустимо {config.ODOMETER_MIN:,}–{config.ODOMETER_MAX:,}:"),
    ("condition", "Состояние?"),
    ("cylinders", "Сколько цилиндров?"),
    ("fuel", "Тип топлива?"),
    ("title_status", "Статус документов (title)?"),
    ("transmission", "Коробка передач?"),
    ("drive", "Привод?"),
    ("size", "Размер кузова?"),
    ("type", "Тип кузова?"),
    ("paint_color", "Цвет кузова?"),
]


class CarPriceBot:
    """Опрашивает все признаки модели и возвращает оценку цены."""

    def __init__(self, model, token: str):
        self.model = model
        self.token = token
        self.features = list(model.feature_names_in_)

        prep = model.named_steps["prep"]
        cat_cols = prep.transformers_[1][2]
        ohe = prep.named_transformers_["cat"]
        self.categories = {
            col: [str(c) for c in cats]
            for col, cats in zip(cat_cols, ohe.categories_)
        }

        self.defaults = {f: "unknown" for f in self.features}
        self.defaults["year"] = 2015
        self.defaults["odometer"] = 100_000

        self.questions = [(f, p) for f, p in QUESTIONS if f in self.features]

    def build_application(self) -> Application:
        app = Application.builder().token(self.token).build()
        app.add_handler(ConversationHandler(
            entry_points=[CommandHandler("start", self.start)],
            states={ASK: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_answer)]},
            fallbacks=[CommandHandler("cancel", self.cancel)],
        ))
        return app

    def run(self) -> None:
        log.info("Бот запущен. Напиши боту /start в Telegram. Остановить — Ctrl+C.")
        self.build_application().run_polling()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        context.user_data.clear()
        context.user_data["step"] = 0
        await update.message.reply_text(
            "Привет! 🚗 Оценю примерную стоимость подержанного авто.\n"
            "Ответь на несколько вопросов. Для года и пробега показаны допустимые пределы, "
            "для остальных полей выбери вариант на клавиатуре."
        )
        await self._ask(update, 0)
        return ASK

    async def on_answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        step = context.user_data["step"]
        feature, _ = self.questions[step]
        text = update.message.text.strip()

        if feature in RANGES:
            digits = text.replace(" ", "").replace(",", "")
            lo, hi = RANGES[feature]
            if not digits.isdigit() or not (lo <= int(digits) <= hi):
                await update.message.reply_text(f"Введите число в пределах {lo:,}–{hi:,}:")
                return ASK
            context.user_data[feature] = int(digits)
        else:
            context.user_data[feature] = text.lower()

        step += 1
        context.user_data["step"] = step
        if step < len(self.questions):
            await self._ask(update, step)
            return ASK
        return await self._predict(update, context)

    async def _ask(self, update: Update, step: int) -> None:
        feature, prompt = self.questions[step]
        await update.message.reply_text(prompt, reply_markup=self._keyboard(feature))

    def _keyboard(self, feature: str):
        if feature in RANGES:
            return ReplyKeyboardRemove()
        options = [c for c in self.categories.get(feature, []) if c != "unknown"]
        rows = [options[i:i + 3] for i in range(0, len(options), 3)]
        if not rows:
            return ReplyKeyboardRemove()
        return ReplyKeyboardMarkup(rows, one_time_keyboard=True, resize_keyboard=True)

    async def _predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        row = dict(self.defaults)
        row.update({f: context.user_data[f] for f, _ in self.questions})

        if is_unsupported_luxury(row):
            await update.message.reply_text(
                "⚠️ Эта версия проекта оценивает автомобили низкого и среднего сегмента "
                "американского рынка б/у авто. Редкие luxury/supercar марки "
                "вроде Ferrari, Lamborghini, Bentley и Rolls-Royce не входят в область "
                "применения модели.\n\nОценить другое авто — /start",
                reply_markup=ReplyKeyboardRemove(),
            )
            return ConversationHandler.END

        X = pd.DataFrame([row])[self.features]
        price = float(np.expm1(self.model.predict(X)[0]))
        segment = detect_market_segment(row)
        segment_label = "средний" if segment == "middle" else "низкий/массовый"

        comment = await asyncio.to_thread(explain_price, row, price)

        await update.message.reply_text(
            f"💰 Примерная стоимость: ${price:,.0f}\n"
            f"📊 Сегмент: {segment_label}\n\n"
            f"{comment}\n\n"
            "Оценить ещё одно авто — /start",
            reply_markup=ReplyKeyboardRemove(),
        )
        return ConversationHandler.END

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Отменено. Заново — /start", reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END
