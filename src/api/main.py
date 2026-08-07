"""
FastAPI-сервис для инференса модели (план проекта, шаг 7).

Не дублирует логику src/models/predict.py — оборачивает уже готовые
load_model() / predict() / predict_label() в HTTP-эндпоинты. Модель и
feature-пайплайн загружаются ОДИН РАЗ при старте сервиса (lifespan),
а не на каждый запрос — иначе каждый /predict заново читал бы
catboost_model.cbm и feature_pipeline.pkl с диска, что на реальном
трафике было бы неоправданно медленно.

Эндпоинты:
  GET  /health   — жив ли сервис и загружена ли модель.
  POST /predict  — вероятность продажи + бинарная рекомендация
                    продвижения (config.PREDICTION_THRESHOLD) для
                    ОДНОГО объявления за запрос (см. docstring
                    src/api/schemas.ItemFeatures — почему один, а не
                    массив: это осознанное упрощение первой версии,
                    не ограничение архитектуры).

Запуск (из корня проекта):
    uvicorn src.api.main:app --reload
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Документация после запуска:
    http://127.0.0.1:8000/docs   (Swagger UI, автогенерируется FastAPI)
"""
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from src import config
from src.api.schemas import HealthResponse, ItemFeatures, PredictionResponse
from src.features.build_features import FeatureEngineeringPipeline
from src.models.predict import load_model, predict

logger = logging.getLogger(__name__)

# Заполняются в lifespan при старте сервиса — обычные модуль-level
# переменные, а не app.state, чтобы не тащить FastAPI-специфику в
# сигнатуры функций ниже; для сервиса из одного процесса (без
# нескольких worker'ов, делящих память) этого достаточно.
_model = None
_feature_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Загружает модель и feature-пайплайн один раз при старте приложения
    (а не лениво на первый запрос) — так ошибка отсутствующего
    catboost_model.cbm/feature_pipeline.pkl проявится сразу при запуске
    сервиса, а не непредсказуемо на первом реальном /predict.
    """
    global _model, _feature_pipeline

    logger.info("Загрузка модели: %s", config.MODEL_PATH)
    if not config.MODEL_PATH.exists():
        raise RuntimeError(
            f"Модель не найдена: {config.MODEL_PATH}. "
            "Сначала обучите её: python -m src.models.train_catboost"
        )
    _model = load_model()

    logger.info("Загрузка feature-пайплайна: %s", config.FEATURE_PIPELINE_PATH)
    if not config.FEATURE_PIPELINE_PATH.exists():
        raise RuntimeError(
            f"Feature pipeline не найден: {config.FEATURE_PIPELINE_PATH}. "
            "Сначала обучите модель: python -m src.models.train_catboost"
        )
    _feature_pipeline = FeatureEngineeringPipeline.load()

    logger.info("Сервис готов принимать запросы.")
    yield

    logger.info("Остановка сервиса.")


app = FastAPI(
    title="Avito Demand Predictor API",
    description="Предсказание вероятности продажи объявления и рекомендация продвижения.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Проверка живости сервиса и факта загрузки модели."""
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(item: ItemFeatures) -> PredictionResponse:
    """
    Предсказывает вероятность продажи для одного объявления.

    item -> DataFrame из одной строки -> predict() (src/models/predict.py,
    тот же путь, что и при обучении: feature_pipeline.transform() строит
    все 25 признаков, включая устойчивые к unseen-значениям агрегаты
    user_ads_count/city_freq/cat_region_freq — см. aggregation_features.py).
    """
    if _model is None or _feature_pipeline is None:
        # На практике недостижимо, если lifespan отработал (он бы упал
        # раньше) — защитная проверка на случай ручного вызова функции
        # в тестах в обход обычного жизненного цикла приложения.
        raise HTTPException(status_code=503, detail="Модель ещё не загружена.")

    df = pd.DataFrame([item.model_dump()])

    try:
        proba = predict(df, model=_model, feature_pipeline=_feature_pipeline)
    except Exception as exc:
        logger.exception("Ошибка при предсказании")
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {exc}") from exc

    probability = float(proba.iloc[0])
    threshold = config.PREDICTION_THRESHOLD

    return PredictionResponse(
        deal_probability=probability,
        recommend_promotion=probability >= threshold,
        threshold_used=threshold,
    )
