"""
Обучение модели (stage 2 — работает только с уже подготовленным parquet).

Воспроизводит 02_feature_engineering.ipynb, раздел 8
("Финальный Full-Train и генерация сабмита"):

  1. Бинаризация таргета по TARGET_THRESHOLD.
  2. Time-based валидационный сплит (80/20 по activation_date).
  3. Обучение CatBoostClassifier с гиперпараметрами из
     config.get_catboost_params() — если models/best_params.json уже
     существует (результат scripts/tune_hyperparameters.py), используются
     они; иначе — дефолтные значения, найденные Optuna в ноутбуке.
  4. Расчёт ROC-AUC / PR-AUC на валидации, Top-10 важности признаков.

Ожидаемый результат на конфигурации из ноутбука (см. FE_отчет.md):
ROC-AUC 0.8354, PR-AUC 0.5134.

Вход — train_eda_base.parquet / test_eda_base.parquet (dtype-оптимизация
и baseline-признаки уже посчитаны). Если их ещё нет — сначала выполните
stage 1: `python -m scripts.build_dataset`.
"""
import logging

import pandas as pd
from catboost import CatBoostClassifier, Pool

from src import config
from src.features.build_features import prepare_feature_splits
from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


def prepare_train_val_pools(train_df: pd.DataFrame = None, test_df: pd.DataFrame = None):
    """
    CatBoost-специфичная обёртка над общей prepare_feature_splits()
    (src/features/build_features.py) — строит Pool поверх тех же 25
    признаков и того же time-based сплита, что и LightGBM/логрегрессия.

    Возвращает (train_pool, val_pool, feature_pipeline).
    """
    df_train_split, df_val_split, y_train, y_val, feature_pipeline = (
        prepare_feature_splits(train_df, test_df)
    )

    train_pool = Pool(
        data=df_train_split,
        label=y_train,
        cat_features=config.CAT_FEATURES,
        text_features=config.TEXT_FEATURES,
    )
    val_pool = Pool(
        data=df_val_split,
        label=y_val,
        cat_features=config.CAT_FEATURES,
        text_features=config.TEXT_FEATURES,
    )

    return train_pool, val_pool, feature_pipeline


def train_model(
    train_df: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
    save: bool = True,
    catboost_params: dict = None,
):
    """
    Обучает финальную модель.

    catboost_params позволяет переопределить гиперпараметры (например,
    для экспериментов); по умолчанию берутся через
    config.get_catboost_params() — best_params.json, если он есть,
    иначе дефолты из ноутбука.

    Возвращает (model, feature_pipeline, metrics, feature_importances).
    """
    train_pool, val_pool, feature_pipeline = prepare_train_val_pools(train_df, test_df)

    params = catboost_params or config.get_catboost_params()
    logger.info("Гиперпараметры CatBoost: %s", params)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool)

    val_preds = model.predict_proba(val_pool)[:, 1]
    metrics = compute_metrics(val_pool.get_label(), val_preds)
    logger.info("ROC-AUC: %.4f | PR-AUC: %.4f", metrics["roc_auc"], metrics["pr_auc"])

    feature_importances = (
        pd.DataFrame({
            "feature": config.FEATURE_COLS,
            "importance": model.get_feature_importance(train_pool),
        })
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(str(config.MODEL_PATH))
        feature_pipeline.save()
        logger.info("Модель сохранена: %s", config.MODEL_PATH)
        logger.info("Feature pipeline сохранён: %s", config.FEATURE_PIPELINE_PATH)

    return model, feature_pipeline, metrics, feature_importances


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()
