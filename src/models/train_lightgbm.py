"""
Обучение LightGBM — модель для сравнения с CatBoost (план проекта, шаг 5).

Использует ТЕ ЖЕ 25 признаков и тот же time-based split, что и
src/models/train.py (через FeatureEngineeringPipeline) — иначе сравнение
моделей было бы нечестным. Расходится только финальное кодирование
признаков под конкретную библиотеку (src/features/lgbm_features.py):
LightGBM не имеет встроенной обработки текста/строковых категорий,
поэтому категории кодируются целыми числами, а текст — через TF-IDF.

Важное отличие от CatBoost-пайплайна: label-encoders и TF-IDF здесь
обучаются СТРОГО на train-части time-based сплита (после разделения),
а не на train_df целиком, как медианы цен/частоты в
FeatureEngineeringPipeline. Это чуть строже к Data Leakage, чем
унаследованный CatBoost-путь — осознанное решение, а не расхождение
по недосмотру (см. отчёт, раздел 7).

Запуск:
    python -m src.models.train_lightgbm
"""
import logging
import pickle

import lightgbm as lgb
import pandas as pd

from src import config
from src.features.build_features import prepare_feature_splits
from src.features.lgbm_features import (
    build_lgbm_matrix,
    fit_label_encoders,
    fit_tfidf_vectorizers,
)
from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


def prepare_lgbm_splits(train_df: pd.DataFrame = None, test_df: pd.DataFrame = None):
    """
    LightGBM-специфичная обёртка над общей prepare_feature_splits()
    (src/features/build_features.py) — добавляет только label-encoders
    (fit строго на train-сплите).

    TF-IDF сюда намеренно не входит: scripts/tune_lightgbm.py подбирает
    размер словаря как часть пространства поиска, поэтому векторизация
    выполняется отдельно на каждом trial — здесь только то, что от
    размера словаря не зависит и не обязано пересчитываться в цикле поиска.

    Возвращает (df_train_split, df_val_split, y_train, y_val, encoders).
    """
    df_train_split, df_val_split, y_train, y_val, _ = prepare_feature_splits(
        train_df, test_df
    )
    encoders = fit_label_encoders(df_train_split)

    return df_train_split, df_val_split, y_train, y_val, encoders


def train_lightgbm_model(
    train_df: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
    save: bool = True,
    lgbm_params: dict = None,
    tfidf_max_features: dict = None,
):
    """
    Обучает LightGBM на том же признаковом пространстве, что и CatBoost.

    lgbm_params / tfidf_max_features позволяют переопределить конфигурацию
    (используется scripts/tune_lightgbm.py); по умолчанию — через
    config.get_lgbm_params() / config.get_tfidf_max_features(), которые
    сами подхватывают models/best_params_lgbm.json, если он уже есть.

    Возвращает (model, encoders_bundle, metrics, feature_importances).
    """
    df_train_split, df_val_split, y_train, y_val, encoders = prepare_lgbm_splits(
        train_df, test_df
    )

    max_features_map = tfidf_max_features or config.get_tfidf_max_features()
    logger.info("Обучение TF-IDF на train-сплите (max_features=%s)...", max_features_map)
    vectorizers = fit_tfidf_vectorizers(
        df_train_split, max_features_map=max_features_map
    )

    X_train, feature_names, cat_indices = build_lgbm_matrix(
        df_train_split, encoders, vectorizers
    )
    X_val, _, _ = build_lgbm_matrix(df_val_split, encoders, vectorizers)

    logger.info(
        "Матрица признаков: %d колонок (%d numeric+cat, %d TF-IDF)",
        X_train.shape[1], len(config.NUM_FEATURES) + len(config.CAT_FEATURES),
        X_train.shape[1] - len(config.NUM_FEATURES) - len(config.CAT_FEATURES),
    )

    params = lgbm_params or config.get_lgbm_params()
    logger.info("Гиперпараметры LightGBM: %s", params)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=cat_indices,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, val_preds)
    logger.info("ROC-AUC: %.4f | PR-AUC: %.4f", metrics["roc_auc"], metrics["pr_auc"])

    feature_importances = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        })
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    encoders_bundle = {
        "encoders": encoders,
        "vectorizers": vectorizers,
        "feature_names": feature_names,
        "cat_indices": cat_indices,
    }

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(config.LGBM_MODEL_PATH))
        with open(config.LGBM_ENCODERS_PATH, "wb") as f:
            pickle.dump(encoders_bundle, f)
        logger.info("Модель сохранена: %s", config.LGBM_MODEL_PATH)
        logger.info("Encoders/TF-IDF сохранены: %s", config.LGBM_ENCODERS_PATH)

    return model, encoders_bundle, metrics, feature_importances


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_lightgbm_model()
