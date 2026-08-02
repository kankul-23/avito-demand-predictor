"""
Разовый Optuna-поиск гиперпараметров LightGBM.

По аналогии с scripts/tune_catboost.py (CatBoost), но с важным
отличием: пространство поиска включает не только гиперпараметры модели
(n_estimators, learning_rate, max_depth, num_leaves, reg_lambda), но и
размер словаря TF-IDF по каждому текстовому полю (title, description,
title_cat).

Причина: при первом сравнении CatBoost vs LightGBM на дефолтных
параметрах LightGBM заметно отстал (ROC-AUC 0.8341 vs 0.8383, PR-AUC
0.5028 vs 0.5157) — а 1100 из 1122 колонок его признаковой матрицы были
TF-IDF при скромных max_features (300/500/300). Тюнить только модельные
гиперпараметры, оставив TF-IDF фиксированным, было бы нечестно по
отношению к LightGBM: узкое текстовое представление — вероятный главный
источник разрыва, а не архитектура бустинга сама по себе.

Дороже, чем тюнинг CatBoost: TF-IDF пересчитывается на КАЖДОМ trial
(в отличие от CatBoost Pool, который строится один раз и переиспользуется
всеми 15 испытаниями) — поиск занимает заметно больше времени за то же
число trials.

Запуск:
    python -m scripts.tune_lightgbm
    python -m scripts.tune_lightgbm --n-trials 20

После запуска src/models/train_lightgbm.py автоматически подхватит
найденные параметры из models/best_params_lgbm.json.
"""
import argparse
import json
import logging

import lightgbm as lgb
import optuna

from src import config
from src.features.lgbm_features import build_lgbm_matrix, fit_tfidf_vectorizers
from src.models.train_lightgbm import prepare_lgbm_splits
from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)

# Гиперпараметры модели — тот же принцип диапазонов, что и у CatBoost
# (scripts/tune_catboost.py), адаптированный под API LightGBM.
MODEL_SEARCH_SPACE = {
    "n_estimators": (500, 1500),
    "learning_rate": (0.01, 0.25),
    "max_depth": (4, 12),
    "num_leaves": (15, 255),
    "reg_lambda": (1.0, 10.0),
}

# Размер словаря TF-IDF — часть пространства поиска (см. docstring выше).
TFIDF_SEARCH_SPACE = {
    "title": (100, 800),
    "description": (200, 1500),
    "title_cat": (100, 800),
}


def _objective(trial: optuna.Trial, df_train_split, df_val_split, y_train, y_val, encoders) -> float:
    tfidf_max_features = {
        col: trial.suggest_int(f"tfidf__{col}", *bounds)
        for col, bounds in TFIDF_SEARCH_SPACE.items()
    }
    vectorizers = fit_tfidf_vectorizers(
        df_train_split, max_features_map=tfidf_max_features
    )
    X_train, _, cat_indices = build_lgbm_matrix(df_train_split, encoders, vectorizers)
    X_val, _, _ = build_lgbm_matrix(df_val_split, encoders, vectorizers)

    model_params = {
        "n_estimators": trial.suggest_int("n_estimators", *MODEL_SEARCH_SPACE["n_estimators"]),
        "learning_rate": trial.suggest_float(
            "learning_rate", *MODEL_SEARCH_SPACE["learning_rate"], log=True
        ),
        "max_depth": trial.suggest_int("max_depth", *MODEL_SEARCH_SPACE["max_depth"]),
        "num_leaves": trial.suggest_int("num_leaves", *MODEL_SEARCH_SPACE["num_leaves"]),
        "reg_lambda": trial.suggest_float("reg_lambda", *MODEL_SEARCH_SPACE["reg_lambda"]),
    }
    params = {**config.LGBM_FIXED_PARAMS, **model_params}

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=cat_indices,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    roc_auc = compute_metrics(y_val, val_preds)["roc_auc"]
    return roc_auc


def tune_lightgbm(n_trials: int = 15, save: bool = True) -> dict:
    """Запускает Optuna-поиск и возвращает найденные параметры модели + TF-IDF."""
    logger.info("Подготовка данных (Feature Engineering + time-based split)...")
    df_train_split, df_val_split, y_train, y_val, encoders = prepare_lgbm_splits()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
    )
    logger.info(
        "Запуск Optuna: %d испытаний, метрика — ROC-AUC "
        "(TF-IDF пересчитывается на каждом trial — ожидайте дольше, чем для CatBoost)",
        n_trials,
    )
    study.optimize(
        lambda trial: _objective(trial, df_train_split, df_val_split, y_train, y_val, encoders),
        n_trials=n_trials,
    )

    best_model_params = {
        k: v for k, v in study.best_params.items() if not k.startswith("tfidf__")
    }
    best_tfidf_max_features = {
        k.replace("tfidf__", ""): v
        for k, v in study.best_params.items()
        if k.startswith("tfidf__")
    }

    result = {
        "best_model_params": best_model_params,
        "best_tfidf_max_features": best_tfidf_max_features,
        "best_value": study.best_value,
        "n_trials": n_trials,
    }
    logger.info("Лучший ROC-AUC: %.4f", study.best_value)
    logger.info("Лучшие параметры модели: %s", best_model_params)
    logger.info("Лучший размер TF-IDF: %s", best_tfidf_max_features)

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.LGBM_BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("Сохранено: %s", config.LGBM_BEST_PARAMS_PATH)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Optuna-поиск гиперпараметров LightGBM")
    parser.add_argument(
        "--n-trials", type=int, default=15,
        help="Число испытаний Optuna (по умолчанию 15 — как для CatBoost, но каждый trial дороже)",
    )
    args = parser.parse_args()

    tune_lightgbm(n_trials=args.n_trials)
