"""
Разовый Optuna-поиск гиперпараметров CatBoost.

Воспроизводит 02_feature_engineering.ipynb, раздел 8 (Optuna, 15 испытаний
TPE, оптимизация ROC-AUC). Использует тот же feature-пайплайн и тот же
time-based split, что и src/models/train_catboost.py (через общую
prepare_train_val_pools) — иначе подобранные параметры могли бы не
соответствовать тому, на чём реально обучается финальная модель.

Запуск (не входит в обычный цикл обучения — считается один раз,
результат переиспользуется):

    python -m scripts.tune_hyperparameters
    python -m scripts.tune_hyperparameters --n-trials 30

После запуска src/models/train_catboost.py автоматически подхватит найденные
параметры из models/best_params.json (см. config.get_catboost_params()).
"""
import argparse
import json
import logging

import optuna
from catboost import CatBoostClassifier

from src import config
from src.models.train_catboost import prepare_train_val_pools
from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)

# Пространство поиска — то же самое, что тюнилось в 02_fe.ipynb, раздел 8.
SEARCH_SPACE = {
    "iterations": (500, 1500),
    "learning_rate": (0.01, 0.25),
    "depth": (4, 12),
    "l2_leaf_reg": (1.0, 10.0),
    "random_strength": (1.0, 10.0),
}


def _objective(trial: optuna.Trial, train_pool, val_pool) -> float:
    tuned_params = {
        "iterations": trial.suggest_int("iterations", *SEARCH_SPACE["iterations"]),
        "learning_rate": trial.suggest_float(
            "learning_rate", *SEARCH_SPACE["learning_rate"], log=True
        ),
        "depth": trial.suggest_int("depth", *SEARCH_SPACE["depth"]),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *SEARCH_SPACE["l2_leaf_reg"]),
        "random_strength": trial.suggest_float(
            "random_strength", *SEARCH_SPACE["random_strength"]
        ),
    }
    params = {**config.CATBOOST_FIXED_PARAMS, **tuned_params}

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, verbose=False)

    val_preds = model.predict_proba(val_pool)[:, 1]
    roc_auc = compute_metrics(val_pool.get_label(), val_preds)["roc_auc"]
    return roc_auc


def tune_hyperparameters(n_trials: int = 15, save: bool = True) -> dict:
    """Запускает Optuna-поиск и возвращает {"best_params": ..., "best_value": ...}."""
    logger.info("Подготовка данных (Feature Engineering + time-based split)...")
    train_pool, val_pool, _ = prepare_train_val_pools()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
    )
    logger.info("Запуск Optuna: %d испытаний, метрика — ROC-AUC", n_trials)
    study.optimize(lambda trial: _objective(trial, train_pool, val_pool), n_trials=n_trials)

    result = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": n_trials,
    }
    logger.info("Лучший ROC-AUC: %.4f", study.best_value)
    logger.info("Лучшие параметры: %s", study.best_params)

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("Сохранено: %s", config.BEST_PARAMS_PATH)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Optuna-поиск гиперпараметров CatBoost")
    parser.add_argument(
        "--n-trials", type=int, default=15,
        help="Число испытаний Optuna (по умолчанию 15, как в 02_fe.ipynb)",
    )
    args = parser.parse_args()

    tune_hyperparameters(n_trials=args.n_trials)
