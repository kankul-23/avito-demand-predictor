"""
Логирование экспериментов в MLflow (план проекта, шаг 6).

Не заменяет compare_models.py и не меняет ни одну train-функцию — это
отдельный, чисто "логирующий" слой поверх уже существующих
train_model() / train_lightgbm_model() / train_logreg_model()
(src/models/train_catboost.py, train_lightgbm.py, train_logreg.py).
Они уже возвращают (model, ..., metrics, feature_importances) — этого
достаточно, чтобы залогировать run, не трогая их внутреннюю логику.

Зачем отдельный файл, а не правка compare_models.py: compare_models.py
уже работает и является источником model_comparison.json, на который
может что-то ссылаться (доклад, отчёт). Обёртка снаружи безопаснее —
если MLflow настроен неправильно или упадёт, обучение моделей и
JSON-сравнение всё равно отработают как раньше.

Что логируется на каждый run:
  - params: гиперпараметры модели (то, что реально уходит в конструктор
    CatBoostClassifier/LGBMClassifier/LogisticRegression)
  - metrics: roc_auc, pr_auc (src/utils/metrics.compute_metrics —
    те же значения, что попадают в model_comparison.json)
  - artifacts: top-10 feature importance (CSV) + сам файл модели,
    если save=True и она уже сохранена в models/

Локальный backend без сервера: MLflow пишет в ./mlruns (тот же путь,
что уже в .gitignore) — mlflow.set_tracking_uri здесь не вызывается,
используется дефолт (файлы на диске в PROJECT_ROOT/mlruns).
Смотреть результаты:
    mlflow ui
и открыть http://localhost:5000 — там таблица всех run'ов с метриками,
можно сортировать/сравнивать графически.

Запуск:
    python -m scripts.mlflow_tracking
    python -m scripts.mlflow_tracking --models catboost lightgbm
    python -m scripts.mlflow_tracking --include-tuning
"""
import argparse
import logging

import mlflow

from src import config
from src.models.train_catboost import train_model
from src.models.train_lightgbm import train_lightgbm_model
from src.models.train_logreg import train_logreg_model, LOGREG_PARAMS

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "avito-demand-predictor"


def _log_feature_importance(feature_importances, run_name: str) -> None:
    """
    Сохраняет top-10 важности признаков как CSV-артефакт run'а.

    Не логируем всю таблицу как metrics (MLflow metrics — это числа по
    имени, а не таблица) — top-10 в виде CSV-файла смотрится в MLflow UI
    как таблица и этого достаточно, полная таблица уже есть в
    model_comparison.json и notebooks.
    """
    import pandas as pd

    top10 = feature_importances.head(10)
    tmp_path = config.MODELS_DIR / f"_tmp_top10_{run_name}.csv"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    top10.to_csv(tmp_path, index=False)
    mlflow.log_artifact(str(tmp_path), artifact_path="feature_importance")
    tmp_path.unlink()


def log_catboost_run() -> dict:
    """Обучает CatBoost (train_model, save=True) и логирует run в MLflow."""
    with mlflow.start_run(run_name="catboost"):
        mlflow.set_tag("model_type", "CatBoost")

        params = config.get_catboost_params()
        mlflow.log_params(params)

        model, _, metrics, feature_importances = train_model(save=True)

        mlflow.log_metrics(metrics)
        _log_feature_importance(feature_importances, "catboost")

        if config.MODEL_PATH.exists():
            mlflow.log_artifact(str(config.MODEL_PATH), artifact_path="model")

        logger.info(
            "CatBoost залогирован: ROC-AUC=%.4f, PR-AUC=%.4f",
            metrics["roc_auc"], metrics["pr_auc"],
        )
        return metrics


def log_lightgbm_run() -> dict:
    """Обучает LightGBM (train_lightgbm_model, save=True) и логирует run."""
    with mlflow.start_run(run_name="lightgbm"):
        mlflow.set_tag("model_type", "LightGBM")

        params = config.get_lgbm_params()
        mlflow.log_params(params)
        mlflow.log_params({
            f"tfidf_{k}": v for k, v in config.get_tfidf_max_features().items()
        })

        model, _, metrics, feature_importances = train_lightgbm_model(save=True)

        mlflow.log_metrics(metrics)
        _log_feature_importance(feature_importances, "lightgbm")

        if config.LGBM_MODEL_PATH.exists():
            mlflow.log_artifact(str(config.LGBM_MODEL_PATH), artifact_path="model")

        logger.info(
            "LightGBM залогирован: ROC-AUC=%.4f, PR-AUC=%.4f",
            metrics["roc_auc"], metrics["pr_auc"],
        )
        return metrics


def log_logreg_run() -> dict:
    """Обучает LogReg (train_logreg_model, save=True) и логирует run."""
    with mlflow.start_run(run_name="logreg_baseline"):
        mlflow.set_tag("model_type", "LogisticRegression")
        mlflow.set_tag("role", "baseline")

        mlflow.log_params(LOGREG_PARAMS)

        model, _, metrics, feature_importances = train_logreg_model(save=True)

        mlflow.log_metrics(metrics)
        _log_feature_importance(feature_importances, "logreg")

        logger.info(
            "LogReg залогирован: ROC-AUC=%.4f, PR-AUC=%.4f",
            metrics["roc_auc"], metrics["pr_auc"],
        )
        return metrics


MODEL_LOGGERS = {
    "catboost": log_catboost_run,
    "lightgbm": log_lightgbm_run,
    "logreg": log_logreg_run,
}


def log_tuning_run(model: str = "catboost", n_trials: int = 15) -> dict:
    """
    Логирует Optuna-тюнинг как один run с итоговым результатом поиска
    (не каждый trial отдельно — 15 микро-run'ов в MLflow только замусорят
    список; сам Optuna уже даёт полную историю trial'ов при необходимости).

    tags "phase"="tuning" отличает такие run'ы от обычного обучения
    (log_catboost_run и т.д.), чтобы в MLflow UI их можно было
    отфильтровать отдельно.
    """
    if model != "catboost":
        raise NotImplementedError(
            "Логирование тюнинга пока реализовано только для CatBoost "
            "(scripts/tune_catboost.py). Для LightGBM (scripts/tune_lightgbm.py) "
            "аналогично добавить при необходимости."
        )

    from scripts.tune_catboost import tune_hyperparameters

    with mlflow.start_run(run_name=f"tuning_{model}"):
        mlflow.set_tag("model_type", "CatBoost")
        mlflow.set_tag("phase", "tuning")
        mlflow.log_param("n_trials", n_trials)

        result = tune_hyperparameters(n_trials=n_trials, save=True)

        mlflow.log_params(result["best_params"])
        mlflow.log_metric("best_roc_auc", result["best_value"])

        logger.info("Тюнинг залогирован: best ROC-AUC=%.4f", result["best_value"])
        return result


def run_all(models=None, include_tuning: bool = False) -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    models = models or list(MODEL_LOGGERS.keys())

    if include_tuning:
        logger.info("=== Optuna-тюнинг CatBoost ===")
        log_tuning_run(model="catboost")

    for name in models:
        logger.info("=== Обучение и логирование: %s ===", name)
        MODEL_LOGGERS[name]()

    logger.info(
        "Готово. Для просмотра результатов запустите 'mlflow ui' "
        "и откройте http://localhost:5000 (эксперимент: %s)",
        EXPERIMENT_NAME,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Обучение моделей с логированием экспериментов в MLflow"
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_LOGGERS.keys()),
        default=None,
        help="Какие модели обучить и залогировать (по умолчанию — все три)",
    )
    parser.add_argument(
        "--include-tuning", action="store_true",
        help="Дополнительно залогировать Optuna-тюнинг CatBoost (15 trials)",
    )
    args = parser.parse_args()

    run_all(models=args.models, include_tuning=args.include_tuning)
