"""
Сравнение CatBoost, LightGBM и логистической регрессии на идентичном
признаковом пространстве и одном и том же time-based split
(план проекта, шаг 5).

Запуск:
    python -m scripts.compare_models

Печатает сравнительную таблицу метрик и сохраняет её в
models/model_comparison.json.
"""
import json
import logging

from src import config
from src.models.train_catboost import train_model
from src.models.train_lightgbm import train_lightgbm_model
from src.models.train_logreg import train_logreg_model

logger = logging.getLogger(__name__)


def compare_models(save: bool = True) -> dict:
    logger.info("=== Обучение CatBoost ===")
    _, _, catboost_metrics, catboost_importance = train_model(save=False)

    logger.info("=== Обучение LightGBM ===")
    _, _, lgbm_metrics, lgbm_importance = train_lightgbm_model(save=False)

    logger.info("=== Обучение логистической регрессии (baseline) ===")
    _, _, logreg_metrics, logreg_importance = train_logreg_model(save=False)

    comparison = {
        "CatBoost": catboost_metrics,
        "LightGBM": lgbm_metrics,
        "LogisticRegression": logreg_metrics,
    }

    logger.info("\n%-20s %10s %10s", "Модель", "ROC-AUC", "PR-AUC")
    for name, m in comparison.items():
        logger.info("%-20s %10.4f %10.4f", name, m["roc_auc"], m["pr_auc"])

    top10 = {
        "CatBoost": catboost_importance.head(10).to_dict(orient="records"),
        "LightGBM": lgbm_importance.head(10).to_dict(orient="records"),
        "LogisticRegression": logreg_importance.head(10).to_dict(orient="records"),
    }

    result = {"metrics": comparison, "top10_feature_importance": top10}

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.MODELS_DIR / "model_comparison.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("Сохранено: %s", out_path)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    compare_models()
