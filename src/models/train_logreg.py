"""
Обучение логистической регрессии — простой baseline для сравнения с
CatBoost и LightGBM (план проекта, шаг 5).

Использует ТЕ ЖЕ 25 признаков и тот же time-based split, что и остальные
две модели (через src.features.build_features.prepare_feature_splits).
Кодирование — третье, отличное от обеих других моделей (см.
src/features/linear_features.py): One-Hot для категорий вместо
целочисленных кодов LightGBM, StandardScaler для числовых.

Запуск:
    python -m src.models.train_logreg
"""
import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import config
from src.features.build_features import prepare_feature_splits
from src.features.linear_features import build_linear_matrix, fit_linear_preprocessors
from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)

# L2-регуляризация по умолчанию; baseline не тюнится Optuna — это
# сознательно самая простая версия сравнения, а не ещё одна тюненая модель.
LOGREG_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 200,
    "random_state": config.RANDOM_SEED,
    "n_jobs": -1,
}


def train_logreg_model(
    train_df: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
    save: bool = True,
):
    """
    Обучает логистическую регрессию на том же признаковом пространстве,
    что и CatBoost/LightGBM.
    Возвращает (model, preprocessors, metrics, feature_importances).
    """
    df_train_split, df_val_split, y_train, y_val, _ = prepare_feature_splits(
        train_df, test_df
    )

    logger.info("Обучение OneHot/StandardScaler/TF-IDF на train-сплите...")
    preprocessors = fit_linear_preprocessors(df_train_split)

    X_train, feature_names = build_linear_matrix(df_train_split, preprocessors)
    X_val, _ = build_linear_matrix(df_val_split, preprocessors)

    logger.info("Матрица признаков: %d колонок (One-Hot + TF-IDF)", X_train.shape[1])

    model = LogisticRegression(**LOGREG_PARAMS)
    model.fit(X_train, y_train)

    val_preds = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, val_preds)
    logger.info("ROC-AUC: %.4f | PR-AUC: %.4f", metrics["roc_auc"], metrics["pr_auc"])

    # Знак коэффициента показывает направление влияния — в отличие от
    # feature_importance деревьев, здесь это осмысленная информация.
    feature_importances = (
        pd.DataFrame({
            "feature": feature_names,
            "coefficient": model.coef_[0],
            "abs_coefficient": abs(model.coef_[0]),
        })
        .sort_values(by="abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )

    if save:
        import pickle
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.MODELS_DIR / "logreg_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(config.MODELS_DIR / "logreg_preprocessors.pkl", "wb") as f:
            pickle.dump(preprocessors, f)
        logger.info("Модель и препроцессоры сохранены в %s", config.MODELS_DIR)

    return model, preprocessors, metrics, feature_importances


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_logreg_model()
