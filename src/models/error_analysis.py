"""
Error Analysis для финальной модели (CatBoost) — план проекта, Шаг 5.

Не пересчитывает метрики агрегатно (для этого есть compute_metrics и
threshold_metrics_table) — смотрит, ГДЕ именно модель ошибается:
в каких category_name и ценовых сегментах чаще всего возникают
False Positive (предсказали продажу, а её не было — риск раздражить
селлера платной услугой) и False Negative (объявление продалось, а
модель его пропустила — упущенная выгода платформы).

Работает поверх val-сплита, который уже даёт prepare_train_val_pools()
(src/models/train_catboost.py) — новых данных/сплитов здесь не создаётся.
"""
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src import config
from src.data.loader import load_processed_data
from src.models.train_catboost import prepare_train_val_pools

logger = logging.getLogger(__name__)


def build_error_frame(
    df_val_raw: pd.DataFrame,
    y_val,
    y_pred_proba,
    threshold: float = config.PREDICTION_THRESHOLD,
) -> pd.DataFrame:
    """
    Собирает построчный датафрейм для анализа ошибок.

    df_val_raw — исходные (не закодированные под конкретную модель)
    строки валидации, содержащие как минимум category_name и price —
    то есть train_df, отфильтрованный по той же val_mask, что и
    y_val/y_pred_proba (см. prepare_feature_splits: маска строится по
    DATE_COL на исходном train_df, поэтому индекс совпадает).

    Возвращает df_val_raw + колонки: y_true, y_pred_proba, y_pred,
    error_type ("TP"/"TN"/"FP"/"FN").
    """
    df = df_val_raw.copy()
    df["y_true"] = np.asarray(y_val)
    df["y_pred_proba"] = np.asarray(y_pred_proba)
    df["y_pred"] = (df["y_pred_proba"] >= threshold).astype("int8")

    conditions = [
        (df["y_true"] == 1) & (df["y_pred"] == 1),
        (df["y_true"] == 0) & (df["y_pred"] == 0),
        (df["y_true"] == 0) & (df["y_pred"] == 1),
        (df["y_true"] == 1) & (df["y_pred"] == 0),
    ]
    choices = ["TP", "TN", "FP", "FN"]
    df["error_type"] = np.select(conditions, choices, default="unknown")

    return df


def error_rate_by_category(
    error_df: pd.DataFrame,
    category_col: str = "category_name",
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Считает по каждой категории: сколько FP, FN, общий n, доли от
    объёма категории и от общего числа FP/FN по датасету.

    Сортирует по fp_count + fn_count — категории, которые вносят
    наибольший вклад в общее число ошибок, а не только по локальному
    error rate (небольшая категория со 100% ошибок — не то же самое,
    что крупная категория с 5% ошибок, но абсолютным числом заметно
    больше вредящая метрике).
    """
    total_fp = (error_df["error_type"] == "FP").sum()
    total_fn = (error_df["error_type"] == "FN").sum()

    grouped = error_df.groupby(category_col, observed=True).agg(
        n=("error_type", "size"),
        fp_count=("error_type", lambda s: (s == "FP").sum()),
        fn_count=("error_type", lambda s: (s == "FN").sum()),
        tp_count=("error_type", lambda s: (s == "TP").sum()),
        tn_count=("error_type", lambda s: (s == "TN").sum()),
    )

    grouped["fp_rate_in_category"] = grouped["fp_count"] / grouped["n"]
    grouped["fn_rate_in_category"] = grouped["fn_count"] / grouped["n"]
    grouped["share_of_all_fp"] = grouped["fp_count"] / total_fp if total_fp else 0.0
    grouped["share_of_all_fn"] = grouped["fn_count"] / total_fn if total_fn else 0.0

    grouped["fp_plus_fn"] = grouped["fp_count"] + grouped["fn_count"]
    grouped = grouped.sort_values("fp_plus_fn", ascending=False).head(top_n)

    return grouped.reset_index()


def error_rate_by_price_bucket(
    error_df: pd.DataFrame,
    price_col: str = "price",
    n_buckets: int = 5,
) -> pd.DataFrame:
    """
    Аналог error_rate_by_category(), но по квантильным ценовым сегментам
    (qcut, а не равным интервалам — цены на Avito сильно скошены вправо,
    равные интервалы дали бы почти пустые бины на верхнем хвосте, см.
    EDA_отчет.md по логике log_price).

    Строки с price=NaN (если есть) попадают в отдельный бакет "missing",
    а не отбрасываются молча.
    """
    df = error_df.copy()

    has_price = df[price_col].notna()
    df.loc[has_price, "price_bucket"] = pd.qcut(
        df.loc[has_price, price_col], q=n_buckets, duplicates="drop"
    ).astype(str)
    df["price_bucket"] = df["price_bucket"].fillna("missing")

    total_fp = (df["error_type"] == "FP").sum()
    total_fn = (df["error_type"] == "FN").sum()

    grouped = df.groupby("price_bucket", observed=True).agg(
        n=("error_type", "size"),
        fp_count=("error_type", lambda s: (s == "FP").sum()),
        fn_count=("error_type", lambda s: (s == "FN").sum()),
    )

    grouped["fp_rate_in_bucket"] = grouped["fp_count"] / grouped["n"]
    grouped["fn_rate_in_bucket"] = grouped["fn_count"] / grouped["n"]
    grouped["share_of_all_fp"] = grouped["fp_count"] / total_fp if total_fp else 0.0
    grouped["share_of_all_fn"] = grouped["fn_count"] / total_fn if total_fn else 0.0

    # qcut даёт Interval — сортируем бакеты по нижней границе, а не
    # по алфавиту строки, иначе порядок сегментов будет случайным.
    def _sort_key(bucket_str):
        if bucket_str == "missing":
            return float("inf")
        return float(bucket_str.strip("([]) ").split(",")[0])

    grouped = grouped.reset_index()
    grouped = grouped.sort_values(
        by="price_bucket", key=lambda col: col.map(_sort_key)
    )

    return grouped.reset_index(drop=True)


def log_error_summary(error_df: pd.DataFrame) -> None:
    """Короткая сводка: сколько всего FP/FN и какая их доля от всех строк."""
    counts = error_df["error_type"].value_counts()
    n = len(error_df)
    logger.info("Всего строк в валидации: %d", n)
    for t in ("TP", "TN", "FP", "FN"):
        c = counts.get(t, 0)
        logger.info("%s: %d (%.2f%%)", t, c, 100 * c / n if n else 0)


def run_error_analysis() -> pd.DataFrame:
    """
    Полный прогон: загружает модель, считает предсказания на валидации,
    строит error_df и печатает сводку + разбивки по категории/цене.

    Вынесено в отдельную функцию (а не оставлено внутри if __name__), чтобы
    при желании можно было вызвать этот же прогон из ноутбука или другого
    скрипта (например, shap_error_analysis.run_shap_error_analysis
    использует ту же логику построения error_df).

    Возвращает error_df — пригодится, если нужно передать его дальше
    в SHAP-анализ без повторного пересчёта предсказаний.
    """
    train_pool, val_pool, _ = prepare_train_val_pools()
    model = CatBoostClassifier()
    model.load_model(str(config.MODEL_PATH))
    y_pred_proba = model.predict_proba(val_pool)[:, 1]
    y_val = val_pool.get_label()

    # исходные (не закодированные под модель) строки того же val-сплита —
    # нужны category_name/price в исходном, читаемом виде, а не в том,
    # как их видит Pool.
    train_df, test_df = load_processed_data()
    split_date = train_df[config.DATE_COL].quantile(config.VALIDATION_SPLIT_QUANTILE)
    df_val_raw = train_df[train_df[config.DATE_COL] >= split_date]

    error_df = build_error_frame(df_val_raw, y_val, y_pred_proba)
    log_error_summary(error_df)

    logger.info("\n=== Ошибки по category_name (топ-15 по FP+FN) ===")
    cat_table = error_rate_by_category(error_df)
    logger.info("\n%s", cat_table.to_string())

    logger.info("\n=== Ошибки по ценовым сегментам ===")
    price_table = error_rate_by_price_bucket(error_df)
    logger.info("\n%s", price_table.to_string())

    return error_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_error_analysis()
