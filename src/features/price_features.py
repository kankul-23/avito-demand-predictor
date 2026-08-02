"""
Ценовые признаки (02_feature_engineering.ipynb, раздел 4 — Relative Price).

Медианы цен по категории/региону считаются ИСКЛЮЧИТЕЛЬНО на train
(во избежание Data Leakage — см. Engineering Principles, раздел 1.3
ноутбука), затем применяются к train и test через fit/transform.

Признак price_zero, построенный в ноутбуке на этом же шаге, сюда не
перенесён: он был удалён Pruning-ом в разделе 5 и не входит в
финальный набор из 25 признаков — воспроизводить его в продакшене
незачем.
"""
import pandas as pd


def fit_price_medians(train_df: pd.DataFrame) -> dict:
    """
    Считает медианы цены по category_name и region строго на train.
    observed=False — как в ноутбуке, чтобы поведение groupby на
    категориальных колонках не менялось между версиями pandas.
    """
    cat_median_map = (
        train_df.groupby("category_name", observed=False)["price"].median().to_dict()
    )
    reg_median_map = (
        train_df.groupby("region", observed=False)["price"].median().to_dict()
    )
    return {"cat_median_map": cat_median_map, "reg_median_map": reg_median_map}


def add_price_features(df: pd.DataFrame, price_medians: dict) -> pd.DataFrame:
    """
    Добавляет price_to_cat_median и price_to_reg_median.

    +1 в знаменателе — обязателен: в данных есть категории с медианной
    ценой 0 (сегмент "отдам даром", см. EDA раздел 3.4). Без +1 деление
    даёт inf, который НЕ отлавливается последующим fillna(-1)
    (clean_and_prepare_types обрабатывает только NaN, не inf).
    """
    df = df.copy()

    cat_med = df["category_name"].map(price_medians["cat_median_map"])
    reg_med = df["region"].map(price_medians["reg_median_map"])

    df["price_to_cat_median"] = (df["price"] / (cat_med + 1)).astype("float32")
    df["price_to_reg_median"] = (df["price"] / (reg_med + 1)).astype("float32")

    return df
