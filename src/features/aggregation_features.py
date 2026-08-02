"""
Признаки на основе агрегаций (02_feature_engineering.ipynb, разделы 5 и 6).

cat_region              — раздел 5, взаимодействие category_name x region.
seller_activity_level   — раздел 6, биннинг item_seq_number.
user_ads_count,
city_freq, cat_region_freq — раздел 6, Frequency Encoding.

Важно (сверено с ноутбуком дословно): user_ads_count и *_freq считаются
на ОБЪЕДИНЕНИИ train + test ("честный расчет frequency encoding и
счетчиков без таргета" — комментарий в ноутбуке), а не только на train.
Это осознанное решение авторов: частоты точнее отражают реальное
распределение, включая объекты, которых нет в train, а таргет при этом
не используется — риска Data Leakage нет.

Следствие для продакшена: fit_aggregations() должен вызываться на
train + test вместе (или на train + новых данных для инференса), а не
только на train, иначе результат разойдётся с тем, что валидировалось
в ноутбуке. См. docstring fit_aggregations().
"""
import numpy as np
import pandas as pd

from src import config


def add_cat_region(df: pd.DataFrame) -> pd.DataFrame:
    """cat_region = category_name + '_' + region, тип category."""
    df = df.copy()
    df["cat_region"] = (
        df["category_name"].astype(str) + "_" + df["region"].astype(str)
    ).astype("category")
    return df


def add_seller_activity_level(
    df: pd.DataFrame,
    bins=config.SELLER_ACTIVITY_BINS,
    labels=config.SELLER_ACTIVITY_LABELS,
) -> pd.DataFrame:
    """Биннинг item_seq_number на low/medium/high/pro."""
    df = df.copy()
    df["seller_activity_level"] = (
        pd.cut(df["item_seq_number"], bins=bins, labels=labels)
        .astype(str)
        .fillna("missing")
    )
    return df


def fit_aggregations(train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> dict:
    """
    Считает user_counts, city_freq_map, cat_region_freq_map.

    Ожидает, что train_df и test_df УЖЕ содержат cat_region
    (см. add_cat_region) — как и в ноутбуке, где cat_region создаётся
    в разделе 5, а частоты по нему — в разделе 6.

    Если test_df не передан (например, инференс на новых данных без
    доступа к полному test), агрегаты считаются только на train_df —
    это расходится с ноутбуком (там всегда train+test) и может немного
    сместить city_freq/cat_region_freq/user_ads_count для объектов,
    которых не было в train. В таком случае стоит либо передавать
    test_df, либо переобучать агрегаты на train + новых данных.
    """
    if test_df is not None:
        full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    else:
        full_df = train_df

    user_counts = full_df["user_id"].value_counts().to_dict()
    city_freq_map = full_df["city"].value_counts(normalize=True).to_dict()
    cat_region_freq_map = full_df["cat_region"].value_counts(normalize=True).to_dict()

    return {
        "user_counts": user_counts,
        "city_freq_map": city_freq_map,
        "cat_region_freq_map": cat_region_freq_map,
    }


def add_aggregation_features(df: pd.DataFrame, aggregations: dict) -> pd.DataFrame:
    """Применяет user_ads_count, city_freq, cat_region_freq к df."""
    df = df.copy()

    df["user_ads_count"] = (
        df["user_id"].map(aggregations["user_counts"]).fillna(1).astype("int32")
    )
    df["city_freq"] = (
        df["city"].map(aggregations["city_freq_map"]).astype("float32").fillna(0)
    )
    df["cat_region_freq"] = (
        df["cat_region"]
        .map(aggregations["cat_region_freq_map"])
        .astype("float32")
        .fillna(0)
    )

    return df
