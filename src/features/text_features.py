"""
Текстовые признаки.

title_len / desc_len исходно создавались в 01_eda.ipynb (Baseline
Handover) — здесь продублированы как stateless-функция на случай,
если пайплайн запускается без предварительного EDA-кэша (см.
build_features.py). Если колонки уже присутствуют (обычный сценарий —
данные пришли из train_eda_base.parquet), функция их не трогает.

title_cat — новый признак, добавленный в 02_feature_engineering.ipynb,
раздел 5 (Interaction Features & Text Refinement). Стал главным
признаком модели по важности (15-17%, см. FE_отчет.md).
"""
import pandas as pd


def add_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """title_len / desc_len — посимвольная длина title/description."""
    df = df.copy()

    if "title_len" not in df.columns:
        df["title_len"] = df["title"].fillna("").str.len().astype("int16")
    if "desc_len" not in df.columns:
        df["desc_len"] = df["description"].fillna("").str.len().astype("int16")

    return df


def add_title_cat(df: pd.DataFrame) -> pd.DataFrame:
    """title_cat = title + ' ' + category_name (02_fe.ipynb, раздел 5)."""
    df = df.copy()

    title_str = df["title"].astype(str).fillna("")
    cat_str = df["category_name"].astype(str).fillna("")
    df["title_cat"] = title_str + " " + cat_str

    return df
