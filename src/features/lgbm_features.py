"""
Препроцессинг признаков под LightGBM.

LightGBM, в отличие от CatBoost, не умеет работать со строковыми
категориями и текстом напрямую:
  - категориальные признаки нужно закодировать целыми числами
    (LightGBM умеет нативные категории, но только через integer-encoded
    колонки + categorical_feature=[...], не через сырые строки);
  - текстовые признаки (title, description, title_cat) у LightGBM вообще
    нет встроенной обработки — нужна ручная векторизация (здесь — TF-IDF).

Это ровно то архитектурное расхождение с CatBoost, которое обсуждалось
при выборе модели в EDA (раздел 6.5) и в итоговом отчёте (раздел 7).
Базовые 25 признаков при этом те же самые (строятся тем же
FeatureEngineeringPipeline) — меняется только их финальное кодирование.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config

# Зарезервированный код для пропусков и категорий, не встреченных при fit()
# (unseen-категории — см. Categorical Drift, EDA раздел 6.3). 0 никогда не
# назначается реальной категории — encoders начинают нумерацию с 1.
UNKNOWN_CODE = 0


def fit_label_encoders(df: pd.DataFrame, cat_cols=config.CAT_FEATURES) -> dict:
    """
    Строит {колонка: {значение: код}} по обучающей выборке.
    Код 0 зарезервирован под unseen/NaN и НЕ используется ни для одной
    реальной категории — на inference неизвестное значение получает 0,
    а не падает с KeyError.
    """
    encoders = {}
    for col in cat_cols:
        categories = df[col].astype(str).fillna("missing").unique()
        encoders[col] = {cat: code for code, cat in enumerate(categories, start=1)}
    return encoders


def transform_label_encoding(
    df: pd.DataFrame, encoders: dict, cat_cols=config.CAT_FEATURES
) -> pd.DataFrame:
    """Применяет encoders: значение -> int32 код, unseen/NaN -> UNKNOWN_CODE."""
    out = pd.DataFrame(index=df.index)
    for col in cat_cols:
        mapping = encoders[col]
        out[col] = (
            df[col].astype(str).fillna("missing").map(mapping).fillna(UNKNOWN_CODE)
        ).astype("int32")
    return out


def fit_tfidf_vectorizers(
    df: pd.DataFrame,
    text_cols=config.TEXT_FEATURES,
    max_features_map=config.TFIDF_MAX_FEATURES,
) -> dict:
    """Обучает по одному TfidfVectorizer на каждое текстовое поле, строго на train."""
    vectorizers = {}
    for col in text_cols:
        vectorizer = TfidfVectorizer(
            max_features=max_features_map[col],
            lowercase=True,
            ngram_range=(1, 1),
        )
        vectorizer.fit(df[col].astype(str).fillna(""))
        vectorizers[col] = vectorizer
    return vectorizers


def transform_tfidf(
    df: pd.DataFrame, vectorizers: dict, text_cols=config.TEXT_FEATURES
) -> dict:
    """Возвращает {колонка: разреженная TF-IDF матрица}."""
    return {
        col: vectorizers[col].transform(df[col].astype(str).fillna(""))
        for col in text_cols
    }


def build_lgbm_matrix(
    df: pd.DataFrame,
    encoders: dict,
    vectorizers: dict,
    num_cols=config.NUM_FEATURES,
    cat_cols=config.CAT_FEATURES,
    text_cols=config.TEXT_FEATURES,
):
    """
    Собирает единую разреженную матрицу признаков для LightGBM:
    numeric + label-encoded categorical + TF-IDF(text), в таком порядке.

    Возвращает (X, feature_names, cat_feature_indices) —
    cat_feature_indices нужен для LGBMClassifier(categorical_feature=...).
    """
    num_block = sp.csr_matrix(df[num_cols].to_numpy(dtype="float32"))

    cat_encoded = transform_label_encoding(df, encoders, cat_cols)
    cat_block = sp.csr_matrix(cat_encoded.to_numpy(dtype="int32"))

    tfidf_blocks = transform_tfidf(df, vectorizers, text_cols)
    tfidf_matrices = [tfidf_blocks[col] for col in text_cols]

    X = sp.hstack([num_block, cat_block, *tfidf_matrices], format="csr")

    feature_names = (
        list(num_cols)
        + list(cat_cols)
        + [
            f"{col}_tfidf_{term}"
            for col in text_cols
            for term in vectorizers[col].get_feature_names_out()
        ]
    )

    # Индексы категориальных колонок в итоговой матрице — сразу после numeric
    cat_start = len(num_cols)
    cat_feature_indices = list(range(cat_start, cat_start + len(cat_cols)))

    return X, feature_names, cat_feature_indices
