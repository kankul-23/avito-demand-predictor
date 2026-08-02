"""
Препроцессинг признаков под логистическую регрессию (baseline-модель,
план проекта, шаг 5).

Третье, ещё одно отличное от CatBoost и LightGBM кодирование:

- Категориальные признаки: One-Hot, а НЕ целочисленные коды, как у
  LightGBM. Для дерева порядок кодов не имеет значения (модель просто
  выбирает пороги), а для линейной модели целочисленный код 7 vs 3
  подразумевал бы несуществующее упорядочивание категорий — это была бы
  содержательная ошибка, а не мелкая деталь.
- Числовые признаки: StandardScaler. Линейная модель чувствительна к
  масштабу (в отличие от деревьев) — price (сотни/тысячи) и city_freq
  (0-1) без масштабирования доминировали бы в регуляризации не по
  значимости, а по единицам измерения.
- Текстовые признаки: TF-IDF — переиспользует ту же логику, что и
  LightGBM (src/features/lgbm_features.py), никакой разницы в подходе
  к тексту между этими двумя моделями нет.

Высокая кардинальность city (1733) и param_2/3 (сотни) даёт после
One-Hot несколько тысяч разреженных колонок — это ожидаемо для линейной
baseline-модели и не проблема (LogisticRegression с L2 работает с
разреженными матрицами эффективно), но стоит иметь в виду при
интерпретации размера итоговой матрицы признаков.
"""
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.features.lgbm_features import fit_tfidf_vectorizers, transform_tfidf


def fit_linear_preprocessors(
    df: pd.DataFrame,
    cat_cols=config.CAT_FEATURES,
    num_cols=config.NUM_FEATURES,
    text_cols=config.TEXT_FEATURES,
    tfidf_max_features: dict = None,
) -> dict:
    """Обучает OneHotEncoder, StandardScaler и TF-IDF строго на train-сплите."""
    onehot = OneHotEncoder(handle_unknown="ignore", dtype="float32")
    onehot.fit(df[cat_cols].astype(str))

    scaler = StandardScaler()
    scaler.fit(df[num_cols].to_numpy(dtype="float32"))

    vectorizers = fit_tfidf_vectorizers(
        df, text_cols=text_cols,
        max_features_map=tfidf_max_features or config.TFIDF_MAX_FEATURES,
    )

    return {"onehot": onehot, "scaler": scaler, "vectorizers": vectorizers}


def build_linear_matrix(
    df: pd.DataFrame,
    preprocessors: dict,
    cat_cols=config.CAT_FEATURES,
    num_cols=config.NUM_FEATURES,
    text_cols=config.TEXT_FEATURES,
):
    """
    Собирает единую разреженную матрицу: scaled numeric + one-hot
    categorical + TF-IDF(text), в таком порядке.

    Возвращает (X, feature_names).
    """
    num_scaled = sp.csr_matrix(
        preprocessors["scaler"].transform(df[num_cols].to_numpy(dtype="float32"))
    )
    cat_onehot = preprocessors["onehot"].transform(df[cat_cols].astype(str))

    tfidf_blocks = transform_tfidf(df, preprocessors["vectorizers"], text_cols)
    tfidf_matrices = [tfidf_blocks[col] for col in text_cols]

    X = sp.hstack([num_scaled, cat_onehot, *tfidf_matrices], format="csr")

    feature_names = (
        list(num_cols)
        + list(preprocessors["onehot"].get_feature_names_out(cat_cols))
        + [
            f"{col}_tfidf_{term}"
            for col in text_cols
            for term in preprocessors["vectorizers"][col].get_feature_names_out()
        ]
    )

    return X, feature_names
