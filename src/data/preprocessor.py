"""
Базовая очистка, приведение типов, обработка пропусков.

Функции покрывают весь путь 01_eda.ipynb (разделы 2 и 8) от сырого CSV
до готового к сохранению в parquet датафрейма:

- optimize_dtypes()          — раздел 2. category/datetime/Int64.
- add_baseline_features()    — раздел 8 (Baseline Handover).
- validate_schema_consistency() — раздел 8, сверка train/test.
- clean_and_prepare_types()  — 02_feature_engineering.ipynb, раздел 8
                                (Full-Train). Финальное приведение типов
                                непосредственно перед CatBoost Pool —
                                это отдельный, гораздо более простой шаг
                                (всё -> str/float64), и он НЕ заменяет
                                optimize_dtypes.
"""
import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Категориальные признаки, приводимые к dtype 'category' (01_eda.ipynb,
# раздел 2). image_top_1 в эту группу не входит: несмотря на то что в
# ноутбуке он изначально фигурирует в общем списке cat_cols, его тут же
# перезаписывают в 'Int64' — то есть по факту это единственный
# категориальный на вид признак, который остаётся числовым.
CATEGORY_DTYPE_COLS = [
    "region",
    "city",
    "parent_category_name",
    "category_name",
    "param_1",
    "param_2",
    "param_3",
    "user_type",
]

DATE_COLS = [config.DATE_COL]


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит типы к оптимизированным (01_eda.ipynb, раздел 2):
    activation_date -> datetime64, категориальные -> category,
    image_top_1 -> nullable Int64 (сохраняет NaN, в отличие от int).

    Сокращает потребление памяти почти вдвое (EDA_отчет.md, раздел 2) и
    даёт train/test одинаковую схему типов.
    """
    df = df.copy()

    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    for col in CATEGORY_DTYPE_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if "image_top_1" in df.columns:
        df["image_top_1"] = df["image_top_1"].astype("Int64")

    return df


def add_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет 4 baseline-признака (01_eda.ipynb, раздел 8):
    log_price, log_item_seq, title_len, desc_len.

    Формулы воспроизведены дословно — без дополнительного clip(), которого
    не было в ноутбуке (в данных нет отрицательных цен, см. EDA раздел 3.4).

    has_description в ноутбуке создавался тут же, но был удалён Pruning-ом
    в 02_fe.ipynb (раздел 5) и не входит в финальный набор из 25
    признаков — здесь не создаётся вообще, чтобы не тратить на него
    память/время впустую.
    """
    df = df.copy()

    df["log_price"] = np.log1p(df["price"].fillna(0)).astype("float32")
    df["log_item_seq"] = np.log1p(df["item_seq_number"]).astype("float32")

    df["title_len"] = df["title"].fillna("").str.len().astype("int16")
    df["desc_len"] = df["description"].fillna("").str.len().astype("int16")

    return df


def validate_schema_consistency(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Сверка train/test после сборки baseline-признаков (01_eda.ipynb,
    раздел 8): единственное ожидаемое расхождение колонок — deal_probability
    (таргет, которого нет в test).

    Несовпадение dtype для категориальных колонок (city, param_1/2/3)
    ОЖИДАЕМО и не является ошибкой: optimize_dtypes() вызывается отдельно
    на train_df и test_df, поэтому astype('category') видит разные
    наблюдаемые значения — это и есть Categorical Drift, задокументированный
    в EDA (раздел 6.3): в test встречаются категории, которых нет в train,
    и наоборот. Сами строковые значения при этом корректны, расходится
    только служебный список категорий внутри dtype.

    В оригинальном ноутбуке (01_eda.ipynb, раздел 8) эта проверка тоже
    не фатальна — там она просто печатает список расхождений, не прерывая
    выполнение. Здесь по той же логике: несовпадения типов логируются,
    а не поднимают исключение. Жёстко проверяется только состав колонок.
    """
    missing_in_test = set(train_df.columns) - set(test_df.columns)
    unexpected = missing_in_test - {config.TARGET_COL}
    if unexpected:
        raise AssertionError(
            f"В test_df неожиданно отсутствуют колонки: {unexpected}"
        )

    common_cols = list(test_df.columns)
    type_mismatches = [
        c for c in common_cols if train_df[c].dtype != test_df[c].dtype
    ]
    if type_mismatches:
        logger.warning(
            "Несовпадающие dtype между train_df и test_df (ожидаемо для "
            "категориальных признаков с Categorical Drift — см. EDA раздел "
            "6.3): %s",
            type_mismatches,
        )


def validate_baseline_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Проверка наличия baseline-признаков (02_fe.ipynb, раздел 2).

    Вызывается на входе Feature Engineering (train_catboost.py) — если она падает,
    значит вы передали данные напрямую из raw CSV, минуя stage 1
    (scripts/build_dataset.py). Сначала прогоните его.
    """
    for name, df in [("train_df", train_df), ("test_df", test_df)]:
        missing = [c for c in config.BASELINE_FEATURES if c not in df.columns]
        if missing:
            raise AssertionError(
                f"В {name} отсутствуют baseline-признаки: {missing}. "
                "Похоже, вы передали сырые данные напрямую в train_model(). "
                "Сначала выполните: python -m scripts.build_dataset"
            )


def clean_and_prepare_types(
    df: pd.DataFrame,
    cat_features=config.CAT_FEATURES,
    text_features=config.TEXT_FEATURES,
    num_features=config.NUM_FEATURES,
) -> pd.DataFrame:
    """
    Финальное приведение типов перед CatBoost Pool
    (02_fe.ipynb, раздел 8, "3. Подготовка датасета Признаков"):

    - категориальные -> str, пропуски -> 'missing'
    - текстовые      -> str, пропуски -> ''
    - числовые       -> float64, пропуски -> -1
    """
    df = df.copy()

    for col in [c for c in cat_features if c in df.columns]:
        df[col] = df[col].astype(str).fillna("missing")

    for col in [c for c in text_features if c in df.columns]:
        df[col] = df[col].astype(str).fillna("")

    for col in [c for c in num_features if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64").fillna(-1)

    return df
