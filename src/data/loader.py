"""
Загрузка данных.

load_processed_data() воспроизводит 02_feature_engineering.ipynb, ячейку
"1. Загрузка данных из Parquet (Baseline Handover)" — обычный вход для
Feature Engineering, когда 01_eda.ipynb уже отработал и сохранил
train_eda_base.parquet / test_eda_base.parquet.

load_raw_data() читает сырые train.csv/test.csv — нужен, если пайплайн
запускается с нуля (без предварительного EDA-кэша), например в тестах
или в CI.
"""
import pandas as pd

from src import config


def load_raw_data(
    train_path=config.TRAIN_RAW_PATH,
    test_path=config.TEST_RAW_PATH,
):
    """
    Читает исходные train.csv / test.csv.

    activation_date парсится в datetime сразу при чтении — это критично:
    train_catboost.py делает time-based split через .quantile() по этой колонке,
    а на строковом dtype quantile() либо упадёт, либо даст неверный сплит
    (лексикографическое, а не хронологическое сравнение дат).
    """
    train_df = pd.read_csv(train_path, parse_dates=[config.DATE_COL])
    test_df = pd.read_csv(test_path, parse_dates=[config.DATE_COL])
    return train_df, test_df


def load_processed_data(
    train_path=config.TRAIN_BASE_PATH,
    test_path=config.TEST_BASE_PATH,
):
    """
    Загружает train_eda_base.parquet / test_eda_base.parquet —
    датасет после EDA-этапа (01_eda.ipynb, раздел 8), уже содержащий
    baseline-признаки (log_price, log_item_seq, title_len, desc_len)
    и синхронизированные типы данных.
    """
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    missing_baseline = [
        c for c in config.BASELINE_FEATURES if c not in train_df.columns
    ]
    if missing_baseline:
        raise ValueError(
            "В train_eda_base.parquet отсутствуют baseline-признаки "
            f"{missing_baseline}. Похоже, файл создан не из "
            "01_eda.ipynb (раздел 8) — используйте add_baseline_features()."
        )

    return train_df, test_df


def save_processed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path=config.TRAIN_BASE_PATH,
    test_path=config.TEST_BASE_PATH,
):
    """Сохраняет обработанные датафреймы в parquet (01_eda.ipynb, раздел 8)."""
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
