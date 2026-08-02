"""
Stage 1: сырые train.csv/test.csv -> data/processed/*.parquet.

Разовая, воспроизводимая подготовка данных (01_eda.ipynb, разделы 2 и 8):
  1. Загрузка сырых CSV.
  2. Оптимизация типов (category / datetime64 / Int64).
  3. Добавление baseline-признаков (log_price, log_item_seq, title_len, desc_len).
  4. Проверка согласованности схемы train/test.
  5. Сохранение в parquet.

Запуск:
    python -m scripts.build_dataset

После этого data/processed/train_eda_base.parquet и test_eda_base.parquet
готовы, и весь дальнейший цикл экспериментов (src.models.train) идёт
только от них — не пересчитывая эту часть заново на каждый запуск.
"""
import logging

from src.data.loader import load_raw_data, save_processed_data
from src.data.preprocessor import (
    add_baseline_features,
    optimize_dtypes,
    validate_schema_consistency,
)

logger = logging.getLogger(__name__)


def build_dataset(save: bool = True):
    """Выполняет весь stage 1 и возвращает (train_df, test_df)."""
    logger.info("Загрузка сырых данных...")
    train_df, test_df = load_raw_data()
    logger.info("train_df: %s, test_df: %s", train_df.shape, test_df.shape)

    logger.info("Оптимизация типов данных...")
    train_df = optimize_dtypes(train_df)
    test_df = optimize_dtypes(test_df)

    logger.info("Добавление baseline-признаков...")
    train_df = add_baseline_features(train_df)
    test_df = add_baseline_features(test_df)

    logger.info("Проверка согласованности схемы train/test...")
    validate_schema_consistency(train_df, test_df)

    if save:
        save_processed_data(train_df, test_df)
        logger.info("Сохранено в data/processed/*.parquet")

    return train_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_dataset()
