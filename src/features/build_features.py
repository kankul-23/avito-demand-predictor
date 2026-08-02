"""
Главный пайплайн сборки признаков.

Воспроизводит последовательность 02_feature_engineering.ipynb
(разделы 2, 4, 5, 6, 8) в виде классического sklearn-like fit/transform:

  fit(train_df, test_df)  — считает все агрегаты ТОЛЬКО на переданных
                             данных (медианы цен — на train_df;
                             частоты/счётчики — на train_df + test_df,
                             как в ноутбуке).
  transform(df)            — детерминированно применяет сохранённые
                             агрегаты к любому df (train, test, или
                             новые данные при инференсе).

Признаки, отклонённые в ноутбуке (Time & Missingness — раздел 3;
price_zero, repeat_listing, has_description — Pruning в разделе 5),
сюда намеренно не перенесены. Подробности и цифры — в FE_отчет.md.
"""
import pickle

import pandas as pd

from src import config
from src.data.preprocessor import add_baseline_features, clean_and_prepare_types
from src.features.aggregation_features import (
    add_aggregation_features,
    add_cat_region,
    add_seller_activity_level,
    fit_aggregations,
)
from src.features.price_features import add_price_features, fit_price_medians
from src.features.text_features import add_text_length_features, add_title_cat


class FeatureEngineeringPipeline:
    """
    Инкапсулирует все обученные на train агрегаты (медианы цен, частоты,
    счётчики пользователей) и умеет детерминированно применять их к
    новым данным — как это делали train_df/test_df в ноутбуке.
    """

    def __init__(self):
        self.price_medians_: dict | None = None
        self.aggregations_: dict | None = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        """
        Считает и сохраняет все агрегаты.

        test_df передаётся опционально, но настоятельно рекомендуется:
        в ноутбуке user_ads_count/city_freq/cat_region_freq считались
        на объединении train+test (без использования таргета — Data
        Leakage это не создаёт). Без test_df результат немного разойдётся
        с валидированным в ноутбуке (см. aggregation_features.fit_aggregations).
        """
        train_prepared = self._prepare_for_fit(train_df)
        test_prepared = self._prepare_for_fit(test_df) if test_df is not None else None

        self.price_medians_ = fit_price_medians(train_prepared)
        self.aggregations_ = fit_aggregations(train_prepared, test_prepared)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Строит все 25 признаков финального набора поверх df."""
        if not self.is_fitted:
            raise RuntimeError("Пайплайн не обучен — вызовите fit() перед transform().")

        df = df.copy()

        # Baseline (log_price, log_item_seq, title_len, desc_len)
        # создаётся здесь, только если данные пришли не из EDA-кэша
        # (train_eda_base.parquet их уже содержит).
        missing_baseline = [c for c in config.BASELINE_FEATURES if c not in df.columns]
        if missing_baseline:
            df = add_baseline_features(df)

        df = add_text_length_features(df)          # на случай отсутствия baseline
        df = add_price_features(df, self.price_medians_)
        df = add_cat_region(df)
        df = add_title_cat(df)
        df = add_seller_activity_level(df)
        df = add_aggregation_features(df, self.aggregations_)

        # Финальная типизация непосредственно перед подачей в модель
        df = clean_and_prepare_types(df)

        return df

    def fit_transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        self.fit(train_df, test_df)
        train_out = self.transform(train_df)
        test_out = self.transform(test_df) if test_df is not None else None
        return (train_out, test_out) if test_df is not None else train_out

    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_for_fit(df: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегаты в ноутбуке считались уже после появления cat_region
        (раздел 6 идёт после раздела 5). Гарантируем то же самое здесь,
        не меняя исходный df.
        """
        df = df.copy()
        missing_baseline = [c for c in config.BASELINE_FEATURES if c not in df.columns]
        if missing_baseline:
            df = add_baseline_features(df)
        if "cat_region" not in df.columns:
            df = add_cat_region(df)
        return df

    # ------------------------------------------------------------------
    def save(self, path=config.FEATURE_PIPELINE_PATH):
        if not self.is_fitted:
            raise RuntimeError("Нечего сохранять — пайплайн не обучен.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"price_medians_": self.price_medians_,
                         "aggregations_": self.aggregations_}, f)

    @classmethod
    def load(cls, path=config.FEATURE_PIPELINE_PATH) -> "FeatureEngineeringPipeline":
        with open(path, "rb") as f:
            state = pickle.load(f)
        pipeline = cls()
        pipeline.price_medians_ = state["price_medians_"]
        pipeline.aggregations_ = state["aggregations_"]
        pipeline.is_fitted = True
        return pipeline


def prepare_feature_splits(train_df: pd.DataFrame = None, test_df: pd.DataFrame = None):
    """
    Общая для ВСЕХ моделей (CatBoost, LightGBM, логистическая регрессия)
    подготовка: загрузка parquet при необходимости, FeatureEngineeringPipeline
    (25 признаков) + бинаризация таргета + time-based split.

    Вынесено сюда одной функцией, чтобы три независимых train_*.py
    гарантированно сравнивались на одном и том же признаковом пространстве
    и одном и том же сплите — реализация каждой модели дальше расходится
    только в специфичном для неё финальном кодировании признаков
    (Pool для CatBoost, label+TF-IDF для LightGBM, OneHot+scaling+TF-IDF
    для логрегрессии).

    Возвращает (df_train_split, df_val_split, y_train, y_val, feature_pipeline).
    """
    from src.data.loader import load_processed_data
    from src.data.preprocessor import validate_baseline_features

    if train_df is None or test_df is None:
        train_df, test_df = load_processed_data()

    validate_baseline_features(train_df, test_df)

    feature_pipeline = FeatureEngineeringPipeline()
    df_features_train = feature_pipeline.fit_transform(train_df, test_df)[0]
    df_features = df_features_train[config.FEATURE_COLS]

    y = (train_df[config.TARGET_COL] >= config.TARGET_THRESHOLD).astype("int8")

    split_date = train_df[config.DATE_COL].quantile(config.VALIDATION_SPLIT_QUANTILE)
    train_mask = train_df[config.DATE_COL] < split_date
    val_mask = train_df[config.DATE_COL] >= split_date

    df_train_split = df_features[train_mask]
    df_val_split = df_features[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    return df_train_split, df_val_split, y_train, y_val, feature_pipeline
