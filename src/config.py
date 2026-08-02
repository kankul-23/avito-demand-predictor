"""
Центральная конфигурация проекта.

Все константы (пути, списки признаков, гиперпараметры) вынесены сюда,
чтобы train_catboost.py / predict.py / build_features.py не расходились между собой.
Значения перенесены дословно из 01_eda.ipynb (раздел 8) и
02_feature_engineering.ipynb (разделы 2, 4, 5, 6, 8).
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Пути
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_RAW_PATH = RAW_DATA_DIR / "train.csv"
TEST_RAW_PATH = RAW_DATA_DIR / "test.csv"

TRAIN_BASE_PATH = PROCESSED_DATA_DIR / "train_eda_base.parquet"
TEST_BASE_PATH = PROCESSED_DATA_DIR / "test_eda_base.parquet"

MODEL_PATH = MODELS_DIR / "catboost_model.cbm"
FEATURE_PIPELINE_PATH = MODELS_DIR / "feature_pipeline.pkl"

# --------------------------------------------------------------------------
# Целевая переменная
# --------------------------------------------------------------------------
TARGET_COL = "deal_probability"
# Порог бинаризации использовался в FE-ноутбуке для промежуточных экспериментов
# (раздел 8, финальный Full-Train). Итоговый продуктовый порог — решение
# отдельного этапа моделирования, здесь фиксируем то же значение, что и в ноутбуке.
TARGET_THRESHOLD = 0.33
# --------------------------------------------------------------------------
# Порог бинаризации предсказаний (продуктовое решение — план проекта,
# "Подбор порога бинаризации"; см. src/utils/threshold_selection.py).
#
# Не путать с TARGET_THRESHOLD выше: тот бинаризует deal_probability при
# ОБУЧЕНИИ ("что считать успешной продажей"). PREDICTION_THRESHOLD — порог
# для predict_proba модели при ИНФЕРЕНСЕ ("когда рекомендовать продвижение
# селлеру").
#
# Значение 0.30 выбрано на валидационном сплите CatBoost по сетке порогов
# (precision=0.484, recall=0.558, F1=0.518) — почти на пике F1 (0.25 даёт
# F1=0.520, разница 0.002), но с заметно лучшим precision (+4 п.п.) и более
# узким потоком рекомендаций (pos_rate 17.7% против 21.8% на 0.25).
# Обоснование: для рекомендации платного продвижения ложные срабатывания
# (совет купить продвижение объявлению, которое не продастся) напрямую
# бьют по доверию продавца к платформе — небольшая прибавка к precision
# ценнее почти нулевой потери в F1.
PREDICTION_THRESHOLD = 0.30


# --------------------------------------------------------------------------
# Baseline-признаки (созданы в 01_eda.ipynb, раздел 8 — Baseline Handover)
# --------------------------------------------------------------------------
# has_description сюда намеренно не входит: он создавался в EDA, но был
# удалён Pruning-ом в 02_fe.ipynb (раздел 5) и не входит в финальный
# набор из 25 признаков (FEATURE_COLS ниже) — считать его в продакшене
# незачем, см. add_baseline_features().
BASELINE_FEATURES = [
    "log_price",
    "log_item_seq",
    "title_len",
    "desc_len",
]

# --------------------------------------------------------------------------
# Seller Activity Level — биннинг item_seq_number (02_fe, раздел 6)
# --------------------------------------------------------------------------
SELLER_ACTIVITY_BINS = [-1, 5, 25, 100, float("inf")]
SELLER_ACTIVITY_LABELS = ["low", "medium", "high", "pro"]

# --------------------------------------------------------------------------
# Итоговый набор признаков модели (02_fe.ipynb, раздел 8, Full-Train)
# --------------------------------------------------------------------------
CAT_FEATURES = [
    "region",
    "city",
    "parent_category_name",
    "category_name",
    "param_1",
    "param_2",
    "param_3",
    "user_type",
    "cat_region",
    "seller_activity_level",
]

TEXT_FEATURES = ["title", "description", "title_cat"]

NUM_FEATURES = [
    "price",
    "log_price",
    "item_seq_number",
    "log_item_seq",
    "title_len",
    "desc_len",
    "image_top_1",
    "price_to_cat_median",
    "price_to_reg_median",
    "user_ads_count",
    "city_freq",
    "cat_region_freq",
]

FEATURE_COLS = CAT_FEATURES + TEXT_FEATURES + NUM_FEATURES

# Признаки, которые были построены и провалидированы в ноутбуках, но отклонены
# по результатам A/B-проверки на валидации — намеренно не воспроизводятся
# в продакшен-пайплайне (см. FE_отчет.md, разделы 3 и 5.1):
#   - day_of_week, day_of_month, month, is_weekend
#   - has_image, price_is_missing, param2_missing, param3_missing
#   - price_zero, repeat_listing
#   - has_description — создавался как baseline-признак в EDA, но удалялся
#     Pruning-ом ещё до обучения; в этом коде не создаётся вообще
#     (см. BASELINE_FEATURES и add_baseline_features)

# --------------------------------------------------------------------------
# Валидация (02_fe.ipynb, раздел 8): time-based split по activation_date
# --------------------------------------------------------------------------
VALIDATION_SPLIT_QUANTILE = 0.8
DATE_COL = "activation_date"

# --------------------------------------------------------------------------
# Гиперпараметры CatBoost
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# Параметры, НЕ входящие в пространство поиска Optuna (02_fe.ipynb, раздел 8) —
# фиксированы независимо от результата тюнинга.
CATBOOST_FIXED_PARAMS = {
    "eval_metric": "AUC",
    "random_seed": RANDOM_SEED,
    "task_type": "GPU",
    "early_stopping_rounds": 50,
    "verbose": 200,
}

# Значения по умолчанию для параметров, которые тюнит Optuna
# (scripts/tune_catboost.py). Это результат 15 испытаний TPE из
# 02_fe.ipynb, раздел 8 — используются, пока models/best_params.json ещё
# не создан (т.е. тюнинг ни разу не запускался в этом окружении).
CATBOOST_TUNED_DEFAULTS = {
    "iterations": 1200,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 6.61857,
    "random_strength": 8.09578,
}

BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"


def get_catboost_params() -> dict:
    """
    Возвращает полный набор гиперпараметров CatBoost:
    CATBOOST_FIXED_PARAMS + (результат Optuna из best_params.json,
    если файл существует, иначе CATBOOST_TUNED_DEFAULTS).

    best_params.json создаётся через:
        python -m scripts.tune_hyperparameters
    """
    import json

    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
            tuned = json.load(f)["best_params"]
    else:
        tuned = CATBOOST_TUNED_DEFAULTS

    return {**CATBOOST_FIXED_PARAMS, **tuned}


# --------------------------------------------------------------------------
# LightGBM — модель для сравнения с CatBoost (см. раздел 5 плана проекта)
# --------------------------------------------------------------------------
# В отличие от CatBoost, LightGBM не имеет встроенной обработки текста и
# категориальных строк — поэтому у него отдельный, специфичный препроцессинг
# (src/features/lgbm_features.py): категории кодируются целыми числами,
# текст — через TF-IDF. Признаки (CAT_FEATURES/TEXT_FEATURES/NUM_FEATURES)
# при этом ровно те же 25, что и у CatBoost — сравнение честное, различается
# только финальное кодирование под требования конкретной библиотеки.
LGBM_MODEL_PATH = MODELS_DIR / "lightgbm_model.txt"
LGBM_ENCODERS_PATH = MODELS_DIR / "lgbm_encoders.pkl"

# Размер словаря TF-IDF по каждому текстовому полю. Подобраны скромно
# (сотни, не тысячи признаков) — это baseline для сравнения архитектур,
# а не попытка выжать максимум качества из текста средствами TF-IDF.
TFIDF_MAX_FEATURES = {
    "title": 300,
    "description": 500,
    "title_cat": 300,
}

# GPU-сборка LightGBM требует отдельной компиляции с OpenCL и здесь не
# предполагается по умолчанию — в отличие от CatBoost, где task_type='GPU'
# работает "из коробки" на стандартном pip-пакете.
LGBM_FIXED_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Значения по умолчанию для тюнингуемых параметров модели — используются,
# пока scripts/tune_lightgbm.py ни разу не запускался в этом окружении.
LGBM_TUNED_DEFAULTS = {
    "n_estimators": 1200,
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 63,
    "reg_lambda": 6.6,
}

LGBM_BEST_PARAMS_PATH = MODELS_DIR / "best_params_lgbm.json"


def get_lgbm_params() -> dict:
    """
    Аналог get_catboost_params(): LGBM_FIXED_PARAMS + (результат
    scripts/tune_lightgbm.py из best_params_lgbm.json, если файл
    существует, иначе LGBM_TUNED_DEFAULTS).
    """
    import json

    if LGBM_BEST_PARAMS_PATH.exists():
        with open(LGBM_BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
            tuned = json.load(f)["best_model_params"]
    else:
        tuned = LGBM_TUNED_DEFAULTS

    return {**LGBM_FIXED_PARAMS, **tuned}


def get_tfidf_max_features() -> dict:
    """
    Размер TF-IDF словаря по текстовым полям. Если scripts/tune_lightgbm.py
    уже подобрал свои значения (см. модуль docstring — TF-IDF считается
    частью пространства поиска, а не только гиперпараметров модели),
    берём их из best_params_lgbm.json; иначе — дефолты TFIDF_MAX_FEATURES.
    """
    import json

    if LGBM_BEST_PARAMS_PATH.exists():
        with open(LGBM_BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "best_tfidf_max_features" in data:
            return data["best_tfidf_max_features"]

    return TFIDF_MAX_FEATURES
