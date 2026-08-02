"""
Инференс на новых данных с помощью сохранённых модели и feature-пайплайна.
"""
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src import config
from src.features.build_features import FeatureEngineeringPipeline


def load_model(path=config.MODEL_PATH) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def predict(
    df: pd.DataFrame,
    model: CatBoostClassifier = None,
    feature_pipeline: FeatureEngineeringPipeline = None,
) -> pd.Series:
    """
    Прогоняет df через тот же feature-пайплайн, что и на обучении,
    и возвращает вероятность deal_probability >= TARGET_THRESHOLD.

    df должен содержать те же сырые поля, что train.csv/test.csv
    (region, city, category_name, price, title, description, ...).
    """
    if model is None:
        model = load_model()
    if feature_pipeline is None:
        feature_pipeline = FeatureEngineeringPipeline.load()

    df_features = feature_pipeline.transform(df)[config.FEATURE_COLS]

    pool = Pool(
        data=df_features,
        cat_features=config.CAT_FEATURES,
        text_features=config.TEXT_FEATURES,
    )

    preds = model.predict_proba(pool)[:, 1]
    return pd.Series(preds, index=df.index, name="deal_probability_pred")


def predict_label(
    df: pd.DataFrame,
    model: CatBoostClassifier = None,
    feature_pipeline: FeatureEngineeringPipeline = None,
    threshold: float = config.PREDICTION_THRESHOLD,
) -> pd.Series:
    ...
    proba = predict(df, model, feature_pipeline)
    return (proba >= threshold).astype("int8").rename("deal_probability_pred_label")