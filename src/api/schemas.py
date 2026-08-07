"""
Pydantic-схемы запроса/ответа для /predict.

ItemFeatures содержит РОВНО те сырые поля, которые реально читает
FeatureEngineeringPipeline.transform() (src/features/build_features.py)
и его подфункции — не всё, что есть в train.csv/test.csv, а только то,
без чего пайплайн не построит 25 финальных признаков:

  - price, item_seq_number   -> add_baseline_features, price_features,
                                 seller_activity_level
  - image_top_1               -> config.NUM_FEATURES напрямую (не
                                 пересчитывается пайплайном — модель
                                 классификации изображений вне скоупа
                                 этого API, поле опционально)
  - title, description       -> text_features, title_cat, TEXT_FEATURES
  - region, city              -> cat_region, price_to_reg_median, city_freq
  - category_name             -> cat_region, price_to_cat_median, title_cat
  - parent_category_name      -> отдельное поле CAT_FEATURES, НЕ совпадает
                                 с category_name (см. config.CAT_FEATURES:
                                 оба присутствуют одновременно — например
                                 parent="Личные вещи",
                                 category="Одежда, обувь, аксессуары")
  - param_1, param_2, param_3 -> CAT_FEATURES напрямую
  - user_type                 -> CAT_FEATURES напрямую
  - user_id                   -> aggregation_features (user_ads_count);
                                 НЕ входит в FEATURE_COLS сама по себе —
                                 нужна только чтобы посчитать агрегат.

activation_date из train.csv НЕ включён: используется в обучении только
для time-based split (prepare_feature_splits), а не внутри
FeatureEngineeringPipeline.transform() — для инференса одного объекта
не нужен.

Пропуски (param_2/param_3 у категорий, где их не бывает, image_top_1 у
объявлений без фото) — Optional, т.к. вся дальнейшая цепочка
(clean_and_prepare_types) уже сама приводит NaN к "missing"/-1 —
дополнительная валидация здесь не нужна: pipeline и так устойчив к
unseen/пустым значениям (см. aggregation_features .fillna(0)/.fillna(1)
и clean_and_prepare_types).
"""
from pydantic import BaseModel, Field


class ItemFeatures(BaseModel):
    """Один объект объявления — вход для /predict."""

    region: str = Field(..., examples=["Свердловская область"])
    city: str = Field(..., examples=["Екатеринбург"])
    parent_category_name: str = Field(..., examples=["Личные вещи"])
    category_name: str = Field(..., examples=["Одежда, обувь, аксессуары"])
    param_1: str | None = Field(default=None, examples=["Женская одежда"])
    param_2: str | None = Field(default=None, examples=["Платья"])
    param_3: str | None = Field(default=None, examples=["Повседневные"])
    user_type: str = Field(..., examples=["Private"])
    user_id: str = Field(..., examples=["a1b2c3d4e5"])

    title: str = Field(..., examples=["Платье летнее новое"])
    description: str | None = Field(
        default=None, examples=["Продаю новое платье, размер 42-44"]
    )

    price: float = Field(..., ge=0, examples=[1500.0])
    item_seq_number: int = Field(..., ge=0, examples=[3])
    image_top_1: float | None = Field(
        default=None,
        examples=[None],
        description=(
            "ID категории изображения (из компьютерного зрения Avito). "
            "Опционально: объявление без фото — нормальная ситуация, "
            "clean_and_prepare_types сам приведёт отсутствующее значение к -1."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "region": "Свердловская область",
                    "city": "Екатеринбург",
                    "parent_category_name": "Личные вещи",
                    "category_name": "Одежда, обувь, аксессуары",
                    "param_1": "Женская одежда",
                    "param_2": "Платья",
                    "param_3": "Повседневные",
                    "user_type": "Private",
                    "user_id": "a1b2c3d4e5",
                    "title": "Платье летнее новое",
                    "description": "Продаю новое платье, размер 42-44",
                    "price": 1500.0,
                    "item_seq_number": 3,
                    "image_top_1": None,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Ответ /predict."""

    deal_probability: float = Field(
        ..., description="Предсказанная вероятность продажи (0.0-1.0)"
    )
    recommend_promotion: bool = Field(
        ...,
        description=(
            "Рекомендация продвигать объявление — "
            "deal_probability >= config.PREDICTION_THRESHOLD"
        ),
    )
    threshold_used: float = Field(
        ..., description="Порог бинаризации, применённый для recommend_promotion"
    )


class HealthResponse(BaseModel):
    """Ответ /health."""

    status: str = Field(..., examples=["ok"])
    model_loaded: bool
