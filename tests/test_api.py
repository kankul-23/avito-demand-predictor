"""
Тесты для src/api/main.py (план проекта, шаг 8).

Модель (CatBoostClassifier) и FeatureEngineeringPipeline ЗДЕСЬ
МОКИРУЮТСЯ, а не грузятся с диска — по двум причинам:

  1. Юнит-тесты API должны проверять логику эндпоинта (валидация
     запроса, применение PREDICTION_THRESHOLD, обработка ошибок) —
     не то, предсказывает ли CatBoost хорошо. За качество модели
     отвечают ROC-AUC/PR-AUC на валидации (см. model_comparison.json),
     не эти тесты.
  2. Тесты не должны требовать наличия обученной модели на диске —
     иначе они бы падали в CI/у нового разработчика, у которого ещё
     не запущен train_catboost.py, и тесты стали бы медленными
     (реальный CatBoost.predict_proba на реальном Pool — не мгновенная
     операция).

Мокируем на уровне src.api.main._model / _feature_pipeline (module-level
переменные, заполняемые lifespan при обычном запуске сервиса) — тесты
подставляют туда фиктивные объекты напрямую, минуя lifespan целиком.

Запуск:
    pytest tests/test_api.py -v
"""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import config
from src.api import main as api_main


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def valid_payload() -> dict:
    """
    Минимальный валидный запрос — те же поля, что в
    ItemFeatures.model_config["json_schema_extra"] (src/api/schemas.py),
    сверенные построчно с config.CAT_FEATURES/NUM_FEATURES/TEXT_FEATURES.
    """
    return {
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


@pytest.fixture
def mock_feature_pipeline():
    """
    Мок FeatureEngineeringPipeline: transform() возвращает df с ровно
    теми колонками, которые ожидает config.FEATURE_COLS — этого
    достаточно, чтобы predict() (src/models/predict.py) мог построить
    Pool и не важно, что реальных агрегатов внутри нет: сама логика
    fit/transform уже не то, что тестирует этот файл (см. module docstring).
    """
    pipeline = MagicMock()

    def fake_transform(df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        out = pd.DataFrame(index=df.index)
        for col in config.CAT_FEATURES + config.TEXT_FEATURES:
            out[col] = "placeholder"
        for col in config.NUM_FEATURES:
            out[col] = 0.0
        return out

    pipeline.transform.side_effect = fake_transform
    return pipeline


@pytest.fixture
def mock_model_low_proba():
    """CatBoostClassifier-мок, всегда предсказывающий низкую вероятность (< PREDICTION_THRESHOLD)."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.98, 0.02]])
    return model


@pytest.fixture
def mock_model_high_proba():
    """CatBoostClassifier-мок, всегда предсказывающий высокую вероятность (>= PREDICTION_THRESHOLD)."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.35, 0.65]])
    return model


@pytest.fixture
def client_with_model(monkeypatch, mock_model_low_proba, mock_feature_pipeline):
    """
    TestClient с подставленными _model/_feature_pipeline — имитирует
    состояние сервиса ПОСЛЕ успешного lifespan, без реального
    обращения к диску.

    Важно: monkeypatch применяется ПОСЛЕ входа в `with TestClient(...)`,
    а не до — lifespan сам присваивает _model/_feature_pipeline при
    старте (см. main.py), и любой patch, сделанный раньше этого
    момента, будет им перезаписан. Внутри `with` lifespan уже
    отработал, дальше можно спокойно подменять модуль-level переменные.
    """
    with TestClient(api_main.app) as client:
        monkeypatch.setattr(api_main, "_model", mock_model_low_proba)
        monkeypatch.setattr(api_main, "_feature_pipeline", mock_feature_pipeline)
        yield client


@pytest.fixture
def client_without_model(monkeypatch):
    """
    TestClient БЕЗ модели — имитирует состояние до отработки lifespan
    (используется в тесте на 503, см. ниже).
    """
    monkeypatch.setattr(api_main, "_model", None)
    monkeypatch.setattr(api_main, "_feature_pipeline", None)
    return TestClient(api_main.app, raise_server_exceptions=False)


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------
def test_health_returns_200(client_with_model):
    response = client_with_model.get("/health")
    assert response.status_code == 200


def test_health_reports_model_loaded_true(client_with_model):
    response = client_with_model.get("/health")
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# --------------------------------------------------------------------------
# /predict — happy path
# --------------------------------------------------------------------------
def test_predict_returns_200_on_valid_payload(client_with_model, valid_payload):
    response = client_with_model.post("/predict", json=valid_payload)
    assert response.status_code == 200


def test_predict_response_has_expected_fields(client_with_model, valid_payload):
    response = client_with_model.post("/predict", json=valid_payload)
    body = response.json()
    assert set(body.keys()) == {
        "deal_probability", "recommend_promotion", "threshold_used",
    }


def test_predict_probability_in_valid_range(client_with_model, valid_payload):
    """
    deal_probability должна быть 0.0-1.0 независимо от того, что вернул
    мок — это защищает от будущей регрессии, если кто-то поменяет
    predict() и забудет взять [:, 1] или перепутает индекс класса.
    """
    response = client_with_model.post("/predict", json=valid_payload)
    proba = response.json()["deal_probability"]
    assert 0.0 <= proba <= 1.0


def test_predict_matches_mocked_model_output(client_with_model, valid_payload):
    """mock_model_low_proba отдаёт 0.02 — проверяем, что это значение доходит до ответа как есть."""
    response = client_with_model.post("/predict", json=valid_payload)
    assert response.json()["deal_probability"] == pytest.approx(0.02, abs=1e-6)


def test_predict_threshold_used_matches_config(client_with_model, valid_payload):
    response = client_with_model.post("/predict", json=valid_payload)
    assert response.json()["threshold_used"] == config.PREDICTION_THRESHOLD


# --------------------------------------------------------------------------
# /predict — применение PREDICTION_THRESHOLD (два сценария: ниже/выше порога)
# --------------------------------------------------------------------------
def test_predict_recommend_false_when_below_threshold(client_with_model, valid_payload):
    """mock_model_low_proba (0.02) < PREDICTION_THRESHOLD (0.30) -> recommend_promotion=False."""
    response = client_with_model.post("/predict", json=valid_payload)
    assert response.json()["recommend_promotion"] is False


def test_predict_recommend_true_when_above_threshold(
    monkeypatch, mock_model_high_proba, mock_feature_pipeline, valid_payload
):
    """mock_model_high_proba (0.65) >= PREDICTION_THRESHOLD (0.30) -> recommend_promotion=True."""
    with TestClient(api_main.app) as client:
        monkeypatch.setattr(api_main, "_model", mock_model_high_proba)
        monkeypatch.setattr(api_main, "_feature_pipeline", mock_feature_pipeline)
        response = client.post("/predict", json=valid_payload)

    assert response.json()["recommend_promotion"] is True
    assert response.json()["deal_probability"] == pytest.approx(0.65, abs=1e-6)


# --------------------------------------------------------------------------
# /predict — опциональные поля (тот самый кейс с image_top_1, который
# уже реально ловили руками в Swagger UI — см. историю правки схемы)
# --------------------------------------------------------------------------
def test_predict_accepts_missing_optional_fields(client_with_model, valid_payload):
    """
    param_2, param_3, description, image_top_1 опциональны в ItemFeatures —
    запрос без них должен быть валиден (422 быть не должно).
    """
    minimal_payload = {
        k: v for k, v in valid_payload.items()
        if k not in {"param_2", "param_3", "description", "image_top_1"}
    }
    response = client_with_model.post("/predict", json=minimal_payload)
    assert response.status_code == 200


# --------------------------------------------------------------------------
# /predict — валидация (422)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("missing_field", [
    "region", "city", "parent_category_name", "category_name",
    "user_type", "user_id", "title", "price", "item_seq_number",
])
def test_predict_422_when_required_field_missing(client_with_model, valid_payload, missing_field):
    payload = {k: v for k, v in valid_payload.items() if k != missing_field}
    response = client_with_model.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_422_when_price_is_negative(client_with_model, valid_payload):
    """price имеет ge=0 в схеме (ItemFeatures) — отрицательная цена должна отклоняться до модели."""
    payload = {**valid_payload, "price": -100.0}
    response = client_with_model.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_422_when_price_is_wrong_type(client_with_model, valid_payload):
    payload = {**valid_payload, "price": "не число"}
    response = client_with_model.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_422_when_item_seq_number_is_negative(client_with_model, valid_payload):
    payload = {**valid_payload, "item_seq_number": -1}
    response = client_with_model.post("/predict", json=payload)
    assert response.status_code == 422


# --------------------------------------------------------------------------
# /predict — модель ещё не загружена (503)
# --------------------------------------------------------------------------
def test_predict_503_when_model_not_loaded(client_without_model, valid_payload):
    """
    На практике недостижимо при обычном запуске (lifespan упал бы раньше,
    см. main.py) — но защитная проверка внутри predict_endpoint должна
    отработать корректно, если её всё же вызвать в обход lifespan.
    """
    response = client_without_model.post("/predict", json=valid_payload)
    assert response.status_code == 503


# --------------------------------------------------------------------------
# /predict — ошибка внутри predict() (500)
# --------------------------------------------------------------------------
def test_predict_500_when_feature_pipeline_raises(monkeypatch, mock_model_low_proba, valid_payload):
    """
    Имитирует реальную ошибку, которая уже случалась на практике
    (KeyError на отсутствующих в схеме полях типа parent_category_name/
    image_top_1, до того как схему поправили) — predict_endpoint должен
    вернуть 500 с текстом ошибки, а не уронить весь процесс.
    """
    broken_pipeline = MagicMock()
    broken_pipeline.transform.side_effect = KeyError(
        "['parent_category_name', 'image_top_1'] not in index"
    )
    with TestClient(api_main.app, raise_server_exceptions=False) as client:
        monkeypatch.setattr(api_main, "_model", mock_model_low_proba)
        monkeypatch.setattr(api_main, "_feature_pipeline", broken_pipeline)
        response = client.post("/predict", json=valid_payload)

    assert response.status_code == 500
    assert "Ошибка инференса" in response.json()["detail"]
