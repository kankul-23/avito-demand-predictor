# Avito Demand Predictor — образ для FastAPI-сервиса инференса (шаг 9 плана).
#
# Контейнеризируется ТОЛЬКО сервис предсказаний (src/api) — не обучение
# моделей. Обучение (python -m src.models.train_catboost) остаётся
# разовым процессом на локальной машине / в CI, не частью этого образа.
#
# Модель ЗАПЕКАЕТСЯ внутрь образа при сборке (COPY models/), а не
# подключается через volume при запуске — осознанное упрощение для
# пет-проекта: образ самодостаточен и сразу готов к работе после
# `docker run`, ценой того, что обновление модели требует пересборки
# образа, а не просто замены файла на хосте.

FROM python:3.10-slim

WORKDIR /app

# Системные зависимости для сборки колёс catboost/lightgbm/scipy —
# без них pip install может падать на компиляции C-расширений на
# slim-образе (в нём нет gcc по умолчанию).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt копируется и устанавливается ОТДЕЛЬНО от остального
# кода — Docker кэширует этот слой, и пока requirements.txt не меняется,
# pip install не перезапускается на каждой пересборке образа после
# правки кода. Экономит минуты на каждой итерации разработки.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код сервиса и то, что реально нужно API в рантайме — не весь репозиторий
# (ноутбуки, EDA, scripts/ для тюнинга и т.д. в образ не идут, см. .dockerignore).
COPY src/ src/

# Обученная модель и feature-пайплайн — то, что реально загружает
# lifespan в src/api/main.py (config.MODEL_PATH, config.FEATURE_PIPELINE_PATH).
# *.cbm/*.pkl обычно в .gitignore (см. models/*.cbm, models/*.pkl) —
# это НЕ мешает COPY внутри Docker: .dockerignore и .gitignore — разные
# механизмы, файлы на диске у вас есть после обучения, COPY их видит.
COPY models/catboost_model.cbm models/catboost_model.cbm
COPY models/feature_pipeline.pkl models/feature_pipeline.pkl

EXPOSE 8000

# --host 0.0.0.0 обязателен: 127.0.0.1 (дефолт uvicorn) слушает только
# порт ВНУТРИ контейнера, наружу через docker run -p ничего не пробросится.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
