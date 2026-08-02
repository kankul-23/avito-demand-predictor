"""
Подбор порога бинаризации предсказанных вероятностей (план проекта,
пункт "Подбор порога бинаризации", отчёт раздел 5 — отмечен как
приоритетный открытый пункт).

Не заменяет compute_metrics() (src/utils/metrics.py) — тот считает
ROC-AUC/PR-AUC, которые от порога не зависят и остаются основными
метриками сравнения моделей. Этот модуль решает другую задачу:
дана уже выбранная модель (CatBoost) — на каком пороге переводить
predict_proba в бинарное решение "рекомендовать продвижение / нет".

Используется на валидационном сплите (df_val_split, y_val), который
уже готовит prepare_feature_splits() — никакого нового сплита данных
здесь не создаётся, чтобы не рисковать утечкой на train.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def threshold_metrics_table(
    y_true,
    y_pred_proba,
    thresholds=None,
) -> pd.DataFrame:
    """
    Считает precision/recall/F1/F0.5/F2 на сетке порогов.

    thresholds по умолчанию — np.arange(0.05, 0.96, 0.05). Диапазон
    начинается не с 0 и не доходит до 1, чтобы не считать вырожденные
    случаи "все объекты положительные"/"все отрицательные", которые
    только засоряют таблицу и никогда не являются кандидатом.

    F0.5 (приоритет precision) и F2 (приоритет recall) добавлены
    сразу, а не только F1 — чтобы не перезапускать расчёт, если
    после просмотра таблицы бизнес-приоритет качнётся в одну из
    сторон (см. обсуждение: продвижение "холодным" объявлениям vs
    охват).

    Возвращает DataFrame с колонками:
    threshold, precision, recall, f1, f0.5, f2, predicted_positive_rate.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    rows = []
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype("int8")

        # zero_division=0: на крайних порогах предсказанных позитивов
        # может не быть вообще — считаем это precision=0, а не падаем.
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = fbeta_score(y_true, y_pred, beta=1.0, zero_division=0)
        f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)

        rows.append({
            "threshold": round(float(t), 2),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f0.5": f05,
            "f2": f2,
            "predicted_positive_rate": y_pred.mean(),
        })

    return pd.DataFrame(rows)


def best_threshold(table: pd.DataFrame, metric: str = "f1") -> dict:
    """
    Возвращает строку таблицы (как dict) с максимальным значением
    указанной метрики ("f1", "f0.5", "f2", "precision", "recall").

    Чистая функция над уже посчитанной threshold_metrics_table() —
    её удобно вызывать повторно для разных metric без пересчёта
    predict_proba/предсказаний на каждом пороге.
    """
    if metric not in table.columns:
        raise ValueError(
            f"Неизвестная метрика '{metric}'. Доступные: "
            f"{[c for c in table.columns if c not in ('threshold', 'predicted_positive_rate')]}"
        )
    row = table.loc[table[metric].idxmax()]
    return row.to_dict()


def log_threshold_table(table: pd.DataFrame) -> None:
    """Печатает таблицу порогов в лог в читаемом виде (для CLI-запуска)."""
    header = f"{'thr':>5} {'precision':>10} {'recall':>10} {'f1':>8} {'f0.5':>8} {'f2':>8} {'pos_rate':>9}"
    logger.info(header)
    for _, r in table.iterrows():
        logger.info(
            "%5.2f %10.4f %10.4f %8.4f %8.4f %8.4f %9.4f",
            r["threshold"], r["precision"], r["recall"],
            r["f1"], r["f0.5"], r["f2"], r["predicted_positive_rate"],
        )


if __name__ == "__main__":
    import logging as _logging

    from src.models.train_catboost import prepare_train_val_pools
    from catboost import CatBoostClassifier

    from src import config

    _logging.basicConfig(level=_logging.INFO)

    # Переиспользуем сохранённую модель, если она уже есть — иначе
    # переобучаем (train_model) и берём val-предсказания напрямую.
    if config.MODEL_PATH.exists():
        train_pool, val_pool, _ = prepare_train_val_pools()
        model = CatBoostClassifier()
        model.load_model(str(config.MODEL_PATH))
    else:
        from src.models.train_catboost import train_model
        model, _, _, _ = train_model(save=False)
        _, val_pool, _ = prepare_train_val_pools()

    val_preds = model.predict_proba(val_pool)[:, 1]
    y_val = val_pool.get_label()

    table = threshold_metrics_table(y_val, val_preds)
    log_threshold_table(table)

    for metric in ("f1", "f0.5", "f2"):
        best = best_threshold(table, metric)
        logger.info(
            "Лучший порог по %s: %.2f (precision=%.4f, recall=%.4f)",
            metric, best["threshold"], best["precision"], best["recall"],
        )
