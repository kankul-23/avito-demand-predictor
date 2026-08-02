"""
Метрики качества модели.

ROC-AUC и PR-AUC выбраны в 01_eda.ipynb (раздел 1) как основные —
не требуют выбора порога и устойчивы к дисбалансу классов
(64.83% объектов имеют deal_probability == 0, см. EDA_отчет.md, раздел 4).
"""
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(y_true, y_pred_proba) -> dict:
    """
    y_true         — бинарные метки (0/1).
    y_pred_proba    — предсказанные вероятности положительного класса.
    """
    return {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
    }
