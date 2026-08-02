"""
SHAP-анализ для проверки гипотез из Error Analysis (error_analysis.py).

Не заменяет error_analysis.py — берёт его уже размеченный error_df
(с колонкой error_type: TP/TN/FP/FN) и отвечает на конкретный вопрос:
на какие ПРИЗНАКИ модель опирается сильнее всего внутри проблемной
группы (например, FP в категории "Автомобили"), в сравнении с TN
той же категории — то есть чем отличается решение модели в ошибочных
случаях от правильных внутри одного и того же сегмента.

Зачем сравнивать с TN внутри категории, а не смотреть SHAP FP-группы
саму по себе: средние |SHAP| по одной только FP-группе показывают,
что вообще влияет на предсказание в этой категории — это может
быть просто "цена важна для всех объявлений этой категории",
а не специфика именно ошибок. Сравнение FP vs TN показывает разницу
в паттерне использования признаков между "модель ошиблась" и
"модель права" — это ближе к настоящей причине систематического
смещения.

Два способа сравнения, отвечающие на РАЗНЫЕ вопросы:

  compare_shap_fp_vs_tn()        — модуль |SHAP|: насколько сильно
                                    признак вообще влияет на решение.
  compare_shap_fp_vs_tn_signed() — знак SHAP: тянет ли признак
                                    предсказание вверх или вниз.

Большой |SHAP| ещё не значит "тянет вверх" — признак может сильно
влиять в обе стороны в разных объектах (среднее ~0), и тогда модульная
версия покажет его как важный, а знаковая — как ненаправленный. Для
утверждений вида "модель верит тексту и завышает скор из-за title"
нужна именно знаковая версия; для утверждений вида "текст вообще
заметнее в FP, чем в TN, вне зависимости от направления" — модульная.

Требует только CatBoost (используется его встроенный get_feature_importance
type="ShapValues") — отдельный пакет shap не нужен.
"""
import logging

import pandas as pd
from catboost import CatBoostClassifier

from src import config
from src.data.loader import load_processed_data
from src.models.error_analysis import build_error_frame
from src.models.train_catboost import prepare_train_val_pools

logger = logging.getLogger(__name__)


def compute_shap_values(model, pool) -> pd.DataFrame:
    """
    Возвращает SHAP-значения CatBoost через встроенный get_feature_importance
    (type='ShapValues') — не требует отдельного shap-пакета, использует
    родной механизм CatBoost, который для text_features агрегирует вклад
    всего текстового признака в одно число (title/description/title_cat
    видны как один столбец каждый, не по словам/токенам).

    Возвращает DataFrame (n_rows x n_features), без служебной последней
    колонки expected_value, которую CatBoost добавляет по умолчанию.
    """
    raw = model.get_feature_importance(pool, type="ShapValues")
    shap_matrix = raw[:, :-1]  # последняя колонка — базовое значение, не признак
    feature_names = pool.get_feature_names()
    return pd.DataFrame(shap_matrix, columns=feature_names)


def _fp_tn_masks(
    error_df: pd.DataFrame,
    category_col: str,
    category_value,
):
    """Общая логика фильтрации по категории + FP/TN для обеих функций сравнения ниже."""
    if category_value is not None:
        mask_category = error_df[category_col] == category_value
    else:
        mask_category = pd.Series(True, index=error_df.index)

    fp_mask = mask_category & (error_df["error_type"] == "FP")
    tn_mask = mask_category & (error_df["error_type"] == "TN")

    if fp_mask.sum() == 0 or tn_mask.sum() == 0:
        raise ValueError(
            f"Недостаточно строк для сравнения (FP={fp_mask.sum()}, "
            f"TN={tn_mask.sum()}) в category_value={category_value!r}. "
            "Проверьте название категории или увеличьте выборку."
        )

    return fp_mask, tn_mask


def _validate_aligned(shap_df: pd.DataFrame, error_df: pd.DataFrame) -> None:
    if len(shap_df) != len(error_df):
        raise ValueError(
            f"shap_df ({len(shap_df)} строк) и error_df ({len(error_df)} строк) "
            "не совпадают по длине — похоже, они посчитаны на разных данных."
        )


def compare_shap_fp_vs_tn(
    shap_df: pd.DataFrame,
    error_df: pd.DataFrame,
    category_col: str = "category_name",
    category_value=None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Сравнивает средний |SHAP| по каждому признаку между FP и TN внутри
    одной категории (category_value). Если category_value=None — сравнение
    делается на всей валидации без фильтра по категории.

    Отвечает на вопрос "насколько сильно признак вообще влияет на решение
    в ошибочных случаях по сравнению с верными" — БЕЗ учёта направления
    влияния. Для направления используйте compare_shap_fp_vs_tn_signed().

    shap_df и error_df должны иметь одинаковый порядок строк (тот же
    Pool/df_val, из которого оба посчитаны) — это ответственность
    вызывающего кода, здесь проверяется только совпадение длины.

    Возвращает DataFrame: feature, mean_abs_shap_fp, mean_abs_shap_tn,
    diff_fp_minus_tn, отсортированный по |diff| — сверху те признаки,
    вклад которых сильнее всего различается между ошибочными и верными
    предсказаниями.
    """
    _validate_aligned(shap_df, error_df)
    error_df = error_df.reset_index(drop=True)
    shap_df = shap_df.reset_index(drop=True)

    fp_mask, tn_mask = _fp_tn_masks(error_df, category_col, category_value)

    mean_abs_fp = shap_df[fp_mask].abs().mean()
    mean_abs_tn = shap_df[tn_mask].abs().mean()

    result = pd.DataFrame({
        "feature": mean_abs_fp.index,
        "mean_abs_shap_fp": mean_abs_fp.values,
        "mean_abs_shap_tn": mean_abs_tn.values,
    })
    result["diff_fp_minus_tn"] = result["mean_abs_shap_fp"] - result["mean_abs_shap_tn"]
    result["abs_diff"] = result["diff_fp_minus_tn"].abs()

    result = result.sort_values("abs_diff", ascending=False).head(top_n)
    return result.drop(columns="abs_diff").reset_index(drop=True)


def compare_shap_fp_vs_tn_signed(
    shap_df: pd.DataFrame,
    error_df: pd.DataFrame,
    category_col: str = "category_name",
    category_value=None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Как compare_shap_fp_vs_tn(), но со ЗНАКОМ SHAP, а не модулем — отвечает
    на другой вопрос: признак тянет предсказание ВВЕРХ или ВНИЗ, а не просто
    "насколько сильно влияет".

    Положительный mean_shap_fp означает, что признак в среднем СПОСОБСТВУЕТ
    ложному срабатыванию (тянет предсказание к 1), а не просто заметен.
    Признак с большим |SHAP| в compare_shap_fp_vs_tn(), но mean_shap близким
    к нулю здесь — влияет сильно, но разнонаправленно (то вверх, то вниз
    на разных объектах), и утверждение вида "модель завышает скор из-за
    этого признака" для него будет НЕ подтверждено данными.

    Ранжирует по |diff_signed| — сверху те признаки, где разница
    направления/силы влияния между FP и TN максимальна.
    """
    _validate_aligned(shap_df, error_df)
    error_df = error_df.reset_index(drop=True)
    shap_df = shap_df.reset_index(drop=True)

    fp_mask, tn_mask = _fp_tn_masks(error_df, category_col, category_value)

    mean_fp = shap_df[fp_mask].mean()
    mean_tn = shap_df[tn_mask].mean()

    result = pd.DataFrame({
        "feature": mean_fp.index,
        "mean_shap_fp": mean_fp.values,
        "mean_shap_tn": mean_tn.values,
    })
    result["diff_signed"] = result["mean_shap_fp"] - result["mean_shap_tn"]
    result["abs_diff"] = result["diff_signed"].abs()

    result = result.sort_values("abs_diff", ascending=False).head(top_n)
    return result.drop(columns="abs_diff").reset_index(drop=True)


def log_comparison(table: pd.DataFrame, label: str) -> None:
    logger.info("\n=== %s ===", label)
    logger.info("\n%s", table.to_string(index=False))


def run_shap_error_analysis(categories=("Предложение услуг", "Автомобили")) -> None:
    """
    Полный прогон: загружает модель, считает SHAP на валидации, печатает
    модульное и знаковое сравнение FP vs TN для каждой категории из
    categories.

    Вынесено в отдельную функцию (а не оставлено внутри if __name__), чтобы
    при желании можно было вызвать этот же прогон из ноутбука или другого
    скрипта, а не только из командной строки.
    """
    train_pool, val_pool, _ = prepare_train_val_pools()
    model = CatBoostClassifier()
    model.load_model(str(config.MODEL_PATH))

    y_pred_proba = model.predict_proba(val_pool)[:, 1]
    y_val = val_pool.get_label()

    train_df, test_df = load_processed_data()
    split_date = train_df[config.DATE_COL].quantile(config.VALIDATION_SPLIT_QUANTILE)
    df_val_raw = train_df[train_df[config.DATE_COL] >= split_date].reset_index(drop=True)

    error_df = build_error_frame(df_val_raw, y_val, y_pred_proba)

    logger.info("Считаю SHAP-значения на валидационном Pool (может занять время)...")
    shap_df = compute_shap_values(model, val_pool)

    for category in categories:
        table_abs = compare_shap_fp_vs_tn(shap_df, error_df, category_value=category)
        log_comparison(table_abs, f"FP vs TN (модуль |SHAP|) — {category}")

        table_signed = compare_shap_fp_vs_tn_signed(shap_df, error_df, category_value=category)
        log_comparison(table_signed, f"FP vs TN (знак SHAP) — {category}")

    logger.info(
        "\nПодсказка по интерпретации:\n"
        "  diff_fp_minus_tn (модуль) > 0  — признак заметнее в FP, чем в TN, "
        "направление не определено.\n"
        "  mean_shap_fp (знак) > 0        — признак в среднем ТЯНЕТ предсказание "
        "к 1 именно в FP-случаях (подтверждает 'модель верит признаку').\n"
        "  mean_shap_fp (знак) ≈ 0 при большом |SHAP| в модульной таблице — "
        "признак влияет сильно, но разнонаправленно; вывод о систематическом "
        "завышении для него данными НЕ подтверждён."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_shap_error_analysis()
