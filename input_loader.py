"""读取并验证评论 CSV，不修改评论文本。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ("review_id", "rating", "content")
STANDARD_COLUMNS = (
    "review_id",
    "rating",
    "title",
    "content",
    "date",
    "verified_purchase",
)
OPTIONAL_COLUMNS = tuple(column for column in STANDARD_COLUMNS if column not in REQUIRED_COLUMNS)


def _blank_mask(series: pd.Series) -> pd.Series:
    """返回同时覆盖缺失值、空字符串和纯空白文本的布尔掩码。"""

    text_values = series.astype("string")
    return series.isna() | text_values.str.strip().eq("").fillna(True)


def _convert_ratings(series: pd.Series) -> pd.Series:
    """将可转换的评分变为数字，同时保留无法转换的原始值。"""

    numeric_ratings = pd.to_numeric(series, errors="coerce")
    blank_ratings = _blank_mask(series)
    conversion_failed = ~blank_ratings & numeric_ratings.isna()

    if not conversion_failed.any():
        return numeric_ratings

    # 混合列仅在存在非法文本时使用 object dtype，避免把原始非法值静默丢失。
    converted = numeric_ratings.astype("object")
    converted.loc[conversion_failed] = series.loc[conversion_failed]
    converted.loc[blank_ratings] = pd.NA
    return converted


def load_reviews(file_path: str | Path) -> pd.DataFrame:
    """从 CSV 读取评论并返回具有统一标准列的 DataFrame。

    文本字段保持原样。缺少的可选列会以空值补充；缺少的必需列不会
    自动伪造，而是交给 :func:`validate_reviews` 报告为严重错误。
    """

    csv_path = Path(file_path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Review CSV not found: {csv_path}")

    try:
        # 先以字符串读取可保留 review_id 前导零和原始评论文本。
        reviews = pd.read_csv(
            csv_path,
            dtype="string",
            encoding="utf-8-sig",
            keep_default_na=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not parse review CSV: {csv_path}") from exc

    if "review_id" in reviews.columns:
        reviews["review_id"] = reviews["review_id"].astype("string")

    if "rating" in reviews.columns:
        reviews["rating"] = _convert_ratings(reviews["rating"])

    for column in OPTIONAL_COLUMNS:
        if column not in reviews.columns:
            reviews[column] = pd.NA

    standard_columns = [column for column in STANDARD_COLUMNS if column in reviews.columns]
    extra_columns = [column for column in reviews.columns if column not in STANDARD_COLUMNS]
    return reviews.loc[:, standard_columns + extra_columns]


def validate_reviews(df: pd.DataFrame) -> dict[str, Any]:
    """检查 CSV 结构和行级数据质量，不修改传入的 DataFrame。

    ``is_valid`` 只表示数据结构允许程序继续。空数据或缺少必需列属于
    严重错误；单行空值、重复 ID 和非法评分只会生成 warning。
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    total_reviews = len(df)
    missing_required_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    errors: list[str] = []

    if df.empty:
        errors.append("CSV contains no review rows.")
    if missing_required_columns:
        errors.append(f"Missing required columns: {', '.join(missing_required_columns)}.")

    if "content" in df.columns:
        missing_content = int(_blank_mask(df["content"]).sum())
    else:
        missing_content = total_reviews

    if "title" in df.columns:
        missing_titles = int(_blank_mask(df["title"]).sum())
    else:
        missing_titles = total_reviews

    if "review_id" in df.columns:
        review_ids = df["review_id"].astype("string")
        missing_id_mask = _blank_mask(review_ids)
        normalized_ids = review_ids.str.strip()
        duplicate_review_ids = int(normalized_ids.loc[~missing_id_mask].duplicated(keep="first").sum())
        missing_review_ids = int(missing_id_mask.sum())
    else:
        duplicate_review_ids = 0
        missing_review_ids = total_reviews

    missing_ratings = 0
    non_numeric_ratings = 0
    ratings_below_range = 0
    ratings_above_range = 0
    invalid_ratings = 0

    if "rating" in df.columns:
        rating_values = df["rating"]
        missing_rating_mask = _blank_mask(rating_values)
        numeric_ratings = pd.to_numeric(rating_values, errors="coerce")
        non_numeric_mask = ~missing_rating_mask & numeric_ratings.isna()
        below_range_mask = numeric_ratings.notna() & numeric_ratings.lt(1)
        above_range_mask = numeric_ratings.notna() & numeric_ratings.gt(5)
        invalid_rating_mask = (
            missing_rating_mask | non_numeric_mask | below_range_mask | above_range_mask
        )

        missing_ratings = int(missing_rating_mask.sum())
        non_numeric_ratings = int(non_numeric_mask.sum())
        ratings_below_range = int(below_range_mask.sum())
        ratings_above_range = int(above_range_mask.sum())
        invalid_ratings = int(invalid_rating_mask.sum())

    warnings: list[str] = []
    warning_counts = (
        (missing_content, "reviews have missing content"),
        (missing_titles, "reviews have missing titles"),
        (missing_review_ids, "reviews have missing review IDs"),
        (duplicate_review_ids, "duplicate review IDs were found"),
        (invalid_ratings, "reviews have invalid ratings"),
    )
    for count, message in warning_counts:
        if count:
            warnings.append(f"{count} {message}.")

    return {
        "total_reviews": total_reviews,
        "missing_content": missing_content,
        "missing_titles": missing_titles,
        "missing_review_ids": missing_review_ids,
        "duplicate_review_ids": duplicate_review_ids,
        "missing_ratings": missing_ratings,
        "invalid_ratings": invalid_ratings,
        "non_numeric_ratings": non_numeric_ratings,
        "ratings_below_range": ratings_below_range,
        "ratings_above_range": ratings_above_range,
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }
