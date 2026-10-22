"""评论文本的基础清洗与可分析性判断。

本模块不做翻译、总结、Aspect 识别或情绪判断。原始文本由
``clean_reviews`` 保留在独立列中。
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

import pandas as pd


_BLOCK_TAGS = {
    "article",
    "br",
    "div",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "li",
    "ol",
    "p",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
_ZERO_WIDTH_SPACES = re.compile(r"[\u200b\ufeff]+")
_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)
_SKIP_REASONS = ("missing_content", "empty_after_cleaning")


class _HTMLTextExtractor(HTMLParser):
    """提取可见文本，并在块级 HTML 元素之间保留词语边界。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in _BLOCK_TAGS:
            self._parts.append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in _BLOCK_TAGS:
            self._parts.append(" ")

    def text(self) -> str:
        return "".join(self._parts)


def _is_missing_scalar(value: object) -> bool:
    """安全判断 None、pd.NA、NaN 和 NaT 等标量缺失值。"""

    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _is_missing_content(value: object) -> bool:
    """区分真正缺失/空字符串与清洗后才变空的内容。"""

    return _is_missing_scalar(value) or (isinstance(value, str) and value == "")


def clean_text(text: object) -> str:
    """执行最小必要清洗，同时保留评论语义与表达方式。

    处理 HTML 标签和实体、Unicode 空格、tab、换行及连续空白；保留
    大小写、标点、数字、否定词和产品型号。
    """

    if _is_missing_scalar(text):
        return ""

    raw_text = text if isinstance(text, str) else str(text)
    parser = _HTMLTextExtractor()
    parser.feed(raw_text)
    parser.close()

    without_html = parser.text()
    without_zero_width_spaces = _ZERO_WIDTH_SPACES.sub(" ", without_html)
    return _WHITESPACE.sub(" ", without_zero_width_spaces).strip()


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """返回带有原文、清洗文本和可分析标记的新 DataFrame。"""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if "content" not in df.columns:
        raise ValueError("Cannot clean reviews without a content column.")

    cleaned_reviews = df.copy(deep=True)
    original_content = df["content"].copy(deep=True)
    cleaned_content = original_content.map(clean_text)
    missing_content = original_content.map(_is_missing_content)
    is_analyzable = cleaned_content.ne("")

    skip_reason = pd.Series("", index=df.index, dtype="string")
    skip_reason.loc[missing_content] = "missing_content"
    skip_reason.loc[~missing_content & ~is_analyzable] = "empty_after_cleaning"

    cleaned_reviews["original_content"] = original_content
    cleaned_reviews["cleaned_content"] = cleaned_content
    cleaned_reviews["is_analyzable"] = is_analyzable.astype(bool)
    cleaned_reviews["skip_reason"] = skip_reason

    if "title" in df.columns:
        original_title = df["title"].copy(deep=True)
        cleaned_reviews["original_title"] = original_title
        cleaned_reviews["cleaned_title"] = original_title.map(clean_text)
    else:
        cleaned_reviews["original_title"] = pd.Series(pd.NA, index=df.index, dtype="object")
        cleaned_reviews["cleaned_title"] = pd.Series("", index=df.index, dtype="string")

    return cleaned_reviews


def summarize_cleaning(df: pd.DataFrame) -> dict[str, Any]:
    """汇总可分析评论数量和简单的跳过原因。"""

    required_columns = {"is_analyzable", "skip_reason"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing cleaning result columns: {', '.join(missing_columns)}")

    analyzable_mask = df["is_analyzable"].eq(True)
    skip_reasons = {
        reason: int(df.loc[~analyzable_mask, "skip_reason"].eq(reason).sum())
        for reason in _SKIP_REASONS
    }

    total_reviews = len(df)
    analyzable_reviews = int(analyzable_mask.sum())
    return {
        "total_reviews": total_reviews,
        "analyzable_reviews": analyzable_reviews,
        "skipped_reviews": total_reviews - analyzable_reviews,
        "skip_reasons": skip_reasons,
    }
