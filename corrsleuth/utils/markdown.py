"""Small deterministic Markdown formatting helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def format_markdown_value(value: float | None) -> str:
    return f"{value:.3f}" if value is not None and pd.notna(value) else "NA"


def escape_markdown_cell(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        is_missing = False
    if isinstance(is_missing, bool) and is_missing:
        return "NA"
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\n", " ")
        .replace("|", "\\|")
        .replace("_", "\\_")
        .replace("*", "\\*")
        .replace("`", "\\`")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    # Escape header cells too: today all call sites pass hardcoded literals, but
    # escaping keeps the helper safe if a header is ever derived from data.
    lines = [
        "| " + " | ".join(escape_markdown_cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(escape_markdown_cell(value) for value in row) + " |"
        )
    return "\n".join(lines)
