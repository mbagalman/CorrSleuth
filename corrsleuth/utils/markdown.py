"""Small deterministic Markdown formatting helpers."""
from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd


def format_markdown_value(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None and pd.notna(value) else "NA"


def escape_markdown_cell(value: Any) -> str:
    if value is None:
        return "NA"
    if not isinstance(value, (dict, list, tuple, set)) and pd.isna(value):
        return "NA"
    return str(value).replace("\n", " ").replace("|", "\\|")


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(escape_markdown_cell(value) for value in row) + " |"
        )
    return "\n".join(lines)
