"""
文档处理器 — 多格式文档 → 统一 Markdown

使用 MinerU 作为主引擎，fallback 到 pymupdf/python-docx。
"""

from src.document_processor.parser import (
    MinerUParser,
    FallbackParser,
    convert_to_markdown,
    extract_text,
    split_chunks,
)

__all__ = [
    "MinerUParser",
    "FallbackParser",
    "convert_to_markdown",
    "extract_text",
    "split_chunks",
]
