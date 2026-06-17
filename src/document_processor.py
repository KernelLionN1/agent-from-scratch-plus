"""
Day7: 文档处理器 — PDF/Word → 纯文本

RAG 管线的第一步：把各种格式的文档转成可检索的文本。

技术选型：
  pymupdf (fitz) — PDF 解析（比 PyPDF2 快 10x，支持表格）
  python-docx    — Word .docx 解析

Java 类比：Apache Tika / POI 做文档解析
"""

import os
from typing import List


def extract_text(filepath: str) -> str:
    """
    根据文件扩展名自动选择解析器

    Args:
        filepath: 文件路径（支持 .pdf / .docx / .txt / .md）

    Returns:
        str: 提取的纯文本
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return _extract_pdf(filepath)
    elif ext in (".docx", ".doc"):
        return _extract_docx(filepath)
    elif ext in (".txt", ".md"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def _extract_pdf(filepath: str) -> str:
    """PDF → 纯文本，逐页提取并合并"""
    import fitz  # pymupdf

    doc = fitz.open(filepath)
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def _extract_docx(filepath: str) -> str:
    """Word .docx → 纯文本，逐段落提取"""
    from docx import Document

    doc = Document(filepath)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def split_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    文本分块：按句号切分，保证语义完整

    分块策略：
    1. 按句号/换行分割句子
    2. 累加句子直到超过 chunk_size
    3. 每个块与前后有 overlap 字符重叠

    Java 类比：Lucene 的分词 + 段落切分

    Args:
        text: 原始文本
        chunk_size: 每块最大字符数（默认 500 ≈ 250-350 tokens）
        overlap: 相邻块重叠字符数

    Returns:
        List[str]: 文本块列表
    """
    # 按句子分割（中英文句号都认）
    import re
    sentences = re.split(r"[。！？\n.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > chunk_size and current:
            chunks.append(current.strip())
            # 重叠：保留最后 overlap 个字符
            current = current[-overlap:] if len(current) > overlap else ""
        current += sentence + "。"
    if current.strip():
        chunks.append(current.strip())

    return chunks


# ── 模块自测 ─────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    # 用 py 文件模拟文本测试（项目里没有真实 PDF）
    print("=" * 50)
    print("DocumentProcessor 自测")
    print("=" * 50)

    # txt 测试
    test_file = os.path.join(os.path.dirname(__file__), "..", "AGENTS.md")
    if os.path.exists(test_file):
        text = extract_text(test_file)
        print(f"\nAGENTS.md: {len(text)} chars")

        chunks = split_chunks(text, chunk_size=200, overlap=30)
        print(f"分块: {len(chunks)} 块")
        for i, c in enumerate(chunks[:3]):
            print(f"  [{i}] {c[:80]}...")

    print("\n✅ 自测完成")
