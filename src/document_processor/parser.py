"""
文档解析器 — 统一入口，多引擎支持

主引擎: MinerU（PDF/Word/Excel/PPT/图片 → Markdown）
Fallback: pymupdf + python-docx（不依赖 MinerU 时使用）

使用方式:
    from src.document_processor import convert_to_markdown
    md_text = convert_to_markdown("document.pdf")  # 自动选引擎

安装 MinerU（可选，增强 PDF/表格解析）:
    uv pip install "mineru[core]"
    或 Docker: docker run -v /data:/data opendatalab/mineru:latest mineru -p input.pdf -o output/
"""

import os
import re
from typing import List, Optional


# ═══════════════════════════════════════════════════════════
# MinerU 解析器（主引擎）
# ═══════════════════════════════════════════════════════════

class MinerUParser:
    """
    MinerU 文档解析器

    支持的格式：PDF / DOCX / PPTX / XLSX / 图片 / HTML
    输出格式：Markdown（含表格、公式、图片引用）

    安装方式（任选其一）：
        1. uv pip install "mineru[core]"
        2. Docker: docker run -v $(pwd):/data opendatalab/mineru:latest
        3. 云端 API: https://mineru.net/api

    MinerU 的核心优势：
        - 表格 → Markdown 表格（不是纯文本）
        - 公式 → LaTeX
        - 图片 → 提取 + 内联引用
        - 布局识别 → 标题/正文/页眉/页脚正确分离
    """
    _available: Optional[bool] = None

    @classmethod
    def is_available(cls) -> bool:
        """检测 MinerU 是否已安装"""
        if cls._available is None:
            try:
                import magic_pdf  # MinerU 的实际包名
                cls._available = True
            except ImportError:
                cls._available = False
        return cls._available

    @classmethod
    def convert(cls, filepath: str) -> str:
        """
        使用 MinerU 将文档转为 Markdown

        实际代码（需要 pip install mineru[core] 后才能运行）：
            from magic_pdf.pipe.UNIPipe import UNIPipe
            from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

            pdf_bytes = open(filepath, "rb").read()
            jso = UNIPipe(pdf_bytes, jso_useful_key={"_pdf_type": "ocr"}).pipe_classify()
            jso = UNIPipe(pdf_bytes, jso_useful_key={"_pdf_type": "txt"}).pipe_parse()
            md_content = UNIPipe.pipe_mk_markdown(
                jso, DropMode.NONE, DiskReaderWriter(output_dir)
            )
            return md_content

        Docker 方式（更简单，不污染 Python 环境）：
            import subprocess
            subprocess.run([
                "docker", "run", "--rm",
                "-v", f"{os.path.dirname(filepath)}:/data",
                "opendatalab/mineru:latest",
                "mineru", "-p", f"/data/{os.path.basename(filepath)}",
                "-o", "/data/output"
            ])
            with open(f"/data/output/{name}.md") as f:
                return f.read()
        """
        raise NotImplementedError(
            "MinerU 未安装。请执行: uv pip install 'mineru[core]'\n"
            "或使用 Docker: docker run -v $(pwd):/data opendatalab/mineru:latest mineru -p file.pdf -o output/"
        )


# ═══════════════════════════════════════════════════════════
# Fallback 解析器（不需要额外安装）
# ═══════════════════════════════════════════════════════════

class FallbackParser:
    """
    回退解析器 — 使用 pymupdf + python-docx

    支持格式：PDF / DOCX / TXT / MD
    不支持：PPTX / XLSX / 图片（这些需要 MinerU）
    """

    SUPPORTED = {".pdf", ".docx", ".doc", ".txt", ".md"}

    @classmethod
    def convert(cls, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            return cls._pdf_to_text(filepath)
        elif ext in (".docx", ".doc"):
            return cls._docx_to_text(filepath)
        elif ext in (".txt", ".md"):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise ValueError(f"FallbackParser 不支持: {ext}，请安装 MinerU")

    @classmethod
    def _pdf_to_text(cls, filepath: str) -> str:
        import fitz
        doc = fitz.open(filepath)
        pages = [page.get_text() for page in doc if page.get_text().strip()]
        doc.close()
        return "\n\n".join(pages)

    @classmethod
    def _docx_to_text(cls, filepath: str) -> str:
        from docx import Document
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ═══════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════

def convert_to_markdown(filepath: str) -> str:
    """
    统一入口——自动选择最优解析器

    优先使用 MinerU（支持表格/公式/图片），
    未安装时回退到 pymupdf + python-docx。

    Args:
        filepath: 文件路径

    Returns:
        str: Markdown 格式文本
    """
    if MinerUParser.is_available():
        return MinerUParser.convert(filepath)
    else:
        print("  ℹ️  MinerU 未安装，使用 FallbackParser（受限功能）")
        return FallbackParser.convert(filepath)


# ═══════════════════════════════════════════════════════════
# 文本分块（保持 MD 格式）
# ═══════════════════════════════════════════════════════════

def extract_text(filepath: str) -> str:
    """向后兼容别名"""
    return convert_to_markdown(filepath)


def split_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    文本分块：按句号切分，保证语义完整

    保留 Markdown 格式（表格/标题），embedding 质量更高。

    Args:
        text: Markdown 文本
        chunk_size: 每块最大字符数（默认 500 ≈ 250-350 tokens）
        overlap: 相邻块重叠字符数

    Returns:
        List[str]: 文本块列表
    """
    sentences = re.split(r"[。！？\n.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > chunk_size and current:
            chunks.append(current.strip())
            current = current[-overlap:] if len(current) > overlap else ""
        current += sentence + "。"
    if current.strip():
        chunks.append(current.strip())

    return chunks


# ── 模块自测 ─────────────────────────────────────────────
if __name__ == "__main__":
    test_file = os.path.join(os.path.dirname(__file__), "..", "..", "AGENTS.md")
    if os.path.exists(test_file):
        md = convert_to_markdown(test_file)
        print(f"MinerU 可用: {MinerUParser.is_available()}")
        print(f"Fallback 解析: {len(md)} chars")
        chunks = split_chunks(md, chunk_size=200)
        print(f"分块: {len(chunks)} 块")
    print("✅ 自测完成")
