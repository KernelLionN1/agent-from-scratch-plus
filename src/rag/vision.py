"""
Day7: 图像理解 — 图片 → 文本描述（可选模块）

将图片转为文本描述，注入 RAG 引擎，实现"搜图"能力。

生产方案：
  - CLIP 模型（本地，免费，但需要 GPU）
  - GPT-4V / Gemini Vision（API，付费，效果好）

教学方案（简化）：
  使用 OpenAI GPT-4V API 生成图片描述，
  或者用 base64 编码传给支持的 LLM。

Java 类比：AWS Rekognition / Google Vision API
"""

import base64
import os
from typing import Optional


def image_to_base64(filepath: str) -> str:
    """图片 → base64 字符串（用于 API 传输）"""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_image(filepath: str, llm=None) -> str:
    """
    用 LLM 视觉能力描述图片内容

    支持 GPT-4V / Claude Vision / Gemini Vision 等多模态模型。
    如果 LLM 不支持图片，回退到文件元信息描述。

    Args:
        filepath: 图片文件路径（支持 .jpg / .png / .webp）
        llm: LLMClient 实例

    Returns:
        str: 图片描述文本
    """
    if llm is None:
        from src.llm_client import LLMClient
        llm = LLMClient()

    ext = os.path.splitext(filepath)[1].lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}
    mime = mime_map.get(ext, "image/jpeg")

    try:
        # OpenAI Vision API 格式
        b64 = image_to_base64(filepath)
        response = llm.chat(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请用中文简要描述这张图片的内容（100字以内）。"},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
            temperature=0.3,
            max_tokens=200,
        )
        # 兼容 dict 和 str 返回
        if isinstance(response, str):
            return response
        return response.get("content", "") or str(response)

    except Exception as e:
        # 回退：当前 LLM（如 DeepSeek 纯文本）不支持图片
        filename = os.path.basename(filepath)
        size_kb = os.path.getsize(filepath) / 1024
        fallback = f"[图片: {filename}, {size_kb:.0f}KB, 格式: {ext}]"
        print(f"  ⚠️  视觉描述失败 ({e})，回退: {fallback}")
        return fallback


def ingest_image_to_rag(filepath: str, engine, llm=None) -> int:
    """
    图片 → 描述文本 → 注入 RAG 引擎

    便捷方法：一步完成"图搜"的摄入。

    Args:
        filepath: 图片路径
        engine: RAGEngine 实例
        llm: LLMClient 实例

    Returns:
        int: 摄入的文本块数
    """
    description = describe_image(filepath, llm)
    # 写入临时文本文件，复用 ingest_file
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(f"[图片文件: {os.path.basename(filepath)}]\n{description}")
        tmp_path = f.name
    count = engine.ingest_file(tmp_path)
    os.unlink(tmp_path)
    return count


# ── 模块自测 ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Vision 自测")
    print("=" * 50)

    # 创建一个 1x1 PNG 测试
    import tempfile
    # 最小 PNG (1x1 白色像素)
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png_data)
        test_img = f.name

    desc = describe_image(test_img)
    print(f"图片描述: {desc}")

    os.unlink(test_img)
    print("\n✅ 自测完成")
