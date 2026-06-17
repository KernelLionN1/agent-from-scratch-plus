"""
Day7 RAG 管线验证

运行方式:
    source .venv/bin/activate
    PYTHONPATH=. python test/test_day7_rag.py
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_docs():
    """创建测试文档（模拟产品手册）"""
    tmpdir = tempfile.mkdtemp()
    p1 = os.path.join(tmpdir, "warranty.txt")
    with open(p1, "w", encoding="utf-8") as f:
        f.write("产品保修政策 v2.0\n")
        f.write("手机保修期: 购买之日起 12 个月\n")
        f.write("笔记本保修期: 购买之日起 24 个月\n")
        f.write("配件保修期: 6 个月（充电器、数据线等）\n")
        f.write("人为损坏不在保修范围内\n")

    p2 = os.path.join(tmpdir, "pricing.txt")
    with open(p2, "w", encoding="utf-8") as f:
        f.write("产品价格表 2026\n")
        f.write("旗舰手机 X1: ¥5999\n")
        f.write("轻薄笔记本 L1: ¥8999\n")
        f.write("无线耳机 E1: ¥1299\n")
        f.write("所有价格含税，中国大陆地区\n")

    return tmpdir, [p1, p2]


def cleanup(tmpdir):
    for f in os.listdir(tmpdir):
        os.unlink(os.path.join(tmpdir, f))
    os.rmdir(tmpdir)


def test_7a_document_parsing():
    """文档解析：PDF/Word/txt → 文本"""
    from src.document_processor import extract_text, split_chunks

    tmpdir, files = create_test_docs()

    text = extract_text(files[0])  # warranty.txt
    assert "12 个月" in text
    assert "24 个月" in text
    print(f"  7a ✅ 文本提取: {len(text)} chars")

    chunks = split_chunks(text, chunk_size=50)  # 小 chunk 测试分块
    assert len(chunks) >= 1, f"至少1块: {len(chunks)}"
    print(f"  7a ✅ 分块: {len(chunks)} 块（85字 → {len(chunks)}块）")

    cleanup(tmpdir)


def test_7b_rag_ingest_and_query():
    """RAG 摄入 + 检索"""
    from src.rag_engine import RAGEngine

    tmpdir, files = create_test_docs()
    engine = RAGEngine(collection_name="test_rag")

    # 摄入两份文档
    n1 = engine.ingest_file(files[0])
    n2 = engine.ingest_file(files[1])
    total = engine.count()
    assert total > 0, f"应有文本块: {total}"
    print(f"  7b ✅ 摄入: {n1}+{n2}={total} 块")

    # 检索
    result = engine.query("手机保修多久？", top_k=2)
    assert "12" in result["context"], f"应包含12个月: {result['context'][:80]}"
    print(f"  7b ✅ 检索命中: {len(result['context'])} chars, sources={result['sources']}")

    # 清理
    engine.clear()
    cleanup(tmpdir)


def test_7c_rag_answer_with_llm():
    """RAG 完整问答（需要 LLM）"""
    from src.rag_engine import RAGEngine
    from src.llm_client import LLMClient

    tmpdir, files = create_test_docs()
    engine = RAGEngine(collection_name="test_rag_qa")
    engine.ingest_file(files[0])

    llm = LLMClient()
    answer = engine.answer("手机保修多久？", llm=llm)
    assert answer and len(answer) > 0, "应有回答"
    print(f"  7c ✅ RAG 回答: {answer[:80]}...")

    engine.clear()
    cleanup(tmpdir)


def test_7d_vision_fallback():
    """图像理解回退（DeepSeek 不支持图片时用文件信息）"""
    from src.vision import describe_image
    import base64

    # 1x1 PNG
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png)
        path = f.name

    desc = describe_image(path)
    assert desc, "应有描述"
    print(f"  7d ✅ 图片: {desc[:60]}")
    os.unlink(path)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day7 RAG 管线验证")
    print("=" * 60 + "\n")

    results = {}
    for name, fn in [
        ("7a 文档解析+分块", test_7a_document_parsing),
        ("7b RAG摄入+检索", test_7b_rag_ingest_and_query),
        ("7c RAG问答(LLM)", test_7c_rag_answer_with_llm),
        ("7d 图像回退", test_7d_vision_fallback),
    ]:
        try:
            fn()
            results[name] = "✅"
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = f"❌ {e}"

    print("\n" + "=" * 60)
    for n, r in results.items():
        print(f"  {r}  {n}")
    print("=" * 60)
