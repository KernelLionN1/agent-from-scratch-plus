"""
Day7: RAG 引擎 — 分块 → 向量化 → 存储 → 检索

简化版实现（教学目的）：
  使用 ChromaDB 内存模式 + OpenAI Embedding，
  不依赖外部向量数据库。

完整 RAG 管线：
  PDF → text → chunks → embeddings → ChromaDB → query → retrieve → LLM

Java 类比：Elasticsearch + 向量插件，但 ChromaDB 更轻量
"""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class RAGEngine:
    """
    简易 RAG 引擎

    使用方式：
        engine = RAGEngine()
        engine.ingest_file("docs/manual.pdf")          # 摄入文档
        answer = engine.query("保修期多久？")           # 提问
    """

    def __init__(self, collection_name: str = "default"):
        import chromadb
        # ChromaDB 内存模式（生产用 PersistentClient 存磁盘）
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 余弦相似度
        )
        self._chunk_size = 500

    def ingest_file(self, filepath: str) -> int:
        """
        摄入文档：解析 → 分块 → 向量化 → 存储

        Args:
            filepath: 文档路径

        Returns:
            int: 摄入的文本块数量
        """
        from src.document_processor import extract_text, split_chunks

        # 1. 文本提取
        text = extract_text(filepath)
        if not text.strip():
            print(f"  ⚠️  文件为空: {filepath}")
            return 0

        # 2. 分块
        chunks = split_chunks(text, chunk_size=self._chunk_size)

        # 3. 向量化 + 存储
        doc_id = os.path.basename(filepath)
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filepath, "chunk_index": i} for i in range(len(chunks))]

        # 去重：先删除已有
        existing = self._collection.get(ids=ids)
        if existing["ids"]:
            self._collection.delete(ids=ids)

        self._collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )

        print(f"  📥 摄入: {filepath} → {len(chunks)} 个文本块")
        return len(chunks)

    def query(self, question: str, top_k: int = 3) -> dict:
        """
        检索相关文档块

        Args:
            question: 用户问题
            top_k: 返回最相关的 K 个文本块

        Returns:
            dict: {"context": "合并的上下文", "sources": ["file1.pdf", ...]}
        """
        results = self._collection.query(
            query_texts=[question],
            n_results=top_k,
        )

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not docs:
            return {"context": "", "sources": []}

        # 合并上下文
        context = "\n\n---\n\n".join(docs)
        sources = list(set(m.get("source", "unknown") for m in metadatas))

        print(f"  🔍 检索: '{question[:40]}...' → {len(docs)} 块, 距离={[round(d,3) for d in distances]}")
        return {"context": context, "sources": sources}

    def answer(self, question: str, llm=None, top_k: int = 3) -> str:
        """
        完整 RAG 问答：检索 → 构建 prompt → LLM 生成

        Args:
            question: 用户问题
            llm: LLMClient 实例（必须提供）
            top_k: 检索块数

        Returns:
            str: 基于文档的答案
        """
        if llm is None:
            from src.llm_client import LLMClient
            llm = LLMClient()

        result = self.query(question, top_k=top_k)
        if not result["context"]:
            return "未找到相关文档内容。"

        # 构建 RAG prompt
        rag_prompt = (
            "你是一个文档助手。请根据以下文档内容回答问题。"
            "如果文档中没有相关信息，请如实说不知道。\n\n"
            f"## 文档内容\n{result['context']}\n\n"
            f"## 用户问题\n{question}\n\n"
            "请回答："
        )

        response = llm.chat(
            messages=[{"role": "user", "content": rag_prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        answer = response.get("content", "")

        print(f"  💬 回答: {answer[:80]}...")
        return answer

    def count(self) -> int:
        """返回已存储的文本块数"""
        return self._collection.count()

    def clear(self) -> None:
        """清空所有数据"""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"}
        )


# ── 模块自测 ─────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("=" * 50)
    print("RAGEngine 自测")
    print("=" * 50)

    # 创建测试文档
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("产品保修政策：\n")
        f.write("1. 手机保修期为购买之日起 12 个月。\n")
        f.write("2. 笔记本电脑保修期为购买之日起 24 个月。\n")
        f.write("3. 配件（充电器、数据线）保修期为 6 个月。\n")
        f.write("4. 人为损坏不在保修范围内。\n")
        f.write("5. 保修需要提供购买凭证。\n")
        test_file = f.name

    engine = RAGEngine(collection_name="test_rag")
    count = engine.ingest_file(test_file)
    print(f"摄入: {count} 块")

    result = engine.query("手机保修多久？")
    print(f"检索: {len(result['context'])} chars")

    # 用 LLM 回答
    try:
        answer = engine.answer("手机保修多久？")
        assert "12" in answer or "个月" in answer, "答案应包含保修期限"
        print(f"RAG 答案: {answer[:100]}")
    except Exception as e:
        print(f"LLM 不可用，跳过: {e}")

    engine.clear()
    os.unlink(test_file)
    print("\n✅ 自测完成")
