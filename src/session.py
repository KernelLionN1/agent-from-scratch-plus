"""
会话管理 + 成本追踪 —— Day5 阶段二

核心功能：
1. SessionManager — 会话的创建/切换/列出/删除
   每个会话对应一个 SQLiteChatMessageHistory，消息持久化隔离
2. 成本追踪 — 每次 LLM 调用的 token 消耗入库，可按会话/全局统计

Java 类比：
  SessionManager ≈ Spring Session + JPA Repository
  TokenUsage ≈ 数据库审计日志（用 AOP 拦截每次 LLM 调用）

设计决策：
  - 用标准库 uuid + sqlite3，零外部依赖
  - 会话 ID 用 UUID4，保证全局唯一
  - 成本追踪独立于消息存储，方便单独查询和统计
"""

import uuid
import sqlite3
import os
from typing import Optional
from src.memory import DB_PATH, SQLiteChatMessageHistory


# ═══════════════════════════════════════════════════════════
# SessionManager —— 会话生命周期管理
# ═══════════════════════════════════════════════════════════

class SessionManager:
    """
    会话管理器

    管理多个对话会话，每个会话有独立的消息历史和成本统计。

    使用方式：
        sm = SessionManager()
        sid = sm.create("学习Python")
        sm.switch(sid)
        # ... 对话 ...
        sm.list_sessions()      # [(id, name, created_at, msg_count), ...]
        sm.delete(sid)          # 删除会话及所有消息
    """

    def __init__(self, db_path: str = DB_PATH):
        """初始化会话管理器，确保数据库就绪"""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # 确保表存在（幂等，首次创建）
        self._ensure_tables()

    def _ensure_tables(self):
        """确保数据库表存在（幂等）"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    model TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    @property
    def active_session_id(self) -> str | None:
        """返回当前活跃的会话 ID（存储在内存中）"""
        return getattr(self, "_active_id", None)

    # ═══════════════════════════════════════════════════════
    # 会话 CRUD
    # ═══════════════════════════════════════════════════════

    def create(self, name: str = "") -> str:
        """
        创建新会话

        Args:
            name: 会话名称（可选，如"学习Python"）

        Returns:
            str: 新会话的 UUID
        """
        session_id = str(uuid.uuid4())
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, name) VALUES (?, ?)",
                (session_id, name)
            )
            conn.commit()

        print(f"[session] 创建会话: id={session_id[:8]}..., name='{name}'")
        return session_id

    def switch(self, session_id: str) -> None:
        """
        切换到指定会话

        后续通过 get_history() 获取的消息历史就是该会话的。

        Args:
            session_id: 会话 UUID（必须已存在）
        """
        # 验证会话存在
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"会话不存在: {session_id}")

        self._active_id = session_id
        # 更新最后活跃时间
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,)
            )
            conn.commit()

        print(f"[session] 切换到: {session_id[:8]}...")

    def list_sessions(self) -> list[dict]:
        """
        列出所有会话（含消息数统计）

        Returns:
            list[dict]: 每个元素包含 id, name, created_at, msg_count, token_total
        """
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT
                    s.id, s.name, s.created_at, s.updated_at,
                    COUNT(m.id) as msg_count,
                    COALESCE(SUM(t.total_tokens), 0) as token_total
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                LEFT JOIN token_usage t ON s.id = t.session_id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
            """).fetchall()

        return [
            {
                "id": row["id"],
                "name": row["name"] or "(未命名)",
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "msg_count": row["msg_count"],
                "token_total": row["token_total"],
            }
            for row in rows
        ]

    def delete(self, session_id: str) -> None:
        """
        删除会话及其所有关联数据（消息 + token 记录）

        Args:
            session_id: 要删除的会话 UUID
        """
        with self._get_conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM token_usage WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()

        # 如果删除的是当前活跃会话，清除活跃状态
        if self.active_session_id == session_id:
            self._active_id = None

        print(f"[session] 删除会话: {session_id[:8]}...")

    def rename(self, session_id: str, new_name: str) -> None:
        """重命名会话"""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET name = ? WHERE id = ?",
                (new_name, session_id)
            )
            conn.commit()

    # ═══════════════════════════════════════════════════════
    # 消息历史访问
    # ═══════════════════════════════════════════════════════

    def get_history(self, session_id: Optional[str] = None) -> SQLiteChatMessageHistory:
        """
        获取指定会话的消息历史对象

        Args:
            session_id: 会话 ID，默认使用当前活跃会话

        Returns:
            SQLiteChatMessageHistory: 可直接传给 ConversationMemory
        """
        sid = session_id or self.active_session_id
        if not sid:
            raise ValueError("没有活跃会话，请先 create() + switch() 或指定 session_id")
        return SQLiteChatMessageHistory(session_id=sid, db_path=self.db_path)

    def get_memory(self, session_id: Optional[str] = None,
                   max_messages: int = 30, max_tokens: int = 4000):
        """
        获取带截断策略的 ConversationMemory（Day5 便捷方法）

        一步完成：取历史 → 包装成 ConversationMemory

        Args:
            session_id: 会话 ID，默认当前活跃会话
            max_messages: 窗口大小
            max_tokens: token 限制

        Returns:
            ConversationMemory: 带 SQLite 后端的记忆管理器
        """
        from src.memory import ConversationMemory
        history = self.get_history(session_id)
        return ConversationMemory(
            max_messages=max_messages,
            max_tokens=max_tokens,
            history_backend=history,
        )

    # ═══════════════════════════════════════════════════════
    # 成本追踪
    # ═══════════════════════════════════════════════════════

    def record_usage(self, session_id: str, usage: dict, model: str = "") -> None:
        """
        记录一次 LLM 调用的 token 消耗

        在 LLMClient.chat() 返回后调用，把 usage 信息入库。

        Args:
            session_id: 会话 ID
            usage: LLMClient 返回的 usage 字典 {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
            model: 模型名称
        """
        if not usage or not usage.get("total_tokens"):
            return  # 没有 token 数据则跳过

        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO token_usage
                   (session_id, prompt_tokens, completion_tokens, total_tokens, model)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id,
                 usage.get("prompt_tokens", 0),
                 usage.get("completion_tokens", 0),
                 usage.get("total_tokens", 0),
                 model)
            )
            conn.commit()

    def get_cost_stats(self, session_id: Optional[str] = None) -> dict:
        """
        查询 token 用量统计

        Args:
            session_id: None=全局统计，指定=单会话统计

        Returns:
            dict: {total_tokens, total_calls, prompt_tokens, completion_tokens}
        """
        with self._get_conn() as conn:
            if session_id:
                row = conn.execute("""
                    SELECT
                        COUNT(*) as calls,
                        COALESCE(SUM(total_tokens), 0) as total_tokens,
                        COALESCE(SUM(prompt_tokens), 0) as prompt_tokens,
                        COALESCE(SUM(completion_tokens), 0) as completion_tokens
                    FROM token_usage WHERE session_id = ?
                """, (session_id,)).fetchone()
            else:
                row = conn.execute("""
                    SELECT
                        COUNT(*) as calls,
                        COALESCE(SUM(total_tokens), 0) as total_tokens,
                        COALESCE(SUM(prompt_tokens), 0) as prompt_tokens,
                        COALESCE(SUM(completion_tokens), 0) as completion_tokens
                    FROM token_usage
                """).fetchone()

        return {
            "total_calls": row["calls"],
            "total_tokens": row["total_tokens"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
        }

    # ═══════════════════════════════════════════════════════
    # 内部工具
    # ═══════════════════════════════════════════════════════

    def _get_conn(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


# ── 模块自测 ─────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    # 使用临时数据库，避免污染项目数据
    test_db = os.path.join(tempfile.gettempdir(), "test_session.db")

    print("=" * 50)
    print("SessionManager 自测")
    print("=" * 50)

    sm = SessionManager(db_path=test_db)

    # 测试1: 创建会话
    s1 = sm.create("测试会话1")
    s2 = sm.create("测试会话2")
    print(f"\n创建了2个会话: {s1[:8]}..., {s2[:8]}...")

    # 测试2: 切换到 s1，写入消息
    sm.switch(s1)
    history = sm.get_history()
    history.add_message({"role": "user", "content": "你好"})
    history.add_message({"role": "assistant", "content": "你好！有什么可以帮你？"})
    print(f"s1 消息数: {history.count_messages()}")

    # 测试3: 切换到 s2，写入不同消息
    sm.switch(s2)
    history2 = sm.get_history()
    history2.add_message({"role": "user", "content": "计算1+1"})
    print(f"s2 消息数: {history2.count_messages()}")

    # 验证隔离：切回 s1，消息数不变
    sm.switch(s1)
    h1 = sm.get_history()
    print(f"切回s1 消息数: {h1.count_messages()} (应该是2)")

    # 测试4: 列表
    sessions = sm.list_sessions()
    print(f"\n会话列表 ({len(sessions)} 个):")
    for s in sessions:
        print(f"  {s['id'][:8]}... | {s['name']:10} | {s['msg_count']}条消息 | {s['token_total']}tokens")

    # 测试5: 成本追踪
    sm.record_usage(s1, {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}, "deepseek-chat")
    sm.record_usage(s1, {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}, "deepseek-chat")
    stats = sm.get_cost_stats(s1)
    print(f"\ns1 成本: {stats['total_calls']}次调用, {stats['total_tokens']}tokens")
    global_stats = sm.get_cost_stats()
    print(f"全局: {global_stats['total_calls']}次调用, {global_stats['total_tokens']}tokens")

    # 测试6: 删除会话
    sm.delete(s2)
    sessions = sm.list_sessions()
    print(f"\n删除后会话数: {len(sessions)} (应该是1)")

    # 清理
    sm.delete(s1)
    os.remove(test_db)
    print("\n✅ SessionManager 自测完成")
