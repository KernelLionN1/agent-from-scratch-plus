"""
对话记忆管理 —— Day4 框架版：基于 LangChain ChatMessageHistory + tiktoken

Day4 核心变化：
  手搓版（Day1-3）：手写消息列表 + 手动截断 + 字符数估算 token
  框架版（Day4）：  InMemoryChatMessageHistory + tiktoken 精确计算

关键收益：
  - tiktoken 精确 token 计数（修复 Day3 #7 遗留的估算不准）
  - LangChain 原生消息对象（HumanMessage/AIMessage）
  - 截断策略保留（兼容 agent.py）
"""

import sqlite3
import json
import os
from typing import Literal

TruncationStrategy = Literal["none", "sliding_window", "token_limit"]

# ── Day5: SQLite 数据库路径 ──────────────────────────────
# 放在项目根目录 data/ 下，.gitignore 会排除
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "agent_memory.db")


# ═══════════════════════════════════════════════════════════
# Day4: LangChain 记忆封装
# ═══════════════════════════════════════════════════════════

class ConversationMemory:
    """
    对话记忆管理器 —— Day4 框架版

    底层：langchain_core.chat_history.InMemoryChatMessageHistory
    Token 计数：tiktoken（精确）

    接口完全兼容 Day1-3 的 ConversationMemory。
    """

    def __init__(self, max_messages: int = 20, max_tokens: int = 4000,
                 history_backend=None, session_id: str | None = None):
        """
        初始化记忆管理器

        Day5 新增：
        - history_backend: 自定义存储后端（如 SQLiteChatMessageHistory）
        - session_id: 会话 ID（用于 SQLite 多会话隔离）
        """
        # ── 存储后端：可选 SQLite 或默认 InMemory ──
        if history_backend is not None:
            self._store = history_backend
        else:
            from langchain_core.chat_history import InMemoryChatMessageHistory
            self._store = InMemoryChatMessageHistory()

        self.max_messages = max_messages
        self.max_tokens = max_tokens

        # ── 内部状态 ──
        self._system_content: str = ""  # 系统提示词单独保存

        # ── Day4: tiktoken 编码器 ──
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4/DeepSeek 通用
        except (ImportError, Exception):
            self._encoder = None

    # ═══════════════════════════════════════════════════════
    # 消息 CRUD
    # ═══════════════════════════════════════════════════════

    def set_system(self, content: str):
        """设置系统提示词"""
        self._system_content = content

    def add_user(self, content: str):
        """记录用户消息"""
        from langchain_core.messages import HumanMessage
        self._store.add_message(HumanMessage(content=content))

    def add_assistant(self, content: str | None = None, tool_calls: list | None = None):
        """记录 LLM 回复"""
        from langchain_core.messages import AIMessage
        msg = AIMessage(content=content or "")
        if tool_calls:
            msg.tool_calls = tool_calls  # type: ignore
        self._store.add_message(msg)

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: str):
        """记录工具执行结果"""
        from langchain_core.messages import ToolMessage
        self._store.add_message(ToolMessage(
            content=result,
            tool_call_id=tool_call_id,
        ))

    # ═══════════════════════════════════════════════════════
    # 截断策略（兼容老接口：返回 dict 列表）
    # ═══════════════════════════════════════════════════════

    def get_messages(self, strategy: TruncationStrategy = "none") -> list[dict]:
        """
        根据截断策略返回消息列表（dict 格式，兼容手搓版）

        Day4: 内部用 tiktoken 精确计算 token
        """
        all_msgs = [
            {"role": "system", "content": self._system_content}
        ] if self._system_content else []

        for m in self._store.messages:
            role = _get_role(m)
            content = m.content if hasattr(m, "content") else str(m)
            msg = {"role": role, "content": content}
            # 保留 tool_calls
            if hasattr(m, "tool_calls") and m.tool_calls:
                msg["tool_calls"] = m.tool_calls
            # 保留 tool_call_id
            if hasattr(m, "tool_call_id"):
                msg["tool_call_id"] = m.tool_call_id
            all_msgs.append(msg)

        if strategy == "none":
            return all_msgs

        elif strategy == "sliding_window":
            # 保留 system + 最近 N 条（非 system）
            system_msgs = [m for m in all_msgs if m["role"] == "system"]
            other_msgs = [m for m in all_msgs if m["role"] != "system"]
            recent = other_msgs[-self.max_messages:] if len(other_msgs) > self.max_messages else other_msgs
            return system_msgs + recent

        elif strategy == "token_limit":
            # Day4: 用 tiktoken 精确计算
            result = [m for m in all_msgs if m["role"] == "system"]
            token_count = self._count_tokens(result)

            for msg in reversed([m for m in all_msgs if m["role"] != "system"]):
                msg_tokens = self._count_tokens([msg])
                if token_count + msg_tokens > self.max_tokens:
                    break
                result.insert(len([m for m in result if m["role"] == "system"]), msg)
                token_count += msg_tokens
            return result

        return all_msgs

    def _count_tokens(self, messages: list[dict]) -> int:
        """
        Day4: tiktoken 精确计数（修复 Day3 #7）

        手搓版: 字符数 / 2 = 估算
        框架版: tiktoken.encode() = 精确
        """
        if self._encoder:
            total = 0
            for msg in messages:
                content = msg.get("content", "") or ""
                total += len(self._encoder.encode(content))
                if msg.get("tool_calls"):
                    total += len(self._encoder.encode(str(msg["tool_calls"])))
            return total
        else:
            # 回退：字符数估算
            total = sum(len(m.get("content", "") or "") for m in messages)
            return total // 2

    # ═══════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════

    def clear(self):
        """清空所有记忆"""
        from langchain_core.chat_history import InMemoryChatMessageHistory
        self._store = InMemoryChatMessageHistory()

    def count_messages(self) -> int:
        """返回非 system 消息数"""
        return len(self._store.messages)

    def __len__(self) -> int:
        return len(self._store.messages) + (1 if self._system_content else 0)

    def __repr__(self) -> str:
        return f"<ConversationMemory(LangChain): {self.count_messages()} 条消息, tiktoken={'✓' if self._encoder else '✗'}>"


# ═══════════════════════════════════════════════════════════
# Day5: SQLite 持久化记忆后端
# ═══════════════════════════════════════════════════════════

class SQLiteChatMessageHistory:
    """
    Day5 新增：SQLite 持久化的消息历史

    替换 InMemoryChatMessageHistory，实现：
    - 进程重启不丢失消息
    - 多会话隔离（按 session_id）
    - 与 ConversationMemory 接口兼容

    Java 类比：InMemory → H2 内存库，SQLite → MySQL/PostgreSQL 磁盘库
    """

    def __init__(self, session_id: str = "default", db_path: str = DB_PATH):
        """
        初始化 SQLite 消息存储

        Args:
            session_id: 会话唯一标识（UUID 或自定义字符串）
            db_path: 数据库文件路径
        """
        self.session_id = session_id
        self.db_path = db_path

        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # 建表（幂等，首次创建）
        self._init_db()

    # ═══════════════════════════════════════════════════════
    # 数据库初始化
    # ═══════════════════════════════════════════════════════

    def _init_db(self):
        """创建消息表和会话表（幂等，IF NOT EXISTS）"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,          -- user / assistant / tool / system
                    content TEXT,                 -- 消息正文
                    tool_calls TEXT,              -- JSON: 工具调用列表
                    tool_call_id TEXT,            -- 工具消息的关联 ID
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,          -- UUID
                    name TEXT,                    -- 会话名称
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

    def _get_conn(self):
        """获取数据库连接（每次新建，用完关闭，线程安全）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 让查询结果可以用 dict 方式访问
        return conn

    # ═══════════════════════════════════════════════════════
    # 消息 CRUD —— 与 InMemoryChatMessageHistory 接口兼容
    # ═══════════════════════════════════════════════════════

    def add_message(self, msg) -> None:
        """
        持久化一条消息

        参数 msg 可以是 LangChain Message 对象或 dict，
        通过 _extract_message_fields 统一提取字段。
        """
        role, content, tool_calls, tool_call_id = _extract_message_fields(msg)

        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (self.session_id, role, content,
                 json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                 tool_call_id)
            )
            conn.commit()

    @property
    def messages(self) -> list:
        """
        返回当前会话的所有消息（LangChain 对象列表）

        与 InMemoryChatMessageHistory.messages 接口一致。

        【注意】@property 把方法伪装成属性，调用方写 history.messages 不加括号。
        代价：每次访问都执行一次 SELECT 查库。InMemory 版直接返回内存列表（O(1)），
        SQLite 版每次都是 IO 操作。如果需要频繁访问，应在外层缓存一份，而非反复调 property。

        Java 类比：getMessages() 每次调 mapper.selectList()，相当于没有 L1 cache 的 JPA。
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                (self.session_id,)
            ).fetchall()

        result = []
        for row in rows:
            role = row["role"]
            content = row["content"] or ""
            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "user":
                result.append(HumanMessage(content=content))
            elif role == "assistant":
                msg_obj = AIMessage(content=content)
                if row["tool_calls"]:
                    try:
                        msg_obj.tool_calls = json.loads(row["tool_calls"])
                    except json.JSONDecodeError:
                        pass
                result.append(msg_obj)
            elif role == "tool":
                result.append(ToolMessage(
                    content=content,
                    tool_call_id=row["tool_call_id"] or "",
                ))
        return result

    def clear(self) -> None:
        """清空当前会话的所有消息"""
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (self.session_id,)
            )
            conn.commit()

    def count_messages(self) -> int:
        """返回当前会话的消息数"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?",
                (self.session_id,)
            ).fetchone()
            return row["cnt"] if row else 0


def _extract_message_fields(msg) -> tuple[str, str, list | None, str | None]:
    """
    从 LangChain Message 对象或 dict 中提取 (role, content, tool_calls, tool_call_id)

    兼容两种来源：
    - LangChain 原生对象（HumanMessage, AIMessage, ToolMessage, SystemMessage）
    - 手搓版 dict（{"role": "user", "content": "..."}）
    """
    # dict 格式
    if isinstance(msg, dict):
        return (
            msg.get("role", "unknown"),
            msg.get("content", ""),
            msg.get("tool_calls"),
            msg.get("tool_call_id"),
        )

    # LangChain Message 对象
    type_name = type(msg).__name__.lower()
    if "human" in type_name:
        return ("user", msg.content, None, None)
    elif "ai" in type_name:
        tool_calls = getattr(msg, "tool_calls", None) or None
        return ("assistant", msg.content, tool_calls, None)
    elif "tool" in type_name:
        tc_id = getattr(msg, "tool_call_id", None) or None
        return ("tool", msg.content, None, tc_id)
    elif "system" in type_name:
        return ("system", msg.content, None, None)

    # 兜底
    content = getattr(msg, "content", str(msg))
    return ("unknown", content, None, None)


def _get_role(msg) -> str:
    """LangChain Message 对象 → 手搓版 role 字符串"""
    type_name = type(msg).__name__.lower()
    if "human" in type_name:
        return "user"
    elif "ai" in type_name:
        return "assistant"
    elif "tool" in type_name:
        return "tool"
    elif "system" in type_name:
        return "system"
    return "unknown"


# ── 模块自测 ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("ConversationMemory（LangChain 版）自测")
    print("=" * 50)

    mem = ConversationMemory(max_messages=4, max_tokens=200)
    mem.set_system("你是一个助手")
    mem.add_user("我叫小明")
    mem.add_assistant("你好小明！")
    mem.add_user("今年25岁")
    mem.add_assistant("了解了")

    print(f"\n消息数: {len(mem)}")
    print(f"tiktoken: {'可用' if mem._encoder else '不可用'}")

    # 测试各策略
    for s in ["none", "sliding_window", "token_limit"]:
        msgs = mem.get_messages(s)
        print(f"\n--- {s}: {len(msgs)} 条 ---")
        for m in msgs[:5]:
            print(f"  [{m['role']}] {str(m.get('content',''))[:40]}")

    print("\n✅ 自测完成")
