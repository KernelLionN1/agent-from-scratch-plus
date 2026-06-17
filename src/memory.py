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

from typing import Literal

TruncationStrategy = Literal["none", "sliding_window", "token_limit"]


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

    def __init__(self, max_messages: int = 20, max_tokens: int = 4000):
        """初始化记忆管理器"""
        # ── LangChain 原生存储 ──
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
