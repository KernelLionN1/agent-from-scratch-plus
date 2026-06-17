"""
对话记忆管理 —— 消息存储 + 上下文截断

核心职责：
- 持久存储多轮对话的所有消息（跨 run() 调用保留）
- 提供多种截断策略，防止上下文超出 LLM 限制

刻意踩坑点：
- 默认不截断 → 长对话触发上下文超限
- 简单截断只保留最近 N 条 → 丢失重要早期信息
- token 估算极其粗糙（字符数/4）→ 中英文混合时严重不准
"""

from typing import Literal

# ── 截断策略类型 ─────────────────────────────────────────
# Literal 相当于 Java 的枚举，限制只能是这三个值之一
TruncationStrategy = Literal["none", "sliding_window", "token_limit"]


class ConversationMemory:
    """
    对话记忆管理器

    类比 Java：这是一个有状态的 @Service，内部维护 List<Message>。
    区别在于它同时负责"截断策略"——类似一个带容量限制的 RingBuffer。

    使用方式：
        mem = ConversationMemory()
        mem.set_system("你是助手")
        mem.add_user("你好")
        mem.add_assistant("你好！")
        messages = mem.get_messages(strategy="sliding_window")  # 拿截断后的
    """

    def __init__(self, max_messages: int = 20, max_tokens: int = 4000):
        """
        初始化记忆管理器

        参数：
            max_messages: 滑动窗口保留的最大消息数（不含 system）
            max_tokens:   token 限制模式的阈值
        """
        # ── 内部存储 ──
        # 所有消息的完整历史（包含 system），只在截断输出时才裁剪
        self._messages: list[dict] = []

        # ── 截断参数 ──
        # 注意：这些值直接影响 Agent 行为和 token 成本
        self.max_messages = max_messages  # 滑动窗口大小
        self.max_tokens = max_tokens      # token 上限

    # ═══════════════════════════════════════════════════════
    # 消息 CRUD（增删改查）
    # ═══════════════════════════════════════════════════════

    def set_system(self, content: str):
        """
        设置系统提示词 —— 必须放在 messages 第一条

        为什么单独一个方法？
        —— system 消息在截断时必须保留，不能用普通 add 混在一起。
        """
        # 如果已有 system 消息则替换，否则插入到最前面
        if self._messages and self._messages[0]["role"] == "system":
            self._messages[0] = {"role": "system", "content": content}
        else:
            self._messages.insert(0, {"role": "system", "content": content})

    def add_user(self, content: str):
        """记录用户消息 —— 每次 run() 调用时追加"""
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str | None = None, tool_calls: list | None = None):
        """
        记录 LLM 的回复 —— 可能是纯文本，也可能是工具调用请求

        两种模式：
        - content 有值：LLM 直接回答，没有调用工具
        - tool_calls 有值：LLM 请求调用工具，content 为 None
        """
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._messages.append(msg)

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: str):
        """
        记录工具执行结果 —— 必须跟在 assistant(tool_calls) 后面

        tool_call_id 用于关联到具体的调用请求，
        LLM 通过这个 ID 知道"这个结果对应我哪次请求"。
        """
        self._messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call_id,
            "name": tool_name,
        })

    # ═══════════════════════════════════════════════════════
    # 截断策略
    # ═══════════════════════════════════════════════════════

    def get_messages(self, strategy: TruncationStrategy = "none") -> list[dict]:
        """
        根据截断策略返回消息列表 —— Agent 每次调 LLM 前调用

        策略：
        - "none":            不截断，返回全部（坑点：消息超长时 LLM 报错）
        - "sliding_window":  保留 system + 最近 N 条消息
        - "token_limit":     按 token 估算值截断
        """
        if strategy == "none":
            # ── 不截断 ──
            # 直接返回全部消息。如果对话超过 LLM 的上下文窗口
            # （如 deepseek-chat 约 64K tokens），API 会直接报错。
            return list(self._messages)

        elif strategy == "sliding_window":
            # ── 滑动窗口截断 ──
            # 保留 system 消息 + 最近 max_messages 条
            # 问题：如果用户在第 1 轮说了重要信息，第 21 轮被丢弃，
            # LLM 就会"失忆"。
            system_msgs = [m for m in self._messages if m["role"] == "system"]
            other_msgs = [m for m in self._messages if m["role"] != "system"]
            # 只取末尾 N 条
            recent = other_msgs[-self.max_messages:] if len(other_msgs) > self.max_messages else other_msgs
            return system_msgs + recent

        elif strategy == "token_limit":
            # ── Token 限制截断 ──
            # 按字符数/4 估算 token 数（极其粗糙！中文一个字符约 1-2 token）
            # 从最新消息开始往前取，直到超出限制
            system_msgs = [m for m in self._messages if m["role"] == "system"]
            other_msgs = [m for m in self._messages if m["role"] != "system"]

            result = list(system_msgs)
            token_count = self._estimate_tokens(system_msgs)

            # 从后往前取（保留最新的消息）
            for msg in reversed(other_msgs):
                msg_tokens = self._estimate_tokens([msg])
                if token_count + msg_tokens > self.max_tokens:
                    break  # 超出限制，停止添加
                result.insert(len(system_msgs), msg)  # 插在 system 后面，保持顺序
                token_count += msg_tokens

            return result

        else:
            # 未知策略 → 保守返回全部
            return list(self._messages)

    # ═══════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """
        Token 数量估算 —— Day3 修复 #7：区分中英文

        修复前公式：总字符数 // 4  （英文约 0.25 token/字，中文约 1-2 token/字）
        修复后公式：中文字数×0.7 + 其他字符×0.25

        注意：这仍然是估算！精确计算需用 tiktoken 库。
        tiktoken 安装：pip install tiktoken
        用法：encoding = tiktoken.encoding_for_model("gpt-4")
              tokens = len(encoding.encode(text))
        """
        # ── 中文字符 Unicode 范围 ──
        # \u4e00-\u9fff: 基本汉字
        # \u3400-\u4dbf: 扩展 A 区
        # \uf900-\ufaff: 兼容汉字
        CJK_RANGES = [
            (0x4e00, 0x9fff),
            (0x3400, 0x4dbf),
            (0xf900, 0xfaff),
        ]

        def _is_chinese(ch: str) -> bool:
            """判断单个字符是否为中文"""
            cp = ord(ch)
            return any(lo <= cp <= hi for lo, hi in CJK_RANGES)

        total_tokens = 0
        for msg in messages:
            content = msg.get("content", "") or ""
            # 分别统计中英文
            chinese_count = sum(1 for c in content if _is_chinese(c))
            other_count = len(content) - chinese_count
            # 中文约 0.7 token/字，英文约 0.25 token/字（取中值）
            total_tokens += chinese_count * 0.7 + other_count * 0.25

            # 如果有 tool_calls，JSON 按英文估算
            if msg.get("tool_calls"):
                total_tokens += len(str(msg["tool_calls"])) * 0.25

        return int(total_tokens)

    def clear(self):
        """清空所有记忆 —— 开始全新对话时调用"""
        self._messages = []

    def count_messages(self) -> int:
        """返回当前消息总数（不含 system）"""
        return len([m for m in self._messages if m["role"] != "system"])

    def __len__(self) -> int:
        """支持 len(memory) 语法"""
        return len(self._messages)

    def __repr__(self) -> str:
        """打印时的友好显示"""
        return f"<ConversationMemory: {len(self)} 条消息, 策略=sliding_window({self.max_messages})>"


# ── 模块自测 ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("ConversationMemory 自测")
    print("=" * 50)

    mem = ConversationMemory(max_messages=4, max_tokens=200)

    # 模拟一段对话
    mem.set_system("你是一个助手")
    mem.add_user("我叫小明")
    mem.add_assistant("你好小明！")
    mem.add_user("我今年25岁")
    mem.add_assistant("了解了，25岁")
    mem.add_user("我喜欢Python")
    mem.add_assistant("Python 很棒！")
    mem.add_user("帮我算1+1")

    print(f"\n原始消息数: {len(mem)}")
    print(f"非system消息数: {mem.count_messages()}")

    # 测试不同策略
    print(f"\n--- 策略: none ---")
    msgs = mem.get_messages("none")
    for m in msgs:
        print(f"  [{m['role']}] {str(m.get('content', ''))[:50]}")

    print(f"\n--- 策略: sliding_window (保留最近4条) ---")
    msgs = mem.get_messages("sliding_window")
    for m in msgs:
        print(f"  [{m['role']}] {str(m.get('content', ''))[:50]}")

    print(f"\n--- 策略: token_limit (200 tokens) ---")
    msgs = mem.get_messages("token_limit")
    for m in msgs:
        print(f"  [{m['role']}] {str(m.get('content', ''))[:50]}")

    print(f"\n{mem}")
    print("✅ 自测完成")
