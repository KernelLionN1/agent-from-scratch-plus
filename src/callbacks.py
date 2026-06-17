"""
Day6: 回调机制 —— 替代 print()，支持可插拔的日志/通知

对标 Hermes Agent 的 8 种 callback surfaces，
提供统一的事件通知接口。

使用方式：
    from src.callbacks import CallbackManager

    cm = CallbackManager()
    cm.on("agent_thinking", lambda step: logger.info(f"Step {step}"))
    agent.run("计算 1+1")  # 自动触发回调
"""

from typing import Callable, Any


class CallbackManager:
    """
    回调管理器

    注册事件处理器，Agent 执行过程中自动触发。
    支持多个处理器监听同一事件。

    Java 类比：Spring Event / Guava EventBus
    """

    # 事件类型常量
    AGENT_THINKING = "agent_thinking"       # Agent 开始思考
    TOOL_EXECUTING = "tool_executing"       # 工具开始执行
    TOOL_RESULT = "tool_result"             # 工具执行完成
    COMPRESS_LIGHT = "compress_light"       # 轻量压缩
    COMPRESS_HEAVY = "compress_heavy"       # 重量压缩
    LLM_CALL_START = "llm_call_start"       # LLM 调用开始
    LLM_CALL_END = "llm_call_end"           # LLM 调用结束
    MEMORY_SAVED = "memory_saved"           # 记忆写入
    ERROR = "error"                         # 错误

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}

    def on(self, event: str, handler: Callable) -> None:
        """注册事件处理器"""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def off(self, event: str, handler: Callable = None) -> None:
        """移除事件处理器（不指定 handler 则清空该事件全部）"""
        if handler is None:
            self._handlers.pop(event, None)
        elif event in self._handlers:
            self._handlers[event] = [h for h in self._handlers[event] if h is not handler]

    def emit(self, event: str, **kwargs) -> None:
        """触发事件（所有注册的处理器都会被调用）"""
        for handler in self._handlers.get(event, []):
            try:
                handler(**kwargs)
            except Exception as e:
                # 回调异常不影响主流程
                print(f"[callback error] {event}: {e}")

    def wrap(self, logger=None):
        """
        一键注册常用回调到指定 logger

        Args:
            logger: Python logging.Logger 实例

        Returns:
            self (支持链式调用)
        """
        if logger is None:
            import logging
            logger = logging.getLogger("agent")

        self.on(self.AGENT_THINKING, lambda step: logger.info(f"Agent thinking step={step}"))
        self.on(self.TOOL_EXECUTING, lambda name, args: logger.info(f"Tool: {name}({args})"))
        self.on(self.TOOL_RESULT, lambda name, result: logger.debug(f"Result: {name} → {str(result)[:100]}"))
        self.on(self.COMPRESS_LIGHT, lambda before, after, saved, ms: logger.debug(f"Light compress: {before}→{after}, {saved}t, {ms}ms"))
        self.on(self.COMPRESS_HEAVY, lambda before, after, saved, ms: logger.info(f"Heavy compress: {before}→{after}, {saved}t, {ms}ms"))
        self.on(self.ERROR, lambda msg: logger.error(msg))
        return self


# ── 全局单例 ─────────────────────────────────────────────
_default_callback_manager = CallbackManager()


def get_callback_manager() -> CallbackManager:
    """获取全局回调管理器"""
    return _default_callback_manager


# ── 模块自测 ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("CallbackManager 自测")
    print("=" * 50)

    cm = CallbackManager()

    # 注册处理器
    events = []
    cm.on("test", lambda msg: events.append(msg))
    cm.on("test", lambda msg: events.append(msg.upper()))

    # 触发
    cm.emit("test", msg="hello")
    assert events == ["hello", "HELLO"]
    print("  ✅ 多处理器正常")

    # 移除
    cm.off("test")
    cm.emit("test", msg="world")  # 不应添加到 events
    assert events == ["hello", "HELLO"]
    print("  ✅ 移除正常")

    # logger 包装
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("test")
    cm.wrap(logger)
    cm.emit("agent_thinking", step=1)
    cm.emit("tool_executing", name="calc", args={"expr": "1+1"})
    print("  ✅ logger 包装正常")

    print("\n✅ 自测完成")
