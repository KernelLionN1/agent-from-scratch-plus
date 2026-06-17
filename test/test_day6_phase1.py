"""
Day6 阶段一验证脚本 —— 技术收尾（增量压缩 + Streaming + MEMORY.md + 回调）

运行方式:
    source .venv/bin/activate
    PYTHONPATH=. python test/test_day6_phase1.py
"""
import sys, os, tempfile, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# 1.1 增量压缩
# ═══════════════════════════════════════════════════════════

def test_1a_incremental_compression():
    """增量压缩: 两次调用 compress_heavy，第二次只压缩新增消息"""
    from src.memory import ConversationMemory
    from src.llm_client import LLMClient

    mem = ConversationMemory()
    llm = LLMClient()

    # 第一轮对话
    mem.add_user("我叫小明")
    mem.add_assistant("你好小明！")
    r1 = mem.compress_heavy(llm=llm)
    assert len(r1) <= 2, f"压缩后应 ≤2 条: {len(r1)}"

    # 验证锚点状态
    assert mem._anchor_summary, "应有摘要"
    assert mem._anchor_msg_count >= 1, f"锚点应 >= 1: {mem._anchor_msg_count}"

    # 第二轮对话（追加）
    mem.add_user("帮我计算 1+2")
    mem.add_assistant("结果是3")
    r2 = mem.compress_heavy(llm=llm)

    # 第二次是增量模式，只压缩新消息
    assert mem._anchor_msg_count > 2, "锚点应前移"
    print(f"  1a ✅ 增量压缩: 锚点={mem._anchor_msg_count}, "
          f"摘要={mem._anchor_summary[:60]}...")


def test_1b_reset_anchor():
    """重置锚点"""
    from src.memory import ConversationMemory
    mem = ConversationMemory()
    mem._anchor_summary = "old"
    mem._anchor_msg_count = 10
    mem.reset_anchor()
    assert mem._anchor_summary == ""
    assert mem._anchor_msg_count == 0
    print("  1b ✅ reset_anchor() 正常")


# ═══════════════════════════════════════════════════════════
# 1.2 Streaming
# ═══════════════════════════════════════════════════════════

def test_1c_streaming():
    """流式输出: token 逐字产出"""
    from src.dynamic_agent import DynamicAgent

    async def _run():
        agent = DynamicAgent(max_steps=3)
        events = []
        async for event in agent.llm.astream(
            messages=[{"role": "user", "content": "说你好"}],
        ):
            events.append(event)

        # 应有 token 事件和 done 事件
        tokens = [e for e in events if e["type"] == "token"]
        done = [e for e in events if e["type"] == "done"]
        assert len(tokens) > 0, "应有 token 事件"
        assert len(done) == 1, "应有 done 事件"
        print(f"  1c ✅ Streaming: {len(tokens)} tokens → '{done[0]['content'][:30]}'")

    asyncio.run(_run())


def test_1d_streaming_with_tool():
    """流式输出 + 工具调用"""
    from src.dynamic_agent import DynamicAgent

    async def _run():
        agent = DynamicAgent(max_steps=3)
        events = []
        async for event in agent.astream("计算 100+200"):
            events.append(event)

        types = [e["type"] for e in events]
        assert "thinking" in types, "应有 thinking"
        assert "tool_result" in types, "应有 tool_result"
        print(f"  1d ✅ Streaming+Tool: {len(events)} events, types={set(types)}")

    asyncio.run(_run())


# ═══════════════════════════════════════════════════════════
# 1.3 MEMORY.md
# ═══════════════════════════════════════════════════════════

def test_1e_file_memory_store():
    """FileMemoryStore CRUD"""
    from src.memory import FileMemoryStore
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
        path = f.name

    store = FileMemoryStore(filepath=path)

    # Save
    store.save("用户偏好", "喜欢简短回答")
    store.save("项目", "agent-from-scratch-plus")

    # Load
    all_mem = store.load()
    assert "用户偏好" in all_mem
    assert "agent-from-scratch-plus" in all_mem

    # Load single
    assert store.load("用户偏好") == "喜欢简短回答"

    # List
    assert len(store.list_keys()) == 2

    # Delete + persist
    store.delete("项目")
    assert len(store.list_keys()) == 1

    # 重新读取验证持久化
    store2 = FileMemoryStore(filepath=path)
    assert len(store2.list_keys()) == 1
    assert store2.load("用户偏好") == "喜欢简短回答"

    os.unlink(path)
    print("  1e ✅ FileMemoryStore CRUD + 持久化")


# ═══════════════════════════════════════════════════════════
# 1.4 回调
# ═══════════════════════════════════════════════════════════

def test_1f_callbacks():
    """CallbackManager 事件注册/触发/移除"""
    from src.callbacks import CallbackManager

    cm = CallbackManager()
    collected = []

    cm.on(CallbackManager.AGENT_THINKING, lambda step: collected.append(f"think:{step}"))
    cm.on(CallbackManager.TOOL_EXECUTING, lambda name, args: collected.append(f"tool:{name}"))

    cm.emit(CallbackManager.AGENT_THINKING, step=1)
    cm.emit(CallbackManager.TOOL_EXECUTING, name="calc", args={"expr": "1+1"})

    assert "think:1" in collected
    assert "tool:calc" in collected

    cm.off(CallbackManager.AGENT_THINKING)
    cm.emit(CallbackManager.AGENT_THINKING, step=2)  # 不应触发
    assert collected.count("think:1") == 1  # 只有第一条
    print(f"  1f ✅ CallbackManager: {collected}")


def test_1g_callback_wrap():
    """一键注册常用回调到 logger"""
    from src.callbacks import CallbackManager, get_callback_manager
    import logging

    cm = CallbackManager()
    logger = logging.getLogger("test_agent")
    logger.setLevel(logging.DEBUG)

    cm.wrap(logger)
    # 验证至少注册了常用事件
    assert cm.AGENT_THINKING in cm._handlers
    assert cm.TOOL_EXECUTING in cm._handlers
    assert cm.ERROR in cm._handlers
    print("  1g ✅ wrap() 注册了 6 个事件处理器")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day6 阶段一：技术收尾（增量压缩 + Streaming + MEMORY.md + 回调）")
    print("=" * 60 + "\n")

    results = {}
    for name, fn in [
        ("1a 增量压缩", test_1a_incremental_compression),
        ("1b 重置锚点", test_1b_reset_anchor),
        ("1c 流式输出", test_1c_streaming),
        ("1d 流式+工具", test_1d_streaming_with_tool),
        ("1e MEMORY.md", test_1e_file_memory_store),
        ("1f 回调", test_1f_callbacks),
        ("1g 回调wrap", test_1g_callback_wrap),
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
