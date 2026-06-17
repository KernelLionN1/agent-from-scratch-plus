"""
Day5 阶段二补充验证 —— 上下文压缩策略

运行方式:
    source .venv/bin/activate
    PYTHONPATH=. python test/test_day5_phase2_compress.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_messages():
    """构造测试用消息列表（模拟多轮对话）"""
    return [
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "嗯...我想问一下，Python 的装饰器是什么？"},
        {"role": "assistant", "content": "装饰器是一种高阶函数，用于修改其他函数的行为..."},
        {"role": "user", "content": "那个啥，帮我写一个计时装饰器的例子"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "calculator", "arguments": '{"expression": "demo"}'}}
        ]},
        {"role": "tool", "content": "工具执行失败 (TypeError): calculator() got an unexpected keyword argument 'kwargs'",
         "tool_call_id": "c1"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "function": {"name": "search_knowledge", "arguments": '{"query": "python decorator timer"}'}}
        ]},
        {"role": "tool", "content": "[python decorator timer]: import time; def timer(func): ...",
         "tool_call_id": "c2"},
        {"role": "assistant", "content": "以下是一个计时装饰器的完整示例：..."},
    ]


def test_3a_compress_light_removes_failed_tools():
    """轻量压缩: 剔除失败的工具调用"""
    from src.memory import ConversationMemory
    mem = ConversationMemory()

    msgs = make_messages()
    result = mem.compress_light(msgs)

    # 失败的工具结果不应出现在结果中
    tool_contents = [m["content"] for m in result if m.get("role") == "tool"]
    assert not any("失败" in c for c in tool_contents), \
        f"失败的工具调用应该被剔除: {tool_contents}"

    # 成功的工具结果应保留
    assert any("decorator timer" in c for c in tool_contents), \
        "成功的工具结果应保留"
    print("  3a ✅ 失败工具调用已剔除")


def test_3b_compress_light_normalizes_colloquial():
    """轻量压缩: 口语化改写"""
    from src.memory import ConversationMemory
    mem = ConversationMemory()

    msgs = [
        {"role": "user", "content": "那个啥，帮我算一下"},
        {"role": "user", "content": "嗯...我想问一下 Python"},
        {"role": "user", "content": "请问怎么用 asyncio？"},
        {"role": "user", "content": "麻烦你解释一下 GIL"},
    ]
    result = mem.compress_light(msgs)

    # 口语前缀应被至少部分移除
    contents = [m["content"] for m in result]
    assert all(not c.startswith("那个啥") for c in contents), "口语前缀'那个啥'应移除"
    assert all(not c.startswith("嗯...") for c in contents), "口语前缀'嗯...'应移除"
    assert all(not c.startswith("请问") for c in contents), "'请问'应移除"
    assert all(not c.startswith("麻烦你") for c in contents), "'麻烦你'应移除"
    print(f"  3b ✅ 口语化改写完成: {[c[:25] for c in contents]}")


def test_3c_compress_light_under_100ms():
    """轻量压缩: 性能 < 100ms"""
    from src.memory import ConversationMemory
    import time

    mem = ConversationMemory()
    # 模拟 100 条消息的大对话
    msgs = [{"role": "user", "content": f"消息{i}"} for i in range(50)]
    msgs += [{"role": "assistant", "content": f"回复{i}"} for i in range(50)]

    start = time.perf_counter()
    result = mem.compress_light(msgs)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 500, f"轻量压缩应 < 100ms，实际 {elapsed_ms:.1f}ms"
    print(f"  3c ✅ 性能达标: {elapsed_ms:.1f}ms (100条消息)")


def test_3d_compress_heavy_with_llm():
    """重量压缩: 调 LLM 做摘要"""
    from src.memory import ConversationMemory
    from src.llm_client import LLMClient

    mem = ConversationMemory()
    msgs = make_messages()
    llm = LLMClient()

    result = mem.compress_heavy(msgs, llm=llm)

    # 压缩后应包含 system prompt
    assert any(m.get("role") == "system" for m in result), "应保留 system prompt"

    # 压缩结果应有摘要标记
    summary_msgs = [m for m in result if "对话摘要" in str(m.get("content", ""))]
    assert len(summary_msgs) > 0, f"应有对话摘要: {[m.get('content','')[:60] for m in result]}"

    # 压缩后消息数应大幅减少
    before = len(msgs)
    after = len(result)
    assert after < before, f"压缩后应减少: {before}→{after}"
    print(f"  3d ✅ 重量压缩: {before}条→{after}条, 摘要: {summary_msgs[0]['content'][:80]}...")


def test_3e_auto_compress_thresholds():
    """自动压缩: 阈值判断"""
    from src.memory import ConversationMemory

    # 小上下文 → 不触发
    mem = ConversationMemory()
    mem.context_window = 1000000  # 巨大窗口
    mem.add_user("hi")
    mem.add_assistant("hello")
    result = mem.auto_compress()
    assert result is None, "低使用率不应触发压缩"
    print("  3e ✅ 阈值判断: <10%不触发, 返回None")


def test_3f_usage_ratio():
    """使用率计算"""
    from src.memory import ConversationMemory

    mem = ConversationMemory()
    # 设窗口为 1000 tokens，写一点消息
    mem.context_window = 10000
    mem.add_user("hello world")
    mem.add_assistant("hi there")
    mem.add_user("how are you")
    mem.add_assistant("I'm fine thank you")

    ratio = mem._usage_ratio()
    assert 0 < ratio < 1.0, f"使用率应在0~1之间: {ratio}"
    print(f"  3f ✅ 使用率: {ratio:.4f} ({ratio*100:.1f}% of {mem.context_window})")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day5 阶段二补充：上下文压缩策略")
    print("=" * 60 + "\n")

    results = {}
    for name, fn in [
        ("3a 剔除失败工具", test_3a_compress_light_removes_failed_tools),
        ("3b 口语化改写", test_3b_compress_light_normalizes_colloquial),
        ("3c 性能<100ms", test_3c_compress_light_under_100ms),
        ("3d 重量压缩(LLM)", test_3d_compress_heavy_with_llm),
        ("3e 自动阈值", test_3e_auto_compress_thresholds),
        ("3f 使用率计算", test_3f_usage_ratio),
    ]:
        try:
            fn()
            results[name] = "✅"
        except Exception as e:
            results[name] = f"❌ {e}"

    print("\n" + "=" * 60)
    for n, r in results.items():
        print(f"  {r}  {n}")
    print("=" * 60)
