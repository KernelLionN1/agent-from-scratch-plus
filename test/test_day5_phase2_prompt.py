"""
Day5 阶段二补充验证 —— Prompt 三层组装

运行方式:
    source .venv/bin/activate
    PYTHONPATH=. python test/test_day5_phase2_prompt.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_4a_basic_assembly():
    """三层组装：stable + context + volatile"""
    from src.prompt_builder import PromptBuilder

    pb = PromptBuilder()
    pb.set_personality("你是测试助手")
    pb.add_tool_guidance("calc", "计算")
    pb.add_rule("规则1")
    pb.load_context("上下文信息")
    prompt = pb.build("剩余3步")

    assert "你是测试助手" in prompt, "应含人格"
    assert "calc" in prompt, "应含工具"
    assert "规则1" in prompt, "应含规则"
    assert "上下文信息" in prompt, "应含上下文"
    assert "剩余3步" in prompt, "应含预算"
    assert "当前时间" in prompt, "应含时间戳"
    print("  4a ✅ 三层组装完整")


def test_4b_stable_cache():
    """stable 层缓存：修改触发重建，不修改则复用"""
    from src.prompt_builder import PromptBuilder

    pb = PromptBuilder()
    pb.set_personality("人格")
    p1 = pb.build()
    p2 = pb.build()

    # 两次 build 不修改中间 → stable 层应相同
    assert p1 == p2, "未修改时 stable 层应缓存复用"

    # 修改后 → 缓存失效
    pb.add_rule("新规则")
    p3 = pb.build()
    assert p3 != p1, "修改后应重新构建"
    assert "新规则" in p3
    print("  4a ✅ stable 缓存机制正常")


def test_4c_context_reloadable():
    """context 层可刷新"""
    from src.prompt_builder import PromptBuilder

    pb = PromptBuilder()
    pb.set_personality("助手")
    pb.load_context("旧上下文")
    p1 = pb.build()

    pb.clear_context()
    pb.load_context("新上下文")
    p2 = pb.build()

    assert "旧上下文" in p1
    assert "旧上下文" not in p2
    assert "新上下文" in p2
    print("  4c ✅ context 层可刷新")


def test_4d_integrate_with_dynamic_agent():
    """PromptBuilder 集成到 DynamicAgent"""
    from src.dynamic_agent import DynamicAgent
    from src.prompt_builder import PromptBuilder

    # 手动构造带 PromptBuilder 的 agent 调用
    # 验证 build_dict() 格式兼容
    pb = PromptBuilder()
    pb.set_personality("你是 ReAct Agent")
    pb.add_tool_guidance("calculator", "计算")
    pb.add_rule("计算问题必须调 calculator")
    system_msg = pb.build_dict()

    assert system_msg["role"] == "system"
    assert isinstance(system_msg["content"], str)
    assert len(system_msg["content"]) > 50
    print("  4d ✅ build_dict() 格式兼容 LLMClient")


def test_4e_empty_layers():
    """空层不输出多余空行"""
    from src.prompt_builder import PromptBuilder

    pb = PromptBuilder()
    # 什么都不设 → build 应该是空字符串
    prompt = pb.build()
    # 只有时间戳（volatile 层总是有的）
    assert "当前时间" in prompt
    # 但不应有空的人格/工具/上下文块
    assert "可用工具" not in prompt
    print("  4e ✅ 空层不输出")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day5 阶段二补充：Prompt 三层组装")
    print("=" * 60 + "\n")

    results = {}
    for name, fn in [
        ("4a 三层组装", test_4a_basic_assembly),
        ("4b 缓存机制", test_4b_stable_cache),
        ("4c 上下文刷新", test_4c_context_reloadable),
        ("4d Agent集成", test_4d_integrate_with_dynamic_agent),
        ("4e 空层处理", test_4e_empty_layers),
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
