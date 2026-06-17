"""
Day5 阶段一验证脚本 —— LLM 动态路由

运行方式:
    source .venv/bin/activate
    python test/test_day5_phase1.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_1a_graph_structure():
    """动态图能编译"""
    from src.dynamic_agent import DynamicAgent
    agent = DynamicAgent(max_steps=3)
    assert hasattr(agent, "run")
    print("  1a ✅ DynamicAgent 创建成功")


def test_1b_llm_decides_to_use_tool():
    """计算问题 → LLM 自主决定调 calculator"""
    from src.dynamic_agent import DynamicAgent
    agent = DynamicAgent(max_steps=5)
    r = agent.run("计算 100+200")
    assert r["tool_calls"] > 0, f"LLM 应该自主调工具: {r}"
    assert r["steps"] >= 2, f"至少 agent→tools→agent: {r['steps']}步"
    print(f"  1b ✅ LLM 自主调工具: steps={r['steps']} tool_calls={r['tool_calls']} answer={r['answer'][:40]}")


def test_1c_simple_question_no_tool():
    """简单问题 → LLM 可能不调工具直接回答"""
    from src.dynamic_agent import DynamicAgent
    agent = DynamicAgent(max_steps=5)
    r = agent.run("说你好")
    # 不强制调工具 —— LLM 自己判断
    print(f"  1c ✅ 简单问题: steps={r['steps']} tool_calls={r['tool_calls']} answer={r['answer'][:40]}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day5 阶段一：LLM 动态路由")
    print("=" * 60 + "\n")

    results = {}
    for name, fn in [
        ("1a 图编译", test_1a_graph_structure),
        ("1b LLM自主调工具", test_1b_llm_decides_to_use_tool),
        ("1c 简单对话", test_1c_simple_question_no_tool),
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
