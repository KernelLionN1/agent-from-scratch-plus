"""
Day4 阶段一验证脚本 —— LangChain 单 Agent 替换

运行方式：
    source .venv/bin/activate
    python test/test_day4_phase1.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
def test_1a_llm_client_langchain():
    """LLMClient 基于 LangChain ChatOpenAI"""
    from src.llm_client import LLMClient

    client = LLMClient()
    assert hasattr(client, "_llm"), "缺少 _llm (LangChain ChatOpenAI)"
    assert hasattr(client, "ainvoke"), "缺少 ainvoke (原生 async)"

    # 同步调用
    r = client.chat([{"role": "user", "content": "说你好"}])
    assert r["content"], "返回空内容"
    assert r["usage"]["total_tokens"] > 0, "token 用量异常"
    print(f"  1a ✅ LangChain LLM: {r['content'][:40]}... tokens={r['usage']['total_tokens']}")


def test_1b_tools_langchain():
    """tools.py 提供 get_langchain_tools()"""
    from src.tools import get_langchain_tools, calculator, search_knowledge

    tools = get_langchain_tools()
    assert len(tools) == 2, f"应有 2 个 tool: {len(tools)}"
    assert tools[0].name in ("calculator", "search_knowledge")
    print(f"  1b ✅ LangChain tools: {[t.name for t in tools]}")

    # 函数仍然可用
    from src.tools import execute_tool
    r = execute_tool("calculator", {"expression": "1+2"})
    assert "3" in r or 3 == float(r), f"calculator 异常: {r}"
    print(f"  1b ✅ execute_tool 兼容: 1+2 = {r}")


def test_1c_agent_executor():
    """agent.py 基于 create_agent（LangChain 1.3+）"""
    from src.agent import ReActAgent
    import inspect

    agent = ReActAgent(max_iterations=5)

    # create_agent 返回的 _agent 是一个 Runnable
    assert hasattr(agent, "_agent"), "缺少 _agent"
    # 有 invoke 方法
    assert hasattr(agent._agent, "invoke"), "_agent 无 invoke"

    # run() 调用 _agent.invoke
    source = inspect.getsource(ReActAgent.run)
    assert "_agent.invoke" in source, "run() 未使用 _agent.invoke"
    assert "while True" not in source, "不应再有 while True 循环"
    print("  1c ✅ create_agent 替代 while 循环")

    # LLM 调用验证
    answer = agent.run("计算 1+2")
    assert answer, "回答为空"
    assert "3" in answer, f"计算结果异常: {answer[:60]}"
    print(f"  1c ✅ Agent 回答: {answer[:60]}...")


def test_1d_interface_compat():
    """接口兼容性：api.py 不报错"""
    from src import api
    assert hasattr(api, "create_agent"), "create_agent 缺失"
    assert hasattr(api, "app"), "FastAPI app 缺失"

    agent = api.create_agent()
    assert hasattr(agent, "run"), "Agent 无 run 方法"
    print("  1d ✅ api.create_agent() 兼容")


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day4 阶段一：LangChain 单 Agent 替换")
    print("=" * 60 + "\n")

    results = {}
    tests = [
        ("1a LLMClient", test_1a_llm_client_langchain),
        ("1b Tools", test_1b_tools_langchain),
        ("1c AgentExecutor", test_1c_agent_executor),
        ("1d API兼容", test_1d_interface_compat),
    ]
    for name, fn in tests:
        try:
            fn()
            results[name] = "✅"
        except Exception as e:
            results[name] = f"❌ {e}"

    print("\n" + "=" * 60)
    for n, r in results.items():
        print(f"  {r}  {n}")
    print("=" * 60)
