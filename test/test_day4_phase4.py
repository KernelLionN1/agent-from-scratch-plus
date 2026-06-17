"""
Day4 阶段四验证脚本 —— API 层适配 LangGraph

运行方式:
    source .venv/bin/activate
    python test/test_day4_phase4.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_4a_api_imports():
    """api.py 能正常导入（不启动服务器）"""
    from src.api import app, create_agent, OrchestrateRequest
    agent = create_agent()
    assert hasattr(agent, "run")
    print("  4a ✅ api 导入正常: create_agent + OrchestrateRequest")


def test_4b_orchestrator_compat():
    """api.py 的 Orchestrator 导入兼容 LangGraph 版"""
    from src.message_bus import MessageBus
    from src.orchestrator import Orchestrator
    orch = Orchestrator(MessageBus())
    # 接口不变
    assert hasattr(orch, "run_serial")
    assert hasattr(orch, "run_parallel")
    print("  4b ✅ Orchestrator 接口兼容（run_serial/run_parallel）")


def test_4c_agent_invoke():
    """agent.py 的 create_agent → run() 仍然可用"""
    from src.api import create_agent
    agent = create_agent()
    answer = agent.run("计算 1+1")
    assert "2" in answer, f"计算结果异常: {answer[:60]}"
    print(f"  4c ✅ Agent run(): {answer[:40]}...")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day4 阶段四：API 层适配")
    print("=" * 60 + "\n")

    results = {}
    for name, fn in [
        ("4a API导入", test_4a_api_imports),
        ("4b 编排兼容", test_4b_orchestrator_compat),
        ("4c Agent调用", test_4c_agent_invoke),
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
