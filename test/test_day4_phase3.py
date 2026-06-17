"""
Day4 阶段三验证脚本 —— LangGraph 多 Agent 编排

运行方式:
    source .venv/bin/activate
    python test/test_day4_phase3.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_3a_langgraph_builds():
    """LangGraph 图能正常构建和编译"""
    from src.orchestrator import build_graph
    graph = build_graph()
    # 编译不应该报错
    app = graph.compile()
    assert app is not None
    print("  3a ✅ LangGraph 图编译成功")


def test_3b_serial_orchestration():
    """串行编排端到端"""
    from src.orchestrator import Orchestrator
    from src.message_bus import MessageBus

    orch = Orchestrator(MessageBus())
    result = orch.run_serial("实现判断回文的函数", with_fix=False)

    assert result["plan"], "plan 为空"
    assert result["code"], "code 为空"
    assert result["review"], "review 为空"
    print(f"  3b ✅ 串行: plan={len(result['plan'])}字 code={len(result['code'])}字 review={len(result['review'])}字")


def test_3c_parallel_orchestration():
    """并行编排端到端"""
    from src.orchestrator import Orchestrator
    from src.message_bus import MessageBus

    orch = Orchestrator(MessageBus())
    result = orch.run_parallel("实现冒泡排序和二分查找", n_coders=2)

    assert len(result.get("codes", [])) == 2, f"codes 数量: {len(result.get('codes',[]))}"
    assert result["review"], "review 为空"
    print(f"  3c ✅ 并行: {len(result['codes'])}个Coder review={len(result['review'])}字")


def test_3d_graph_structure():
    """图结构正确：节点和边完整"""
    from src.orchestrator import build_graph
    graph = build_graph()
    # 检查节点存在
    nodes = graph.nodes if hasattr(graph, 'nodes') else graph.__dict__.get('_nodes', {})
    assert len(nodes) >= 5, f"节点数不足: {len(nodes)}"
    print(f"  3d ✅ 图结构: {len(nodes)} 个节点")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day4 阶段三：LangGraph 多 Agent 编排")
    print("=" * 60 + "\n")

    results = {}
    for name, fn in [
        ("3a 图编译", test_3a_langgraph_builds),
        ("3b 串行编排", test_3b_serial_orchestration),
        ("3c 并行编排", test_3c_parallel_orchestration),
        ("3d 图结构", test_3d_graph_structure),
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
