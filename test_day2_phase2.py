"""
Day2 阶段二验证脚本 —— 串行任务分发

验证内容：
1. Agent 实例化：Planner / Coder / Reviewer 能正常创建
2. 消息总线传递：Agent 之间能通过 MessageBus 收发消息
3. Orchestrator 结构：编排器正确创建和管理 Agent
4. LLM 调用：各 Agent 能成功调用 LLM 并返回内容
5. 串行流程：run_serial() 完整流程不报错
6. 【踩坑】角色越权观察：检查 Planner 是否越权写代码

运行方式：
    source .venv/bin/activate
    python test_day2_phase2.py
"""

import sys
import time

from src.message_bus import MessageBus, Message
from src.agents import PlannerAgent, CoderAgent, ReviewerAgent, BaseAgent
from src.agents.base import BaseAgent
from src.roles import AgentRole
from src.orchestrator import Orchestrator


# ═══════════════════════════════════════════════════════════
# 验证 1：Agent 实例化
# ═══════════════════════════════════════════════════════════
def test_agent_creation():
    """
    验证三个 Agent 能正常创建，且配置正确
    """
    print("=" * 60)
    print("验证 1：Agent 实例化")
    print("=" * 60)

    bus = MessageBus()

    planner = PlannerAgent(bus, "planner-1")
    coder = CoderAgent(bus, "coder-1")
    reviewer = ReviewerAgent(bus, "reviewer-1")

    # 检查名称
    assert planner.name == "planner-1"
    assert coder.name == "coder-1"
    assert reviewer.name == "reviewer-1"

    # 检查角色
    assert planner.role == AgentRole.PLANNER
    assert coder.role == AgentRole.CODER
    assert reviewer.role == AgentRole.REVIEWER

    # 检查共享同一个 MessageBus
    assert planner.bus is bus
    assert coder.bus is bus
    assert reviewer.bus is bus

    # 检查 LLM 客户端存在
    assert planner.llm is not None
    assert coder.llm is not None
    assert reviewer.llm is not None

    # 检查配置
    assert planner.config.temperature == 0.7  # Planner 温度高
    assert coder.config.temperature == 0.3
    assert reviewer.config.temperature == 0.3

    print(f"  ✅ Planner: {planner}")
    print(f"  ✅ Coder:   {coder}")
    print(f"  ✅ Reviewer:{reviewer}")
    print(f"  ✅ 共享 MessageBus: {id(bus)}")
    print(f"  ✅ 实例化验证通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 2：消息总线传递
# ═══════════════════════════════════════════════════════════
def test_message_passing():
    """
    验证 Agent 之间能通过 MessageBus 正常通信
    """
    print("=" * 60)
    print("验证 2：Agent 消息传递")
    print("=" * 60)

    bus = MessageBus()
    planner = PlannerAgent(bus, "planner-1")
    coder = CoderAgent(bus, "coder-1")
    reviewer = ReviewerAgent(bus, "reviewer-1")

    # Planner → Coder：发送任务
    msg1 = planner.send_message("coder-1", "task", "实现排序算法")
    coder_tasks = coder.receive_tasks()
    assert len(coder_tasks) == 1
    assert coder_tasks[0].content == "实现排序算法"
    assert coder_tasks[0].sender == "planner-1"
    print(f"  ✅ Planner → Coder: {msg1}")

    # Coder → Reviewer：发送代码
    msg2 = coder.send_message("reviewer-1", "code", "def sort(): pass")
    reviewer_msgs = reviewer.receive_messages()
    code_msgs = [m for m in reviewer_msgs if m.type == "code"]
    assert len(code_msgs) == 1
    print(f"  ✅ Coder → Reviewer: {msg2}")

    # Reviewer → Coder：发送审核意见
    msg3 = reviewer.send_message("coder-1", "review", "缺少边界检查")
    coder_reviews = coder.receive_reviews()
    assert len(coder_reviews) == 1
    print(f"  ✅ Reviewer → Coder: {msg3}")

    # 全链路消息统计
    print(f"  📊 消息总线统计: {bus.stats()}")
    print(f"  ✅ 消息传递验证通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 3：Orchestrator 结构
# ═══════════════════════════════════════════════════════════
def test_orchestrator_structure():
    """
    验证编排器能正确创建和管理 Agent
    """
    print("=" * 60)
    print("验证 3：Orchestrator 结构")
    print("=" * 60)

    bus = MessageBus()
    orch = Orchestrator(bus)

    # 检查三个 Agent 已创建
    assert orch.planner is not None
    assert orch.coder is not None
    assert orch.reviewer is not None

    # 检查类型
    assert isinstance(orch.planner, PlannerAgent)
    assert isinstance(orch.coder, CoderAgent)
    assert isinstance(orch.reviewer, ReviewerAgent)

    # 检查共享 MessageBus
    assert orch.planner.bus is bus
    assert orch.coder.bus is bus
    assert orch.reviewer.bus is bus

    print(f"  ✅ Planner:  {orch.planner}")
    print(f"  ✅ Coder:    {orch.coder}")
    print(f"  ✅ Reviewer: {orch.reviewer}")
    print(f"  ✅ Orchestrator 结构验证通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 4：LLM 调用
# ═══════════════════════════════════════════════════════════
def test_llm_calls():
    """
    验证各 Agent 能成功调用 LLM（需要网络和 API Key）

    这个测试会真实调用 DeepSeek API，每次消耗 token。
    """
    print("=" * 60)
    print("验证 4：LLM 调用（真实 API 请求）")
    print("=" * 60)

    bus = MessageBus()
    planner = PlannerAgent(bus, "planner-test")
    coder = CoderAgent(bus, "coder-test")
    reviewer = ReviewerAgent(bus, "reviewer-test")

    print("\n── 4a：Planner 拆分任务 ──")
    plan = planner.execute("实现一个函数，判断一个数是否为质数")
    assert plan, "Planner 返回空结果"
    assert len(plan) > 20, f"Planner 输出太短: {len(plan)} 字符"
    print(f"  ✅ Planner 输出: {len(plan)} 字符")
    print(f"  预览: {plan[:150]}...")

    print("\n── 4b：Coder 生成代码 ──")
    code = coder.execute("实现一个函数 is_prime(n)，判断 n 是否为质数")
    assert code, "Coder 返回空结果"
    assert len(code) > 20, f"Coder 输出太短: {len(code)} 字符"
    # 【踩坑】不检查代码是否真的能运行，只检查包含关键词
    print(f"  ✅ Coder 输出: {len(code)} 字符")
    print(f"  预览: {code[:150]}...")

    print("\n── 4c：Reviewer 审核代码 ──")
    review = reviewer.execute(code)
    assert review, "Reviewer 返回空结果"
    assert len(review) > 10, f"Reviewer 输出太短: {len(review)} 字符"
    print(f"  ✅ Reviewer 输出: {len(review)} 字符")
    print(f"  预览: {review[:150]}...")

    print(f"\n  ✅ LLM 调用验证通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 5：串行编排完整流程
# ═══════════════════════════════════════════════════════════
def test_serial_flow():
    """
    验证完整的串行编排流程

    这个测试会：
    1. Planner 拆分需求
    2. Coder 生成代码
    3. Reviewer 审核
    4. Coder 修复
    """
    print("=" * 60)
    print("验证 5：串行编排完整流程")
    print("=" * 60)

    bus = MessageBus()
    orch = Orchestrator(bus)

    requirement = "实现一个函数，将列表中的元素去重并排序"

    result = orch.run_serial(requirement, with_fix=True)

    # 验证返回结构
    assert "plan" in result
    assert "code" in result
    assert "review" in result
    assert "fixed_code" in result
    assert "stats" in result

    # 验证内容非空
    assert result["plan"], "plan 为空"
    assert result["code"], "code 为空"
    assert result["review"], "review 为空"

    # 打印摘要
    print(f"\n  📋 规划长度: {len(result['plan'])} 字符")
    print(f"  💻 代码长度: {len(result['code'])} 字符")
    print(f"  🔍 审核长度: {len(result['review'])} 字符")
    if result["fixed_code"]:
        print(f"  🔧 修复代码长度: {len(result['fixed_code'])} 字符")
    print(f"  📊 统计: {result['stats']}")

    print(f"\n  ✅ 串行编排验证通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 6：【踩坑】角色越权观察
# ═══════════════════════════════════════════════════════════
def test_role_boundary_pitfall():
    """
    【踩坑】观察角色越权行为

    用 Coder 的角色 prompt 问 Planner 该问的问题，
    看 Coder 会怎么回答。
    """
    print("=" * 60)
    print("验证 6：【踩坑】角色越权观察")
    print("=" * 60)

    bus = MessageBus()
    coder = CoderAgent(bus, "coder-test")

    # 问 Coder 一个规划类的问题（这应该是 Planner 的职责）
    plan_question = (
        "请帮我规划一个完整的 Web 应用项目，"
        "包括需要的模块、技术选型和开发步骤。"
    )

    response = coder.call_llm(plan_question)

    # 【踩坑】Coder 的 system prompt 没写「只能编码」，
    # 所以它可能会回答规划类问题——这就是越权
    print(f"  💡 问了 Coder 一个规划问题")
    print(f"  📝 Coder 回复: {response[:200]}...")

    # 检查 Coder 是否越权回答了规划问题
    # 如果回复中包含规划类关键词（模块、步骤、架构），说明 Coder 越权了
    plan_keywords = ["模块", "步骤", "架构", "技术选型", "流程"]
    found_keywords = [kw for kw in plan_keywords if kw in response]
    if found_keywords:
        print(f"  ⚠️ 【踩坑确认】Coder 越权回答了规划问题！")
        print(f"     回复包含规划关键词: {found_keywords}")
    else:
        print(f"  ℹ️ Coder 没有明显越权（可能只给了代码方案）")

    print(f"  ✅ 越权观察完成\n")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Day2 阶段二验证：串行任务分发")
    print("=" * 60 + "\n")

    failed = []

    # 结构测试（不需要网络）
    try:
        test_agent_creation()
    except Exception as e:
        failed.append(("Agent 实例化", e))

    try:
        test_message_passing()
    except Exception as e:
        failed.append(("消息传递", e))

    try:
        test_orchestrator_structure()
    except Exception as e:
        failed.append(("Orchestrator 结构", e))

    # LLM 测试（需要网络 + API Key）
    try:
        test_llm_calls()
    except Exception as e:
        print(f"\n  ⚠️ LLM 调用测试失败: {e}")
        print(f"  （可能原因：网络问题、API Key 无效、配额不足）")
        failed.append(("LLM 调用", e))

    try:
        test_serial_flow()
    except Exception as e:
        print(f"\n  ⚠️ 串行流程测试失败: {e}")
        failed.append(("串行编排", e))

    try:
        test_role_boundary_pitfall()
    except Exception as e:
        failed.append(("越权观察", e))

    # ── 总结 ──
    print("=" * 60)
    if failed:
        print(f"  ⚠️ {len(failed)} 项测试失败:")
        for name, err in failed:
            print(f"     - {name}: {err}")
        sys.exit(1)
    else:
        print("  🎉 阶段二全部验证通过！")
        print("  ⚠️ 踩坑已确认：任务拆分不约束 + 角色边界模糊 + 无输出格式约束")
    print("=" * 60)
