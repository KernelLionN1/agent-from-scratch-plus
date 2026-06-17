"""
Day3 阶段三验证脚本 —— Agent 行为修复

验证内容：
1. 角色 Prompt 收紧（Planner/Coder/Reviewer）
2. 输出格式约束（代码块提取/结构化审核）
3. 消息总线修复（增量消费/去重/依赖检查）

运行方式：
    source .venv/bin/activate
    PYTHONPATH=. python test/test_day3_phase3.py
"""

import sys
import inspect
import asyncio
import threading
import time

# ── 导入验证目标 ──────────────────────────────────────────
from src.roles import AgentRole, ROLE_PROMPTS, get_role_config
from src.agents.planner import PlannerAgent
from src.agents.coder import CoderAgent, _extract_code_block
from src.agents.reviewer import ReviewerAgent
from src.message_bus import MessageBus, Message


# ═══════════════════════════════════════════════════════════
# 验证 14-16：角色 Prompt 收紧
# ═══════════════════════════════════════════════════════════
def test_14_16_roles_prompt():
    """
    验证三个角色的 prompt 已收紧（#14-16）

    检查点：
    - Planner: 包含 "### 子任务" 格式要求 + 数量限制 + "禁止写代码"
    - Coder: 要求纯代码块输出 + "禁止回答规划类问题"
    - Reviewer: 4 维度结构化格式 + "禁止自己写新代码"
    """
    print("=" * 60)
    print("验证 14-16：角色 Prompt 收紧")
    print("=" * 60)

    # Planner prompt
    planner_prompt = ROLE_PROMPTS[AgentRole.PLANNER]
    assert "### 子任务" in planner_prompt, "Planner 缺少 ### 子任务 格式"
    assert "3-8" in planner_prompt or "3-8" in planner_prompt, "Planner 缺少 3-8 数量限制"
    assert "禁止写代码" in planner_prompt, "Planner 缺少禁止越权约束"
    print(f"  14a ✅ Planner: 格式要求 + 数量限制 + 禁止越权（{len(planner_prompt)}字）")

    # Coder prompt
    coder_prompt = ROLE_PROMPTS[AgentRole.CODER]
    assert "```python" in coder_prompt, "Coder 缺少代码块格式要求"
    assert "禁止回答规划" in coder_prompt, "Coder 缺少禁止越权约束"
    print(f"  15a ✅ Coder: 代码块格式 + 禁止越权（{len(coder_prompt)}字）")

    # Reviewer prompt
    reviewer_prompt = ROLE_PROMPTS[AgentRole.REVIEWER]
    assert "正确性" in reviewer_prompt, "Reviewer 缺少正确性维度"
    assert "可读性" in reviewer_prompt, "Reviewer 缺少可读性维度"
    assert "性能" in reviewer_prompt, "Reviewer 缺少性能维度"
    assert "安全性" in reviewer_prompt, "Reviewer 缺少安全性维度"
    assert "禁止自己写新代码" in reviewer_prompt, "Reviewer 缺少禁止越权约束"
    print(f"  16a ✅ Reviewer: 4 维度 + 禁止越权（{len(reviewer_prompt)}字）")

    # 所有 prompt 比修复前长
    for name, prompt in [("Planner", planner_prompt), ("Coder", coder_prompt), ("Reviewer", reviewer_prompt)]:
        assert len(prompt) > 80, f"{name} prompt 太短: {len(prompt)}字"
    print("  16b ✅ 三个 prompt 长度都 > 80 字")

    print("  ✅ 验证 14-16 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 21：Coder 代码块提取
# ═══════════════════════════════════════════════════════════
def test_21_code_extraction():
    """
    验证 _extract_code_block 代码提取函数（#21）

    检查点：
    - ```python 代码块提取
    - ``` 无语言标签提取
    - 无代码块时返回原文
    """
    print("=" * 60)
    print("验证 21：代码块提取（#21）")
    print("=" * 60)

    # 场景1：标准 python 代码块
    text1 = "```python\ndef hello():\n    print('hi')\n```"
    result1 = _extract_code_block(text1)
    assert "def hello():" in result1, f"提取失败: {result1}"
    assert "```" not in result1, "提取结果中不应出现 ```"
    print("  21a ✅ python 代码块提取成功")

    # 场景2：无语言标签
    text2 = "```\ndef foo():\n    pass\n```"
    result2 = _extract_code_block(text2)
    assert "def foo():" in result2
    print("  21b ✅ 无语言标签提取成功")

    # 场景3：无代码块 —— 返回原文
    text3 = "def bar():\n    return 42"
    result3 = _extract_code_block(text3)
    assert result3 == text3, f"无代码块时应返回原文: {result3}"
    print("  21c ✅ 无代码块返回原文")

    print("  ✅ 验证 21 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 18-19：消息总线增量消费 + 去重
# ═══════════════════════════════════════════════════════════
def test_18_19_message_bus():
    """
    验证消息总线增量消费 + 去重（#18-19）

    检查点：
    - receive() 第二次调用返回空（offset 已推进）
    - send() 重复消息返回 None
    - receive_all() 返回全部消息
    """
    print("=" * 60)
    print("验证 18-19：消息总线增量消费 + 去重")
    print("=" * 60)

    bus = MessageBus()

    # ── 去重测试 ──
    msg1 = Message(sender="planner", receiver="coder", type="task", content="实现排序")
    id1 = bus.send(msg1)
    assert id1 is not None, "第一条消息应该成功发送"
    print(f"  19a ✅ 第一条消息发送成功: {id1}")

    id2 = bus.send(msg1)  # 完全相同
    assert id2 is None, "重复消息应该返回 None"
    print("  19b ✅ 重复消息被拦截（返回 None）")

    # 不同 content 不重复
    msg2 = Message(sender="planner", receiver="coder", type="task", content="实现搜索")
    id3 = bus.send(msg2)
    assert id3 is not None, "不同内容应该发送成功"
    print("  19c ✅ 不同内容正常发送")

    # ── 增量消费测试 ──
    bus2 = MessageBus()
    for i in range(5):
        bus2.send(Message(sender="test", receiver="receiver1", type="info", content=f"消息{i}"))

    # 第一次 receive
    batch1 = bus2.receive("receiver1")
    assert len(batch1) == 5, f"第一次应返回 5 条: {len(batch1)}"
    print(f"  18a ✅ 第一次 receive: {len(batch1)} 条")

    # 第二次 receive —— 应为空
    batch2 = bus2.receive("receiver1")
    assert len(batch2) == 0, f"第二次应返回 0 条: {len(batch2)}"
    print(f"  18b ✅ 第二次 receive: {len(batch2)} 条（增量消费生效）")

    # 发送新消息后再 receive
    bus2.send(Message(sender="test", receiver="receiver1", type="info", content="新消息"))
    batch3 = bus2.receive("receiver1")
    assert len(batch3) == 1, f"应只返回 1 条新消息: {len(batch3)}"
    print(f"  18c ✅ 新消息后 receive: {len(batch3)} 条")

    # receive_all 返回全部（不推进 offset）
    all_msgs = bus2.receive_all("receiver1")
    assert len(all_msgs) == 6, f"receive_all 应返回全部 6 条: {len(all_msgs)}"
    print(f"  18d ✅ receive_all: {len(all_msgs)} 条（全量）")

    print("  ✅ 验证 18-19 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 20：消息依赖检查
# ═══════════════════════════════════════════════════════════
def test_20_wait_for_message():
    """
    验证 wait_for_message 依赖检查（#20）

    检查点：
    - 消息未到达时阻塞等待
    - 消息到达后立即返回
    - 超时返回 None
    """
    print("=" * 60)
    print("验证 20：消息依赖检查（#20）")
    print("=" * 60)

    bus = MessageBus()

    # 20a: 消息已存在 → 立即返回
    bus.send(Message(sender="coder", receiver="reviewer", type="code", content="print('ok')"))
    msg = bus.wait_for_message("reviewer", "code", timeout=1.0)
    assert msg is not None, "消息已存在应立即返回"
    assert msg.type == "code", f"消息类型不对: {msg.type}"
    print("  20a ✅ 消息存在时立即返回")

    # 20b: 消息在后台到达
    bus2 = MessageBus()

    def delayed_send():
        time.sleep(0.3)
        bus2.send(Message(sender="coder", receiver="reviewer", type="code", content="延迟到达"))

    t = threading.Thread(target=delayed_send)
    t.start()

    msg = bus2.wait_for_message("reviewer", "code", timeout=2.0)
    assert msg is not None, "延迟消息应在超时前返回"
    assert msg.content == "延迟到达"
    t.join()
    print("  20b ✅ 延迟消息成功等待")

    # 20c: 超时返回 None
    msg = bus2.wait_for_message("reviewer", "nonexistent_type", timeout=0.5)
    assert msg is None, "超时应该返回 None"
    print("  20c ✅ 超时返回 None")

    print("  ✅ 验证 20 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day3 阶段三：Agent 行为修复 验证")
    print("=" * 60 + "\n")

    results = {}

    tests = [
        ("#14-16 角色Prompt", test_14_16_roles_prompt),
        ("#21 代码块提取", test_21_code_extraction),
        ("#18-19 增量+去重", test_18_19_message_bus),
        ("#20 依赖检查", test_20_wait_for_message),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = "✅"
        except AssertionError as e:
            results[name] = f"❌ {e}"
        except Exception as e:
            results[name] = f"💥 {type(e).__name__}: {e}"

    # ── 汇总 ──
    print("=" * 60)
    print("验证汇总")
    print("=" * 60)
    passed = 0
    for name, result in results.items():
        status = "✅" if result == "✅" else result
        if result == "✅":
            passed += 1
        print(f"  {status:6s} {name}")
    print(f"\n  通过: {passed}/{len(results)}")
    print("=" * 60)

    if passed == len(results):
        print("\n🎉 阶段三全部验证通过！")
    else:
        print(f"\n⚠️ 有 {len(results) - passed} 项未通过，请检查。")
        sys.exit(1)
