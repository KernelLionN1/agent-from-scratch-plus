"""
Day2 阶段一验证脚本 —— 角色定义 + 消息总线

验证内容：
1. 三个角色的 system prompt 能正常加载
2. 消息总线基本收发功能
3. 【踩坑】消息重复消费验证
4. 广播 + 按类型过滤 + 增量拉取

运行方式：
    source .venv/bin/activate
    python test_day2_phase1.py
"""

import sys
import time

# ── 导入验证目标 ──────────────────────────────────────────
from src.roles import AgentRole, get_role_config, get_role_prompt
from src.message_bus import MessageBus, Message


# ═══════════════════════════════════════════════════════════
# 验证 1：角色定义
# ═══════════════════════════════════════════════════════════
def test_roles():
    """
    验证角色枚举和 prompt 模板

    检查点：
    - AgentRole 枚举有三个值
    - 每个角色都有非空的 system prompt
    - RoleConfig 能正确捆绑 prompt + temperature
    """
    print("=" * 60)
    print("验证 1：角色定义")
    print("=" * 60)

    # 检查枚举值数量
    roles = list(AgentRole)
    assert len(roles) == 3, f"预期 3 个角色，实际 {len(roles)}"

    # 逐个检查每个角色
    for role in roles:
        config = get_role_config(role)

        # prompt 不能为空
        assert config.prompt, f"{role.name} 的 prompt 为空！"
        # 每个角色 prompt 长度至少 30 字 —— 确保不是空字符串或占位符
        assert len(config.prompt) >= 30, \
            f"{role.name} 的 prompt 太短（{len(config.prompt)}字），疑似未填写"

        # temperature 在合理范围
        assert 0 <= config.temperature <= 1, \
            f"{role.name} 的 temperature={config.temperature} 超出范围"

        # 便捷函数返回相同结果
        assert get_role_prompt(role) == config.prompt, \
            f"{role.name} 的 get_role_prompt 和 get_role_config 不一致"

        print(f"  ✅ {role.name:10s} | temp={config.temperature} | prompt={len(config.prompt)}字")

    print(f"  ✅ 角色定义验证通过（共 {len(roles)} 个角色）\n")


# ═══════════════════════════════════════════════════════════
# 验证 2：消息总线基本收发
# ═══════════════════════════════════════════════════════════
def test_message_bus_basic():
    """
    验证消息总线的基本发送和接收功能

    检查点：
    - 消息正确投递到指定接收方
    - 消息字段完整
    - 未发送消息的队列是空的
    """
    print("=" * 60)
    print("验证 2：消息总线基本收发")
    print("=" * 60)

    bus = MessageBus()

    # 发送消息
    msg = Message(
        sender="planner",
        receiver="coder",
        type="task",
        content="计算 1+1",
    )
    bus.send(msg)

    # Planner 队列应该为空（消息是发给 Coder 的）
    planner_msgs = bus.receive("planner")
    assert len(planner_msgs) == 0, f"Planner 不应该收到消息，却有 {len(planner_msgs)} 条"

    # Coder 队列应该有 1 条消息
    coder_msgs = bus.receive("coder")
    assert len(coder_msgs) == 1, f"Coder 应该有 1 条消息，实际 {len(coder_msgs)} 条"

    # 消息字段完整
    received = coder_msgs[0]
    assert received.sender == "planner"
    assert received.receiver == "coder"
    assert received.type == "task"
    assert "1+1" in received.content
    assert received.id, "消息 ID 为空"
    assert received.timestamp > 0, "消息时间戳异常"

    print(f"  ✅ 消息内容: {received}")
    print(f"  ✅ 消息 ID: {received.id}")
    print(f"  ✅ 消息投递正确（Planner→Coder 隔离生效）")
    print(f"  ✅ 基本收发验证通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 3：【踩坑】消息重复消费
# ═══════════════════════════════════════════════════════════
def test_message_duplication_bug():
    """
    【刻意踩坑验证】消息重复消费

    验证消息总线不删除已读消息，导致重复消费。
    这是 Day2 的已知问题，Day3 统一修复。
    """
    print("=" * 60)
    print("验证 3：【踩坑】消息重复消费")
    print("=" * 60)

    bus = MessageBus()
    bus.send(Message(sender="planner", receiver="coder", type="task", content="任务A"))

    # 第一次读取
    first_read = bus.receive("coder")
    assert len(first_read) == 1, f"第一次读取应该有 1 条消息"

    # 第二次读取 —— Day3 已修复：偏移量前进，不再重复返回
    second_read = bus.receive("coder")
    assert len(second_read) == 0, f"Day3修复后第二次应返回0条（已消费），实际{len(second_read)}条"

    # 验证 receive_all 仍可获取全部（不更新偏移）
    all_msgs = bus.receive_all("coder")
    assert len(all_msgs) == 1, "receive_all 应返回全部消息"

    # 验证 receive_all 仍可获取全部（不更新偏移）
    all_msgs = bus.receive_all("coder")
    assert len(all_msgs) == 1, "receive_all 应返回全部消息"
    assert all_msgs[0].id == first_read[0].id, "ID 应相同"

    print(f"  第一次 receive: {len(first_read)} 条 → ID={first_read[0].id}")
    print(f"  第二次 receive: {len(second_read)} 条（Day3 修复后不再重复）")
    print(f"  receive_all: {len(all_msgs)} 条（不更新偏移）")
    print(f"  ✅ Day3 修复验证通过：消息不再重复消费\n")


# ═══════════════════════════════════════════════════════════
# 验证 4：广播 + 过滤 + 增量拉取
# ═══════════════════════════════════════════════════════════
def test_advanced_features():
    """
    验证消息总线的高级功能：
    - 广播（一条消息发多个接收方）
    - 按类型过滤
    - 时间增量拉取
    """
    print("=" * 60)
    print("验证 4：广播 + 过滤 + 增量拉取")
    print("=" * 60)

    bus = MessageBus()

    # ── 广播测试 ──
    bus.broadcast(
        sender="planner",
        type_="info",
        content="全员通知：项目启动",
        receivers=["coder", "reviewer"],
    )
    # Day3 修复后 receive() 会更新偏移，用 receive_all 做验证（不更新偏移）
    coder_all = bus.receive_all("coder")
    reviewer_all = bus.receive_all("reviewer")
    assert len(coder_all) == 1, f"广播后 coder 应有1条: {len(coder_all)}"
    assert len(reviewer_all) == 1, f"广播后 reviewer 应有1条: {len(reviewer_all)}"
    print(f"  ✅ 广播成功：Coder={len(coder_all)}条, Reviewer={len(reviewer_all)}条")

    # ── 按类型过滤 ──
    bus.send(Message(sender="planner", receiver="coder", type="task", content="写代码"))
    bus.send(Message(sender="reviewer", receiver="coder", type="review", content="审核意见"))

    tasks = bus.receive_by_type("coder", "task")
    reviews = bus.receive_by_type("coder", "review")
    assert len(tasks) == 1, f"task 类型应该 1 条，实际 {len(tasks)}"
    assert len(reviews) == 1, f"review 类型应该 1 条，实际 {len(reviews)}"
    print(f"  ✅ 按类型过滤：task={len(tasks)}条, review={len(reviews)}条")

    # ── 增量拉取 ──
    marker = time.time()
    bus.send(Message(sender="planner", receiver="coder", type="task", content="新任务"))
    new_msgs = bus.receive_since("coder", marker)
    assert len(new_msgs) == 1, f"增量消息应该 1 条，实际 {len(new_msgs)}"
    print(f"  ✅ 增量拉取：拿到 {len(new_msgs)} 条时间戳>{marker}的消息")

    print(f"  ✅ 高级功能验证通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 5：统计信息
# ═══════════════════════════════════════════════════════════
def test_stats():
    """
    验证消息总线统计功能正确
    """
    print("=" * 60)
    print("验证 5：统计信息")
    print("=" * 60)

    bus = MessageBus()
    bus.send(Message(sender="planner", receiver="coder", type="task", content="任务1"))
    bus.send(Message(sender="planner", receiver="coder", type="task", content="任务2"))
    bus.send(Message(sender="planner", receiver="reviewer", type="task", content="任务3"))

    stats = bus.stats()
    assert stats["total_messages"] == 3
    assert stats["queues"]["coder"] == 2
    assert stats["queues"]["reviewer"] == 1

    print(f"  📊 {stats}")
    print(f"  ✅ 统计验证通过\n")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Day2 阶段一验证：角色定义 + 消息总线")
    print("=" * 60 + "\n")

    try:
        test_roles()
        test_message_bus_basic()
        test_message_duplication_bug()
        test_advanced_features()
        test_stats()

        print("=" * 60)
        print("  🎉 阶段一全部验证通过！")
        print("  ⚠️ 踩坑已确认：消息重复消费 + 角色 Prompt 边界模糊")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ 验证失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 意外错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
