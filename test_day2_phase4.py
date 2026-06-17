"""
Day2 阶段四验证脚本 —— 全局状态管理

验证内容：
1. SharedState 基础读写
2. 乐观锁版本控制
3. 【踩坑】bypass_lock 绕锁修改
4. 【踩坑】版本号冲突
5. 串行编排 + 状态追踪（真实 LLM）
6. 并行编排 + 状态追踪（真实 LLM）
7. 并发冲突演示（纯逻辑）
8. 深拷贝隔离验证

运行方式：
    source .venv/bin/activate
    python test_day2_phase4.py
"""

import sys
import time
import threading
from src.shared_state import SharedState
from src.message_bus import MessageBus
from src.orchestrator import Orchestrator


# ═══════════════════════════════════════════════════════════
# 验证 1：SharedState 基础读写
# ═══════════════════════════════════════════════════════════
def test_basic_read_write():
    """SharedState 基本的 set/get 功能"""
    print("=" * 60)
    print("验证 1：SharedState 基础读写")
    print("=" * 60)

    state = SharedState()

    # 写入并读取
    v = state.set("name", "agent-from-scratch-plus")
    assert state.get("name") == "agent-from-scratch-plus"
    print(f"  set('name', ...) → get('name') = '{state.get('name')}'")

    # 写多个键
    state.set("version", 1)
    state.set("tags", ["agent", "python", "multi-agent"])
    keys = state.keys()
    print(f"  写入 3 个键: {keys}")
    assert len(keys) == 3

    # 检查键存在
    assert state.has("name")
    assert not state.has("nonexistent")

    # 默认值
    assert state.get("missing", "default") == "default"

    # 版本号递增
    assert v > 0
    print(f"  初始 set 返回版本号: {v}")
    print(f"  当前版本号: {state.get_version()}")

    print(f"  ✅ 基础读写正常\n")


# ═══════════════════════════════════════════════════════════
# 验证 2：乐观锁版本控制
# ═══════════════════════════════════════════════════════════
def test_optimistic_lock():
    """乐观锁：版本匹配则更新，不匹配则拒绝"""
    print("=" * 60)
    print("验证 2：乐观锁版本控制")
    print("=" * 60)

    state = SharedState()

    # 正常更新
    v = state.set("data", "v1")
    ok = state.update("data", "v2", expected_version=v)
    assert ok, "版本匹配应该成功"
    assert state.get("data") == "v2"
    print(f"  正常 update: data='{state.get('data')}' ✅")

    # 版本冲突：用旧版本号更新
    ok = state.update("data", "v3", expected_version=v)  # v 是旧版本
    assert not ok, "版本冲突应该返回 False"
    assert state.get("data") == "v2"  # 值不变
    print(f"  冲突 update（旧版本号 {v}）: 拒绝 ✅, data 仍是 '{state.get('data')}'")

    # 两个 Agent 的并发场景模拟
    v_a = state.set("shared", "A写的")
    # Agent B 抢先改了
    state.set("shared", "B抢先改的")
    # Agent A 用旧版本更新
    ok = state.update("shared", "A又写了一次", expected_version=v_a)
    assert not ok
    assert state.get("shared") == "B抢先改的"
    print(f"  并发冲突模拟: A 的更新被拒绝 ✅, 最终值='{state.get('shared')}'")

    print(f"  冲突计数: {state.stats()['conflict_count']}")
    print(f"  ✅ 乐观锁正常\n")


# ═══════════════════════════════════════════════════════════
# 验证 3：【踩坑】bypass_lock 绕锁修改
# ═══════════════════════════════════════════════════════════
def test_bypass_lock_pitfall():
    """绕过锁直接修改 —— 版本号不递增，乐观锁失效"""
    print("=" * 60)
    print("验证 3：【踩坑】bypass_lock 绕锁修改")
    print("=" * 60)

    state = SharedState()

    state.set("protected", "安全写入")
    v_before = state.get_version()
    print(f"  安全 set: protected='安全写入', 版本={v_before}")

    # 绕过锁直接修改
    state.bypass_lock("protected", "被悄悄覆盖了")
    v_after = state.get_version()
    print(f"  bypass_lock 后: protected='{state.get('protected')}', 版本={v_after}")

    assert state.get("protected") == "被悄悄覆盖了"
    assert v_before == v_after, "绕过锁不应该递增版本号！"
    print(f"  ⚠️ 值被覆盖，但版本号 {v_before} → {v_after}（未变）")
    print(f"  ⚠️ 乐观锁检测不到这次修改，相当于绕过了所有保护")

    # 此时如果用旧版本号 update → 会成功（误以为没变化）
    ok = state.update("protected", "乐观锁也挡不住", expected_version=v_before)
    print(f"  用旧版本 {v_before} update → {'成功(锁被绕过!)' if ok else '失败'}")
    print(f"  ⚠️ 乐观锁阻止不了 bypass_lock 的修改")
    print(f"  ⚠️ 对应真实场景：Agent 拿到 _data 引用直接 dict[key]=value")

    print(f"  ✅ 踩坑确认\n")


# ═══════════════════════════════════════════════════════════
# 验证 4：深拷贝隔离
# ═══════════════════════════════════════════════════════════
def test_deep_copy_isolation():
    """get() 返回深拷贝，外部修改不影响内部状态"""
    print("=" * 60)
    print("验证 4：深拷贝隔离")
    print("=" * 60)

    state = SharedState()

    # 写入一个列表
    original = [1, 2, 3]
    state.set("items", original)

    # 取出后修改
    retrieved = state.get("items")
    retrieved.append(999)
    print(f"  外部修改后: {retrieved}")

    # 内部应不变
    internal = state.get("items")
    assert internal == [1, 2, 3], f"内部被污染了: {internal}"
    print(f"  内部仍为: {internal} ✅")

    # 嵌套 dict 的隔离
    state.set("config", {"db": {"host": "localhost", "port": 5432}})
    config = state.get("config")
    config["db"]["host"] = "evil.com"
    assert state.get("config")["db"]["host"] == "localhost"
    print(f"  嵌套 dict 隔离: 内部 host 仍是 'localhost' ✅")

    print(f"  ✅ 深拷贝隔离正常\n")


# ═══════════════════════════════════════════════════════════
# 验证 5：并发冲突演示（Orchestrator 集成）
# ═══════════════════════════════════════════════════════════
def test_orchestrator_concurrent_conflict():
    """通过 Orchestrator 演示并发冲突场景"""
    print("=" * 60)
    print("验证 5：Orchestrator 并发冲突演示")
    print("=" * 60)

    bus = MessageBus()
    orch = Orchestrator(bus, with_shared_state=True)

    assert orch.shared_state is not None, "应创建 SharedState"
    assert orch.planner.shared_state is not None, "Agent 应有共享状态"
    assert orch.coder.shared_state is not None, "Agent 应有共享状态"
    assert orch.reviewer.shared_state is not None, "Agent 应有共享状态"

    results = orch.demonstrate_concurrent_conflict()

    assert results["optimistic_ok"], "正常乐观锁应成功"
    assert results["version_conflict"], "版本冲突应被检测到"
    assert results["bypass_pitfall"], "绕过锁应成功修改（踩坑）"

    print(f"\n  最终状态: {orch.shared_state}")
    print(f"  ✅ 并发冲突演示通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 6：串行编排 + 状态追踪（真实 LLM）
# ═══════════════════════════════════════════════════════════
def test_serial_with_state():
    """完整串行编排 + SharedState 进度追踪"""
    print("=" * 60)
    print("验证 6：串行编排 + 状态追踪（真实 LLM）")
    print("=" * 60)

    bus = MessageBus()
    orch = Orchestrator(bus, with_shared_state=True)

    result = orch.run_serial_with_state("实现一个函数，判断字符串是否为回文")

    # 验证返回结构
    assert result["plan"], "plan 不应为空"
    assert result["code"], "code 不应为空"
    assert result["review"], "review 不应为空"
    assert result["fixed_code"], "fixed_code 不应为空"

    # 验证状态追踪
    snapshot = result["state_snapshot"]
    assert "requirement" in snapshot
    assert "plan" in snapshot
    assert "code" in snapshot
    assert "review" in snapshot
    assert "progress" in snapshot
    assert snapshot.get("phase") == "done"

    progress = orch.shared_state.get_progress()
    assert progress["total"] == 4
    assert progress["completed"] == 4, f"应完成 4 步，实际 {progress['completed']}"

    print(f"\n  阶段: {snapshot['phase']}")
    print(f"  进度: {progress['completed']}/{progress['total']}")
    print(f"  状态键: {list(snapshot.keys())}")
    print(f"  状态统计: {orch.shared_state.stats()}")

    print(f"  ✅ 串行编排 + 状态追踪通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 7：并行编排 + 状态追踪（真实 LLM）
# ═══════════════════════════════════════════════════════════
def test_parallel_with_state():
    """并行编排 + SharedState 进度追踪"""
    print("=" * 60)
    print("验证 7：并行编排 + 状态追踪（真实 LLM）")
    print("=" * 60)

    bus = MessageBus()
    orch = Orchestrator(bus, with_shared_state=True)

    result = orch.run_parallel_with_state(
        "实现以下两个独立功能：1) 计算斐波那契数列 2) 判断素数",
        n_coders=2,
    )

    # 验证返回结构
    assert len(result["codes"]) == 2, f"期望 2 份代码，实际 {len(result['codes'])}"
    assert result["combined"], "combined 不应为空"

    # 验证状态追踪
    snapshot = result["state_snapshot"]
    assert snapshot.get("phase") == "done"
    assert "code_task1" in snapshot or "code_task2" in snapshot  # 至少记录了一份代码

    progress = orch.shared_state.get_progress()
    assert progress["total"] == 3
    assert progress["completed"] == 3

    print(f"\n  阶段: {snapshot['phase']}")
    print(f"  进度: {progress['completed']}/{progress['total']}")
    print(f"  记录的结果: {len(progress['results'])} 条")

    print(f"  ✅ 并行编排 + 状态追踪通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 8：【踩坑】Agent 直接改全局状态
# ═══════════════════════════════════════════════════════════
def test_agent_direct_modification():
    """
    【踩坑验证】Agent 不通过 SharedState API，而是直接持有引用修改

    对应真实场景：各 Agent 共享一个 dict，直接 self._data[key] = value
    不加锁，不检查版本，导致 A 的修改被 B 覆盖。
    """
    print("=" * 60)
    print("验证 8：【踩坑】Agent 直接改全局 dict")
    print("=" * 60)

    # ── 模拟全局 dict 被多个 Agent 持有的场景 ──
    # 这是刻意踩坑：不用 SharedState，用裸 dict
    global_data = {"counter": 0, "logs": []}

    def agent_a_work():
        """Agent A 读 counter、计算、写回"""
        # 模拟读到 counter
        local = global_data["counter"]
        # 模拟耗时操作（API 调用等）
        time.sleep(0.2)
        # 写回 —— 但 B 可能已经改了！
        global_data["counter"] = local + 1
        global_data["logs"].append("Agent A 写入了")

    def agent_b_work():
        """Agent B 同样逻辑，但速度更快"""
        local = global_data["counter"]
        # B 比较快
        time.sleep(0.05)
        global_data["counter"] = local + 1
        global_data["logs"].append("Agent B 写入了")

    # 启动两个线程模拟并发
    t_a = threading.Thread(target=agent_a_work)
    t_b = threading.Thread(target=agent_b_work)

    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    # 期望：两个 Agent 各 +1，结果应该是 2
    expected = 2
    actual = global_data["counter"]
    print(f"  期望 counter: {expected}")
    print(f"  实际 counter: {actual}")
    print(f"  操作日志: {global_data['logs']}")

    if actual != expected:
        print(f"  ⚠️ 【踩坑确认】状态错乱！counter={actual}，应该是 {expected}")
        print(f"      原因：两个 Agent 读-改-写 没有加锁，A 的写入覆盖了 B 的写入")
    else:
        print(f"  ℹ️  本次恰好没冲突（但无锁保护，随时可能出错）")
        # 手动制造冲突场景来演示
        print(f"  💡 如果 A 和 B 的耗时差异更大，就会看到覆盖现象")

    # ── 对比：用 SharedState 的正确方式 ──
    print(f"\n  对比：用 SharedState 的正确方式")
    state = SharedState()
    state.set("safe_counter", 0)

    # Agent A：乐观锁更新
    v_a = state.get_version()
    val_a = state.get("safe_counter")
    time.sleep(0.2)  # A 耗时
    ok_a = state.update("safe_counter", val_a + 1, expected_version=v_a)
    print(f"  Agent A: get()={val_a}, 耗时 0.2s, update → {'成功' if ok_a else '版本冲突'}")

    # Agent B：乐观锁更新（更快）
    v_b = state.get_version()
    val_b = state.get("safe_counter")
    time.sleep(0.05)  # B 更快
    ok_b = state.update("safe_counter", val_b + 1, expected_version=v_b)
    print(f"  Agent B: get()={val_b}, 耗时 0.05s, update → {'成功' if ok_b else '版本冲突'}")

    # 结果分析
    final_val = state.get("safe_counter")
    print(f"  最终 safe_counter: {final_val}")
    if ok_a and ok_b:
        print(f"  ✅ 两个都成功（恰好没冲突）")
    elif not ok_a or not ok_b:
        print(f"  ⚠️ 乐观锁检测到冲突！需要重试机制（Day3 实现）")
        print(f"     但至少数据是安全的，不会出现覆盖丢失")

    print(f"  ✅ 踩坑对比完成\n")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Day2 阶段四验证：全局状态管理")
    print("=" * 60 + "\n")

    failed = []

    # ── 结构测试（不需要网络）──
    for name, fn in [
        ("基础读写", test_basic_read_write),
        ("乐观锁", test_optimistic_lock),
        ("bypass_lock踩坑", test_bypass_lock_pitfall),
        ("深拷贝隔离", test_deep_copy_isolation),
        ("并发冲突演示", test_orchestrator_concurrent_conflict),
        ("Agent直接改全局dict", test_agent_direct_modification),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"\n  ⚠️ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed.append((name, e))

    # ── LLM 测试（需要网络）──
    for name, fn in [
        ("串行编排+状态追踪", test_serial_with_state),
        ("并行编排+状态追踪", test_parallel_with_state),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"\n  ⚠️ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed.append((name, e))

    # ── 总结 ──
    print("=" * 60)
    if failed:
        print(f"  ⚠️ {len(failed)} 项测试失败:")
        for name, err in failed:
            print(f"     - {name}: {err}")
        sys.exit(1)
    else:
        print("  🎉 阶段四全部验证通过！")
        print("  ⚠️ 踩坑已确认：绕过锁修改 + 版本号冲突 + Agent 直接改全局 dict")
    print("=" * 60)
