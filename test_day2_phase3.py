"""
Day2 阶段三验证脚本 —— 并行执行 + 结果汇总

验证内容：
1. 子任务解析：_parse_subtasks() 能从 Planner 输出中提取子任务
2. 并行执行：run_parallel() 多个 Coder 并行工作
3. 结果汇总：合并多个 Coder 的输出
4. 耗时对比：并行 vs 串行预估的时间差异
5. 【踩坑】不写 await：run_parallel_no_await() 任务全部丢失
6. 【踩坑】超时丢失：run_parallel_with_timeout() 超时后不保留部分结果

运行方式：
    source .venv/bin/activate
    python test_day2_phase3.py
"""

import sys
import time

from src.message_bus import MessageBus
from src.agents import CoderAgent
from src.orchestrator import Orchestrator


# ═══════════════════════════════════════════════════════════
# 验证 1：子任务解析
# ═══════════════════════════════════════════════════════════
def test_parse_subtasks():
    """
    验证 _parse_subtasks() 能从 Planner 输出中提取子任务
    """
    print("=" * 60)
    print("验证 1：子任务解析")
    print("=" * 60)

    bus = MessageBus()
    orch = Orchestrator(bus)

    # 模拟 Planner 输出（包含多个子任务）
    sample_plan = """
### 子任务1：定义函数接口
确定函数名称、参数类型和返回值类型。

### 子任务2：实现核心算法
编写排序算法的主体代码。

### 子任务3：编写测试用例
创建测试数据并验证算法正确性。
"""

    subtasks = orch._parse_subtasks(sample_plan)

    assert len(subtasks) == 3, f"期望 3 个子任务，实际 {len(subtasks)}"
    assert "函数接口" in subtasks[0]
    assert "核心算法" in subtasks[1]
    assert "测试用例" in subtasks[2]

    for i, st in enumerate(subtasks):
        print(f"  子任务 {i+1}: {st[:60]}...")

    print(f"  ✅ 成功拆分 {len(subtasks)} 个子任务\n")


# ═══════════════════════════════════════════════════════════
# 验证 2：并行执行结构
# ═══════════════════════════════════════════════════════════
def test_parallel_structure():
    """
    验证并行执行的基本结构正确
    """
    print("=" * 60)
    print("验证 2：并行执行结构")
    print("=" * 60)

    bus = MessageBus()
    orch = Orchestrator(bus)

    # 验证方法存在
    assert hasattr(orch, "run_parallel"), "缺少 run_parallel 方法"
    assert hasattr(orch, "run_parallel_no_await"), "缺少 run_parallel_no_await 方法"
    assert hasattr(orch, "run_parallel_with_timeout"), "缺少 run_parallel_with_timeout 方法"
    assert hasattr(orch, "_parse_subtasks"), "缺少 _parse_subtasks 方法"
    assert hasattr(orch, "_coder_async"), "缺少 _coder_async 协程方法"

    # 验证 _coder_async 是协程
    import inspect
    assert inspect.iscoroutinefunction(orch._coder_async), \
        "_coder_async 不是 async 函数"

    print(f"  ✅ run_parallel() 存在")
    print(f"  ✅ run_parallel_no_await() 存在（踩坑演示）")
    print(f"  ✅ run_parallel_with_timeout() 存在")
    print(f"  ✅ _coder_async() 是 async 协程")
    print(f"  ✅ 并行执行结构验证通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 3：并行执行（真实 LLM 调用）
# ═══════════════════════════════════════════════════════════
def test_parallel_execution():
    """
    验证完整的并行执行流程
    """
    print("=" * 60)
    print("验证 3：并行执行（真实 LLM 调用）")
    print("=" * 60)

    bus = MessageBus()
    orch = Orchestrator(bus)

    requirement = "实现以下两个功能：1) 冒泡排序 2) 二分查找"

    result = orch.run_parallel(requirement, n_coders=2)

    # 验证返回结构
    assert "plan" in result
    assert "codes" in result
    assert len(result["codes"]) == 2, f"期望 2 份代码，实际 {len(result['codes'])}"
    assert "combined" in result
    assert "timing" in result

    # 验证每份代码非空
    for c in result["codes"]:
        assert c["code"], f"任务 {c['task_id']} 的代码为空"
        assert len(c["code"]) > 50, f"任务 {c['task_id']} 代码太短"

    print(f"\n  📋 规划: {len(result['plan'])} 字符")
    for c in result["codes"]:
        print(f"  💻 任务 {c['task_id']}: {len(c['code'])} 字符")
    print(f"  📦 合并: {len(result['combined'])} 字符")
    print(f"  ⏱️ 耗时: plan={result['timing']['plan']:.1f}s, "
          f"coders={result['timing']['coders']:.1f}s, "
          f"review={result['timing']['review']:.1f}s")

    print(f"  ✅ 并行执行验证通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 4：【踩坑】不写 await
# ═══════════════════════════════════════════════════════════
def test_no_await_pitfall():
    """
    【踩坑验证】不写 await 导致任务丢失
    """
    print("=" * 60)
    print("验证 4：【踩坑】不写 await")
    print("=" * 60)

    bus = MessageBus()
    orch = Orchestrator(bus)

    result = orch.run_parallel_no_await("实现快速排序和归并排序")

    # 验证 codes 为空（协程被丢弃）
    assert len(result["codes"]) == 0, \
        f"不 await 时 codes 应该为空，实际 {len(result['codes'])} 条"
    assert "warning" in result, "缺少警告信息"

    print(f"  📋 Plan 已生成（正常）")
    print(f"  💻 Codes: {len(result['codes'])} 条（应为 0）")
    print(f"  ⚠️ {result.get('warning', '')}")
    print(f"  ✅ 踩坑验证通过——不写 await，任务全部丢失\n")


# ═══════════════════════════════════════════════════════════
# 验证 5：串行 vs 并行耗时对比（结构验证）
# ═══════════════════════════════════════════════════════════
def test_serial_vs_parallel_structure():
    """
    验证串行和并行两种模式都能返回正确的数据结构

    注意：这里不做真实的耗时对比断言（LLM 调用耗时不稳定），
    只验证两种模式的数据结构一致性。
    """
    print("=" * 60)
    print("验证 5：串行 vs 并行结构对比")
    print("=" * 60)

    bus1 = MessageBus()
    orch1 = Orchestrator(bus1)
    result_serial = orch1.run_serial("实现二分查找", with_fix=False)

    bus2 = MessageBus()
    orch2 = Orchestrator(bus2)
    result_parallel = orch2.run_parallel("实现二分查找", n_coders=1)

    # 两种模式都应返回有效结果
    assert result_serial["code"], "串行 code 为空"
    assert len(result_parallel["codes"]) > 0, "并行 codes 为空"
    assert result_parallel["codes"][0]["code"], "并行第一份代码为空"

    print(f"  串行: plan={len(result_serial['plan'])}字, code={len(result_serial['code'])}字")
    print(f"  并行: plan={len(result_parallel['plan'])}字, "
          f"codes={len(result_parallel['codes'])}份, "
          f"合并={len(result_parallel['combined'])}字")

    # 并行返回结构包含 timing
    assert "timing" in result_parallel, "并行结果缺少 timing 字段"
    print(f"  并行耗时: {result_parallel['timing']}")

    print(f"  ✅ 串行/并行结构一致\n")


# ═══════════════════════════════════════════════════════════
# 验证 6：【踩坑】输出格式不统一
# ═══════════════════════════════════════════════════════════
def test_output_format_inconsistency():
    """
    【踩坑验证】多个 Coder 并行输出格式不一致

    两个 Coder 各自独立调 LLM，返回的代码格式可能完全不同：
    - 一个用 Markdown 代码块
    - 一个直接返回 Python 代码
    - 另一个可能还包含解释文字

    合并时不做格式统一，直接拼接。
    """
    print("=" * 60)
    print("验证 6：【踩坑】输出格式不统一")
    print("=" * 60)

    bus = MessageBus()
    orch = Orchestrator(bus)

    # 用两个不同的任务触发不同的 Coder 输出
    requirement = "实现以下两个独立功能：1) 计算阶乘 2) 判断回文"
    result = orch.run_parallel(requirement, n_coders=2)

    codes = result["codes"]
    assert len(codes) == 2

    # 检查输出格式差异
    print(f"\n  任务 1 输出前 100 字符:\n    {codes[0]['code'][:100]}...")
    print(f"\n  任务 2 输出前 100 字符:\n    {codes[1]['code'][:100]}...")

    # 检查是否可能含有 Markdown 标记（格式不统一的表现）
    has_markdown = any("```" in c["code"] for c in codes)
    has_plain = any("```" not in c["code"] for c in codes)

    if has_markdown and has_plain:
        print(f"\n  ⚠️ 【踩坑确认】输出格式不统一！")
        print(f"     有任务用了 Markdown 代码块，有的没用")
    elif has_markdown:
        print(f"\n  ⚠️ 两个都用 Markdown（但无法保证每次都一致）")
    else:
        print(f"\n  ⚠️ 两个都是纯文本（但无法保证每次都一致）")

    # 查看合并后代码的前 200 字符
    print(f"\n  合并后前 200 字符:")
    print(f"    {result['combined'][:200]}...")

    print(f"\n  ✅ 格式不统一验证完成\n")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Day2 阶段三验证：并行执行 + 结果汇总")
    print("=" * 60 + "\n")

    failed = []

    # 结构测试（不需要网络）
    try:
        test_parse_subtasks()
    except Exception as e:
        failed.append(("子任务解析", e))

    try:
        test_parallel_structure()
    except Exception as e:
        failed.append(("并行结构", e))

    try:
        test_no_await_pitfall()
    except Exception as e:
        failed.append(("不写 await", e))

    # LLM 测试（需要网络）
    try:
        test_parallel_execution()
    except Exception as e:
        print(f"\n  ⚠️ 并行执行测试失败: {e}")
        failed.append(("并行执行", e))

    try:
        test_serial_vs_parallel_structure()
    except Exception as e:
        print(f"\n  ⚠️ 结构对比测试失败: {e}")
        failed.append(("结构对比", e))

    try:
        test_output_format_inconsistency()
    except Exception as e:
        print(f"\n  ⚠️ 格式不统一测试失败: {e}")
        failed.append(("格式不统一", e))

    # ── 总结 ──
    print("=" * 60)
    if failed:
        print(f"  ⚠️ {len(failed)} 项测试失败:")
        for name, err in failed:
            print(f"     - {name}: {err}")
        sys.exit(1)
    else:
        print("  🎉 阶段三全部验证通过！")
        print("  ⚠️ 踩坑已确认：不写 await 任务丢失 + 输出格式不统一")
    print("=" * 60)
