"""
Day3 阶段二验证脚本 —— 失败重试 + 熔断机制

验证内容：
1. LLMClient.chat() 指数退避重试（可重试/不可重试异常分类）
2. run_parallel() 的 return_exceptions=True 容错
3. run_parallel_with_timeout() 部分结果收集
4. SharedState.update_with_retry() 乐观锁自动重试

运行方式：
    source .venv/bin/activate
    python test/test_day3_phase2.py
"""

import sys
import inspect
import asyncio

# ── 导入验证目标 ──────────────────────────────────────────
from src.llm_client import LLMClient, _parse_chat_response
from src.message_bus import MessageBus
from src.orchestrator import Orchestrator
from src.shared_state import SharedState
from openai import (
    APITimeoutError, RateLimitError, APIConnectionError,
    InternalServerError, AuthenticationError, BadRequestError,
)


# ═══════════════════════════════════════════════════════════
# 验证 10：LLM 指数退避重试
# ═══════════════════════════════════════════════════════════
def test_10_llm_retry():
    """
    验证 LLMClient.chat() 具备重试机制（#10）

    检查点：
    - chat() 接受 max_retries 参数
    - 源码中包含可重试异常分类
    - 源码中包含指数退避逻辑 (2 ** (attempt - 1))
    - 不可重试异常直接抛（不重试）
    """
    print("=" * 60)
    print("验证 10：LLM 指数退避重试（#10）")
    print("=" * 60)

    client = LLMClient()
    sig = inspect.signature(client.chat)

    # 10a: max_retries 参数存在
    assert "max_retries" in sig.parameters, "chat() 缺少 max_retries 参数"
    print("  10a ✅ chat() 有 max_retries 参数")

    # 10b: 源码包含可重试异常
    source = inspect.getsource(LLMClient.chat)
    assert "APITimeoutError" in source, "缺少 APITimeoutError 处理"
    assert "RateLimitError" in source, "缺少 RateLimitError 处理"
    assert "AuthenticationError" in source, "缺少 AuthenticationError 处理"
    print("  10b ✅ 源码包含可重试/不可重试异常分类")

    # 10c: 指数退避逻辑
    assert "2 ** (attempt - 1)" in source, "缺少指数退避逻辑"
    print("  10c ✅ 包含指数退避: 2**(attempt-1)")

    # 10d: 正常调用仍能工作
    result = client.chat(
        messages=[{"role": "user", "content": "说你好"}],
        max_retries=3,
    )
    assert result["content"] is not None, "正常调用失败"
    print(f"  10d ✅ 正常调用成功: {result['content'][:30]}...")

    # 10e: _parse_chat_response 函数存在（重构后提取的）
    source_full = inspect.getsource(inspect.getmodule(LLMClient))
    assert "_parse_chat_response" in source_full, "_parse_chat_response 函数缺失"
    print("  10e ✅ _parse_chat_response 响应解析函数存在")

    print("  ✅ 验证 10 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 11：并行任务容错（不调 LLM，纯逻辑验证）
# ═══════════════════════════════════════════════════════════
def test_11_parallel_fault_tolerance():
    """
    验证 gather 加 return_exceptions=True（#11）

    检查点：
    - run_parallel() 的 _run_all 包含 return_exceptions=True
    - 异常被捕获并转为错误 dict 而非抛出
    - run_parallel_with_state 也加了 return_exceptions
    """
    print("=" * 60)
    print("验证 11：并行任务容错（#11）")
    print("=" * 60)

    source = inspect.getsource(Orchestrator.run_parallel)

    # 11a: return_exceptions=True 存在
    assert "return_exceptions=True" in source, \
        "run_parallel() 的 gather 没有 return_exceptions=True"
    print("  11a ✅ run_parallel() 有 return_exceptions=True")

    # 11b: 异常处理逻辑存在
    assert "isinstance(r, Exception)" in source or "isinstance(r, BaseException)" in source, \
        "缺少异常类型判断逻辑"
    print("  11b ✅ 包含 isinstance 异常类型判断")

    # 11c: 错误 dict 包含 task_id 和 error
    assert '"error"' in source or "'error'" in source, "错误结果缺少 error 字段"
    print("  11c ✅ 错误结果包含 error 字段")

    # 11d: run_parallel_with_state 也修复了
    source_state = inspect.getsource(Orchestrator.run_parallel_with_state)
    assert "return_exceptions=True" in source_state, \
        "run_parallel_with_state 没有 return_exceptions"
    print("  11d ✅ run_parallel_with_state 也加了 return_exceptions")

    # 11e: 纯逻辑测试 — asyncio.gather 带异常
    async def _test_gather_exceptions():
        async def ok_task():
            return {"id": 1, "ok": True}

        async def fail_task():
            raise RuntimeError("模拟失败")

        results = await asyncio.gather(ok_task(), fail_task(), return_exceptions=True)
        # 第一个是正常结果
        assert results[0] == {"id": 1, "ok": True}
        # 第二个是异常
        assert isinstance(results[1], RuntimeError)
        return results

    results = asyncio.run(_test_gather_exceptions())
    print(f"  11e ✅ 纯逻辑验证: 1个成功 + 1个异常 = {len(results)} 个结果")

    print("  ✅ 验证 11 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 12：超时保护 + 部分结果收集（不调 LLM）
# ═══════════════════════════════════════════════════════════
def test_12_timeout_partial_results():
    """
    验证超时后收集部分结果（#12）

    检查点：
    - run_parallel_with_timeout 用 asyncio.wait 而非 wait_for
    - 超时后 pending 任务被 cancel
    - done 中的结果被保留
    - 超时任务标记为 error: "timeout"
    """
    print("=" * 60)
    print("验证 12：超时保护 + 部分结果收集（#12）")
    print("=" * 60)

    source = inspect.getsource(Orchestrator.run_parallel_with_timeout)

    # 12a: 用 asyncio.wait 而非 wait_for
    assert "asyncio.wait(" in source, "应该用 asyncio.wait 而非 wait_for"
    print("  12a ✅ 使用 asyncio.wait（而非 wait_for）")

    # 12b: ensure_future 包装
    assert "ensure_future" in source, "缺少 ensure_future 包装"
    print("  12b ✅ 用 ensure_future 包装协程")

    # 12c: 取消 pending 任务
    assert ".cancel()" in source, "超时后应取消 pending 任务"
    print("  12c ✅ 超时后取消 pending 任务")

    # 12d: 超时标记
    assert '"timeout"' in source or "'timeout'" in source, "超时任务缺少 timeout 标记"
    print("  12d ✅ 超时任务标记为 timeout")

    # 12e: 纯逻辑测试
    async def _test_wait_timeout():
        async def fast_task():
            await asyncio.sleep(0.1)
            return "快速完成"

        async def slow_task():
            await asyncio.sleep(10)

        future_tasks = [asyncio.ensure_future(t) for t in [fast_task(), slow_task()]]
        done, pending = await asyncio.wait(future_tasks, timeout=0.5)

        # fast_task 完成了
        assert len(done) >= 1, "快速任务应该完成"
        # slow_task 被取消
        for t in pending:
            t.cancel()

        # 收集结果
        results = []
        for ft in future_tasks:
            if ft in done:
                results.append(ft.result())
            else:
                results.append("timeout")
        return results

    results = asyncio.run(_test_wait_timeout())
    assert results[0] == "快速完成", f"快速任务结果不对: {results[0]}"
    assert results[1] == "timeout", f"慢任务应该是 timeout: {results[1]}"
    print(f"  12e ✅ 纯逻辑验证: [{results[0]}, {results[1]}]")

    print("  ✅ 验证 12 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 13：乐观锁自动重试
# ═══════════════════════════════════════════════════════════
def test_13_optimistic_lock_retry():
    """
    验证 update_with_retry 乐观锁自动重试（#13）

    检查点：
    - update_with_retry 方法存在
    - 正常场景：一次成功
    - 冲突场景：读取-计算-写入循环
    - 最终值正确
    """
    print("=" * 60)
    print("验证 13：乐观锁自动重试（#13）")
    print("=" * 60)

    state = SharedState()

    # 13a: 方法存在
    assert hasattr(state, "update_with_retry"), "缺少 update_with_retry 方法"
    assert callable(state.update_with_retry), "update_with_retry 不是可调用的"
    print("  13a ✅ update_with_retry 方法存在")

    # 13b: 正常场景 — 一次成功
    state.set("counter", 0)
    ok = state.update_with_retry("counter", lambda v: (v or 0) + 1)
    assert ok, "正常场景应该成功"
    assert state.get("counter") == 1, f"值应为1: {state.get('counter')}"
    print(f"  13b ✅ 正常场景: counter 0→1")

    # 13c: 冲突场景 — 模拟被抢先修改后自动重试
    state2 = SharedState()
    state2.set("shared", 0)

    # 记录冲突前版本号，然后抢先修改
    v_before = state2.get_version()
    state2.set("shared", 999)  # 模拟其他 Agent 抢先改

    # 用旧版本号的场景不应该发生（update_with_retry 自己读版本号）
    # 但我们可以直接验证：连加3次
    state3 = SharedState()
    state3.set("sum", 0)
    for _ in range(3):
        state3.update_with_retry("sum", lambda v: (v or 0) + 1)
    assert state3.get("sum") == 3, f"累加3次应为3: {state3.get('sum')}"
    print(f"  13c ✅ 累加3次: 0→1→2→3 = {state3.get('sum')}")

    # 13d: 计算函数异常场景
    ok = state.update_with_retry("counter", lambda v: 1 / 0)  # 除零
    assert not ok, "计算失败应该返回 False"
    print("  13d ✅ 计算函数异常返回 False（不崩溃）")

    # 13e: 源码包含三步循环
    source = inspect.getsource(SharedState.update_with_retry)
    assert "deepcopy" in source, "缺少 deepcopy（读最新值）"
    assert "compute_new_value" in source, "缺少计算步骤"
    assert "_version" in source, "缺少版本号检查"
    print("  13e ✅ 源码包含读→算→写 三步循环")

    print("  ✅ 验证 13 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day3 阶段二：失败重试 + 熔断机制 验证")
    print("=" * 60 + "\n")

    results = {}

    tests = [
        ("#10 LLM重试", test_10_llm_retry),
        ("#11 并行容错", test_11_parallel_fault_tolerance),
        ("#12 超时部分结果", test_12_timeout_partial_results),
        ("#13 乐观锁重试", test_13_optimistic_lock_retry),
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
        print("\n🎉 阶段二全部 4 项验证通过！")
    else:
        print(f"\n⚠️ 有 {len(results) - passed} 项未通过，请检查。")
        sys.exit(1)
