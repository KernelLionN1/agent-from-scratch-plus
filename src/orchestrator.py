"""
编排器 —— Day2 阶段二 + 阶段三

核心职责：
- 阶段二：串行编排 — Planner → Coder → Reviewer → Coder(修复)
- 阶段三：并行编排 — 多个 Coder 并行处理独立子任务

刻意踩坑点：
- 【踩坑】执行顺序混乱（阶段二）：不设依赖检查，
  Reviewer 可能在 Coder 完成前就尝试审核。
- 【踩坑】无等待逻辑（阶段三）：并行时不写 await，
  主线程直接退出——阶段三才加这个坑。

类比 Java：
    Orchestrator 相当于一个有状态的 Service，
    编排多个 Agent 的执行流程，类似工作流引擎。
"""

from src.message_bus import MessageBus, Message
from src.agents import PlannerAgent, CoderAgent, ReviewerAgent
from src.shared_state import SharedState  # 阶段四新增

# 阶段三新增：异步并行支持
import asyncio
import re


class Orchestrator:
    """
    多 Agent 编排器

    负责协调 Planner → Coder → Reviewer 的完整流程。

    使用方式：
        bus = MessageBus()
        orch = Orchestrator(bus)
        result = orch.run_serial("实现一个排序算法")
    """

    def __init__(self, bus: MessageBus, with_shared_state: bool = False):
        """
        初始化编排器

        参数：
            bus:               共享的消息总线
            with_shared_state: 是否启用全局状态管理（阶段四）
        """
        self.bus = bus

        # 创建三个 Agent 实例
        # 所有 Agent 共享同一个 MessageBus
        self.planner = PlannerAgent(bus, "planner-1")
        self.coder = CoderAgent(bus, "coder-1")
        self.reviewer = ReviewerAgent(bus, "reviewer-1")

        # ── 共享状态 ──（阶段四新增）
        self.shared_state: SharedState | None = None
        if with_shared_state:
            self.shared_state = SharedState()
            # 注入到所有 Agent
            for agent in [self.planner, self.coder, self.reviewer]:
                agent.shared_state = self.shared_state

        # 执行统计
        self.stats = {
            "plan_chars": 0,      # Planner 输出字符数
            "code_chars": 0,      # Coder 输出字符数
            "review_chars": 0,    # Reviewer 输出字符数
            "fixed_code_chars": 0,# 修复后代码字符数
        }

    # ═══════════════════════════════════════════════════════
    # 阶段二：串行编排
    # ═══════════════════════════════════════════════════════

    def run_serial(self, requirement: str, with_fix: bool = True) -> dict:
        """
        串行执行完整的多 Agent 流程

        流程：
            用户需求
               ↓
            Planner → 拆分子任务
               ↓
            Coder   → 生成代码
               ↓
            Reviewer → 审核代码
               ↓
            Coder   → 根据审核意见修复代码（可选）
               ↓
            最终结果

        参数：
            requirement: 用户的原始需求
            with_fix:    是否在审核后执行修复步骤

        返回：
            {
                "plan":      规划结果,
                "code":      生成的代码,
                "review":    审核意见,
                "fixed_code": 修复后的代码（如果 with_fix=True）,
                "stats":     执行统计,
            }
        """
        print("\n" + "=" * 60)
        print(f"🚀 开始串行编排：{requirement[:50]}...")
        print("=" * 60)

        # ── 步骤 1：Planner 拆分需求 ──
        print("\n📋 步骤 1/4：Planner 拆分需求...")
        plan = self.planner.execute(requirement)
        self.stats["plan_chars"] = len(plan)
        print(f"   Planner 输出: {len(plan)} 字符")
        print(f"   规划摘要: {plan[:200]}...")

        # ── 步骤 2：Coder 生成代码 ──
        print("\n💻 步骤 2/4：Coder 生成代码...")
        # 【踩坑】直接把 Planner 的输出喂给 Coder，不做格式校验
        code = self.coder.execute(plan)
        self.stats["code_chars"] = len(code)
        print(f"   Coder 输出: {len(code)} 字符")

        # ── 步骤 3：Reviewer 审核代码 ──
        print("\n🔍 步骤 3/4：Reviewer 审核代码...")
        # 【踩坑】不检查 Coder 是否真的完成了，
        # 直接让 Reviewer 审核——如果 code 是空的也照样审
        review = self.reviewer.execute(code)
        self.stats["review_chars"] = len(review)
        print(f"   Reviewer 输出: {len(review)} 字符")

        # ── 步骤 4：Coder 修复代码 ──
        fixed_code = ""
        if with_fix:
            print("\n🔧 步骤 4/4：Coder 根据审核意见修复代码...")
            fixed_code = self.coder.fix_code(review)
            self.stats["fixed_code_chars"] = len(fixed_code)
            print(f"   修复后代码: {len(fixed_code)} 字符")

        print("\n" + "=" * 60)
        print("✅ 串行编排完成")
        print("=" * 60)

        return {
            "plan": plan,
            "code": code,
            "review": review,
            "fixed_code": fixed_code,
            "stats": dict(self.stats),
        }

    def run_serial_with_bus(self, requirement: str) -> dict:
        """
        串行编排 —— 通过消息总线传递（接近真实多 Agent 通信）

        和 run_serial 的区别：
        - run_serial：Orchestrator 直接调用 Agent 的方法
        - run_serial_with_bus：通过 MessageBus 传递，Agent 从总线读取

        用于对比「直接调用」和「消息通信」两种方式。
        """
        print("\n" + "=" * 60)
        print(f"🚀 开始串行编排（消息总线模式）：{requirement[:50]}...")
        print("=" * 60)

        # ── Planner ──
        self.planner.plan_and_dispatch(requirement, receiver="coder-1")

        # ── Coder ──
        tasks = self.coder.receive_tasks()
        if tasks:
            code = self.coder.execute(tasks[-1].content)
            self.coder.send_message("reviewer-1", "code", code)
        else:
            print("  ⚠️ Coder 没有收到任务")

        # ── Reviewer ──
        code_msgs = [
            m for m in self.reviewer.receive_messages()
            if m.type == "code"
        ]
        if code_msgs:
            review = self.reviewer.execute(code_msgs[-1].content)
            self.reviewer.send_message("coder-1", "review", review)
        else:
            print("  ⚠️ Reviewer 没有收到代码")

        # ── 读取审核意见 ──
        reviews = self.coder.receive_reviews()
        fixed_code = ""
        if reviews:
            fixed_code = self.coder.fix_code(reviews[-1].content)

        return {
            "plan": tasks[-1].content if tasks else "",
            "code": code_msgs[-1].content if code_msgs else "",
            "review": reviews[-1].content if reviews else "",
            "fixed_code": fixed_code,
        }

    # ═══════════════════════════════════════════════════════
    # 阶段三：并行执行
    # ═══════════════════════════════════════════════════════

    def _parse_subtasks(self, plan_text: str) -> list[str]:
        """
        从 Planner 输出中粗略提取子任务

        参数：
            plan_text: Planner 的输出文本

        返回：
            子任务描述列表

        【踩坑】这里用最简单的正则匹配，没有做严格的格式解析。
        Planner 输出的格式稍有变化就会解析失败。
        """
        subtasks = []

        # 尝试匹配「### 子任务 N：标题」或「**子任务 N**」格式
        # 用正则按标题分割
        parts = re.split(r'(?:###?\s*子任务\s*\d+|【子任务\d】)', plan_text)

        if len(parts) <= 1:
            # 如果正则没匹配到，尝试按双换行分割作为兜底
            parts = plan_text.split("\n\n")

        for part in parts:
            part = part.strip()
            # 跳过太短的内容（可能是标题残留）
            if len(part) > 30:
                subtasks.append(part)

        # 如果没拆出来子任务，就把整个 plan 作为一个任务
        if not subtasks:
            subtasks = [plan_text]

        print(f"   📦 拆分出 {len(subtasks)} 个子任务")
        return subtasks

    async def _coder_async(self, coder: CoderAgent, subtask: str, task_id: int) -> dict:
        """
        异步执行单个 Coder 任务

        参数：
            coder:   Coder Agent 实例
            subtask: 子任务描述
            task_id: 任务编号（用于追踪）

        返回：
            {"task_id": int, "subtask": str, "code": str}

        【踩坑】LLM 调用本身是同步的，用 asyncio.to_thread() 包装。
        真正的异步 LLM 调用应该用 aiohttp/httpx 的异步客户端。
        """
        print(f"   🧵 任务 {task_id} 开始...")
        # to_thread: 把同步调用丢到线程池，不阻塞事件循环
        # 相当于 Java 的 CompletableFuture.supplyAsync()
        code = await asyncio.to_thread(coder.execute, subtask)
        print(f"   ✅ 任务 {task_id} 完成 ({len(code)} 字符)")
        return {
            "task_id": task_id,
            "subtask": subtask,
            "code": code,
        }

    def run_parallel(self, requirement: str, n_coders: int = 2, timeout: float = 120) -> dict:
        """
        并行执行：Planner 拆分 → 多个 Coder 并行处理 → 汇总结果

        流程：
            用户需求
               ↓
            Planner → 拆分成 N 个子任务
               ↓
            ┌──────┼──────┐
            ↓      ↓      ↓
          Coder1 Coder2 Coder3  ← 并行执行
            ↓      ↓      ↓
            └──────┼──────┘
               ↓
            Reviewer → 审核汇总后的代码
               ↓
            最终结果

        参数：
            requirement: 用户的原始需求
            n_coders:    并行工作的 Coder 数量
            timeout:     整体超时秒数

        返回：
            {
                "plan":     规划结果,
                "codes":    [{task_id, subtask, code}, ...],
                "combined": 合并后的代码,
                "review":   审核意见,
                "timing":   {plan, coders, review} 耗时统计,
            }
        """
        import time

        print("\n" + "=" * 60)
        print(f"🚀 开始并行编排：{requirement[:50]}...")
        print("=" * 60)

        timing = {}

        # ── 步骤 1：Planner 拆分（串行，只有一次 LLM 调用）──
        print("\n📋 步骤 1/3：Planner 拆分需求...")
        t0 = time.time()
        plan = self.planner.execute(requirement)
        timing["plan"] = time.time() - t0
        print(f"   Planner 输出: {len(plan)} 字符 ({timing['plan']:.1f}s)")

        # ── 提取子任务 ──
        subtasks = self._parse_subtasks(plan)
        if len(subtasks) > n_coders:
            subtasks = subtasks[:n_coders]  # 只取前 N 个

        # ── 步骤 2：多个 Coder 并行执行 ──
        print(f"\n💻 步骤 2/3：{len(subtasks)} 个 Coder 并行生成代码...")
        t0 = time.time()

        # 为每个子任务创建一个独立的 Coder 实例
        # 避免共享同一个 LLMClient 导致的串行化问题
        coders = [CoderAgent(self.bus, f"coder-{i+1}") for i in range(len(subtasks))]

        # asyncio.gather：同时启动所有协程，等全部完成
        # Day3 修复 #11：return_exceptions=True → 一个 Coder 挂了不影响其他
        async def _run_all():
            tasks = [
                self._coder_async(coders[i], subtasks[i], i + 1)
                for i in range(len(subtasks))
            ]
            # return_exceptions=True: 异常不抛出，作为返回值的一部分
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # 分离成功和异常结果
            codes = []
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    print(f"  ⚠️ Coder{i+1} 失败: {r}（已隔离，不影响其他 Coder）")
                    codes.append({
                        "task_id": i + 1,
                        "subtask": subtasks[i],
                        "code": f"# [错误] Coder{i+1} 执行失败: {r}",
                        "error": str(r),
                    })
                else:
                    codes.append(r)
            return codes

        # 启动事件循环并等待全部完成
        codes = asyncio.run(_run_all())
        timing["coders"] = time.time() - t0
        print(f"   ⏱️ 并行耗时: {timing['coders']:.1f}s（串行预估: {timing['coders'] * len(subtasks):.0f}s）")

        # ── 汇总 ──
        print(f"\n📦 汇总 {len(codes)} 份代码...")
        combined = "\n\n".join([
            f"# === 子任务 {c['task_id']} ===\n{c['code']}"
            for c in codes
        ])
        print(f"   合并后: {len(combined)} 字符")

        # ── 步骤 3：Reviewer 审核汇总结果 ──
        print(f"\n🔍 步骤 3/3：Reviewer 审核汇总代码...")
        t0 = time.time()
        review = self.reviewer.execute(combined)
        timing["review"] = time.time() - t0
        print(f"   Reviewer 输出: {len(review)} 字符 ({timing['review']:.1f}s)")

        print("\n" + "=" * 60)
        print("✅ 并行编排完成")
        print(f"   总耗时: {sum(timing.values()):.1f}s")
        print("=" * 60)

        return {
            "plan": plan,
            "codes": codes,
            "combined": combined,
            "review": review,
            "timing": timing,
        }

    def run_parallel_no_await(self, requirement: str) -> dict:
        """
        【踩坑演示】并行时不写 await，主线程直接退出

        这个方法故意不 await，展示不等待的后果：
        - 协程被创建但没有被等待
        - 主函数返回时协程还没执行完
        - 返回的结果是空的

        用法：和 run_parallel() 对比输出，观察差异。
        """
        import time

        print("\n" + "=" * 60)
        print(f"🚀 【踩坑】并行编排（不等待）：{requirement[:50]}...")
        print("=" * 60)

        # Planner 拆分
        plan = self.planner.execute(requirement)
        subtasks = self._parse_subtasks(plan)[:2]

        # ⚠️ 【踩坑】创建协程但不 await
        # 协程对象被创建后立即被丢弃，永远不会执行
        print(f"\n💻 创建 {len(subtasks)} 个 Coder 协程但不等待...")

        coders = [CoderAgent(self.bus, f"coder-{i+1}") for i in range(len(subtasks))]
        discarded = []  # 协程对象被丢弃在这里
        for i in range(len(subtasks)):
            coro = self._coder_async(coders[i], subtasks[i], i + 1)
            discarded.append(coro)  # 创建了但没有 await！
            print(f"   ⚠️ 任务 {i+1} 协程已创建但未被等待（已丢弃）")

        print(f"\n   💀 主线程直接返回，{len(discarded)} 个协程永远没执行")
        print(f"   ⚠️ 【踩坑确认】不写 await，任务全部丢失")

        return {
            "plan": plan,
            "codes": [],  # 空的！没等到任何结果
            "combined": "",
            "review": "",
            "warning": "所有 Coder 协程被丢弃，未执行",
        }

    def run_parallel_with_timeout(self, requirement: str, timeout: float = 5) -> dict:
        """
        并行执行 + 超时控制

        参数：
            timeout: 单个任务超时秒数（默认 5s，故意设小以触发超时）

        返回：
            和 run_parallel 相同，但超时的任务返回错误信息而非代码

        【踩坑】超时处理简单粗暴：asyncio.wait_for 抛出 TimeoutError，
        不做部分结果收集，丢失已完成任务的结果。
        """
        import time

        print("\n" + "=" * 60)
        print(f"🚀 并行编排（超时={timeout}s）：{requirement[:50]}...")
        print("=" * 60)

        plan = self.planner.execute(requirement)
        subtasks = self._parse_subtasks(plan)[:2]
        coders = [CoderAgent(self.bus, f"coder-{i+1}") for i in range(len(subtasks))]

        async def _run_with_timeout():
            tasks = [
                self._coder_async(coders[i], subtasks[i], i + 1)
                for i in range(len(subtasks))
            ]
            # Day3 修复 #12：用 asyncio.wait 代替 asyncio.wait_for
            # wait_for 超时后抛异常，丢弃所有结果
            # wait 超时后返回 (done, pending)，已完成的不丢
            # ensure_future：把协程包装成 Task，asyncio.wait 需要 Future 类型
            future_tasks = [asyncio.ensure_future(t) for t in tasks]
            done, pending = await asyncio.wait(
                future_tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )

            # 取消还在跑的任务
            for task in pending:
                task.cancel()
                print(f"  ⏰ 任务超时被取消")

            # 按原始顺序收集结果（包括已完成和超时的）
            codes = []
            for i, future_t in enumerate(future_tasks):
                if future_t in done:
                    try:
                        result = future_t.result()
                        codes.append(result)
                        print(f"  ✅ Coder{i+1} 完成")
                    except Exception as e:
                        print(f"  ⚠️ Coder{i+1} 异常: {e}")
                        codes.append({
                            "task_id": i + 1,
                            "subtask": subtasks[i],
                            "code": f"# [错误] {e}",
                            "error": str(e),
                        })
                else:
                    print(f"  ⏰ Coder{i+1} 超时")
                    codes.append({
                        "task_id": i + 1,
                        "subtask": subtasks[i],
                        "code": f"# [超时] Coder{i+1} 未在 {timeout}s 内完成",
                        "error": "timeout",
                    })

            return codes

        t0 = time.time()
        codes = asyncio.run(_run_with_timeout())
        elapsed = time.time() - t0

        if not codes:
            print(f"   💀 所有任务因超时丢失 ({elapsed:.1f}s)")

        return {
            "plan": plan,
            "codes": codes,
            "combined": "",
            "review": "",
            "timing": {"total": elapsed},
            "timeout": timeout,
        }

    # ═══════════════════════════════════════════════════════
    # 阶段四：带状态追踪的编排
    # ═══════════════════════════════════════════════════════

    def run_serial_with_state(self, requirement: str) -> dict:
        """
        串行编排 + 全局状态追踪

        在 run_serial 的基础上，每一步都通过 SharedState 记录进度。
        外部可以随时通过 shared_state.get_progress() 查看当前状态。

        要求：创建 Orchestrator 时 with_shared_state=True
        """
        if not self.shared_state:
            raise RuntimeError("需要 with_shared_state=True 创建 Orchestrator")

        print("\n" + "=" * 60)
        print(f"🚀 串行编排（带状态追踪）：{requirement[:50]}...")
        print("=" * 60)

        # ── 初始化进度 ──
        self.shared_state.init_progress(4)  # 4 个步骤
        self.shared_state.set("requirement", requirement)
        self.shared_state.set("phase", "planning")

        # ── 步骤 1：Planner ──
        print("\n📋 步骤 1/4：Planner 拆分需求...")
        plan = self.planner.execute(requirement)
        self.shared_state.set("phase", "coding")
        self.shared_state.mark_task_done("planner", 1, f"规划完成，{len(plan)}字")
        # 【踩坑】记录 Planner 输出到状态中
        self.shared_state.set("plan", plan)
        self.stats["plan_chars"] = len(plan)

        # ── 步骤 2：Coder ──
        print("\n💻 步骤 2/4：Coder 生成代码...")
        code = self.coder.execute(plan)
        self.shared_state.mark_task_done("coder-1", 2, f"代码生成完成，{len(code)}字")
        self.shared_state.set("code", code)
        self.stats["code_chars"] = len(code)

        # ── 步骤 3：Reviewer ──
        print("\n🔍 步骤 3/4：Reviewer 审核代码...")
        self.shared_state.set("phase", "reviewing")
        review = self.reviewer.execute(code)
        self.shared_state.mark_task_done("reviewer", 3, f"审核完成，{len(review)}字")
        self.shared_state.set("review", review)
        self.stats["review_chars"] = len(review)

        # ── 步骤 4：Coder 修复 ──
        print("\n🔧 步骤 4/4：Coder 修复代码...")
        self.shared_state.set("phase", "fixing")
        fixed_code = self.coder.fix_code(review)
        self.shared_state.mark_task_done("coder-1", 4, f"修复完成，{len(fixed_code)}字")
        self.shared_state.set("fixed_code", fixed_code)
        self.stats["fixed_code_chars"] = len(fixed_code)

        # ── 完成 ──
        self.shared_state.set("phase", "done")

        print("\n" + "=" * 60)
        print("✅ 串行编排完成（状态已记录）")
        progress = self.shared_state.get_progress()
        print(f"   进度: {progress.get('completed')}/{progress.get('total')}")
        print("=" * 60)

        return {
            "plan": plan,
            "code": code,
            "review": review,
            "fixed_code": fixed_code,
            "stats": dict(self.stats),
            "state_snapshot": self.shared_state.snapshot(),
        }

    def run_parallel_with_state(self, requirement: str, n_coders: int = 2) -> dict:
        """
        并行编排 + 全局状态追踪

        在 run_parallel 的基础上，用 SharedState 追踪各 Coder 的进度。

        要求：创建 Orchestrator 时 with_shared_state=True
        """
        if not self.shared_state:
            raise RuntimeError("需要 with_shared_state=True 创建 Orchestrator")

        import time

        print("\n" + "=" * 60)
        print(f"🚀 并行编排（带状态追踪）：{requirement[:50]}...")
        print("=" * 60)

        timing = {}

        # ── 初始化进度 ──
        self.shared_state.set("requirement", requirement)
        self.shared_state.set("phase", "planning")
        self.shared_state.init_progress(3)  # planner / coders / reviewer 三个阶段

        # ── 步骤 1：Planner ──
        print("\n📋 步骤 1/3：Planner 拆分需求...")
        t0 = time.time()
        plan = self.planner.execute(requirement)
        timing["plan"] = time.time() - t0
        self.shared_state.set("plan", plan)
        self.shared_state.set("phase", "coding")
        self.shared_state.mark_task_done("planner", 1, f"规划完成，{len(plan)}字")

        # 提取子任务
        subtasks = self._parse_subtasks(plan)
        if len(subtasks) > n_coders:
            subtasks = subtasks[:n_coders]

        # ── 步骤 2：并行 Coder ──
        print(f"\n💻 步骤 2/3：{len(subtasks)} 个 Coder 并行生成代码...")
        t0 = time.time()

        coders = [CoderAgent(self.bus, f"coder-{i+1}") for i in range(len(subtasks))]
        # 注入 SharedState 到动态创建的 Coders
        for c in coders:
            c.shared_state = self.shared_state

        async def _run_all():
            tasks = [
                self._coder_async(coders[i], subtasks[i], i + 1)
                for i in range(len(subtasks))
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)  # Day3 修复 #11

        codes = asyncio.run(_run_all())
        timing["coders"] = time.time() - t0

        # Day3 修复 #11：过滤掉异常，只保留成功结果
        valid_codes = [c for c in codes if not isinstance(c, BaseException)]
        error_count = len(codes) - len(valid_codes)
        if error_count > 0:
            print(f"  ⚠️ {error_count} 个 Coder 失败，{len(valid_codes)} 个成功")

        # 记录各 Coder 的结果到状态
        for c in valid_codes:
            self.shared_state.set(f"code_task{c['task_id']}", c["code"][:500])
        self.shared_state.mark_task_done("coders", 0,
                                         f"{len(valid_codes)}/{len(codes)} 个 Coder 完成")

        # ── 汇总 ──（只用有效结果）
        combined = "\n\n".join([
            f"# === 子任务 {c['task_id']} ===\n{c['code']}"
            for c in valid_codes
        ])
        self.shared_state.set("combined", combined[:2000])

        # ── 步骤 3：Reviewer ──
        print(f"\n🔍 步骤 3/3：Reviewer 审核...")
        self.shared_state.set("phase", "reviewing")
        t0 = time.time()
        review = self.reviewer.execute(combined)
        timing["review"] = time.time() - t0
        self.shared_state.set("review", review)
        self.shared_state.mark_task_done("reviewer", 3, f"审核完成，{len(review)}字")
        self.shared_state.set("phase", "done")

        print("\n" + "=" * 60)
        print("✅ 并行编排完成（状态已记录）")
        progress = self.shared_state.get_progress()
        print(f"   进度: {progress.get('completed')}/{progress.get('total')}")
        print("=" * 60)

        return {
            "plan": plan,
            "codes": codes,
            "combined": combined,
            "review": review,
            "timing": timing,
            "state_snapshot": self.shared_state.snapshot(),
        }

    def demonstrate_concurrent_conflict(self) -> dict:
        """
        【踩坑演示】模拟两个 Agent 同时更新状态导致版本冲突

        不调 LLM，纯逻辑演示乐观锁冲突的场景。
        """
        if not self.shared_state:
            raise RuntimeError("需要 with_shared_state=True 创建 Orchestrator")

        print("\n" + "=" * 60)
        print("【踩坑演示】并发状态冲突")
        print("=" * 60)

        results = {}

        # ── 场景 1：乐观锁正常 ──
        print("\n场景 1：乐观锁正常更新")
        v = self.shared_state.set("counter", 0)
        print(f"  初始版本: {v}, counter=0")
        ok = self.shared_state.update("counter", 1, expected_version=v)
        print(f"  update(counter=1, v={v}) → {'✅ 成功' if ok else '❌ 失败'}")
        results["optimistic_ok"] = ok

        # ── 场景 2：版本冲突 ──
        print("\n场景 2：版本冲突（Agent A 和 B 同时改）")
        v_a = self.shared_state.set("shared_counter", 0)
        print(f"  Agent A 读到版本 {v_a}，counter=0")
        # Agent B 抢先改了 —— 这次是真的调 set()
        self.shared_state.set("shared_counter", 999)
        print(f"  Agent B 抢先改了 → set(shared_counter, 999)，版本变成 {self.shared_state.get_version()}")
        # A 用旧版本号尝试更新 → 冲突！
        ok = self.shared_state.update("shared_counter", 42, expected_version=v_a)
        print(f"  Agent A 尝试 update(counter=42, v={v_a}) → {'✅ 成功' if ok else '⚠️ 冲突（预期）'}")
        assert self.shared_state.get("shared_counter") == 999, "值应该是 B 的结果"
        results["version_conflict"] = not ok

        # ── 场景 3：【踩坑】绕过锁直接改 ──
        print("\n场景 3：【踩坑】绕过锁直接修改（bypass_lock）")
        v_before = self.shared_state.get_version()
        self.shared_state.set("protected_data", "正常写入的值")
        print(f"  正常 set: protected_data = '正常写入的值', 版本 {self.shared_state.get_version()}")
        # 绕过锁直接改
        self.shared_state.bypass_lock("protected_data", "被悄悄覆盖了！")
        v_after = self.shared_state.get_version()
        print(f"  bypass_lock: protected_data = '{self.shared_state.get('protected_data')}'")
        print(f"  版本号变化: {v_before} → {v_after}")
        print(f"  ⚠️ 值被覆盖了，但版本号没递增！乐观锁失效！")
        results["bypass_pitfall"] = self.shared_state.get("protected_data") == "被悄悄覆盖了！"

        # ── 场景 4：并发写入无锁保护 ──
        print(f"\n场景 4：最终的 SharedState 统计")
        print(f"  {self.shared_state}")

        print("\n" + "=" * 60)
        all_ok = all(results.values())
        print(f"  {'✅ 所有踩坑已确认' if all_ok else '⚠️ 部分验证失败'}")
        print("=" * 60)

        return results

    # ═══════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════

    def reset(self):
        """重置消息总线和 Agent 状态"""
        self.bus.clear()
        self.stats = {k: 0 for k in self.stats}


# ── 模块自测 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Orchestrator 自测")
    print("=" * 50)

    bus = MessageBus()
    orch = Orchestrator(bus)

    requirement = "实现一个函数，判断一个字符串是否为回文"

    result = orch.run_serial(requirement, with_fix=True)

    print(f"\n📊 统计: {result['stats']}")
