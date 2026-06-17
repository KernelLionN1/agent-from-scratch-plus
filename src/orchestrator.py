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

    def __init__(self, bus: MessageBus):
        """
        初始化编排器

        参数：
            bus: 共享的消息总线
        """
        self.bus = bus

        # 创建三个 Agent 实例
        # 所有 Agent 共享同一个 MessageBus
        self.planner = PlannerAgent(bus, "planner-1")
        self.coder = CoderAgent(bus, "coder-1")
        self.reviewer = ReviewerAgent(bus, "reviewer-1")

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
        # 相当于 Java 的 CompletableFuture.allOf()
        async def _run_all():
            tasks = [
                self._coder_async(coders[i], subtasks[i], i + 1)
                for i in range(len(subtasks))
            ]
            return await asyncio.gather(*tasks)

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
            try:
                # 【踩坑】wait_for 超时后直接抛异常，已完成的结果也丢了
                return await asyncio.wait_for(
                    asyncio.gather(*tasks),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                print(f"   ⏰ 超时！({timeout}s)")
                # 【踩坑】不收集已完成的部分结果，全部丢弃
                return []

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
