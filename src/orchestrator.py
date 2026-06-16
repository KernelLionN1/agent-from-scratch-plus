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
