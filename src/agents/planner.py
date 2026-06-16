"""
Planner Agent —— Day2 阶段二

核心职责：
- 接收用户需求 → 调 LLM 拆分成可执行的子任务列表
- 使用 Planner 角色的 system prompt（温度 0.7，偏创意）

刻意踩坑点：
- 【踩坑】不约束拆分粒度：prompt 只说要「拆分子任务」，
  不规定最少/最多几个、不规定每个子任务的详细程度。
  导致 Planner 可能拆得太粗（1 个任务 = 没拆）或太细（20 个微任务）。
- 【踩坑】输出格式自由：不要求 Planner 用固定格式（如 JSON），
  下游 Coder 解析 Planner 输出时可能出错。

类比 Java：
    PlannerAgent 相当于一个拆解需求的服务类，
    实现 Strategy 模式里的「规划策略」。
"""

from src.agents.base import BaseAgent
from src.roles import AgentRole
from src.message_bus import MessageBus


class PlannerAgent(BaseAgent):
    """
    规划 Agent —— 负责将用户需求拆分为子任务

    流程：
    1. 接收用户需求（来自 Orchestrator 或 MessageBus）
    2. 调用 LLM（Planner 角色 + 温度 0.7）
    3. 返回子任务列表

    使用方式：
        bus = MessageBus()
        planner = PlannerAgent(bus, "planner-1")
        subtasks = planner.execute("做一个在线商城系统")
        # subtasks 是 LLM 生成的子任务描述
    """

    def __init__(self, bus: MessageBus, name: str = "planner-1"):
        """
        创建 Planner Agent

        参数：
            bus:  共享消息总线
            name: Agent 标识（默认 "planner-1"）
        """
        super().__init__(bus, name, AgentRole.PLANNER)

    def execute(self, requirement: str) -> str:
        """
        分析用户需求，拆分成子任务

        参数：
            requirement: 用户的原始需求描述

        返回：
            LLM 生成的子任务列表（文本格式）

        【踩坑】不限定拆分粒度和输出格式
        """
        # ── 构建 prompt ──
        # 用 Planner 角色的 system prompt + 用户需求
        return self.call_llm(
            user_prompt=(
                f"请分析以下用户需求，并将其拆分为可执行的子任务：\n\n"
                f"用户需求：{requirement}\n\n"
                f"请列出每个子任务的：\n"
                f"1. 任务名称\n"
                f"2. 任务描述\n"
                f"3. 预期输入/输出\n"
                # 【踩坑】没有要求具体的格式（JSON/YAML/编号），
                # 也没有限制子任务数量
            )
        )

    def plan_and_dispatch(self, requirement: str, coder_name: str = "coder-1") -> str:
        """
        一站式操作：拆分需求 → 发送任务到消息总线

        参数：
            requirement: 用户需求
            coder_name:  接收任务的 Coder 名称

        返回：
            规划结果的文本

        这是阶段二串行流程的第一步。
        """
        # 步骤 1：拆分
        plan = self.execute(requirement)
        print(f"  [Planner] 📋 规划完成，{len(plan)} 字符")

        # 步骤 2：通过消息总线发送给 Coder
        self.send_message(coder_name, "task", plan)
        print(f"  [Planner] 📤 任务已发送到 {coder_name}")

        return plan


# ── 模块自测 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Planner Agent 自测")
    print("=" * 50)

    from src.message_bus import MessageBus

    bus = MessageBus()
    planner = PlannerAgent(bus, "planner-1")

    # 测试简单的需求拆分
    requirement = "实现一个计算器程序，支持加减乘除四则运算"
    print(f"\n用户需求: {requirement}")
    print("-" * 50)

    result = planner.execute(requirement)
    print(f"\n规划结果:\n{result}")
