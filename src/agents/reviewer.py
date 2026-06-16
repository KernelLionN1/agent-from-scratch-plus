"""
Reviewer Agent —— Day2 阶段二

核心职责：
- 接收代码 → 调 LLM 审核 → 输出改进建议
- 使用 Reviewer 角色的 system prompt（温度 0.3，偏精确）

刻意踩坑点：
- 【踩坑】审核意见不结构化：不要求输出 JSON 格式，
  导致下游 Coder 解析审核意见困难。
- 【踩坑】不做自动重审：Reviewer 只审一次，
  修复后的代码不会自动再审核。

类比 Java：
    ReviewerAgent 相当于 Code Review 服务，
    输入代码，输出审核意见。
"""

from src.agents.base import BaseAgent
from src.roles import AgentRole
from src.message_bus import MessageBus


class ReviewerAgent(BaseAgent):
    """
    审核 Agent —— 审查代码质量

    流程：
    1. 从 MessageBus 拉取 code 类型的消息（或直接 execute）
    2. 调用 LLM（Reviewer 角色 + 温度 0.3）
    3. 返回审核意见

    使用方式：
        bus = MessageBus()
        reviewer = ReviewerAgent(bus, "reviewer-1")
        feedback = reviewer.execute("def add(a,b): return a+b")
    """

    def __init__(self, bus: MessageBus, name: str = "reviewer-1"):
        """
        创建 Reviewer Agent

        参数：
            bus:  共享消息总线
            name: Agent 标识（默认 "reviewer-1"）
        """
        super().__init__(bus, name, AgentRole.REVIEWER)

    def execute(self, code: str) -> str:
        """
        审查代码并返回改进建议

        参数：
            code: Coder 生成的代码

        返回：
            LLM 的审核意见

        【踩坑】审核意见格式不固定，下游解析困难
        """
        return self.call_llm(
            user_prompt=(
                f"请审查以下 Python 代码，指出问题和改进建议：\n\n"
                f"```python\n{code}\n```\n\n"
                f"请从以下维度审查：\n"
                f"1. 正确性：代码是否能正确运行？\n"
                f"2. 可读性：命名是否清晰？注释是否充分？\n"
                f"3. 性能：是否有明显的性能问题？\n"
                f"4. 安全性：是否存在安全风险？\n\n"
                f"请给出具体的修改建议。"
                # 【踩坑】没有要求输出结构化格式（如 JSON）
                # 导致 Coder 解析审核意见时可能出错
            )
        )

    def review_and_respond(self, coder_name: str = "coder-1") -> str:
        """
        一站式操作：从消息总线读取代码 → 审核 → 发送意见回 Coder

        参数：
            coder_name: Coder 的名称（用于发送审核意见）

        返回：
            审核意见

        这是阶段二串行流程的第三步（审核环节）。
        """
        # 从消息总线获取代码
        code_msgs = [
            m for m in self.bus.receive(self.name)
            if m.type == "code"
        ]

        if not code_msgs:
            print(f"  [Reviewer] ⚠️ 没有待审核的代码")
            return ""

        # 取最新的代码消息
        latest_code = code_msgs[-1].content
        print(f"  [Reviewer] 📥 收到代码，{len(latest_code)} 字符")

        # 审核
        feedback = self.execute(latest_code)
        print(f"  [Reviewer] 🔍 审核完成，{len(feedback)} 字符")

        # 发送审核意见回 Coder
        self.send_message(coder_name, "review", feedback)
        print(f"  [Reviewer] 📤 审核意见已发送到 {coder_name}")

        return feedback


# ── 模块自测 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Reviewer Agent 自测")
    print("=" * 50)

    from src.message_bus import MessageBus

    bus = MessageBus()
    reviewer = ReviewerAgent(bus, "reviewer-1")

    # 一段有问题的代码
    sample_code = """\
def add(a, b):
    return a + b

def divide(a, b):
    return a / b

# 测试
print(add(5, 3))
print(divide(10, 0))  # 除以零！
"""

    print(f"\n待审核代码:\n{sample_code}")
    print("-" * 50)

    feedback = reviewer.execute(sample_code)
    print(f"\n审核意见:\n{feedback}")
