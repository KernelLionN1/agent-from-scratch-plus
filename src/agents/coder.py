"""
Coder Agent —— Day2 阶段二

核心职责：
- 接收子任务描述 → 调 LLM 生成 Python 代码
- 使用 Coder 角色的 system prompt（温度 0.3，偏精确）

刻意踩坑点：
- 【踩坑】不校验输入：直接信任 Planner 传来的任务描述，
  如果 Planner 输出格式混乱，Coder 照样处理。
- 【踩坑】不做代码安全校验：生成的代码可能含安全漏洞，
  不检查 eval/exec/os.system 等危险调用。

类比 Java：
    CoderAgent 相当于代码生成服务，
    输入是需求规格，输出是代码字符串。
"""

from src.agents.base import BaseAgent
from src.roles import AgentRole
from src.message_bus import MessageBus


class CoderAgent(BaseAgent):
    """
    编码 Agent —— 根据任务描述生成代码

    流程：
    1. 从 MessageBus 拉取 task 类型消息（或直接 execute）
    2. 调用 LLM（Coder 角色 + 温度 0.3）
    3. 返回生成的代码

    使用方式：
        bus = MessageBus()
        coder = CoderAgent(bus, "coder-1")
        code = coder.execute("实现二分查找算法")
    """

    def __init__(self, bus: MessageBus, name: str = "coder-1"):
        """
        创建 Coder Agent

        参数：
            bus:  共享消息总线
            name: Agent 标识（默认 "coder-1"）
        """
        super().__init__(bus, name, AgentRole.CODER)

    def execute(self, task_description: str) -> str:
        """
        根据任务描述生成代码

        参数：
            task_description: Planner 传来的子任务描述

        返回：
            LLM 生成的 Python 代码

        【踩坑】不做输入校验，不限定输出格式
        """
        return self.call_llm(
            user_prompt=(
                f"请根据以下任务描述编写 Python 代码：\n\n"
                f"任务描述：\n{task_description}\n\n"
                f"要求：\n"
                f"- 代码包含详细的中文注释\n"
                f"- 包含函数定义和简单的测试代码（if __name__ == '__main__'）\n"
                f"- 代码可以直接运行\n"
                # 【踩坑】没有要求代码必须通过语法检查
                # 【踩坑】没有要求返回格式（纯代码 / 带解释 / JSON）
            )
        )

    def code_and_submit(self, task_description: str, reviewer_name: str = "reviewer-1") -> str:
        """
        一站式操作：生成代码 → 发送给 Reviewer

        参数：
            task_description: 任务描述
            reviewer_name:    Reviewer 的名称

        返回：
            生成的代码

        这是阶段二串行流程的第二步（编码环节）。
        """
        code = self.execute(task_description)
        print(f"  [Coder] 💻 代码生成完成，{len(code)} 字符")

        self.send_message(reviewer_name, "code", code)
        print(f"  [Coder] 📤 代码已发送到 {reviewer_name}")

        return code

    def fix_code(self, review_feedback: str) -> str:
        """
        根据审核意见修复代码 —— 串行流程的第四步

        参数：
            review_feedback: Reviewer 的审核意见

        返回：
            修复后的代码
        """
        return self.call_llm(
            user_prompt=(
                f"请根据以下审核意见修改代码：\n\n"
                f"审核意见：\n{review_feedback}\n\n"
                f"请提供修改后的完整代码，"
                f"包含中文注释说明修改了哪些问题。"
            )
        )


# ── 模块自测 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Coder Agent 自测")
    print("=" * 50)

    from src.message_bus import MessageBus

    bus = MessageBus()
    coder = CoderAgent(bus, "coder-1")

    task = "实现一个函数 fibonacci(n)，返回第 n 个斐波那契数。要求包含详细注释。"
    print(f"\n任务描述: {task}")
    print("-" * 50)

    code = coder.execute(task)
    print(f"\n生成的代码:\n{code}")
