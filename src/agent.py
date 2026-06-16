"""
ReAct Agent 核心循环 —— 思考 → 行动 → 观察 的自动化

这是整个 Day1 最重要的文件。它把阶段一的两个独立模块
（LLM 客户端 + 工具系统）串联成一个能自主行动的 Agent。

刻意踩坑点：
- 不设最大轮次限制 → 可能死循环
- 无格式容错 → LLM 返回异常格式直接崩溃
- 不校验工具参数 → 恶意/错误参数直接执行
"""

import json

# ── 导入阶段一的两个核心模块 ─────────────────────────────
# 注意：这两个模块互相不认识，Agent 是它们的"调度者"
from src.llm_client import LLMClient          # 负责和 LLM 对话
from src.tools import get_tool_definitions, execute_tool  # 负责执行工具


class ReActAgent:
    """
    ReAct（Reasoning + Acting）Agent 实现

    核心思想：
        1. 把用户问题发给 LLM
        2. LLM 要么直接回答 → 结束
        3. LLM 要么要求调工具 → 我们执行工具，结果反馈给 LLM
        4. 重复 2-3，直到 LLM 给出最终答案

    类比 Java：
        这是一个有状态的服务类，持有 LLMClient 和工具定义。
        相当于 @Service 注入了两个依赖。
    """

    def __init__(self):
        """
        初始化 Agent —— 准备 LLM 客户端和工具列表

        工具定义只加载一次（注册表是全局的），
        但 LLM 客户端可以换模型/Key（当前从 .env 读）
        """
        # 创建 LLM 客户端 —— 负责发 HTTP 请求
        self.llm = LLMClient()

        # 加载所有已注册的工具定义 —— 告诉 LLM "你能用这些"
        self.tools = get_tool_definitions()

        # ── 系统提示词 ──
        # 告诉 LLM 它的角色和行为规则。相当于 Java 里的 @SystemPrompt
        self.system_prompt = (
            "你是一个有用的 AI 助手。"
            "当用户问需要计算或查询的问题时，请使用提供的工具。"
            "使用工具获取结果后，用自然语言向用户解释结果。"
        )

    def run(self, user_message: str) -> str:
        """
        执行一次完整的 Agent 对话 —— 自动调用工具直到得出最终答案

        参数：
            user_message: 用户输入的问题，如 "帮我计算 123*456"

        返回：
            LLM 的最终回复文本

        这是用户看到的唯一入口 —— 内部循环对用户透明。
        """

        # ── 初始化消息列表 ──
        # messages 就是整个对话的"上下文窗口"，LLM 能看到的历史
        # system 消息在最前面，定义了 LLM 的行为规则
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        # ── 记录循环轮次 ──
        # 用于追踪 LLM 调了几次工具，帮助理解 Agent 的"思考深度"
        iteration = 0

        # ╔══════════════════════════════════════════════════╗
        # ║          ReAct 核心循环                          ║
        # ║  Think → Act → Observe → Think → ...            ║
        # ╚══════════════════════════════════════════════════╝
        #
        # 注意：这里故意没有设最大轮次限制 ← 刻意踩坑！
        # 如果 LLM 陷入循环（不断调工具但永远不给出答案），
        # 这个 while True 会永远跑下去。
        # 阶段三会加上 max_iterations 保护。
        while True:
            iteration += 1
            print(f"  [Agent] 第 {iteration} 轮思考...")

            # ── 步骤 1：调用 LLM ──
            # 把当前消息列表（含历史）发给 LLM，让它决定下一步
            response = self.llm.chat(
                messages=messages,
                tools=self.tools,
                temperature=0.1,  # 低温度 → 更确定性的输出，适合工具调用
            )

            # ── 步骤 2：检查 LLM 的回答方式 ──
            # LLM 有两种回应：
            #   A. 直接给出文字答案（content 不为空）→ 结束循环
            #   B. 要求调用工具（tool_calls 不为空）→ 继续循环

            # 情况 A：LLM 直接回答了
            if response["content"]:
                print(f"  [Agent] LLM 给出最终答案")
                return response["content"]

            # 情况 B：LLM 想调用工具
            if response["tool_calls"]:
                # ── 步骤 3：执行 LLM 请求的工具 ──
                # 注意：LLM 可能一次请求多个工具调用
                for tc in response["tool_calls"]:
                    tool_name = tc["name"]
                    # arguments 是 JSON 字符串，需要解析
                    # 坑点：不做异常处理，如果 LLM 传了非法 JSON 直接崩溃
                    tool_args = json.loads(tc["arguments"])

                    print(f"  [Agent] 🔧 调用工具: {tool_name}({tool_args})")

                    # 真正执行工具 —— 这里才从"LLM 的想法"变成"实际动作"
                    tool_result = execute_tool(tool_name, tool_args)
                    print(f"  [Agent] 📤 工具结果: {tool_result}")

                    # ── 步骤 4：把工具结果反馈给 LLM ──
                    # 这是 ReAct 循环的关键：把工具结果作为一条新消息
                    # 追加到对话历史，下一轮 LLM 就能"看到"这个结果
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tc["arguments"],
                                },
                            }
                        ],
                    })
                    messages.append({
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tc["id"],
                    })

            # ── 循环继续 ──
            # messages 里已经多了 assistant（带 tool_calls）+ tool（结果）
            # 下一轮 LLM 会"看到"调用历史和结果，基于此决定下一步


# ── 便捷函数：快速运行 Agent ──────────────────────────────
def run_agent(question: str) -> str:
    """
    一键启动 Agent，输入问题返回答案

    用法：
        >>> answer = run_agent("123 * 456 等于多少？")
    """
    agent = ReActAgent()
    return agent.run(question)


# ── 模块自测（直接运行 python src/agent.py 时触发） ─────
if __name__ == "__main__":
    print("=" * 50)
    print("ReAct Agent 自测")
    print("=" * 50)

    questions = [
        "帮我计算 (100 + 200) * 3 等于多少",
        "请介绍一下 Python 是什么",
    ]

    for q in questions:
        print(f"\n👤 用户: {q}")
        answer = run_agent(q)
        print(f"🤖 Agent: {answer}")
        print("-" * 50)
