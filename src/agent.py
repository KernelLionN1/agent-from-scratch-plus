"""
ReAct Agent 核心循环 —— 集成对话记忆

阶段三升级：
- 不再每次 run() 重新构建 messages
- 使用 ConversationMemory 持久存储对话历史
- 支持多轮对话（连续多次 run() 共享上下文）
- 支持上下文截断策略

刻意踩坑点（阶段二保留 + 阶段三新增）：
- 不设最大轮次限制 → 可能死循环
- 默认截断策略为 "none" → 长对话超出上下文窗口
- token 估算粗糙 → 中文对话 token 数严重低估
"""

import json

# ── 导入依赖模块 ─────────────────────────────────────────
from src.llm_client import LLMClient
from src.tools import get_tool_definitions, execute_tool
from src.memory import ConversationMemory, TruncationStrategy


class ReActAgent:
    """
    带记忆的 ReAct Agent

    阶段二 → 阶段三的关键变化：
    - 之前：每次 run() 从零构建 messages，对话完就忘
    - 现在：所有消息存入 ConversationMemory，多次 run() 共享

    类比 Java：
        @Service 类，注入 LLMClient 和 ConversationMemory 两个依赖。
        ConversationMemory 相当于一个有状态 Repository，
        持久化会话数据，支持不同的查询策略（截断方式）。
    """

    def __init__(
        self,
        max_iterations: int = 10,                 # 阶段三新增：最大循环轮次
        truncation_strategy: TruncationStrategy = "none",  # 阶段三新增：截断策略
    ):
        """
        初始化 Agent

        参数：
            max_iterations:      防止死循环的轮次上限（默认 10）
            truncation_strategy: 截断策略，可选 "none"|"sliding_window"|"token_limit"
        """
        # ── 依赖组件 ──
        self.llm = LLMClient()
        self.tools = get_tool_definitions()

        # ── 记忆系统（阶段三新增）──
        # ConversationMemory 存储所有对话，跨 run() 保留
        self.memory = ConversationMemory(max_messages=20, max_tokens=4000)
        self.truncation_strategy = truncation_strategy

        # ── 系统提示词 ──
        self.system_prompt = (
            "你是一个有用的 AI 助手。"
            "当用户问需要计算或查询的问题时，请使用提供的工具。"
            "使用工具获取结果后，用自然语言向用户解释结果。"
        )
        # 系统提示词存入 memory —— 它始终在消息列表第一位
        self.memory.set_system(self.system_prompt)

        # ── 循环控制 ──
        self.max_iterations = max_iterations

    # ═══════════════════════════════════════════════════════
    # 核心入口
    # ═══════════════════════════════════════════════════════

    def run(self, user_message: str) -> str:
        """
        执行一次 Agent 对话 —— 自动调用工具直到得出最终答案

        参数：
            user_message: 用户输入

        返回：
            LLM 的最终回复

        阶段三变化：
        - 不再每次构建 messages 列表，改为往 memory 里追加
        - 每次调 LLM 前从 memory 获取（可能经过截断的）消息
        - 支持多轮对话：第二次 run() 时 memory 里保留着之前的对话
        """
        # ── 追加用户消息到记忆 ──
        # 注意：这是持久化的，下次 run() 时还能看到
        self.memory.add_user(user_message)

        # ── 记录循环轮次 ──
        iteration = 0

        # ╔══════════════════════════════════════════════════╗
        # ║          ReAct 核心循环                          ║
        # ╚══════════════════════════════════════════════════╝
        while True:
            iteration += 1
            print(f"  [Agent] 第 {iteration} 轮思考...")

            # ── 阶段三新增：轮次保护 ──
            # 防止 LLM 陷入死循环（比如不断调工具但永远不回答）
            if iteration > self.max_iterations:
                print(f"  [Agent] ⚠️ 超过最大轮次 {self.max_iterations}，强制终止")
                return "抱歉，处理超时，请尝试简化问题后重试。"

            # ── 步骤 1：从记忆获取消息（可能截断）──
            # 阶段三关键：不再用本地 messages 列表，而是从 memory 取
            messages = self.memory.get_messages(strategy=self.truncation_strategy)

            # ── 步骤 2：调用 LLM ──
            response = self.llm.chat(
                messages=messages,
                tools=self.tools,
                temperature=0.1,
            )

            # ── 步骤 3：处理 LLM 响应 ──

            # 情况 A：LLM 直接给出文字回答
            if response["content"]:
                print(f"  [Agent] LLM 给出最终答案")
                # 阶段三新增：把 LLM 回答也存入记忆
                self.memory.add_assistant(content=response["content"])
                return response["content"]

            # 情况 B：LLM 请求调用工具
            if response["tool_calls"]:
                # ── 阶段三新增：把 LLM 的工具调用意图存入记忆 ──
                self.memory.add_assistant(
                    content=None,
                    tool_calls=[
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in response["tool_calls"]
                    ],
                )

                # ── 步骤 4：执行工具 + 结果反馈 ──
                for tc in response["tool_calls"]:
                    tool_name = tc["name"]
                    # 解析 JSON 参数 —— 坑点：无异常处理
                    tool_args = json.loads(tc["arguments"])

                    print(f"  [Agent] 🔧 调用工具: {tool_name}({tool_args})")

                    # 执行工具
                    tool_result = execute_tool(tool_name, tool_args)
                    print(f"  [Agent] 📤 工具结果: {tool_result}")

                    # 阶段三新增：工具结果存入记忆（而非本地 messages）
                    self.memory.add_tool_result(
                        tool_call_id=tc["id"],
                        tool_name=tool_name,
                        result=tool_result,
                    )

    # ═══════════════════════════════════════════════════════
    # 记忆管理
    # ═══════════════════════════════════════════════════════

    def clear_memory(self):
        """
        清空对话记忆 —— 开始全新对话

        会保留 system prompt，只清除 user/assistant/tool 消息。
        """
        self.memory.clear()
        self.memory.set_system(self.system_prompt)
        print("[Agent] 记忆已清空")

    def set_truncation(self, strategy: TruncationStrategy):
        """
        切换截断策略 —— 用于对比测试不同策略的效果

        用法：
            agent.set_truncation("sliding_window")
            agent.run("继续上一轮的问题...")
        """
        self.truncation_strategy = strategy
        print(f"[Agent] 截断策略切换为: {strategy}")

    def memory_stats(self) -> dict:
        """
        返回记忆状态信息 —— 调试和监控用

        返回：
            {"total_messages": N, "non_system": N, "strategy": "..."}
        """
        return {
            "total_messages": len(self.memory),
            "non_system": self.memory.count_messages(),
            "strategy": self.truncation_strategy,
        }


# ── 便捷函数 ─────────────────────────────────────────────
def run_agent(question: str) -> str:
    """一键启动带记忆的 Agent"""
    agent = ReActAgent()
    return agent.run(question)


# ── 模块自测 ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("ReAct Agent（带记忆）自测")
    print("=" * 50)

    agent = ReActAgent(max_iterations=10, truncation_strategy="none")

    # 测试多轮对话
    rounds = [
        "帮我计算 100 + 200 等于多少",
        "刚才的结果再乘以 3 等于多少",   # ← 需要记住上一轮
    ]

    for q in rounds:
        print(f"\n👤 用户: {q}")
        answer = agent.run(q)
        print(f"🤖 Agent: {answer}")
        print(f"📊 {agent.memory_stats()}")
        print("-" * 50)
