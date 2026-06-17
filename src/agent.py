"""
ReAct Agent —— Day4 框架版：基于 LangChain create_agent (v1.3+)

Day4 核心变化：
  手搓版（Day1-3）：while True + 手动消息构建 + 手动工具调用解析
  框架版（Day4）：  langchain.agents.create_agent 一行搞定

LangChain 1.3+ API 变化：
  旧: from langchain.agents import create_react_agent, AgentExecutor
  新: from langchain.agents import create_agent
  create_agent 内部自动处理 ReAct 循环 + 工具调用

Java 类比：手写 for 循环 + if/else → Spring StateMachine 声明式流程
"""

from src.llm_client import LLMClient
from src.tools import get_tool_definitions, get_langchain_tools
from src.memory import ConversationMemory, TruncationStrategy

# ── Day4：LangChain v1.3+ Agent ────────────────────────────
from langchain.agents import create_agent


class ReActAgent:
    """
    ReAct Agent —— Day4 框架版（LangChain 1.3+）

    使用 create_agent 替代手搓 while 循环。
    接口完全兼容 Day1-3 的 ReActAgent。
    """

    def __init__(
        self,
        max_iterations: int = 10,
        truncation_strategy: TruncationStrategy = "none",
    ):
        """初始化 Agent"""
        self.llm = LLMClient()
        self.tools = get_tool_definitions()

        # ── 记忆系统 ──
        self.memory = ConversationMemory(max_messages=20, max_tokens=4000)
        self.truncation_strategy = truncation_strategy

        # ── system prompt ──
        self.system_prompt = (
            "你是 ReAct Agent，必须通过工具与外界交互。\n\n"
            "核心规则：\n"
            "1. 计算类问题 → 必须调用 calculator 工具\n"
            "2. 知识类问题 → 必须调用 search_knowledge 工具\n"
            "3. 不要用自己的知识直接回答\n"
            "4. 工具返回错误时如实告诉用户\n"
        )
        self.memory.set_system(self.system_prompt)

        self.max_iterations = max_iterations

        # ═══════════════════════════════════════════════════
        # Day4: LangChain 1.3+ create_agent 一行建 Agent
        # ═══════════════════════════════════════════════════
        lc_tools = get_langchain_tools()

        self._agent = create_agent(
            model=self.llm._llm,        # LangChain ChatOpenAI 对象
            tools=lc_tools,
            system_prompt=self.system_prompt,
        )
        # create_agent 返回的是一个 Runnable，直接 .invoke() 就行
        # 内部自动处理：ReAct循环、工具调用、结果反馈

    def run(self, user_message: str) -> str:
        """
        执行一次 Agent 对话 —— 接口完全兼容 Day1-3

        Day4 变化：_agent.invoke() 替代 while 循环
        """
        self.memory.add_user(user_message)

        # 构建对话历史
        messages = self.memory.get_messages(strategy=self.truncation_strategy)
        history = [
            m["content"] for m in messages
            if m["role"] in ("user", "assistant")
        ]

        # ═══════════════════════════════════════════════════
        # Day4 核心：一行替代 70 行 while 循环
        # ═══════════════════════════════════════════════════
        result = self._agent.invoke({
            "messages": [
                {"role": "user", "content": user_message},
            ],
        })

        # 提取最终回复
        # create_agent 返回的 messages 列表中最后一条是 AI 的最终回复
        output_msgs = result.get("messages", [])
        answer = ""
        for msg in reversed(output_msgs):
            if hasattr(msg, "content") and msg.content:
                answer = msg.content
                break

        self.memory.add_assistant(content=answer)
        return answer

    # ═══════════════════════════════════════════════════════
    # 记忆管理（同 Day3）
    # ═══════════════════════════════════════════════════════

    def clear_memory(self):
        self.memory.clear()
        self.memory.set_system(self.system_prompt)

    def set_truncation(self, strategy: TruncationStrategy):
        self.truncation_strategy = strategy

    def memory_stats(self) -> dict:
        return {
            "total_messages": len(self.memory),
            "non_system": self.memory.count_messages(),
            "strategy": self.truncation_strategy,
        }


# ── 便捷函数 ─────────────────────────────────────────────
def run_agent(question: str) -> str:
    agent = ReActAgent()
    return agent.run(question)


# ── 模块自测 ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("ReAct Agent（LangChain 1.3+ create_agent）自测")
    print("=" * 50)

    agent = ReActAgent(max_iterations=5)

    q = "帮我计算 100 + 200 等于多少"
    print(f"\n👤 用户: {q}")
    answer = agent.run(q)
    print(f"🤖 Agent: {answer}")
    print("✅ 自测完成")
