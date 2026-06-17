"""
动态路由 Agent —— Day5：LangGraph 的真正用法

Day4 vs Day5 核心区别：
  Day4（固定工作流）: planner → coder → reviewer → END
                     边的方向是程序员写死的
  Day5（LLM 动态路由）: agent → LLM判断 → tool / END → agent(循环)
                     边的方向是 LLM 实时决定的

OpenHarness 对照：
  OpenHarness 的 Agent Loop Engine 就是这套逻辑的工程实现。
  我们把它拆开成一个可见的 LangGraph 图，理解"LLM 怎么自己决定下一步"。
"""

import asyncio
from typing import TypedDict, Annotated, Literal
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.llm_client import LLMClient
from src.tools import execute_tool, get_tool_definitions, get_langchain_tools
from src.memory import ConversationMemory


# ═══════════════════════════════════════════════════════════
# State —— 图中流转的共享状态
# ═══════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """动态 Agent 的共享状态"""
    input: str                        # 用户输入
    messages: Annotated[list, add]    # 对话历史（自动追加，不覆盖）
    tool_results: Annotated[list, add]  # 工具执行结果历史（自动追加）
    final_answer: str                 # 最终回复
    steps: int                        # 已执行步数
    max_steps: int                    # 最大步数


# ═══════════════════════════════════════════════════════════
# 节点函数
# ═══════════════════════════════════════════════════════════

def agent_node(state: AgentState, llm: LLMClient, tools: list) -> dict:
    """
    Agent 节点：调 LLM，LLM 自己决定是调工具还是直接回答

    这是动态路由的核心 —— 没有固定的下一步，
    LLM 返回的 tool_calls 决定 should_continue() 往哪走。
    """
    print(f"\n🧠 Agent 思考中（第{state['steps']}步）...")

    # 构建消息
    system_msg = {
        "role": "system",
        "content": (
            "你是 ReAct Agent。你有以下工具可用：calculator（计算）、search_knowledge（搜索知识库）。\n"
            "规则：\n"
            "1. 计算类问题 → 必须调 calculator\n"
            "2. 知识类问题 → 必须调 search_knowledge\n"
            "3. 获取结果后，判断信息是否足够回答用户\n"
            "4. 足够 → 直接回答；不够 → 继续调工具\n"
        )
    }

    messages = [system_msg]

    # 【Day5踩坑】state["input"] 存了用户问题，但初始 messages 为空
    # LLM 只看 state["messages"] 不看 state["input"]
    # → 第一次调用时 messages 只有 system prompt，LLM 不知道用户问了什么
    # → 解法：messages 为空时把 state["input"] 作为首条 user 消息加入
    history = state.get("messages", [])
    if not history and state.get("input"):
        messages.append({"role": "user", "content": state["input"]})

    for msg in history:
        if isinstance(msg, dict):
            messages.append(msg)
        elif isinstance(msg, tuple):
            messages.append({"role": msg[0], "content": msg[1]})

    # 调 LLM
    response = llm.chat(messages=messages, tools=tools)

    new_messages = []
    if response["content"]:
        new_messages.append({"role": "assistant", "content": response["content"]})

    if response.get("tool_calls"):
        for tc in response["tool_calls"]:
            new_messages.append({
                "role": "assistant",
                "content": response.get("content", ""),
                "tool_calls": [{
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }],
            })

    return {"messages": new_messages, "steps": state["steps"] + 1}


def tool_node(state: AgentState) -> dict:
    """
    工具执行节点：执行 LLM 请求的工具调用

    执行完后返回 agent 节点继续思考（形成循环）。
    """
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else {}

    results = []
    tool_calls = last_msg.get("tool_calls", [])
    if not tool_calls:
        return {}

    for tc in tool_calls:
        fn = tc.get("function", tc)
        name = fn.get("name", "unknown")
        args_str = fn.get("arguments", "{}")

        import json
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {}

        # 【Day5踩坑】DeepSeek 首次工具调用可能用 kwargs 包装参数：
        #     {"kwargs": {"expression": "100+200"}}
        # 导致 calculator() got an unexpected keyword argument 'kwargs'
        # → 解法：检测到 kwargs 包装时自动解包
        if isinstance(args, dict) and "kwargs" in args and len(args) == 1:
            args = args["kwargs"]

        print(f"  🔧 执行工具: {name}({args})")
        result = execute_tool(name, args)
        print(f"  📤 结果: {result[:100]}")

        results.append({
            "role": "tool",
            "content": str(result),
            "tool_call_id": tc.get("id", ""),
        })

    return {"messages": results, "tool_results": [r["content"] for r in results]}


# ═══════════════════════════════════════════════════════════
# 路由函数 —— LLM 的决定在这里变成图的走向
# ═══════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    LLM 的 tool_calls 决定图的走向

    这是整个动态路由的关键：
    - LLM 返回 tool_calls → 走 tools 节点
    - LLM 返回纯文本    → 走 END

    没有硬编码的 "planner 之后一定是 coder" ——
    一切由 LLM 的返回决定。
    """
    # 步数保护
    if state["steps"] >= state.get("max_steps", 10):
        print(f"  ⏰ 超过最大步数 {state['max_steps']}，强制结束")
        return "end"

    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_msg = messages[-1]
    # LLM 有 tool_calls → 需要执行工具
    if isinstance(last_msg, dict) and last_msg.get("tool_calls"):
        return "tools"

    return "end"


# ═══════════════════════════════════════════════════════════
# 图构建
# ═══════════════════════════════════════════════════════════

def build_dynamic_graph(llm: LLMClient) -> StateGraph:
    """
    构建 LLM 动态路由图

    图结构：
         ┌──────────────────────┐
         │                      ↓
      agent ──(LLM判断)──→ tools ──→ agent (循环)
         │
         └──→ END (LLM 认为够了)

    与 Day4 固定图的关键区别：
      agent → tools → agent → ... 是 LLM 自己决定的循环
      程序员不写死 "走几步"，LLM 自己判断何时结束
    """
    tools = get_tool_definitions()

    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("agent", lambda s: agent_node(s, llm, tools))
    graph.add_node("tools", tool_node)

    # 入口
    graph.set_entry_point("agent")

    # ═══════════════════════════════════════════════════════
    # 动态路由：LLM 决定下一步
    # ═══════════════════════════════════════════════════════
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )

    # 工具执行后 → 回到 agent 继续思考
    graph.add_edge("tools", "agent")

    return graph


# ═══════════════════════════════════════════════════════════
# 对外接口
# ═══════════════════════════════════════════════════════════

class DynamicAgent:
    """
    LLM 自主决策的 Agent

    与 Day4 ReActAgent 的区别：
      - Day4: create_agent 内部黑盒，看不到决策过程
      - Day5: 每步决策在图里可见（agent → tools → agent → END）
    """

    def __init__(self, max_steps: int = 10):
        self.llm = LLMClient()
        self.max_steps = max_steps

    def run(self, user_input: str) -> dict:
        """执行一次对话，返回最终结果和决策轨迹"""
        graph = build_dynamic_graph(self.llm)
        app = graph.compile()

        result = app.invoke({
            "input": user_input,
            "messages": [],
            "tool_results": [],
            "final_answer": "",
            "steps": 0,
            "max_steps": self.max_steps,
        })

        # 提取最终回复
        answer = ""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content and not msg.get("tool_calls"):
                    answer = content
                    break

        return {
            "answer": answer,
            "steps": result["steps"],
            "tool_calls": len(result.get("tool_results", [])),
            "messages": result["messages"],
        }


# ── 便捷函数 ─────────────────────────────────────────────
def run_dynamic_agent(question: str) -> dict:
    agent = DynamicAgent()
    return agent.run(question)


# ── 模块自测 ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("动态路由 Agent 自测")
    print("=" * 50)

    agent = DynamicAgent()

    # 测试1: 需要工具的问题
    print("\n--- 测试1: 计算 ---")
    r = agent.run("计算 100+200 等于多少")
    print(f"  回答: {r['answer'][:80]}")
    print(f"  步数: {r['steps']}, 工具调用: {r['tool_calls']}次")

    # 测试2: 知识查询
    print("\n--- 测试2: 知识查询 ---")
    r = agent.run("什么是 DeepSeek？")
    print(f"  回答: {r['answer'][:80]}")
    print(f"  步数: {r['steps']}, 工具调用: {r['tool_calls']}次")

    print("\n✅ 自测完成")
