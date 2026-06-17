"""
编排器 —— Day4 框架版：LangGraph StateGraph 替代手搓 Orchestrator

Day4 核心变化：
  手搓版（Day2-3）：Orchestrator + MessageBus + SharedState 三个模块协作
  框架版（Day4）：  一张 LangGraph 图替代全部

手搓版三个模块各做什么 → 框架版怎么替代：
  Orchestrator: if/else 控制串行/并行 → StateGraph 的节点+边
  MessageBus:   Agent 间消息传递       → 不需要了，State 自动传递
  SharedState:  全局共享状态+乐观锁     → 不需要了，State dict 就是状态

Java 类比：手写 if/else 工作流 → Spring StateMachine 声明式状态图
"""

import asyncio
import re
import time
from typing import TypedDict, Annotated, Literal
from operator import add

from src.agents.planner import PlannerAgent
from src.agents.coder import CoderAgent
from src.agents.reviewer import ReviewerAgent
from src.message_bus import MessageBus

# ── Day4: LangGraph 核心 ──────────────────────────────────
from langgraph.graph import StateGraph, END


# ═══════════════════════════════════════════════════════════
# State 定义 —— 图中流转的共享状态
# 等价于手搓版的 SharedState + MessageBus 的结合体
# ═══════════════════════════════════════════════════════════

class OrchestratorState(TypedDict):
    """LangGraph 图的共享状态"""
    requirement: str          # 用户需求
    plan: str                 # Planner 输出
    subtasks: list[str]       # 解析后的子任务列表
    code: str                 # Coder 输出（串行）
    codes: list[dict]         # Coder 输出列表（并行）
    review: str               # Reviewer 输出
    fixed_code: str           # 修复后代码
    mode: str                 # "serial" 或 "parallel"
    n_coders: int             # 并行 Coder 数量
    with_fix: bool            # 是否执行修复步骤
    error: str                # 错误信息


# ═══════════════════════════════════════════════════════════
# 节点函数 —— 每个节点等价于手搓版的一个执行步骤
# ═══════════════════════════════════════════════════════════

def _planner_node(state: OrchestratorState, bus: MessageBus) -> dict:
    """Planner 节点：拆分需求 → 子任务列表"""
    print("\n📋 Planner 拆分需求...")
    planner = PlannerAgent(bus, "planner-1")
    plan = planner.execute(state["requirement"])
    subtasks = _parse_subtasks(plan)[:state.get("n_coders", 2)]
    print(f"   规划: {len(plan)}字, 拆出{len(subtasks)}个子任务")
    return {"plan": plan, "subtasks": subtasks}


def _coder_node_serial(state: OrchestratorState, bus: MessageBus) -> dict:
    """Coder 节点（串行）：一个 Coder 处理全部"""
    print("\n💻 Coder 生成代码...")
    coder = CoderAgent(bus, "coder-1")
    code = coder.execute(state["plan"])
    print(f"   代码: {len(code)}字")
    return {"code": code}


def _coder_node_parallel(state: OrchestratorState, bus: MessageBus) -> dict:
    """Coder 节点（并行）：多个 Coder 各自处理子任务"""
    subtasks = state["subtasks"]
    print(f"\n💻 {len(subtasks)} 个 Coder 并行生成代码...")

    coders = [CoderAgent(bus, f"coder-{i+1}") for i in range(len(subtasks))]

    async def _run():
        import asyncio
        tasks = []
        for i in range(len(subtasks)):
            tasks.append(_coder_async(coders[i], subtasks[i], i + 1))
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = asyncio.run(_run())
    codes = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            codes.append({"task_id": i + 1, "code": f"# 错误: {r}", "error": str(r)})
        else:
            codes.append(r)
    print(f"   完成: {len(codes)}份代码")
    return {"codes": codes}


def _reviewer_node(state: OrchestratorState, bus: MessageBus) -> dict:
    """Reviewer 节点：审核代码"""
    print("\n🔍 Reviewer 审核...")
    reviewer = ReviewerAgent(bus, "reviewer-1")
    code_to_review = state.get("code") or _combine_codes(state.get("codes", []))
    review = reviewer.execute(code_to_review)
    print(f"   审核: {len(review)}字")
    return {"review": review}


def _fixer_node(state: OrchestratorState, bus: MessageBus) -> dict:
    """Coder 修复节点：根据审核意见修代码"""
    print("\n🔧 Coder 修复代码...")
    coder = CoderAgent(bus, "coder-fixer")
    fixed = coder.fix_code(state["review"])
    print(f"   修复后: {len(fixed)}字")
    return {"fixed_code": fixed}


# ═══════════════════════════════════════════════════════════
# 路由函数 —— 等价于手搓版的 if/else 控制流
# ═══════════════════════════════════════════════════════════

def _route_after_planner(state: OrchestratorState) -> str:
    """Planner 之后走串行还是并行？"""
    return "coder_parallel" if state["mode"] == "parallel" else "coder_serial"


def _route_after_review(state: OrchestratorState) -> str:
    """审核后是否执行修复？"""
    return "fixer" if state.get("with_fix", False) else END


# ═══════════════════════════════════════════════════════════
# 图构建 —— 声明式定义节点和边
# 等价于手搓版 run_serial() / run_parallel() 中的控制流
# ═══════════════════════════════════════════════════════════

_bus = MessageBus()  # 共享消息总线（Agent 工厂函数用）


def build_graph() -> StateGraph:
    """
    构建 LangGraph 编排图

    串行模式:
      planner → coder_serial → reviewer → [fixer →] END

    并行模式:
      planner → coder_parallel → reviewer → END
    """
    graph = StateGraph(OrchestratorState)

    # ── 注册节点 ──
    graph.add_node("planner", lambda s: _planner_node(s, _bus))
    graph.add_node("coder_serial", lambda s: _coder_node_serial(s, _bus))
    graph.add_node("coder_parallel", lambda s: _coder_node_parallel(s, _bus))
    graph.add_node("reviewer", lambda s: _reviewer_node(s, _bus))
    graph.add_node("fixer", lambda s: _fixer_node(s, _bus))

    # ── 注册边（流程控制）──
    graph.set_entry_point("planner")

    # planner → 条件路由（串行 or 并行）
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"coder_serial": "coder_serial", "coder_parallel": "coder_parallel"},
    )

    # coder → reviewer
    graph.add_edge("coder_serial", "reviewer")
    graph.add_edge("coder_parallel", "reviewer")

    # reviewer → 条件路由（修不修？）
    graph.add_conditional_edges(
        "reviewer",
        _route_after_review,
        {"fixer": "fixer", END: END},
    )

    graph.add_edge("fixer", END)

    return graph


# ═══════════════════════════════════════════════════════════
# 对外接口 —— 兼容手搓版 Orchestrator
# ═══════════════════════════════════════════════════════════

class Orchestrator:
    """
    Day4 编排器 —— LangGraph 版

    接口与 Day2-3 的 Orchestrator 完全兼容。
    内部用 StateGraph 替代 if/else 控制流。
    """

    def __init__(self, bus: MessageBus, with_shared_state: bool = False):
        self.bus = bus

    def run_serial(self, requirement: str, with_fix: bool = True) -> dict:
        """串行编排（LangGraph 版）"""
        graph = build_graph()
        app = graph.compile()

        state = app.invoke({
            "requirement": requirement,
            "mode": "serial",
            "with_fix": with_fix,
            "n_coders": 1,
        })

        return {
            "plan": state.get("plan", ""),
            "code": state.get("code", ""),
            "review": state.get("review", ""),
            "fixed_code": state.get("fixed_code", ""),
            "stats": {
                "plan_chars": len(state.get("plan", "")),
                "code_chars": len(state.get("code", "")),
                "review_chars": len(state.get("review", "")),
                "fixed_code_chars": len(state.get("fixed_code", "")),
            },
        }

    def run_parallel(self, requirement: str, n_coders: int = 2, timeout: float = 120) -> dict:
        """并行编排（LangGraph 版）"""
        graph = build_graph()
        app = graph.compile()

        t0 = time.time()
        state = app.invoke({
            "requirement": requirement,
            "mode": "parallel",
            "n_coders": n_coders,
            "with_fix": False,
        })
        elapsed = time.time() - t0

        codes = state.get("codes", [])
        combined = "\n\n".join([
            f"# === 子任务 {c.get('task_id', '?')} ===\n{c.get('code', '')}"
            for c in codes
        ])

        return {
            "plan": state.get("plan", ""),
            "codes": codes,
            "combined": combined,
            "review": state.get("review", ""),
            "timing": {"plan": 0, "coders": elapsed, "review": 0},
        }


# ── 工具函数（同手搓版）──────────────────────────────────

def _parse_subtasks(plan_text: str) -> list[str]:
    """正则拆分 Planner 输出为子任务列表"""
    parts = re.split(r'(?:###?\s*子任务\s*\d+|【子任务\d】)', plan_text)
    subtasks = [p.strip() for p in parts if len(p.strip()) > 30]
    if not subtasks:
        return [plan_text]
    return subtasks


async def _coder_async(coder, subtask: str, task_id: int) -> dict:
    """异步 Coder 执行（用于并行）"""
    code = await asyncio.to_thread(coder.execute, subtask)
    return {"task_id": task_id, "subtask": subtask, "code": code}


def _combine_codes(codes: list[dict]) -> str:
    """合并多份代码"""
    return "\n\n".join([
        f"# === 子任务 {c.get('task_id', '?')} ===\n{c.get('code', '')}"
        for c in codes
    ])
