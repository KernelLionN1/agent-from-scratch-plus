"""
Day5: Prompt 分层组装 —— 借鉴 Hermes Agent 的 3 层架构

三层设计：
  stable   — 人格 + 工具指导（对话中不变，缓存不重复计算）
  context  — 上下文文件、记忆（按需刷新）
  volatile — 时间戳、预算警告（每次动态生成）

Java 类比：StringBuilder 分段拼接，每段独立管理生命周期。
"""

from datetime import datetime
from typing import Optional


class PromptBuilder:
    """
    三层 Prompt 组装器

    使用方式：
        pb = PromptBuilder()
        pb.set_personality("你是编程助手")
        pb.add_tool_guidance("calculator", "计算数学表达式")
        pb.load_context("用户叫小明，偏好简短回答")
        prompt = pb.build()  # 每次调用自动拼 volatile 层
    """

    def __init__(self):
        # ── stable 层（缓存，只设置一次）──
        self._personality: str = ""
        self._tool_descriptions: list[str] = []
        self._rules: list[str] = []

        # ── context 层（可刷新）──
        self._context_blocks: list[str] = []

        # ── stable 层缓存 ──
        self._cached_stable: Optional[str] = None
        self._dirty: bool = True  # 标记是否有新内容需要重建缓存

    # ═══════════════════════════════════════════════════════
    # stable 层：对话中不变
    # ═══════════════════════════════════════════════════════

    def set_personality(self, text: str) -> None:
        """设置 Agent 人格（如"你是编程助手"）"""
        self._personality = text
        self._dirty = True

    def add_tool_guidance(self, name: str, description: str) -> None:
        """
        添加工具说明

        Args:
            name: 工具名（如 calculator）
            description: 功能描述（如 "执行数学计算"）
        """
        self._tool_descriptions.append(f"- {name}: {description}")
        self._dirty = True

    def add_rule(self, rule: str) -> None:
        """添加行为规则"""
        self._rules.append(rule)
        self._dirty = True

    def _build_stable(self) -> str:
        """组装 stable 层（有缓存则复用）"""
        if not self._dirty and self._cached_stable is not None:
            return self._cached_stable

        parts = []

        if self._personality:
            parts.append(self._personality)

        if self._tool_descriptions:
            parts.append("\n可用工具：\n" + "\n".join(self._tool_descriptions))

        if self._rules:
            parts.append("\n规则：\n" + "\n".join(
                f"{i+1}. {r}" for i, r in enumerate(self._rules)
            ))

        self._cached_stable = "\n".join(parts)
        self._dirty = False
        return self._cached_stable

    # ═══════════════════════════════════════════════════════
    # context 层：按需加载，可刷新
    # ═══════════════════════════════════════════════════════

    def load_context(self, text: str) -> None:
        """
        加载上下文信息（可多次调用追加）

        典型用法：
        - 加载 AGENTS.md 内容
        - 加载 MEMORY.md（用户偏好）
        - 加载当前项目信息
        """
        if text.strip():
            self._context_blocks.append(text)

    def clear_context(self) -> None:
        """清空 context 层（切换项目/会话时）"""
        self._context_blocks.clear()

    def _build_context(self) -> str:
        """组装 context 层"""
        if not self._context_blocks:
            return ""
        return "[上下文]\n" + "\n".join(self._context_blocks)

    # ═══════════════════════════════════════════════════════
    # volatile 层：每次动态生成
    # ═══════════════════════════════════════════════════════

    def _build_volatile(self, budget_info: str = "") -> str:
        """
        组装 volatile 层

        Args:
            budget_info: 预算警告信息（如 "还有3步到上限"）
        """
        parts = [f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]

        if budget_info:
            parts.append(budget_info)

        return "\n".join(parts)

    # ═══════════════════════════════════════════════════════
    # 对外接口
    # ═══════════════════════════════════════════════════════

    def build(self, budget_info: str = "") -> str:
        """
        组装完整 system prompt

        三层顺序：stable → context → volatile

        Args:
            budget_info: 可选的预算警告

        Returns:
            str: 完整的 system prompt
        """
        layers = [
            self._build_stable(),
            self._build_context(),
            self._build_volatile(budget_info),
        ]
        return "\n\n".join(l for l in layers if l)

    def build_dict(self, budget_info: str = "") -> dict:
        """
        返回 {"role": "system", "content": "..."} 格式，
        可直接传给 LLMClient.chat()
        """
        return {"role": "system", "content": self.build(budget_info)}


# ── 模块自测 ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("PromptBuilder 三层组装自测")
    print("=" * 50)

    pb = PromptBuilder()

    # stable 层
    pb.set_personality("你是 ReAct Agent，擅长数学计算和知识检索。")
    pb.add_tool_guidance("calculator", "执行数学表达式计算")
    pb.add_tool_guidance("search_knowledge", "搜索 Python 知识库")
    pb.add_rule("计算类问题 → 必须调 calculator")
    pb.add_rule("知识类问题 → 必须调 search_knowledge")

    # context 层
    pb.load_context("用户偏好：简短回答，不啰嗦")
    pb.load_context("项目: agent-from-scratch-plus Day5 教学")

    # 第一次 build（缓存未命中）
    p1 = pb.build()
    print(f"\n第一次 build ({len(p1)} chars):")
    print(p1[:200])

    # 第二次 build（缓存命中，stable 层不重建）
    p2 = pb.build("剩余迭代: 7/10")
    print(f"\n第二次 build ({len(p2)} chars) — 加预算警告:")
    print(p2[-100:])

    # 修改 stable → 缓存失效
    pb.add_rule("获取结果后判断信息是否足够，不够继续调工具")
    p3 = pb.build()
    print(f"\n修改后 rebuild ({len(p3)} chars):")
    print(p3[:200])

    print("\n✅ 自测完成")
