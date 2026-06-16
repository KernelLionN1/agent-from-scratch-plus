"""
角色定义与 System Prompt 模板 —— Day2 阶段一

核心职责：
- 定义多 Agent 系统中的角色枚举（Planner / Coder / Reviewer）
- 为每个角色提供专属 system prompt
- 提供角色配置的便捷访问

刻意踩坑点：
- 【踩坑】角色 Prompt 边界模糊：prompt 写得不够精确，Agent 可能越权
  比如 Planner 可能直接写代码，Coder 可能自己规划任务
  保留此坑观察越权行为，Day3 用严格 prompt 修复

类比 Java：
    AgentRole 相当于一个 Enum 类，
    ROLE_PROMPTS 相当于 Enum 的 getDescription() 方法返回不同字符串，
    RoleConfig 相当于一个不可变 DTO（record class）。
"""

# ── 枚举定义 ──────────────────────────────────────────────
from enum import Enum


class AgentRole(Enum):
    """
    多 Agent 系统中的角色枚举

    每个角色有不同的职责：
    - PLANNER：  接收需求，拆分子任务
    - CODER：    接收子任务，生成代码
    - REVIEWER： 接收代码，审核反馈

    类比 Java：
        public enum AgentRole { PLANNER, CODER, REVIEWER }
    """
    PLANNER = "planner"     # 规划者：拆解需求 → 子任务列表
    CODER = "coder"         # 编码者：子任务 → 代码实现
    REVIEWER = "reviewer"   # 审核者：代码 → 审核意见


# ── System Prompt 模板 ────────────────────────────────────
# 【踩坑】这里刻意把 prompt 写得比较模糊，不严格限定角色边界
# 目的：观察 Agent 是否会越权（比如 Planner 直接写代码）
# Day3 会加上「你只能做 X，不能做 Y」的严格约束

ROLE_PROMPTS: dict[AgentRole, str] = {
    AgentRole.PLANNER: (
        "你是一个任务规划专家。"
        "你的职责是分析用户需求，将复杂任务拆分为可执行的子任务。"
        "对于每个子任务，给出清晰的描述和预期输出格式。"
        # 【踩坑】没有写「你只能规划，不能执行」
        # 这可能导致 Planner 越权直接写代码
    ),

    AgentRole.CODER: (
        "你是一个 Python 编程专家。"
        "你的职责是根据任务描述生成高质量的 Python 代码。"
        "代码应该清晰、有注释、可以直接运行。"
        # 【踩坑】没有写「你只能编码，不能规划或审核」
        # Coder 可能会对任务描述提出质疑或自行修改规划
    ),

    AgentRole.REVIEWER: (
        "你是一个代码审核专家。"
        "你的职责是审查代码质量，指出问题和改进建议。"
        "关注：正确性、可读性、性能、安全性。"
        # 【踩坑】没有写「你只能审核，不能改代码」
        # Reviewer 可能会直接修改代码而不是给出建议
    ),
}


# ── 角色配置类 ────────────────────────────────────────────
class RoleConfig:
    """
    角色的完整配置 —— 捆绑角色、prompt、推荐参数

    相当于 Java 里一个不可变的配置 DTO：
        record RoleConfig(AgentRole role, String prompt, double temperature) {}

    用途：外部创建 Agent 实例时，根据角色获取全套配置。
    """

    def __init__(self, role: AgentRole):
        """
        根据角色枚举创建配置

        参数：
            role: 角色枚举值

        初始化后会填充：
            self.role        — 角色枚举
            self.prompt      — 对应的 system prompt
            self.temperature — 推荐温度值（Planner=0.7 偏创意，Coder/Reviewer=0.3 偏精确）
        """
        self.role = role
        self.prompt = ROLE_PROMPTS[role]

        # 不同角色推荐不同的 temperature：
        # - Planner 需要发散思维拆分任务 → 高一点
        # - Coder 和 Reviewer 需要精确 → 低一点
        self.temperature = 0.7 if role == AgentRole.PLANNER else 0.3

    def __repr__(self) -> str:
        """打印友好的角色信息"""
        return f"RoleConfig(role={self.role.value}, temp={self.temperature})"


# ── 便捷函数 ──────────────────────────────────────────────
def get_role_prompt(role: AgentRole) -> str:
    """
    获取指定角色的 system prompt

    参数：
        role: AgentRole 枚举值

    返回：
        对应的 prompt 字符串

    用法：
        prompt = get_role_prompt(AgentRole.PLANNER)
    """
    return ROLE_PROMPTS[role]


def get_role_config(role: AgentRole) -> RoleConfig:
    """
    获取指定角色的完整配置

    参数：
        role: AgentRole 枚举值

    返回：
        RoleConfig 实例，包含 prompt + temperature 推荐值

    用法：
        config = get_role_config(AgentRole.CODER)
        messages = [{"role": "system", "content": config.prompt}, ...]
    """
    return RoleConfig(role)


# ── 模块自测 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("角色定义模块自测")
    print("=" * 50)

    # 遍历所有角色，打印配置
    for role in AgentRole:
        config = get_role_config(role)
        print(f"\n🎭 {role.name} ({role.value})")
        print(f"   Temperature: {config.temperature}")
        print(f"   Prompt 长度: {len(config.prompt)} 字符")
        print(f"   Prompt 预览: {config.prompt[:80]}...")

    # 验证便捷函数
    print(f"\n🔍 get_role_prompt(PLANNER) 前60字:")
    print(f"   {get_role_prompt(AgentRole.PLANNER)[:60]}")
