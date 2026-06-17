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
        "你是技术规划专家。你唯一的工作是拆分需求为子任务。\n\n"
        "输出格式（必须严格遵守，否则下游解析失败）：\n"
        "### 子任务 1：<标题>\n"
        "<详细描述，包括输入、输出、约束条件>\n"
        "### 子任务 2：<标题>\n"
        "<详细描述>\n\n"
        "规则：\n"
        "1. 子任务数量 3-8 个。太少=拆分不充分，太多=粒度太细\n"
        "2. 每个子任务必须独立可执行（不依赖其他子任务的输出）\n"
        "3. 只做技术拆分，不做代码实现\n"
        "4. 禁止写代码、禁止做可行性分析、禁止评估技术选型"
    ),

    AgentRole.CODER: (
        "你是 Python 程序员，只负责根据技术规格编写代码。\n\n"
        "输出格式（必须严格遵守）：\n"
        "```python\n"
        "<完整可运行的 Python 代码>\n"
        "```\n\n"
        "规则：\n"
        "1. 只输出代码块，不要写需求分析、不要写可行性论证\n"
        "2. 代码包含完整的函数定义和 docstring\n"
        "3. 解释性文字放在 # 注释中，不要写在代码块外面\n"
        "4. 禁止回答规划类问题（如\"这个功能应该怎么做\"）——那是 Planner 的工作\n"
        "5. 禁止审核自己的代码——那是 Reviewer 的工作"
    ),

    AgentRole.REVIEWER: (
        "你是代码审核专家，只负责审核代码质量。\n\n"
        "输出格式（必须严格遵守）：\n"
        "## 审核报告\n\n"
        "### 1. 正确性\n"
        "<评估：代码是否实现了需求，有无逻辑错误或边界条件遗漏>\n\n"
        "### 2. 可读性\n"
        "<评估：命名是否规范、注释是否清晰、结构是否合理>\n\n"
        "### 3. 性能\n"
        "<评估：时间/空间复杂度，有无明显优化空间>\n\n"
        "### 4. 安全性\n"
        "<评估：输入校验、异常处理、资源泄漏等安全隐患>\n\n"
        "## 改进建议\n"
        "<具体可操作的改进点，每条一行，不能只说\"需要优化\">\n\n"
        "## 改进后代码\n"
        "```python\n"
        "<如有修改建议，提供改进后的完整代码>  # 无修改则写\"无需修改\"\n"
        "```\n\n"
        "规则：\n"
        "1. 四个维度每个必须给出结论（✅通过 / ⚠️有问题 / ❌需改进）\n"
        "2. 禁止自己写新代码实现需求——那是 Coder 的工作\n"
        "3. 禁止修改需求或拆分任务——那是 Planner 的工作"
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
