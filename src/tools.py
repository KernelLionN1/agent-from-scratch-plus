"""
工具注册中心 + 内置工具

核心设计思路（对标 Java 的 IoC 容器）：
- TOOL_REGISTRY 是一个全局字典，等同于 Map<String, ToolMeta>
- register_tool() = @Service + @Bean 注册
- execute_tool()  = 从容器取 bean，反射调用

刻意踩坑点：
- 计算器：不做参数校验，字符串参数直接 eval（危险 + 脆弱）
- 文本查询：返回值不清洗为字符串，直接返回 dict（LLM 难以理解）
- 工具描述：先写模糊的描述，观察 LLM 误调用（面试重点：为什么工具描述很重要）
"""

# ── 导入依赖 ──────────────────────────────────────────────
import json  # 用于解析 LLM 返回的 JSON 参数
import re    # Day3 修复 #1：用正则做表达式白名单校验，替代裸 eval


# ============================================================
# 第一部分：工具注册基础设施（Registry Pattern）
# ============================================================

# 全局工具注册表 —— 整个应用唯一的一份
# 类型：dict[str, dict]  →  {"calculator": {name, description, parameters, func}, ...}
TOOL_REGISTRY: dict = {}


def register_tool(name: str, description: str, parameters: dict, func):
    """
    注册一个工具到全局注册表

    这是「工具注册」的唯一入口，类似 Java 的 applicationContext.registerBean()

    参数：
        name:        工具唯一名称，如 "calculator"，LLM 通过这个名字调用
        description: 工具用途描述 —— LLM 靠这个决定"什么时候用这个工具"
        parameters:  JSON Schema 格式的参数定义 —— 告诉 LLM "这个工具需要什么输入"
        func:        实际执行的 Python 函数

    设计要点：
        - 用全局 dict 而非类属性，简单直接，不需要单例模式
        - 所有工具在模块加载时注册（见文件底部），启动即完成
    """
    TOOL_REGISTRY[name] = {
        "name": name,
        "description": description,
        "parameters": parameters,
        "func": func,  # 保存函数引用，后续 execute_tool() 取出执行
    }


def get_tool_definitions() -> list[dict]:
    """
    返回 OpenAI 格式的工具定义列表（不含 func）

    这个方法把内部注册表转成 LLM API 要求的格式。
    注意：故意不返回 func，因为 func 是 Python 对象，不能序列化发给 API。

    返回格式：
    [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "执行数学计算",
                "parameters": {"type": "object", "properties": {...}, "required": [...]}
            }
        },
        ...
    ]
    """
    return [
        {
            "type": "function",  # OpenAI 固定字段
            "function": {
                "name": meta["name"],
                "description": meta["description"],
                "parameters": meta["parameters"],
            },
        }
        for meta in TOOL_REGISTRY.values()
    ]


def execute_tool(name: str, arguments: dict) -> str:
    """
    根据工具名和参数，执行对应的工具函数

    这是整个工具系统的调度入口 —— LLM 说要调哪个工具，这里就执行哪个。

    参数：
        name:      工具名，如 "calculator"
        arguments: LLM 传过来的参数，如 {"expression": "123*456"}

    返回：工具执行结果，强制转为字符串

    刻意踩坑点：
        - `func(**arguments)` 是 Python 的参数解包语法，
          相当于 func(expression="123*456")，
          如果 LLM 传了不存在的参数名，这里直接报错（不做容错）
        - str(result) 强转 —— 如果工具返回 dict 或复杂对象，
          str() 后的格式 LLM 可能无法理解
    """
    # 工具不存在时给出明确错误信息，而非 KeyError 堆栈
    if name not in TOOL_REGISTRY:
        return f"Error: 未知工具 '{name}'"

    # 从注册表取出函数引用，解包参数后执行
    func = TOOL_REGISTRY[name]["func"]
    result = func(**arguments)

    # 强制转字符串返回 —— 如果 result 是 dict，str() 后可能很难读
    return str(result)


# ============================================================
# 第二部分：内置工具实现
# ============================================================

# ── 工具 1：计算器（Day3 修复 #1：safe_eval 替代裸 eval）─
def calculator(expression: str) -> float:
    """
    计算器工具 —— 用安全校验后的 eval() 执行数学表达式

    Day3 修复 #1：从裸 eval() 升级为三步安全校验
      步骤1: 白名单正则 — 只允许数字、运算符、括号、空格
      步骤2: 受限命名空间 — {"__builtins__": {}} 禁止所有内置函数
      步骤3: try/except 保护 — 除零等运行时错误转为友好提示

    修复前风险：eval("__import__('os').system('ls')") 会直接执行系统命令
    修复后行为：白名单校验不通过 → 抛出 ValueError
    """
    # ── 步骤1：白名单字符校验 ──
    # 只允许：数字(0-9)、小数点、四则运算符(+-*/)、幂(^)、
    #         取模(%)、括号(())、空格、下划线
    if not re.match(r'^[\d\s+\-*/()\.\%\^_]+$', expression):
        raise ValueError(f"表达式包含不安全字符: '{expression}'")

    # ── 步骤2+3：受限 eval + 异常保护 ──
    try:
        # __builtins__={} → 禁止所有内置函数（如 __import__, open, exec 等）
        # 这样即使正则被绕过，eval 也无法执行任何函数调用
        return eval(expression, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        raise ValueError("除数不能为零")
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {e}")
    except Exception as e:
        raise ValueError(f"计算失败: {e}")


# 注册计算器工具 —— 模块加载时自动执行
register_tool(
    name="calculator",
    description="执行数学计算",
    # ↑ 坑点：描述太模糊。LLM 不知道这个计算器能做什么运算、输入格式是什么
    #   更好的描述："执行四则运算和数学表达式求值，输入为数学表达式字符串，如 '123*456'、'(1+2)*3'"
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                # ↑ 坑点：没有 description 字段，LLM 不知道 expression 应该长什么样
                #   应该加: "description": "数学表达式，如 '1+2*3'，支持 + - * / ** () 等运算符"
            }
        },
        "required": ["expression"],
    },
    func=calculator,
)


# ── 工具 2：文本查询 ─────────────────────────────────────
# 模拟知识库 —— 实际工程中这会是一个向量数据库或搜索引擎
# 下划线前缀 _ 表示这是模块私有变量（Python 约定，不是强制的）
_KNOWLEDGE_BASE = {
    "python": "Python 是一门解释型、面向对象的高级编程语言，由 Guido van Rossum 于 1991 年发布。",
    "deepseek": "DeepSeek 是由深度求索公司开发的 AI 大模型，支持对话、代码生成等功能。",
    "react": "ReAct（Reasoning + Acting）是一种将推理和行动交替进行的 AI Agent 模式。",
    "fastapi": "FastAPI 是一个现代、高性能的 Python Web 框架，基于 Starlette 和 Pydantic。",
}


def search_knowledge(query: str) -> str:
    """
    文本查询工具 —— 从本地知识库中搜索（Day3 修复 #2：返回格式化文本）

    Day3 修复 #2：返回类型从 dict 改为 str
      修复前：return result  # dict → str() → "{'key': 'value'}"  LLM 难理解
      修复后：return 格式化文本   # 每行 "[key]: value"，LLM 一目了然
      空结果返回友好提示，而非空 dict

    仍保留的教学踩坑：
    1. 搜索逻辑极其简陋 —— 仅做子串匹配
    2. 搜索方向问题 —— query in key（反了），但加了 value 方向弥补
    """
    result = {}
    for key, value in _KNOWLEDGE_BASE.items():
        # 双向匹配：query 在 key 中，或 query 在 value 中
        if query.lower() in key.lower() or query.lower() in value.lower():
            result[key] = value

    # ── Day3 修复 #2：格式化为 LLM 友好的文本 ──
    if not result:
        return f"未找到与「{query}」相关的知识。知识库当前包含：{list(_KNOWLEDGE_BASE.keys())}"

    lines = [f"搜索「{query}」的结果（{len(result)} 条）："]
    for key, value in result.items():
        lines.append(f"  [{key}]: {value}")
    return "\n".join(lines)


# 注册文本查询工具
register_tool(
    name="search_knowledge",
    description="搜索知识库",
    # ↑ 坑点：描述太模糊，没说明知识库里有什么、支持什么类型的查询
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                # ↑ 同样没有 description，LLM 不知道 query 是该写关键词还是自然语言
            }
        },
        "required": ["query"],
    },
    func=search_knowledge,
)


# ============================================================
# 第三部分：便捷入口
# ============================================================

def list_tools() -> list[str]:
    """列出所有已注册工具的名称 —— 调试和日志用"""
    return list(TOOL_REGISTRY.keys())


# ── 模块加载时输出注册信息 ──
# 这样做的好处：import 时就确认工具注册成功，不用等到运行出错才发现
print(f"[tools] 已注册 {len(TOOL_REGISTRY)} 个工具: {list_tools()}")
