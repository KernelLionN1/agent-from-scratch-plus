"""
工具注册 + 内置工具 —— Day4 框架版：基于 LangChain @tool 装饰器

Day4 核心变化：
  手搓版（Day1-3）：手动 JSON Schema + 全局 TOOL_REGISTRY + 手动 execute_tool()
  框架版（Day4）：  @tool 装饰器自动生成 Schema + LangChain 工具绑定

关键收益：
  - @tool 装饰器自动从函数签名+docstring 生成 JSON Schema
  - 不再需要手动写 parameters dict
  - LangChain 的 bind_tools() 自动注入工具定义
  - 仍然保留 execute_tool() 兼容老接口

Java 类比：手写 OpenAPI JSON Schema → @ApiOperation + Swagger 自动生成
"""

import re  # Day3 修复 #1：白名单正则

# ── 工具注册基础设施（保留，兼容老代码）───────────────────
TOOL_REGISTRY: dict = {}


def register_tool(name: str, description: str, parameters: dict, func):
    """注册工具（兼容老接口）"""
    TOOL_REGISTRY[name] = {
        "name": name,
        "description": description,
        "parameters": parameters,
        "func": func,
    }


def get_tool_definitions() -> list[dict]:
    """返回 OpenAI 格式的工具定义列表"""
    return [
        {
            "type": "function",
            "function": {
                "name": meta["name"],
                "description": meta["description"],
                "parameters": meta["parameters"],
            },
        }
        for meta in TOOL_REGISTRY.values()
    ]


def execute_tool(name: str, arguments: dict) -> str:
    """执行工具（兼容老接口）"""
    if name not in TOOL_REGISTRY:
        return f"Error: 未知工具 '{name}'"
    func = TOOL_REGISTRY[name]["func"]
    try:
        result = func(**arguments)
        return str(result)
    except Exception as e:
        return f"工具执行失败 ({type(e).__name__}): {e}"


def list_tools() -> list[str]:
    """列出所有已注册工具"""
    return list(TOOL_REGISTRY.keys())


# ── 内置工具（Day4：用 @tool 装饰器）─────────────────────

# Day4 新风格：@tool 装饰器，LangChain 自动生成 Schema
# 同时注册到 TOOL_REGISTRY，兼容手搓版 agent 和 api

def calculator(expression: str) -> float:
    """
    执行数学计算 —— Day4：docstring 自动变成 tool description

    Args:
        expression: 数学表达式，如 "1+2*3"、"100/4"
    """
    if not re.match(r'^[\d\s+\-*/()\.\%\^_]+$', expression):
        raise ValueError(f"表达式包含不安全字符: '{expression}'")
    try:
        return eval(expression, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        raise ValueError("除数不能为零")
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {e}")


# 知识库
_KNOWLEDGE_BASE = {
    "python": "Python 是一门解释型、面向对象的高级编程语言，由 Guido van Rossum 于 1991 年发布。",
    "deepseek": "DeepSeek 是由深度求索公司开发的 AI 大模型，支持对话、代码生成等功能。",
    "react": "ReAct（Reasoning + Acting）是一种将推理和行动交替进行的 AI Agent 模式。",
    "fastapi": "FastAPI 是一个现代、高性能的 Python Web 框架，基于 Starlette 和 Pydantic。",
}


def search_knowledge(query: str) -> str:
    """
    搜索知识库 —— Day4：docstring 描述工具用途

    知识库包含 Python、DeepSeek、ReAct、FastAPI 等条目。

    Args:
        query: 搜索关键词，如 "python"、"deepseek"
    """
    result = {}
    for key, value in _KNOWLEDGE_BASE.items():
        if query.lower() in key.lower() or query.lower() in value.lower():
            result[key] = value

    if not result:
        return f"未找到与「{query}」相关的知识。知识库当前包含：{list(_KNOWLEDGE_BASE.keys())}"

    lines = [f"搜索「{query}」的结果（{len(result)} 条）："]
    for key, value in result.items():
        lines.append(f"  [{key}]: {value}")
    return "\n".join(lines)


# ── 注册到兼容层 ────────────────────────────────────────
register_tool(
    name="calculator",
    description="执行数学计算，支持四则运算。输入为数学表达式字符串，如 '123*456'。",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '1+2*3'",
            }
        },
        "required": ["expression"],
    },
    func=calculator,
)

register_tool(
    name="search_knowledge",
    description="搜索本地知识库（Python/DeepSeek/ReAct/FastAPI），输入查询关键词。",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "查询关键词",
            }
        },
        "required": ["query"],
    },
    func=search_knowledge,
)

# ── Day4 新增：LangChain 原生工具列表 ──────────────────
# 用于 llm_client.bind_tools()，不再依赖手搓 JSON Schema


def get_langchain_tools():
    """
    Day4：返回 LangChain 原生 tool 对象列表

    用于 AgentExecutor 等高层 API，不需要手动 bind_tools
    注意：当前用 StructuredTool.from_function 包装，实际执行仍回调手搓函数
    """
    from langchain_core.tools import StructuredTool

    return [
        StructuredTool.from_function(
            func=calculator,
            name="calculator",
            description="执行数学计算，如 '1+2*3'。输入 expression 参数。",
        ),
        StructuredTool.from_function(
            func=search_knowledge,
            name="search_knowledge",
            description="搜索知识库（Python/DeepSeek/ReAct/FastAPI）。输入 query 参数。",
        ),
    ]


print(f"[tools] 已注册 {len(TOOL_REGISTRY)} 个工具: {list_tools()}")
