"""
LLM 客户端封装 —— 对接 DeepSeek API（OpenAI 兼容协议）

核心职责：把「调用 LLM」这件事封装成一个类，外部只需要传 messages 和 tools，
不需要关心 API Key 从哪来、HTTP 请求怎么发、返回值怎么解析。

刻意踩坑点：
- 不做请求重试（让调用方自己处理错误）
- 不做超时控制（后面阶段再加）
"""

# ── 导入依赖 ──────────────────────────────────────────────
import os
from dotenv import load_dotenv  # 从 .env 文件读取环境变量
from openai import OpenAI        # OpenAI SDK，兼容 DeepSeek

# 模块加载时自动读取 .env 文件，之后 os.getenv() 就能拿到配置
load_dotenv()


# ── LLM 客户端类 ──────────────────────────────────────────
class LLMClient:
    """
    封装 DeepSeek API 调用

    使用方式：
        client = LLMClient()
        response = client.chat(messages=[...], tools=[...])
    """

    def __init__(self):
        """
        初始化客户端 —— 从环境变量读取配置并创建 OpenAI 连接

        对应 Java 里读取 application.yml 后构建 HttpClient，
        区别是 Python 用 os.getenv() 直读环境变量。
        """
        # 从 .env 读 API Key —— 绝不能硬编码在代码里
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        # DeepSeek 的 API 地址 —— 和 OpenAI 不冲突，base_url 指向别处
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        # 模型名，默认 deepseek-chat
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        # 没有 Key 直接报错 —— 早失败比晚失败好调试
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY 未设置，请检查 .env 文件")

        # 创建 OpenAI 客户端实例 —— 底层封装了 HTTP 连接池
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict:
        """
        调用 LLM，返回统一格式的 dict

        这是整个 Agent 系统最核心的方法 —— 所有 LLM 交互都走这里。

        参数：
            messages:   对话历史，OpenAI 格式 [{"role":"user","content":"..."}]
            tools:      可选，工具定义列表。传了就开启 function calling
            temperature: 0~2，越高越随机。0=确定性强，适合代码/数学
            max_tokens:  限制 LLM 最大输出长度，防止无限生成

        返回：统一 dict，外部不需要关心里面是怎么解析的
            {
                "content":    str | None,   # LLM 直接回复的文本
                "tool_calls": list | None,  # LLM 想调用的工具列表
                "model":      str,          # 实际使用的模型名
                "usage":      dict,         # {"prompt_tokens", "completion_tokens", "total_tokens"}
            }

        content 和 tool_calls 互斥：LLM 要么说话要么调工具，不会同时做两件事。
        """
        # ── 构建请求参数 ──
        # 相当于 Java 里构建一个 Request DTO
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # 如果有工具定义就带上 —— 这告诉 LLM "你可以用这些工具"
        if tools:
            kwargs["tools"] = tools

        # ── 发出 HTTP 请求 ──
        # 底层：POST https://api.deepseek.com/v1/chat/completions
        # 请求体是 JSON，响应体也是 JSON
        # OpenAI SDK 帮我们做了序列化/反序列化
        response = self.client.chat.completions.create(**kwargs)

        # ── 解析响应 ──
        # response.choices 是一个列表（通常只有一个元素）
        choice = response.choices[0]
        msg = choice.message  # OpenAI SDK 的消息对象

        # ── 构建统一返回格式 ──
        # 把 SDK 对象转成纯 dict，外部调用方不需要 import openai
        return {
            # msg.content 可能是 None（当 LLM 决定调工具时）
            "content": msg.content,
            # 从 message 对象里提取 tool_calls，封装成自己的格式
            "tool_calls": _parse_tool_calls(msg),
            # 记录实际使用的模型（DeepSeek 内部可能做模型路由）
            "model": response.model,
            # token 用量 —— 方便做成本监控
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
        }


# ── 工具函数：解析 LLM 的工具调用请求 ─────────────────────
def _parse_tool_calls(message) -> list[dict] | None:
    """
    从 OpenAI 的 message 对象中提取 tool_calls

    OpenAI SDK 原格式：
        message.tool_calls = [
            ToolCall(
                id="call_xxx",
                function=FunctionCall(
                    name="calculator",
                    arguments='{"expression": "1+1"}'  # 注意：这是 JSON 字符串，不是 dict
                )
            )
        ]

    我们转成纯 dict 列表：
        [{"id": "call_xxx", "name": "calculator", "arguments": '{"expression": "1+1"}'}]

    为什么不在这一层就 json.loads(arguments)？
    —— 留给调用方决定怎么解析，保持数据原样传递。
    """
    # LLM 没想调工具，直接返回 None
    if not message.tool_calls:
        return None

    result = []
    for tc in message.tool_calls:
        result.append({
            "id": tc.id,                          # 工具调用的唯一 ID，后续关联结果用
            "name": tc.function.name,             # 工具名，如 "calculator"
            "arguments": tc.function.arguments,   # JSON 字符串，需要调用方自己 json.loads()
        })
    return result
