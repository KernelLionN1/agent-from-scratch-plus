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
import time  # Day3 修复 #10：指数退避重试用
from dotenv import load_dotenv  # 从 .env 文件读取环境变量
from openai import OpenAI        # OpenAI SDK，兼容 DeepSeek
# Day3 修复 #10：按可重试/不可重试分类 OpenAI 异常
from openai import (
    APITimeoutError,       # 网络超时 — 可重试
    RateLimitError,        # 429 限流 — 可重试
    APIConnectionError,    # 连接失败 — 可重试
    InternalServerError,   # 5xx — 可重试
    AuthenticationError,   # 401 — 不可重试
    BadRequestError,       # 400 — 不可重试
)

# 模块加载时自动读取 .env 文件，之后 os.getenv() 就能拿到配置
load_dotenv()


# ── LLM 客户端类 ──────────────────────────────────────────
class LLMClient:
    """
    封装 DeepSeek API 调用（Day3 修复 #10：加重试机制）

    Day3 修复 #10：从裸调升级为指数退避重试
      可重试错误（网络抖动）：超时/429限流/连接失败/5xx → 最多 3 次重试
      不可重试错误（参数/认证）：401/400 → 直接抛，不浪费重试
      退避间隔：1s → 2s → 4s（指数增长，防止惊群效应）

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
        max_retries: int = 3,  # Day3 修复 #10：最大重试次数
    ) -> dict:
        """
        调用 LLM（Day3 修复 #10：失败时指数退避重试）

        参数：
            messages:   对话历史
            tools:      可选工具定义
            temperature: 0~2
            max_tokens:  最大输出长度
            max_retries: 最大重试次数（默认 3）

        返回：统一 dict {content, tool_calls, model, usage}

        重试策略（Day3 修复 #10）：
            - 可重试：APITimeoutError / RateLimitError / APIConnectionError / InternalServerError
            - 不可重试：AuthenticationError / BadRequestError → 直接抛
            - 退避间隔：2^0=1s → 2^1=2s → 2^2=4s
            - 3 次全部失败后抛 RuntimeError
        """
        # ── 构建请求参数（不变）──
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        # ═══════════════════════════════════════════════════
        # Day3 修复 #10：指数退避重试循环
        # ═══════════════════════════════════════════════════
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                # ── 发出 HTTP 请求 ──
                response = self.client.chat.completions.create(**kwargs)

                # 成功 → 解析返回
                return _parse_chat_response(response)

            except (AuthenticationError, BadRequestError) as e:
                # 不可重试：API Key 错了、参数格式不对 → 重试没用
                raise RuntimeError(
                    f"LLM 调用失败（不可重试）: {type(e).__name__}: {e}"
                )

            except (APITimeoutError, RateLimitError, APIConnectionError,
                    InternalServerError) as e:
                # 可重试：网络抖动 / 限流 / 服务端临时故障
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** (attempt - 1)  # 指数退避: 1s → 2s → 4s
                    print(f"  [LLM] {type(e).__name__}，{wait}s 后重试 ({attempt}/{max_retries})...")
                    time.sleep(wait)
                else:
                    print(f"  [LLM] 已重试 {max_retries} 次，全部失败")

        # 所有重试耗尽
        raise RuntimeError(
            f"LLM 调用失败（已重试 {max_retries} 次）: "
            f"{type(last_error).__name__}: {last_error}"
        )


# ── 工具函数：解析 LLM 响应 ─────────────────────────────
def _parse_chat_response(response) -> dict:
    """
    Day3 修复 #10：从 chat() 内联提取为独立函数

    把 OpenAI SDK 的 response 对象转成统一的 dict 格式，
    外部调用方不需要 import openai。
    """
    choice = response.choices[0]
    msg = choice.message

    return {
        # msg.content 可能是 None（当 LLM 决定调工具时）
        "content": msg.content,
        # 从 message 对象里提取 tool_calls
        "tool_calls": _parse_tool_calls(msg),
        # 记录实际使用的模型
        "model": response.model,
        # token 用量
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        },
    }


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
