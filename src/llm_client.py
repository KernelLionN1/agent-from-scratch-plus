"""
LLM 客户端封装 —— Day4 框架版：基于 LangChain ChatDeepSeek

Day4 核心变化：
  手搓版（Day1-3）：openai.OpenAI 裸调 + 手动重试 + 自定义 dict 返回
  框架版（Day4）：  ChatOpenAI(DeepSeek) + LangChain 内置重试 + AIMessage 原生对象

关键收益：
  - 不再需要手动写指数退避重试 → LangChain 内置 tenacity
  - 不再需要 asyncio.to_thread → LangChain 原生 async（ainvoke）
  - 不再需要 _parse_chat_response → LangChain 返回结构化 AIMessage

Java 类比：从手写 HttpClient + JSON 解析 → 用 Spring RestTemplate
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # DeepSeek 兼容 OpenAI 协议

load_dotenv()

# ── LangChain LLM 客户端 ──────────────────────────────────

class LLMClient:
    """
    LLM 调用封装 —— Day4 框架版

    底层：langchain_openai.ChatOpenAI（指向 DeepSeek API）
    优势：原生 async（ainvoke）、内置重试、token 用量自动追踪

    使用方式：
        client = LLMClient()
        response = client.chat([{"role": "user", "content": "你好"}])
        # response 仍然返回 dict，保持调用方兼容
    """

    def __init__(self):
        """初始化 ChatOpenAI 客户端，指向 DeepSeek"""
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY 未设置")

        # ═══════════════════════════════════════════════════
        # Day4 核心：LangChain 的 ChatOpenAI 封装
        # 内置重试（max_retries）+ 原生 async（ainvoke）
        # ═══════════════════════════════════════════════════
        self._llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.7,
            max_tokens=1024,
            max_retries=3,        # Day4：LangChain 内置重试，不再手写指数退避
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict:
        """
        同步调用 LLM（兼容手搓版接口）

        参数与返回格式同 Day1-3，保证调用方（agent.py/api.py）不改动。
        """
        # 更新参数
        self._llm.temperature = temperature
        self._llm.max_tokens = max_tokens

        # Day4：LangChain 原生工具绑定
        if tools:
            llm_with_tools = self._llm.bind_tools(
                [_convert_tool_lc(t) for t in tools]
            )
        else:
            llm_with_tools = self._llm

        # 同步调用 → 内部阻塞等待
        response = llm_with_tools.invoke([_to_lc_message(m) for m in messages])
        return _to_dict(response)

    async def ainvoke(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Day4 新增：原生异步调用 —— 不需要 asyncio.to_thread！

        这是框架版最大的性能提升：LangChain 的 ainvoke 是真正的 async，
        不占线程池，可以直接 await。
        """
        self._llm.temperature = temperature
        self._llm.max_tokens = max_tokens

        if tools:
            llm_with_tools = self._llm.bind_tools(
                [_convert_tool_lc(t) for t in tools]
            )
        else:
            llm_with_tools = self._llm

        response = await llm_with_tools.ainvoke(
            [_to_lc_message(m) for m in messages]
        )
        return _to_dict(response)

    async def astream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """
        Day6 新增：流式输出

        使用 LangChain 的 astream_events() 逐 token 产出，
        同时追踪完整的 tool_calls 累积。

        Yields:
            dict: {"type": "token", "content": "你"} 或
                  {"type": "tool_call", "name": "calculator", "arguments": "..."} 或
                  {"type": "done", "content": "完整回复", "usage": {...}}
        """
        self._llm.temperature = temperature
        self._llm.max_tokens = max_tokens

        if tools:
            llm_with_tools = self._llm.bind_tools(
                [_convert_tool_lc(t) for t in tools]
            )
        else:
            llm_with_tools = self._llm

        full_content = ""
        tool_calls = []
        current_tool = None

        async for event in llm_with_tools.astream_events(
            [_to_lc_message(m) for m in messages],
            version="v2",
        ):
            kind = event.get("event")

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                # 文本 token
                if chunk.content:
                    full_content += chunk.content
                    yield {"type": "token", "content": chunk.content}

                # 工具调用增量
                if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                    for tc in chunk.tool_call_chunks:
                        if tc.get("name"):
                            current_tool = {
                                "name": tc["name"],
                                "arguments": "",
                                "id": tc.get("id", ""),
                            }
                            tool_calls.append(current_tool)
                        if tc.get("args") and current_tool:
                            current_tool["arguments"] += tc["args"]

            elif kind == "on_chat_model_end":
                # 流结束，发送完整结果
                yield {
                    "type": "done",
                    "content": full_content,
                    "tool_calls": tool_calls if tool_calls else None,
                    "usage": {},
                }


# ── 格式转换工具函数（手搓 dict ↔ LangChain 对象）────────

def _to_lc_message(msg: dict):
    """手搓版 dict → LangChain 消息对象"""
    from langchain_core.messages import (
        HumanMessage, AIMessage, SystemMessage, ToolMessage,
    )
    role = msg["role"]
    content = msg.get("content", "") or ""
    if role == "system":
        return SystemMessage(content=content)
    elif role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        msg_obj = AIMessage(content=content)
        if msg.get("tool_calls"):
            msg_obj.tool_calls = [
                {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "args": _safe_json_parse(tc["function"]["arguments"]),
                }
                for tc in msg["tool_calls"]
            ]
        return msg_obj
    elif role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=msg["tool_call_id"],
        )
    return HumanMessage(content=str(content))


def _to_dict(response) -> dict:
    """
    LangChain AIMessage → 手搓版统一 dict

    保持返回格式与 Day1-3 完全一致：
    {content, tool_calls, model, usage}
    """
    tool_calls = None
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_calls = [
            {
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "arguments": _safe_json_dumps(tc.get("args", {})),  
            }
            for tc in response.tool_calls
        ]

    # LangChain 的 usage_metadata 包含 token 信息
    usage = {}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        usage = {
            "prompt_tokens": um.get("input_tokens", 0),
            "completion_tokens": um.get("output_tokens", 0),
            "total_tokens": um.get("total_tokens", 0),
        }

    return {
        "content": response.content,
        "tool_calls": tool_calls,
        "model": getattr(response, "response_metadata", {}).get("model_name", ""),
        "usage": usage,
    }


def _convert_tool_lc(tool_def: dict):
    """
    手搓版 OpenAI 格式工具定义 → LangChain tool 对象

    手搓版格式：
      {type: "function", function: {name: "...", description: "...", parameters: {...}}}
    """
    from langchain_core.tools import StructuredTool

    name = tool_def["function"]["name"]
    desc = tool_def["function"]["description"]
    # 提取参数 schema
    params = tool_def["function"]["parameters"]["properties"]
    # 构建一个占位 tool（实际执行仍走手搓注册表）
    return StructuredTool.from_function(
        func=lambda **kwargs: f"[tool:{name}] called with {kwargs}",
        name=name,
        description=desc,
    )


def _safe_json_parse(s: str) -> dict:
    """安全解析 JSON 字符串"""
    import json
    try:
        return json.loads(s) if isinstance(s, str) else s
    except json.JSONDecodeError:
        return {}


def _safe_json_dumps(obj: dict) -> str:
    """安全序列化为 JSON 字符串"""
    import json
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)


# ── 模块自测 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("LLMClient (LangChain版) 自测")
    print("=" * 50)

    client = LLMClient()

    # 同步调用
    result = client.chat(
        messages=[{"role": "user", "content": "说你好"}],
    )
    print(f"同步: {result['content'][:50]}...")
    print(f"tokens: {result['usage']}")
    print("✅ 自测完成")
