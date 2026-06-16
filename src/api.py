"""
FastAPI 接口层 —— 把 Agent 封装成 HTTP 服务

刻意踩坑点（全保留，用于学习）：
- 全局 Agent 实例 → 多用户并发时会互相覆盖 memory
- 同步 def 接口 → 阻塞事件循环，压测时表现差
- 无超时控制 → LLM 卡住时请求一直挂起
- 无并发隔离 → 用户 A 的问题可能污染用户 B 的对话
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── 导入我们的 Agent ──
from src.agent import ReActAgent

# ── 创建 FastAPI 应用 ──
app = FastAPI(
    title="Agent From Scratch API",
    description="从零手搓的 ReAct Agent HTTP 服务",
    version="0.1.0",
)

# ╔══════════════════════════════════════════════════════════╗
# ║  刻意踩坑 1：全局 Agent 实例                              ║
# ║  所有请求共享同一个 agent，memory 会互相污染！            ║
# ║  用户 A 说"我叫张三" → 用户 B 问"我叫什么？" → 返回张三  ║
# ╚══════════════════════════════════════════════════════════╝
agent = ReActAgent(max_iterations=10, truncation_strategy="none")

# ╔══════════════════════════════════════════════════════════╗
# ║  刻意踩坑 2：用同步 def 而非 async def                    ║
# ║  Agent.run() 里的 while 循环是阻塞的，                     ║
# ║  会卡住 FastAPI 的事件循环，导致其他请求排队。             ║
# ╚══════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════
# Pydantic 模型 —— 请求和响应的数据结构
# ═══════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """
    聊天请求体

    类比 Java DTO：
        public class ChatRequest {
            private String message;
            // getter/setter...
        }

    Pydantic 自动做参数校验 —— message 缺失或类型不对直接返回 422
    """
    message: str = Field(
        ...,
        description="用户输入的问题",
        min_length=1,
        max_length=2000,
        examples=["帮我计算 123 * 456"],
    )


class ChatResponse(BaseModel):
    """
    聊天响应体

    统一格式，方便前端/客户端解析
    """
    answer: str = Field(description="Agent 的最终回复")
    status: str = Field(default="ok", description="请求状态：ok 或 error")


class ErrorResponse(BaseModel):
    """错误响应体 —— 发生异常时的统一格式"""
    detail: str = Field(description="错误详情")


# ═══════════════════════════════════════════════════════════
# API 端点
# ═══════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """
    根路径 —— 健康检查

    生产环境通常用这个端点做负载均衡的健康探测。
    返回 {"status": "ok"} 表示服务活着。
    """
    return {"status": "ok", "service": "Agent From Scratch API"}


@app.get("/health")
async def health():
    """健康检查端点 —— 比 / 更详细"""
    return {
        "status": "ok",
        "agent_ready": agent.llm is not None,
        "tools_count": len(agent.tools),
        "memory_messages": agent.memory_stats(),
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Agent 内部错误"},
        504: {"model": ErrorResponse, "description": "请求超时"},
    },
)
def chat(request: ChatRequest) -> ChatResponse:
    """
    核心端点：发送消息给 Agent，获取回复

    用法：
        curl -X POST http://localhost:8000/chat \
             -H "Content-Type: application/json" \
             -d '{"message": "帮我计算 123 * 456"}'

    ⚠️ 刻意踩坑：
    - 使用同步 def（非 async），Agent.run() 会阻塞事件循环
    - 无超时保护，Agent 卡住时请求会一直挂起
    - 全局 agent 实例，并发请求共享 memory
    """
    try:
        # 调用 Agent —— 这里可能耗时几秒到几十秒
        # 坑点：没有超时控制，如果 LLM 不响应，这行永远不会返回
        answer = agent.run(request.message)

        return ChatResponse(answer=answer, status="ok")

    except Exception as e:
        # ── 基础异常处理 ──
        # 坑点：只做了简单的 500 返回，没有区分不同的异常类型
        # 实际工程应该区分：
        #   - HTTPError → 上游服务问题
        #   - TimeoutError → 超时
        #   - ValidationError → 参数问题
        raise HTTPException(status_code=500, detail=f"Agent 错误: {str(e)}")


@app.post("/chat/reset")
def reset_chat():
    """
    重置对话记忆 —— 开始新对话

    因为全局 agent 共享 memory，这个接口会清空所有人的对话。
    坑点：用户 A 调用 reset，用户 B 的对话也被清了。
    """
    agent.clear_memory()
    return {"status": "ok", "message": "对话记忆已重置"}


# ── 模块自测说明 ─────────────────────────────────────────
# 不要直接 python src/api.py，用 uvicorn 启动：
#   uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
# 或直接运行 main.py
