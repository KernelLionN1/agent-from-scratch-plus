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
import asyncio  # Day3 修复 #9：async/await + 超时控制

# ── 导入我们的 Agent ──
from src.agent import ReActAgent
from src.llm_client import LLMClient  # Day3 修复：健康检查用

# ── 创建 FastAPI 应用 ──
app = FastAPI(
    title="Agent From Scratch API",
    description="从零手搓的 ReAct Agent HTTP 服务",
    version="0.1.0",
)

# ╔══════════════════════════════════════════════════════════╗
# ║  Day3 修复 #8：不再使用全局 Agent 实例！                   ║
# ║  每个请求创建新的 agent，memory 隔离，避免用户间污染。      ║
# ╚══════════════════════════════════════════════════════════╝
# 旧代码（已移除）: agent = ReActAgent(...) 全局实例
# 新方案：工厂函数，每请求创建

def create_agent() -> ReActAgent:
    """
    Day3 修复 #8：工厂函数，每次调用创建新的 Agent 实例

    对比修复前：
      agent = ReActAgent(...)  # 模块级全局变量
      所有请求共享 → memory 互相污染 → 用户A的信息泄露给用户B

    修复后：
      每个 POST /chat 请求 → create_agent() → 独立 memory
    """
    return ReActAgent(max_iterations=10, truncation_strategy="none")

# ── Day3 修复：以下注释块已移除。
# 原因：#8 已修复（工厂函数代替全局实例），#9 已修复（async + 超时）


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
    """健康检查端点 —— Day3 修复：不再依赖全局 agent"""
    # 用轻量检查代替全局 agent 状态
    try:
        llm = LLMClient()
        llm_ready = llm.api_key is not None
    except Exception:
        llm_ready = False
    
    return {
        "status": "ok",
        "llm_ready": llm_ready,
        "mode": "per-request agent (isolated)",  # Day3 修复 #8
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Agent 内部错误"},
        504: {"model": ErrorResponse, "description": "请求超时"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:  # Day3 修复 #9: async def
    """
    核心端点：发送消息给 Agent，获取回复

    Day3 修复 #8 + #9：
    - 每个请求创建独立的 Agent（内存隔离）
    - async def + asyncio.to_thread（不阻塞事件循环）
    - asyncio.wait_for 超时保护（60s）

    用法：
        curl -X POST http://localhost:8000/chat \\
             -H "Content-Type: application/json" \\
             -d '{"message": "帮我计算 123 * 456"}'
    """
    try:
        # Day3 修复 #8：每个请求创建新 Agent，而非共享全局实例
        agent = create_agent()

        # Day3 修复 #9：async + 超时
        # asyncio.to_thread: 把同步 agent.run() 丢进线程池，不阻塞事件循环
        # asyncio.wait_for: 加 60s 超时，防止 LLM 卡住时请求永久挂起
        answer = await asyncio.wait_for(
            asyncio.to_thread(agent.run, request.message),
            timeout=60.0,
        )
        return ChatResponse(answer=answer, status="ok")

    except asyncio.TimeoutError:
        # Day3 修复 #9：超时返回 504，而非让请求永久挂起
        raise HTTPException(
            status_code=504,
            detail="请求超时（60s），请简化问题后重试",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 错误: {str(e)}")


@app.post("/chat/reset")
async def reset_chat():  # Day3 修复 #9: async def
    """
    重置对话记忆 —— 开始新对话

    Day3 修复 #8：由于每个请求已经创建独立 Agent，这个端点意义不大。
    保留用于测试隔离性 —— 现在调用它不会影响其他用户。
    """
    return {
        "status": "ok",
        "message": "每个请求已使用独立 Agent，无需手动重置。（Day3 修复 #8）",
    }


# ── 模块自测说明 ─────────────────────────────────────────
# 不要直接 python src/api.py，用 uvicorn 启动：
#   uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
# 或直接运行 main.py
