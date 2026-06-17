"""
FastAPI 接口层 —— 把 Agent 封装成 HTTP 服务

刻意踩坑点（全保留，用于学习）：
- 全局 Agent 实例 → 多用户并发时会互相覆盖 memory
- 同步 def 接口 → 阻塞事件循环，压测时表现差
- 无超时控制 → LLM 卡住时请求一直挂起
- 无并发隔离 → 用户 A 的问题可能污染用户 B 的对话
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import asyncio
import traceback  # Day3 阶段四：异常处理器日志

# ── 导入我们的 Agent ──
from src.agent import ReActAgent
from src.llm_client import LLMClient  # Day3 修复：健康检查用
# Day3 阶段四：多 Agent 编排
from src.orchestrator import Orchestrator
from src.message_bus import MessageBus
import uuid  # 任务 ID 生成

# ── 创建 FastAPI 应用 ──
app = FastAPI(
    title="Agent From Scratch API",
    description="从零手搓的 ReAct Agent HTTP 服务",
    version="0.1.0",
)

# ═══════════════════════════════════════════════════════════
# Day3 阶段四：全局异常处理器（AOP 统一接管）
# 等价于 Spring 的 @ControllerAdvice —— 所有端点自动继承
# ═══════════════════════════════════════════════════════════

@app.exception_handler(asyncio.TimeoutError)
async def timeout_exception_handler(request: Request, exc: asyncio.TimeoutError):
    """asyncio.TimeoutError → HTTP 504"""
    print(f"[API] ⏰ 超时: {request.method} {request.url.path}")
    return JSONResponse(status_code=504, content={"detail": "请求超时，请简化问题后重试"})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTPException 透传（保留端点手动抛出的 404 等状态码）"""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """兜底：未捕获异常 → HTTP 500，打印堆栈方便排查"""
    print(f"[API] 💥 异常: {request.method} {request.url.path} — {type(exc).__name__}: {exc}")
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": f"服务器内部错误: {type(exc).__name__}"})


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


# Day3 阶段四：编排请求模型
class OrchestrateRequest(BaseModel):
    """
    多 Agent 编排请求

    mode: "serial" — Planner→Coder→Reviewer→Coder(修复)
          "parallel" — Planner→[Coder1,Coder2,...]→Reviewer
    """
    requirement: str = Field(..., min_length=1, max_length=2000, description="用户需求")
    mode: str = Field(default="serial", description="编排模式：serial 或 parallel")
    n_coders: int = Field(default=2, ge=1, le=5, description="并行 Coder 数量")
    with_fix: bool = Field(default=True, description="串行模式是否执行修复步骤")


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
    agent = create_agent()
    answer = await asyncio.wait_for(
        asyncio.to_thread(agent.run, request.message),
        timeout=60.0,
    )
    return ChatResponse(answer=answer, status="ok")
    # 异常由全局处理器自动接管（TimeoutError→504, Exception→500）


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


@app.post(
    "/chat/orchestrate",
    responses={
        200: {"description": "编排成功"},
        500: {"model": ErrorResponse, "description": "编排失败"},
        504: {"model": ErrorResponse, "description": "编排超时"},
    },
)
async def orchestrate(request: OrchestrateRequest):
    """
    Day3 阶段四：多 Agent 编排端点

    支持两种模式：
    - serial:   Planner → Coder → Reviewer → Coder(修复)
    - parallel: Planner → [Coder1, Coder2, ...] → Reviewer

    用法：
        curl -X POST http://localhost:8000/chat/orchestrate \\
             -H "Content-Type: application/json" \\
             -d '{"requirement": "实现冒泡排序和二分查找", "mode": "serial"}'
    """
    bus = MessageBus()
    orch = Orchestrator(bus)

    if request.mode == "serial":
        result = await asyncio.wait_for(
            asyncio.to_thread(orch.run_serial, request.requirement, request.with_fix),
            timeout=180.0,
        )
        return {
            "status": "ok", "mode": "serial",
            "plan": result["plan"],
            "code": result.get("code", ""),
            "review": result.get("review", ""),
            "fixed_code": result.get("fixed_code", ""),
            "stats": result.get("stats", {}),
        }

    else:  # parallel
        result = await asyncio.wait_for(
            asyncio.to_thread(orch.run_parallel, request.requirement, request.n_coders),
            timeout=180.0,
        )
        codes = [c for c in result.get("codes", []) if not isinstance(c, Exception)]
        return {
            "status": "ok", "mode": "parallel",
            "plan": result["plan"],
            "codes": codes,
            "combined": result.get("combined", ""),
            "review": result.get("review", ""),
            "timing": result.get("timing", {}),
        }
    # 异常由全局处理器自动接管（TimeoutError→504, Exception→500）

# ── Day3 阶段四：编排任务进度存储（内存）──
_orchestrate_tasks: dict[str, dict] = {}


@app.get("/chat/progress/{task_id}")
async def get_progress(task_id: str):
    """
    Day3 阶段四：查询编排任务进度

    用法：
        curl http://localhost:8000/chat/progress/<task_id>
    """
    task = _orchestrate_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return {
        "task_id": task_id,
        "status": task.get("status", "unknown"),
        "phase": task.get("phase", ""),
        "progress": task.get("progress", {}),
    }


@app.get("/chat/progress")
async def list_progress():
    """
    列出所有编排任务进度（调试用）
    """
    return {
        "count": len(_orchestrate_tasks),
        "tasks": {
            tid: {"status": t.get("status"), "phase": t.get("phase")}
            for tid, t in _orchestrate_tasks.items()
        },
    }
# 不要直接 python src/api.py，用 uvicorn 启动：
#   uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
# 或直接运行 main.py
