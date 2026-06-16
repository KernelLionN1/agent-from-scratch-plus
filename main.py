"""
Agent 服务启动入口

用法：
    python main.py

等价于：
    uvicorn src.api:app --host 0.0.0.0 --port 8000

启动后访问：
    http://localhost:8000          → 健康检查
    http://localhost:8000/docs     → Swagger 文档（可以网页上直接调 API）
    http://localhost:8000/health   → 详细健康信息
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Agent From Scratch API 启动中...")
    print("=" * 50)
    print("📍 Swagger 文档: http://localhost:8000/docs")
    print("📍 健康检查:    http://localhost:8000/health")
    print("📍 聊天接口:    POST http://localhost:8000/chat")
    print("=" * 50)

    # ── 启动 uvicorn 服务器 ──
    # host="0.0.0.0"   → 监听所有网络接口（外部可访问）
    # port=8000        → HTTP 端口
    # reload=False     → 不自动重载（学习阶段手动重启更可控）
    #          改为 True 可在代码修改时自动重启（开发用）
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
