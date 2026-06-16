"""
阶段四 测试脚本：验证 FastAPI 接口
用法：python test_phase4.py

使用 FastAPI TestClient，不需要先启动服务器。
"""

from fastapi.testclient import TestClient
from src.api import app

# ── TestClient 模拟 HTTP 请求，不经过网络 ──
# 相当于 Java 的 MockMvc，在进程内测试 Controller
client = TestClient(app)

print("=" * 60)
print("阶段四 测试：FastAPI 接口")
print("=" * 60)

# ── 测试 1：健康检查 ──
print("\n--- 测试 1：GET / (健康检查) ---")
response = client.get("/")
print(f"状态码: {response.status_code}")
print(f"响应: {response.json()}")

# ── 测试 2：详细健康信息 ──
print("\n--- 测试 2：GET /health ---")
response = client.get("/health")
print(f"状态码: {response.status_code}")
print(f"响应: {response.json()}")

# ── 测试 3：发送聊天请求 ──
print("\n--- 测试 3：POST /chat (简单问题) ---")
response = client.post(
    "/chat",
    json={"message": "你好，请用一句话介绍自己"},
)
print(f"状态码: {response.status_code}")
data = response.json()
print(f"回答: {data['answer'][:100]}...")

# ── 测试 4：发送需要工具的问题 ──
print("\n--- 测试 4：POST /chat (需要计算器) ---")
response = client.post(
    "/chat",
    json={"message": "帮我计算 999 * 888"},
)
print(f"状态码: {response.status_code}")
data = response.json()
print(f"回答: {data['answer'][:100]}...")

# ── 测试 5：空消息校验 ──
print("\n--- 测试 5：POST /chat (空消息 → 应返回 422) ---")
response = client.post(
    "/chat",
    json={"message": ""},
)
print(f"状态码: {response.status_code} (预期 422)")
print(f"响应: {response.json()}")

# ── 测试 6：重置记忆 ──
print("\n--- 测试 6：POST /chat/reset ---")
response = client.post("/chat/reset")
print(f"状态码: {response.status_code}")
print(f"响应: {response.json()}")

print("\n" + "=" * 60)
print("阶段四测试完成 ✅")
print("=" * 60)
print("\n💡 启动真实服务: python main.py")
print("💡 然后访问: http://localhost:8000/docs")
