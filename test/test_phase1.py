"""
阶段一 测试脚本：验证 LLM 调用 + 工具调用
用法：python test_phase1.py
"""

import json
from src.llm_client import LLMClient
from src.tools import get_tool_definitions, execute_tool, list_tools

print("=" * 50)
print("阶段一 测试：LLM 调用 + 工具注册")
print("=" * 50)

# --- 1. 验证工具注册 ---
print(f"\n📦 已注册工具: {list_tools()}")
print(f"📦 工具定义: {json.dumps(get_tool_definitions(), ensure_ascii=False, indent=2)}")

# --- 2. 验证 LLM 直调（无工具） ---
print("\n--- 测试 1：纯文本对话 ---")
client = LLMClient()
response = client.chat(
    messages=[{"role": "user", "content": "你好，1+1等于几？"}],
    max_tokens=100,
)
print(f"模型: {response['model']}")
print(f"回复: {response['content']}")
print(f"用量: {response['usage']}")

# --- 3. 验证 LLM + 工具调用 ---
print("\n--- 测试 2：带工具的对话（计算器） ---")
response = client.chat(
    messages=[{"role": "user", "content": "请帮我计算 123 * 456 等于多少"}],
    tools=get_tool_definitions(),
    max_tokens=200,
)
print(f"content: {response['content']}")
print(f"tool_calls: {json.dumps(response['tool_calls'], ensure_ascii=False, indent=2)}")

# 如果 LLM 返回了工具调用，执行它
if response["tool_calls"]:
    tc = response["tool_calls"][0]
    print(f"\n🔧 执行工具: {tc['name']}({tc['arguments']})")
    result = execute_tool(tc["name"], json.loads(tc["arguments"]))
    print(f"📤 工具结果: {result}")

# --- 4. 验证文本查询工具 ---
print("\n--- 测试 3：文本查询工具 ---")
response = client.chat(
    messages=[{"role": "user", "content": "你知道 FastAPI 是什么吗？帮我查一下"}],
    tools=get_tool_definitions(),
    max_tokens=200,
)
print(f"content: {response['content']}")
print(f"tool_calls: {json.dumps(response['tool_calls'], ensure_ascii=False, indent=2)}")

if response["tool_calls"]:
    tc = response["tool_calls"][0]
    print(f"\n🔧 执行工具: {tc['name']}({tc['arguments']})")
    result = execute_tool(tc["name"], json.loads(tc["arguments"]))
    print(f"📤 工具结果: {result}")

print("\n" + "=" * 50)
print("阶段一测试完成 ✅")
print("=" * 50)
