"""
阶段二 测试脚本：验证 ReAct Agent 循环
用法：python test_phase2.py
"""

from src.agent import ReActAgent

print("=" * 60)
print("阶段二 测试：ReAct Agent 循环")
print("=" * 60)

# ── 创建 Agent 实例 ──
# 只需创建一次，可以反复调用 run()
agent = ReActAgent()
print(f"[测试] Agent 已创建，可用工具数: {len(agent.tools)}\n")

# ── 测试用例 1：需要计算器工具 ──
print("=" * 60)
print("测试 1：数学计算（触发 calculator 工具）")
print("=" * 60)
result = agent.run("帮我计算 45678 + 12345 等于多少")
print(f"\n📌 最终回答: {result}\n")

# ── 测试用例 2：需要知识查询工具 ──
# 注意：这是阶段一踩过的坑，知识库搜索很简陋
print("=" * 60)
print("测试 2：知识查询（触发 search_knowledge 工具）")
print("=" * 60)
result = agent.run("帮我查一下，什么是 DeepSeek？")
print(f"\n📌 最终回答: {result}\n")

# ── 测试用例 3：不需要工具，直接回答 ──
print("=" * 60)
print("测试 3：纯对话（不需要工具）")
print("=" * 60)
result = agent.run("你好，请用一句话介绍一下自己")
print(f"\n📌 最终回答: {result}\n")

# ── 测试用例 4：多步推理（先查后算） ──
print("=" * 60)
print("测试 4：复合问题（可能需要多轮工具调用）")
print("=" * 60)
result = agent.run("Python 是做什么的？顺便帮我算一下 2024 - 1991 等于多少")
print(f"\n📌 最终回答: {result}\n")

print("=" * 60)
print("阶段二测试完成 ✅")
print("=" * 60)
