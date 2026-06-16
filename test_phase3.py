"""
阶段三 测试脚本：验证对话记忆 + 上下文截断
用法：python test_phase3.py
"""

from src.agent import ReActAgent

print("=" * 60)
print("阶段三 测试：对话记忆 + 上下文截断")
print("=" * 60)

# ── 测试 1：多轮对话记忆 ──
print("\n" + "=" * 60)
print("测试 1：多轮对话 —— 验证 Agent 记住之前的内容")
print("=" * 60)

agent = ReActAgent(max_iterations=10, truncation_strategy="none")

# 第一轮：告诉 Agent 一个信息
r1 = agent.run("我今年30岁，请记住这个信息")
print(f"🤖 回复1: {r1}")

# 第二轮：问刚才的信息 —— Agent 应该记得
r2 = agent.run("我刚才说我多少岁？")
print(f"🤖 回复2: {r2}")
print(f"📊 记忆状态: {agent.memory_stats()}")

# 清空记忆，准备下一个测试
agent.clear_memory()

# ── 测试 2：上下文截断 —— 滑动窗口 ──
print("\n" + "=" * 60)
print("测试 2：滑动窗口截断 —— 旧消息被丢弃")
print("=" * 60)

agent2 = ReActAgent(max_iterations=10, truncation_strategy="sliding_window")
# 把 max_messages 设小，观察截断效果
agent2.memory.max_messages = 4

# 填充多条消息
agent2.run("第一条：我叫张三")
agent2.run("第二条：我住北京")
agent2.run("第三条：我喜欢编程")
# 此时第1条应该已经被截断丢弃了

r = agent2.run("我叫什么名字？住在哪里？")
print(f"🤖 回复: {r}")
print(f"📊 记忆状态: {agent2.memory_stats()}")

# ── 测试 3：上下文截断 —— token 限制 ──
print("\n" + "=" * 60)
print("测试 3：Token 限制截断 —— 超限自动裁剪")
print("=" * 60)

agent3 = ReActAgent(max_iterations=10, truncation_strategy="token_limit")
agent3.memory.max_tokens = 300  # 设很小的限制

agent3.run("写一段很长的自我介绍，包含姓名年龄爱好工作经历等等详细信息，至少200字")
# 继续问，看 token 限制下是否还能正常工作
r = agent3.run("请用一句话总结我们刚才的对话")
print(f"🤖 回复: {r}")
print(f"📊 记忆状态: {agent3.memory_stats()}")

# ── 测试 4：多轮工具调用 ──
print("\n" + "=" * 60)
print("测试 4：多轮对话中混合工具调用")
print("=" * 60)

agent4 = ReActAgent(max_iterations=10, truncation_strategy="none")

r1 = agent4.run("帮我计算 10 * 20")
print(f"🤖 回复1: {r1}")

r2 = agent4.run("刚才的结果加上 30 等于多少？")
print(f"🤖 回复2: {r2}")

r3 = agent4.run("那最初的两个数 10 和 20，换成加法是多少？")
print(f"🤖 回复3: {r3}")

print(f"\n📊 最终记忆状态: {agent4.memory_stats()}")

print("\n" + "=" * 60)
print("阶段三测试完成 ✅")
print("=" * 60)
