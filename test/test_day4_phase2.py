"""
Day4 阶段二验证脚本 —— LangChain 记忆管理替换

运行方式:
    source .venv/bin/activate
    python test/test_day4_phase2.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_2a_memory_langchain():
    """ConversationMemory 基于 InMemoryChatMessageHistory"""
    from src.memory import ConversationMemory

    mem = ConversationMemory(max_messages=4, max_tokens=200)
    mem.set_system("你是助手")
    mem.add_user("你好")
    mem.add_assistant("你好！")

    assert len(mem) == 3  # system + user + assistant
    assert mem.count_messages() == 2
    print(f"  2a ✅ LangChain 记忆: {len(mem)} 条, count={mem.count_messages()}")


def test_2b_tiktoken():
    """tiktoken 精确 token 计数"""
    from src.memory import ConversationMemory

    mem = ConversationMemory()
    assert mem._encoder is not None, "tiktoken 编码器未加载"

    # 中文 token 计数
    chinese = [{"role": "user", "content": "你好今天天气怎么样"}]
    tokens = mem._count_tokens(chinese)
    assert tokens > 5, f"中文 token 太少: {tokens}"  # tiktoken 对中文约 1-2 token/字
    print(f"  2b ✅ tiktoken: 中文 9字 = {tokens} tokens（手搓估算约 {len(chinese[0]['content'])//2}）")


def test_2c_truncation():
    """三种截断策略仍然可用"""
    from src.memory import ConversationMemory

    mem = ConversationMemory(max_messages=2, max_tokens=20)  # Day4: tiktoken 精确，20 足够限制
    mem.set_system("系统提示词")
    for i in range(5):
        mem.add_user(f"问题{i}")
        mem.add_assistant(f"回答{i}")

    assert len(mem.get_messages("none")) == 11
    assert len(mem.get_messages("sliding_window")) <= 3  # system + 最多2条
    assert len(mem.get_messages("token_limit")) < 11      # 被 token 限制
    print(f"  2c ✅ 截断: none={len(mem.get_messages('none'))} sliding={len(mem.get_messages('sliding_window'))} token={len(mem.get_messages('token_limit'))}")


def test_2d_agent_compat():
    """Agent 仍然能正常使用记忆"""
    from src.agent import ReActAgent
    agent = ReActAgent(max_iterations=3)

    # 多轮对话
    agent.run("1+1等于几")
    stats = agent.memory_stats()
    assert stats["non_system"] > 0, "记忆应有内容"
    print(f"  2d ✅ Agent 兼容: {stats}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day4 阶段二：LangChain 记忆替换")
    print("=" * 60 + "\n")

    results = {}
    for name, fn in [
        ("2a LangChain记忆", test_2a_memory_langchain),
        ("2b tiktoken", test_2b_tiktoken),
        ("2c 截断策略", test_2c_truncation),
        ("2d Agent兼容", test_2d_agent_compat),
    ]:
        try:
            fn()
            results[name] = "✅"
        except Exception as e:
            results[name] = f"❌ {e}"

    print("\n" + "=" * 60)
    for n, r in results.items():
        print(f"  {r}  {n}")
    print("=" * 60)
