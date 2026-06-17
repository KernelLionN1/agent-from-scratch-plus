"""
Day5 阶段二验证脚本 —— 持久化记忆（SQLite）

运行方式:
    source .venv/bin/activate
    PYTHONPATH=. python test/test_day5_phase2.py
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用临时数据库，不污染项目数据
TEST_DB = os.path.join(tempfile.gettempdir(), "test_day5_phase2.db")


def cleanup():
    """清理测试数据库"""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_2a_sqlite_history_basic():
    """SQLiteChatMessageHistory 基本 CRUD"""
    from src.memory import SQLiteChatMessageHistory

    history = SQLiteChatMessageHistory(session_id="test-2a", db_path=TEST_DB)
    assert history.count_messages() == 0

    # 写入消息
    history.add_message({"role": "user", "content": "你好"})
    history.add_message({"role": "assistant", "content": "你好！"})
    assert history.count_messages() == 2, f"应该有2条消息，实际{history.count_messages()}"

    # 验证 messages 属性
    msgs = history.messages
    assert len(msgs) == 2
    assert msgs[0].content == "你好"
    print("  2a ✅ SQLiteChatMessageHistory CRUD 正常")

    # 清理
    history.clear()
    assert history.count_messages() == 0


def test_2b_conversation_memory_with_sqlite():
    """ConversationMemory 对接 SQLite 后端"""
    from src.memory import SQLiteChatMessageHistory, ConversationMemory

    history = SQLiteChatMessageHistory(session_id="test-2b", db_path=TEST_DB)
    mem = ConversationMemory(
        max_messages=5, max_tokens=500,
        history_backend=history,
    )

    mem.add_user("我叫小明")
    mem.add_assistant("你好小明！")

    msgs = mem.get_messages("none")
    assert len(msgs) >= 2, f"至少2条消息: {len(msgs)}"
    print("  2b ✅ ConversationMemory + SQLite 后端正常")


def test_2c_session_manager_lifecycle():
    """SessionManager 生命周期（创建/切换/列表/删除）"""
    from src.session import SessionManager

    sm = SessionManager(db_path=TEST_DB)

    # 创建
    s1 = sm.create("学习Python")
    s2 = sm.create("学习LangGraph")
    assert len(sm.list_sessions()) == 2

    # 切换 + 写入
    sm.switch(s1)
    h1 = sm.get_history()
    h1.add_message({"role": "user", "content": "什么是装饰器？"})
    h1.add_message({"role": "assistant", "content": "装饰器是一种高阶函数..."})

    sm.switch(s2)
    h2 = sm.get_history()
    h2.add_message({"role": "user", "content": "LangGraph是什么？"})
    assert h2.count_messages() == 1

    # 验证隔离
    sm.switch(s1)
    assert sm.get_history().count_messages() == 2, "s1 消息应该不变"

    # 列表验证
    sessions = sm.list_sessions()
    s1_info = next(s for s in sessions if s["id"] == s1)
    assert s1_info["msg_count"] == 2

    print("  2c ✅ SessionManager 生命周期正常")

    # 清理
    sm.delete(s2)
    assert len(sm.list_sessions()) == 1


def test_2d_data_persistence():
    """
    数据持久化验证：写入 → 重建实例 → 读取

    模拟进程重启场景：创建 SQLiteChatMessageHistory → 写入 → 销毁 → 新建 → 读取
    """
    from src.memory import SQLiteChatMessageHistory

    sid = "test-persist"

    # 第一次写入
    h1 = SQLiteChatMessageHistory(session_id=sid, db_path=TEST_DB)
    h1.add_message({"role": "user", "content": "持久化测试"})
    assert h1.count_messages() == 1
    del h1  # 销毁实例

    # 第二次读取（模拟进程重启）
    h2 = SQLiteChatMessageHistory(session_id=sid, db_path=TEST_DB)
    assert h2.count_messages() == 1, f"重启后消息应该还在，实际{h2.count_messages()}"
    assert h2.messages[0].content == "持久化测试"
    print("  2d ✅ 数据持久化验证通过（模拟重启）")

    h2.clear()


def test_2e_cost_tracking():
    """成本追踪：记录 + 查询 token 消耗"""
    from src.session import SessionManager

    sm = SessionManager(db_path=TEST_DB)
    sid = sm.create("成本测试")

    # 模拟多次 LLM 调用
    sm.record_usage(sid, {"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200}, "deepseek-chat")
    sm.record_usage(sid, {"prompt_tokens": 300, "completion_tokens": 80, "total_tokens": 380}, "deepseek-chat")
    sm.record_usage(sid, {}, "deepseek-chat")  # 空 usage → 跳过

    # 查询单会话统计
    stats = sm.get_cost_stats(sid)
    assert stats["total_calls"] == 2
    assert stats["total_tokens"] == 580
    assert stats["prompt_tokens"] == 450
    assert stats["completion_tokens"] == 130
    print(f"  2e ✅ 成本追踪: {stats['total_calls']}次调用, {stats['total_tokens']}tokens")

    sm.delete(sid)


def test_2f_get_memory_convenience():
    """get_memory() 便捷方法：一步获取带截断的 ConversationMemory"""
    from src.session import SessionManager

    sm = SessionManager(db_path=TEST_DB)
    sid = sm.create("便捷测试")
    sm.switch(sid)

    # 写入消息
    h = sm.get_history()
    h.add_message({"role": "user", "content": "测试消息"})

    # 一步获取 ConversationMemory（含截断策略）
    mem = sm.get_memory(session_id=sid, max_messages=10, max_tokens=1000)
    msgs = mem.get_messages("none")
    assert len(msgs) >= 1
    print("  2f ✅ get_memory() 便捷方法正常")

    sm.delete(sid)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day5 阶段二：持久化记忆（SQLite）")
    print("=" * 60 + "\n")

    # 确保干净的测试环境
    cleanup()

    results = {}
    for name, fn in [
        ("2a SQLite CRUD", test_2a_sqlite_history_basic),
        ("2b Memory+SQLite", test_2b_conversation_memory_with_sqlite),
        ("2c SessionManager生命周期", test_2c_session_manager_lifecycle),
        ("2d 数据持久化", test_2d_data_persistence),
        ("2e 成本追踪", test_2e_cost_tracking),
        ("2f 便捷方法", test_2f_get_memory_convenience),
    ]:
        try:
            fn()
            results[name] = "✅"
        except Exception as e:
            results[name] = f"❌ {e}"

    # 最终清理
    cleanup()

    print("\n" + "=" * 60)
    for n, r in results.items():
        print(f"  {r}  {n}")
    print("=" * 60)
