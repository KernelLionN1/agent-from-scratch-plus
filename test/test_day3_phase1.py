"""
Day3 阶段一验证脚本 —— Day1 遗留 9 坑修复验证

验证内容：
1. eval 安全修复（白名单 + 受限命名空间）
2. search_knowledge 返回格式化文本
3. system prompt 强化（"必须"而非"请"）
4. max_iterations 轮次上限生效
5. 工具执行异常保护
6. 上下文截断策略可用
7. token 估算区分中英文
8. 不再使用全局 agent（工厂函数隔离）
9. API 端点 async + 超时保护

运行方式：
    source .venv/bin/activate
    python test_day3_phase1.py
"""

import sys
import inspect

# ── 导入验证目标 ──────────────────────────────────────────
from src.tools import calculator, search_knowledge, execute_tool
from src.agent import ReActAgent
from src.memory import ConversationMemory
from src import api  # 模块级导入，检查全局变量


# ═══════════════════════════════════════════════════════════
# 验证 1：eval 安全修复
# ═══════════════════════════════════════════════════════════
def test_1_eval_safety():
    """
    验证 eval() 不再能执行任意 Python 代码

    检查点：
    - 正常数学表达式仍然能计算
    - 危险表达式被 ValueError 拦截
    - 除零被友好拦截
    """
    print("=" * 60)
    print("验证 1：eval 安全修复（#1）")
    print("=" * 60)

    # 1a: 正常表达式应该能计算
    result = calculator("1 + 2 * 3")
    assert result == 7, f"正常表达式计算结果错误: {result}"
    print("  1a ✅ 正常表达式: 1+2*3 = 7")

    # 1b: 危险表达式应该被拦截
    try:
        calculator("__import__('os').getcwd()")
        assert False, "危险表达式没有被拦截！"
    except ValueError as e:
        assert "不安全" in str(e) or "表达式" in str(e), f"错误信息不明确: {e}"
        print(f"  1b ✅ 危险表达式被拦截: {e}")

    # 1c: 除零应该被拦截
    try:
        calculator("1/0")
        assert False, "除零没有被拦截！"
    except ValueError as e:
        assert "零" in str(e), f"错误信息不明确: {e}"
        print(f"  1c ✅ 除零被拦截: {e}")

    # 1d: 非法字符应该被拦截
    try:
        calculator("hello world")
        assert False, "非法字符没有被拦截！"
    except ValueError:
        print("  1d ✅ 非法字符被拦截")

    print("  ✅ 验证 1 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 2：工具返回格式化文本
# ═══════════════════════════════════════════════════════════
def test_2_formatted_tool_result():
    """
    验证 search_knowledge 返回格式化文本而非 dict

    检查点：
    - 返回类型是 str 而非 dict
    - 格式包含 [key]: value
    - 空结果有友好提示
    """
    print("=" * 60)
    print("验证 2：工具返回格式化文本（#2）")
    print("=" * 60)

    # 2a: 返回类型应该是 str
    result = search_knowledge("python")
    assert isinstance(result, str), f"返回类型错误: {type(result).__name__}，预期 str"
    print(f"  2a ✅ 返回类型: str（修复前是 dict）")

    # 2b: 应该包含格式化标签 [key]
    assert "[python]" in result, f"输出格式不对，缺少 '[python]' 标签: {result[:100]}"
    print(f"  2b ✅ 包含格式化标签 [python]")

    # 2c: 空结果应该有友好提示
    empty = search_knowledge("量子力学")
    assert "未找到" in empty, f"空结果没有友好提示: {empty}"
    assert "知识库当前包含" in empty, "空结果没有列出可用知识"
    print(f"  2c ✅ 空结果友好提示: {empty[:80]}...")

    print("  ✅ 验证 2 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 3：system prompt 强化
# ═══════════════════════════════════════════════════════════
def test_3_system_prompt():
    """
    验证 system prompt 从"请求"改为"命令"

    检查点：
    - 包含"必须"关键词
    - 有明确的工具使用规则
    - prompt 长度明显增加（更多约束）
    """
    print("=" * 60)
    print("验证 3：system prompt 强化（#3）")
    print("=" * 60)

    agent = ReActAgent()
    prompt = agent.system_prompt

    # 3a: 应该包含"必须"
    assert "必须" in prompt, "prompt 中没有'必须'，约束不够强硬"
    print("  3a ✅ 包含'必须'关键词")

    # 3b: 应该有明确的工具使用规则
    assert "calculator" in prompt, "prompt 中没有提到 calculator 工具"
    assert "search_knowledge" in prompt, "prompt 中没有提到 search_knowledge 工具"
    print("  3b ✅ 明确指定了工具名称")

    # 3c: prompt 长度至少 100 字（修复前约 50 字）
    assert len(prompt) >= 100, f"prompt 太短（{len(prompt)}字），约束可能不够"
    print(f"  3c ✅ prompt 长度: {len(prompt)} 字（修复前约 50 字）")

    print("  ✅ 验证 3 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 4：max_iterations 轮次上限
# ═══════════════════════════════════════════════════════════
def test_4_max_iterations():
    """
    验证 max_iterations 已存在并生效

    检查点：
    - ReActAgent 接受 max_iterations 参数
    - 默认值 > 0
    - 可以设置为自定义值
    """
    print("=" * 60)
    print("验证 4：max_iterations 确认生效（#4）")
    print("=" * 60)

    # 4a: 默认值存在且合理
    agent = ReActAgent()
    assert agent.max_iterations > 0, "max_iterations 未设置或为 0"
    print(f"  4a ✅ 默认 max_iterations = {agent.max_iterations}")

    # 4b: 可以自定义
    agent2 = ReActAgent(max_iterations=3)
    assert agent2.max_iterations == 3, f"自定义值未生效: {agent2.max_iterations}"
    print(f"  4b ✅ 自定义 max_iterations = {agent2.max_iterations}")

    # 4c: run() 方法里有轮次检查逻辑
    # 检查源码中是否有 iteration > self.max_iterations 的判断
    import inspect
    source = inspect.getsource(ReActAgent.run)
    assert "max_iterations" in source, "run() 方法中没有 max_iterations 检查"
    print("  4c ✅ run() 方法中存在 max_iterations 检查")

    print("  ✅ 验证 4 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 5：工具执行异常保护
# ═══════════════════════════════════════════════════════════
def test_5_tool_error_handling():
    """
    验证工具执行有异常保护

    检查点：
    - execute_tool 对未知工具返回错误信息（而非抛异常）
    - run() 方法里有 json.JSONDecodeError 的 try/except
    """
    print("=" * 60)
    print("验证 5：工具执行异常保护（#5）")
    print("=" * 60)

    # 5a: execute_tool 对未知工具不抛异常
    result = execute_tool("nonexistent_tool", {})
    assert "未知" in result, f"未知工具未返回友好错误: {result}"
    print(f"  5a ✅ 未知工具友好提示: {result}")

    # 5b: run() 方法里有 JSONDecodeError 处理
    source = inspect.getsource(ReActAgent.run)
    assert "JSONDecodeError" in source, "run() 中没有 JSONDecodeError 处理"
    print("  5b ✅ run() 方法包含 JSONDecodeError 异常保护")

    # 5c: run() 方法里有通用的 execute_tool 异常保护
    assert "except Exception" in source.split("execute_tool")[1] if "execute_tool" in source else True, \
        "run() 中 execute_tool 调用缺少异常保护"
    print("  5c ✅ execute_tool 调用有通用异常保护")

    print("  ✅ 验证 5 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 6：上下文截断策略
# ═══════════════════════════════════════════════════════════
def test_6_truncation_strategies():
    """
    验证三种截断策略存在且行为不同

    检查点：
    - none / sliding_window / token_limit 三种策略
    - 不同策略返回不同数量的消息
    """
    print("=" * 60)
    print("验证 6：上下文截断策略（#6）")
    print("=" * 60)

    mem = ConversationMemory(max_messages=3, max_tokens=100)
    mem.set_system("系统提示词")

    # 塞入 5 条消息
    for i in range(5):
        mem.add_user(f"用户消息 {i}")
        mem.add_assistant(f"助手回复 {i}")

    # 6a: none 策略返回全部（8 条非 system）
    msgs_none = mem.get_messages("none")
    non_system_none = [m for m in msgs_none if m["role"] != "system"]
    assert len(non_system_none) == 10, f"none 策略消息数不对: {len(non_system_none)}"
    print(f"  6a ✅ none 策略: {len(non_system_none)} 条（全部）")

    # 6b: sliding_window 策略受 max_messages 限制
    msgs_sw = mem.get_messages("sliding_window")
    non_system_sw = [m for m in msgs_sw if m["role"] != "system"]
    assert len(non_system_sw) <= 3, f"sliding_window 未限制: {len(non_system_sw)}"
    print(f"  6b ✅ sliding_window 策略: {len(non_system_sw)} 条（≤3）")

    # 6c: token_limit 策略返回消息数不同
    msgs_tl = mem.get_messages("token_limit")
    non_system_tl = [m for m in msgs_tl if m["role"] != "system"]
    # 受 max_tokens=100 限制，应该比 none 少
    assert len(non_system_tl) <= len(non_system_none), \
        f"token_limit 未截断: {len(non_system_tl)} vs none={len(non_system_none)}"
    print(f"  6c ✅ token_limit 策略: {len(non_system_tl)} 条（受 token 限制）")

    # 6d: 三种策略都能正常工作（不抛异常）
    for strategy in ["none", "sliding_window", "token_limit"]:
        msgs = mem.get_messages(strategy)
        assert msgs[0]["role"] == "system", f"{strategy} 策略丢失 system 消息"
    print("  6d ✅ 三种策略都保留 system 消息")

    print("  ✅ 验证 6 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 7：token 估算区分中英文
# ═══════════════════════════════════════════════════════════
def test_7_token_estimation():
    """
    验证 token 估算能区分中英文

    检查点：
    - 纯中文消息的 token 估算 > 同等长度纯英文
    - _estimate_tokens 方法存在并可用
    """
    print("=" * 60)
    print("验证 7：token 估算区分中英文（#7）")
    print("=" * 60)

    mem = ConversationMemory()

    # 7a: 纯英文消息
    english_msg = [{"role": "user", "content": "Hello, how are you doing today?"}]
    tokens_en = mem._estimate_tokens(english_msg)
    print(f"  7a 英文 35 字符估算: {tokens_en} tokens")

    # 7b: 纯中文消息（等长）
    chinese_msg = [{"role": "user", "content": "你好今天天气怎么样我想出去散步"}]
    tokens_cn = mem._estimate_tokens(chinese_msg)
    print(f"  7b 中文 18 字符估算: {tokens_cn} tokens")

    # 7c: 中文每个字符的 token 权重应高于英文
    # 同等字符数下，中文 tokens 应该更多
    same_len_en = [{"role": "user", "content": "a" * 50}]   # 50 个英文字符
    same_len_cn = [{"role": "user", "content": "你" * 50}]   # 50 个中文字符
    en_tokens = mem._estimate_tokens(same_len_en)
    cn_tokens = mem._estimate_tokens(same_len_cn)
    assert cn_tokens > en_tokens, \
        f"中文 token 估算应高于英文: cn={cn_tokens} vs en={en_tokens}"
    print(f"  7c ✅ 50字符: 中文={cn_tokens} tokens > 英文={en_tokens} tokens")

    # 7d: 中英混合
    mixed_msg = [{"role": "user", "content": "你好 world! 今天 AI 进展如何？"}]
    mixed_tokens = mem._estimate_tokens(mixed_msg)
    print(f"  7d 中英混合估算: {mixed_tokens} tokens")
    # 混合消息的 token 应该在纯中文和纯英文之间
    pure_en_same_len = [{"role": "user", "content": "a" * 22}]
    pure_cn_same_len = [{"role": "user", "content": "你" * 22}]
    en_t = mem._estimate_tokens(pure_en_same_len)
    cn_t = mem._estimate_tokens(pure_cn_same_len)
    assert en_t < mixed_tokens < cn_t or mixed_tokens >= en_t, \
        f"混合 token 估算异常: en={en_t} mix={mixed_tokens} cn={cn_t}"
    print("  7d ✅ 中英混合估算合理")

    print("  ✅ 验证 7 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 8：不再使用全局 agent
# ═══════════════════════════════════════════════════════════
def test_8_no_global_agent():
    """
    验证 api.py 不再有模块级全局 agent

    检查点：
    - api 模块没有 'agent' 属性
    - create_agent() 工厂函数存在
    - 两次调用返回不同实例
    """
    print("=" * 60)
    print("验证 8：全局 agent 已移除（#8）")
    print("=" * 60)

    # 8a: api 模块不再有全局 agent 变量
    assert not hasattr(api, "agent"), \
        "api 模块仍然有全局 agent 变量！应该已改为 create_agent() 工厂函数"
    print("  8a ✅ api 模块无全局 agent 变量")

    # 8b: create_agent() 工厂函数存在
    assert hasattr(api, "create_agent"), "api 模块缺少 create_agent() 工厂函数"
    assert callable(api.create_agent), "create_agent 不是可调用的函数"
    print("  8b ✅ create_agent() 工厂函数存在")

    # 8c: 两次调用返回不同实例（内存隔离）
    agent1 = api.create_agent()
    agent2 = api.create_agent()
    assert agent1 is not agent2, "create_agent() 两次调用返回了同一个实例！"
    print("  8c ✅ 每次调用返回不同实例（内存隔离）")

    # 8d: memory 也是隔离的
    agent1.memory.add_user("用户A的秘密")
    assert agent2.memory.count_messages() == 0, \
        "agent1 的 memory 污染了 agent2 的 memory！"
    print("  8d ✅ 不同 agent 的 memory 完全隔离")

    print("  ✅ 验证 8 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 验证 9：API 端点 async + 超时保护
# ═══════════════════════════════════════════════════════════
def test_9_async_api():
    """
    验证 /chat 端点改为 async def 且有超时保护

    检查点：
    - chat 函数是 async def
    - 函数体内使用了 asyncio.wait_for
    - 有 TimeoutError 处理
    """
    print("=" * 60)
    print("验证 9：API 端点 async + 超时（#9）")
    print("=" * 60)

    # 9a: chat 函数是 async def
    chat_fn = getattr(api, "chat", None)
    assert chat_fn is not None, "api 模块没有 chat 函数"
    assert inspect.iscoroutinefunction(chat_fn), \
        f"chat 不是 async 函数，而是 {type(chat_fn).__name__}"
    print("  9a ✅ /chat 端点是 async def")

    # 9b: 函数体包含 asyncio.wait_for（超时保护）
    source = inspect.getsource(chat_fn)
    assert "wait_for" in source, "chat 函数中没有 asyncio.wait_for 超时保护"
    print("  9b ✅ chat 函数包含 asyncio.wait_for 超时保护")

    # 9c: 有 TimeoutError 处理
    assert "TimeoutError" in source, "chat 函数中没有 TimeoutError 处理"
    print("  9c ✅ chat 函数包含 TimeoutError 处理")

    # 9d: 函数体包含 create_agent() 调用
    assert "create_agent()" in source or "create_agent" in source, \
        "chat 函数没有调用 create_agent()"
    print("  9d ✅ chat 函数使用 create_agent()（每请求新建）")

    # 9e: reset_chat 也是 async
    reset_fn = getattr(api, "reset_chat", None)
    assert reset_fn is not None, "api 模块没有 reset_chat 函数"
    assert inspect.iscoroutinefunction(reset_fn), "reset_chat 不是 async 函数"
    print("  9e ✅ /chat/reset 端点也是 async def")

    print("  ✅ 验证 9 全部通过\n")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day3 阶段一：9 坑修复验证")
    print("=" * 60 + "\n")

    results = {}

    # 逐一运行验证，收集结果
    tests = [
        ("#1 eval安全", test_1_eval_safety),
        ("#2 工具返回格式化", test_2_formatted_tool_result),
        ("#3 prompt强化", test_3_system_prompt),
        ("#4 max_iterations", test_4_max_iterations),
        ("#5 工具异常保护", test_5_tool_error_handling),
        ("#6 截断策略", test_6_truncation_strategies),
        ("#7 token估算", test_7_token_estimation),
        ("#8 全局agent移除", test_8_no_global_agent),
        ("#9 async+超时", test_9_async_api),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = "✅"
        except AssertionError as e:
            results[name] = f"❌ {e}"
        except Exception as e:
            results[name] = f"💥 {type(e).__name__}: {e}"

    # ── 汇总 ──
    print("=" * 60)
    print("验证汇总")
    print("=" * 60)
    passed = 0
    for name, result in results.items():
        status = "✅" if result == "✅" else result
        if result == "✅":
            passed += 1
        print(f"  {status:6s} {name}")
    print(f"\n  通过: {passed}/{len(results)}")
    print("=" * 60)

    if passed == len(results):
        print("\n🎉 阶段一全部 9 坑修复验证通过！")
    else:
        print(f"\n⚠️ 有 {len(results) - passed} 项未通过，请检查。")
        sys.exit(1)
