"""
Agent 基类 —— Day2 阶段二

核心职责：
- 封装 Agent 的通用行为：LLM 调用、角色配置、消息收发
- Planner / Coder / Reviewer 都继承此类
- 子类只需重写 execute() 定义具体行为

刻意踩坑点：
- 【踩坑】不做输入校验：Agent 直接信任上游传过来的内容
  如果 Planner 输出格式不对，Coder 照样吞下去处理
- 【踩坑】不做输出规范化：每个 Agent 输出格式由 LLM 自由决定
  导致下游解析困难——Day3 加 JSON Schema 约束

类比 Java：
    BaseAgent 是抽象父类，子类实现抽象方法 execute()。
    相当于 Spring 的 AbstractService + template method 模式。
"""

# ── 导入依赖 ──────────────────────────────────────────────
from src.llm_client import LLMClient
from src.roles import AgentRole, get_role_config, RoleConfig
from src.message_bus import MessageBus, Message


class BaseAgent:
    """
    多 Agent 系统中的 Agent 基类

    每个 Agent 实例包含：
    - 一个角色配置（决定 system prompt 和 temperature）
    - 一个 LLM 客户端（调用 DeepSeek）
    - 一个名称标识（用于消息总线的 sender/receiver）
    - 一个消息总线引用（共享的通信通道）

    子类需要实现：
        execute(input_text: str) → str
        定义 Agent 收到消息后的具体处理逻辑

    使用方式：
        planner = PlannerAgent(bus, "planner-1")
        result = planner.execute("实现一个排序算法")
        # result 是 LLM 根据 Planner 的 system prompt 生成的规划结果
    """

    def __init__(self, bus: MessageBus, name: str, role: AgentRole):
        """
        初始化 Agent

        参数：
            bus:  共享的消息总线（所有 Agent 共用一个实例）
            name: 此 Agent 的唯一标识（如 "planner-1"）
            role: 此 Agent 的角色枚举
        """
        # ── 标识 ──
        self.name = name          # Agent 名字，用于消息总线的 sender/receiver
        self.role = role          # 角色枚举（PLANNER / CODER / REVIEWER）

        # ── 角色配置 ──
        self.config: RoleConfig = get_role_config(role)

        # ── LLM 客户端 ──
        # 每个 Agent 有自己的 LLMClient 实例
        # 注意：这不是多实例共享连接池，而是独立调用
        self.llm = LLMClient()

        # ── 消息总线 ──
        # 【踩坑】所有 Agent 共享同一个 MessageBus 引用
        # 这意味着一个 Agent 可以读到发给其他 Agent 的消息
        self.bus = bus

        # ── 共享状态 ──（阶段四新增，可选）
        # 如果 Orchestrator 传了 SharedState，Agent 可以读写全局进度
        # 如果没传（如阶段二的用法），Agent 正常运行不受影响
        self.shared_state = None  # 由 Orchestrator 在创建 Agent 后设置

    # ═══════════════════════════════════════════════════════
    # 子类必须实现的抽象方法
    # ═══════════════════════════════════════════════════════

    def execute(self, input_text: str) -> str:
        """
        处理输入并返回结果 —— 子类必须重写

        参数：
            input_text: 输入文本（任务描述 / 代码 / 审核意见等）

        返回：
            处理结果文本（子任务列表 / 代码 / 审核反馈等）

        子类通常做三件事：
        1. 构建消息（system prompt + user input）
        2. 调用 LLM
        3. 返回 LLM 的输出

        【踩坑】这里不校验输入输出格式，
        完全信任 LLM 返回的就是可用的
        """
        raise NotImplementedError("子类必须实现 execute() 方法")

    # ═══════════════════════════════════════════════════════
    # 消息总线操作
    # ═══════════════════════════════════════════════════════

    def send_message(self, receiver: str, type_: str, content: str) -> Message:
        """
        通过消息总线向另一个 Agent 发送消息

        参数：
            receiver: 接收方 Agent 的名称
            type_:    消息类型（"task" / "code" / "review"）
            content:  消息内容

        返回：
            发送的 Message 实例

        用法：
            planner.send_message("coder-1", "task", "实现二分查找")
        """
        msg = Message(
            sender=self.name,
            receiver=receiver,
            type=type_,
            content=content,
        )
        self.bus.send(msg)
        return msg

    def receive_messages(self) -> list[Message]:
        """
        从消息总线获取发给自己的消息

        返回：
            发给此 Agent 的消息列表

        【踩坑】使用 receive_since 而不是 receive，
        但 since 是 0，所以首次调用会拿到全部历史消息。
        这里没有做真正的增量消费。
        """
        return self.bus.receive(self.name)

    def receive_tasks(self) -> list[Message]:
        """
        只获取 task 类型的消息 —— Coder 专用

        返回：
            type="task" 的消息列表
        """
        return self.bus.receive_by_type(self.name, "task")

    def receive_reviews(self) -> list[Message]:
        """
        只获取 review 类型的消息 —— Coder 修复专用

        返回：
            type="review" 的消息列表
        """
        return self.bus.receive_by_type(self.name, "review")

    # ═══════════════════════════════════════════════════════
    # LLM 调用
    # ═══════════════════════════════════════════════════════

    def call_llm(self, user_prompt: str, extra_context: str = "") -> str:
        """
        用角色的 system prompt 调用 LLM

        参数：
            user_prompt:    用户/上游传来的具体任务
            extra_context:  可选的额外上下文（如历史消息、全局状态等）

        返回：
            LLM 的文字回复（content 字段）

        这是 BaseAgent 最核心的方法 —— 所有子类的 execute()
        最终都会调用它来和 LLM 交互。
        """
        # ── 构建消息列表 ──
        # system 消息：角色的专属 prompt（告诉 LLM 它是什么角色）
        messages = [{"role": "system", "content": self.config.prompt}]

        # 如果有额外上下文（如全局状态），追加到消息中
        if extra_context:
            messages.append({
                "role": "system",
                "content": f"当前上下文信息：\n{extra_context}"
            })

        # user 消息：具体的任务描述
        messages.append({"role": "user", "content": user_prompt})

        # ── 调用 LLM ──
        response = self.llm.chat(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=2048,
        )

        # ── 返回文本内容 ──
        # 【踩坑】如果 LLM 返回了 tool_calls（不应该，因为没传 tools），
        # 这里会返回 None，导致后续处理出错
        return response.get("content", "") or ""

    # ═══════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════

    def __repr__(self) -> str:
        """打印友好的 Agent 信息"""
        return f"{self.__class__.__name__}(name={self.name}, role={self.role.value})"


# ── 模块自测 ──────────────────────────────────────────────
if __name__ == "__main__":
    from src.roles import AgentRole

    bus = MessageBus()

    # 创建一个测试用的 Agent（用 Planner 角色）
    class TestAgent(BaseAgent):
        """测试子类 —— 简单回显"""
        def execute(self, input_text: str) -> str:
            return f"[{self.role.value}] 收到: {input_text}"

    agent = TestAgent(bus, "test-1", AgentRole.PLANNER)
    print(f"Agent: {agent}")
    print(f"Role config: {agent.config}")

    # 测试 execute
    result = agent.execute("你好")
    print(f"execute 结果: {result}")

    # 测试消息收发
    agent.send_message("coder-1", "task", "测试任务")
    msgs = bus.receive("coder-1")
    print(f"收到消息: {msgs[0]}")
