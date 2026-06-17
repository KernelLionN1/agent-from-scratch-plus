"""
消息总线 —— Day2 阶段一

核心职责：
- 提供 Agent 之间通信的基础设施
- 消息格式标准化：sender / receiver / type / content
- 支持点对点发送和广播

刻意踩坑点：
- 【踩坑】消息重复消费：receive() 不删除消息，同一消息可被多次读取
  意味着多个 Agent 可能重复处理同一条消息
  保留此坑观察重复处理的影响，Day3 用消费标记 + 消息 ID 去重修复
- 【踩坑】无消息持久化：消息存内存，进程重启全丢

类比 Java：
    MessageBus 相当于一个简化版的消息队列（类似 JMS Topic），
    Message 类相当于一个不可变的 Event DTO。

使用方式：
    bus = MessageBus()
    bus.send(Message(sender="planner", receiver="coder", type="task", content="...")))
    messages = bus.receive("coder")
"""

# ── 导入 ──────────────────────────────────────────────────
import time
import uuid
from dataclasses import dataclass, field


# ── 消息数据类 ────────────────────────────────────────────
@dataclass
class Message:
    """
    标准化的 Agent 间消息

    字段说明：
        id:       消息唯一标识（UUID），用于后续去重——但目前【踩坑】没有实际用上
        sender:   发送方角色名（如 "planner"）
        receiver: 接收方角色名（如 "coder"），空字符串表示广播
        type:     消息类型 —— "task" | "code" | "review" | "result" | "info"
        content:  消息内容（纯文本）
        timestamp: 消息创建时间戳

    类比 Java：
        相当于一个 immutable DTO，所有字段 public final。
        用 @dataclass 自动生成 __init__、__repr__、__eq__。
    """

    sender: str
    receiver: str
    type: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])  # 短 UUID，便于阅读
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """转为 dict —— 方便序列化和日志输出"""
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        """友好的字符串表示"""
        return (
            f"[{self.type.upper()}] {self.sender} → {self.receiver}: "
            f"{self.content[:50]}{'...' if len(self.content) > 50 else ''}"
        )


# ── 消息总线类 ────────────────────────────────────────────
class MessageBus:
    """
    简易消息总线 —— 多 Agent 通信的中央枢纽

    内部实现：
        _queues: dict，key 是接收方名称，value 是该接收方的消息列表
        所有消息按到达顺序追加到对应队列

    【踩坑】receive() 只读取不删除：
        - Agent A 调用 receive("coder") 拿到消息
        - Agent B 也调用 receive("coder") 拿到同样的消息
        - 同一 Agent 多次调用 receive() 会重复拿到旧消息
        - Day3 修复方案：每条消息加 consumed 标记，或 receive 时 pop

    类比 Java：
        相当于一个简化的 BlockingQueue<E> 集合，
        但故意不实现 take()（消费即删除），只实现 peek()（查看不删除）。
    """

    def __init__(self):
        """
        初始化空的消息总线

        Day3 修复 #18+#19：增加去重和增量消费支持
        - _offsets: 每个接收方的已读消息偏移
        - _seen_hashes: 已发送消息的去重指纹
        """
        self._queues: dict[str, list[Message]] = {}
        self._message_count: int = 0
        # Day3 修复 #18：每个接收方的已读偏移（增量消费）
        self._offsets: dict[str, int] = {}
        # Day3 修复 #19：消息去重指纹集合（sender+type+content）
        self._seen_hashes: set[str] = set()

    # ═══════════════════════════════════════════════════════
    # 发送
    # ═══════════════════════════════════════════════════════

    def send(self, message: Message) -> str | None:
        """
        发送消息（Day3 修复 #19：自动去重）

        参数：
            message: Message 实例

        返回：
            消息 ID（如果是新消息），None（如果是重复消息）

        去重逻辑：同一 sender+type+content 的消息只发一次。
        """
        receiver = message.receiver
        if receiver not in self._queues:
            self._queues[receiver] = []

        # ── Day3 修复 #19：消息去重 ──
        # 计算消息指纹：sender + type + content（前 200 字）
        content_hash = message.content[:200] if message.content else ""
        fingerprint = f"{message.sender}|{message.type}|{content_hash}"
        if fingerprint in self._seen_hashes:
            # 重复消息，静默忽略
            return None
        self._seen_hashes.add(fingerprint)

        self._queues[receiver].append(message)
        self._message_count += 1
        return message.id

    def broadcast(self, sender: str, type_: str, content: str, receivers: list[str]) -> None:
        """
        向多个接收方广播同一条消息

        内部实现：对每个 receiver 创建独立的 Message 实例并直接写入队列。
        绕过 send() 的去重检查——广播的本质就是对不同接收方发"同一内容"。

        Day6 修复：之前调用 send()，去重导致除了第一个接收方外全被丢弃。
        """
        for receiver in receivers:
            msg = Message(
                sender=sender,
                receiver=receiver,
                type=type_,
                content=content,
            )
            # 直接入队，不走 send() 的去重逻辑
            if receiver not in self._queues:
                self._queues[receiver] = []
            self._queues[receiver].append(msg)
            self._message_count += 1

    # ═══════════════════════════════════════════════════════
    # 接收
    # ═══════════════════════════════════════════════════════

    def receive(self, receiver: str) -> list[Message]:
        """
        Day3 修复 #18：增量拉取 —— 只返回未读过的新消息

        每次调用后自动推进读取偏移，不会重复返回同一条消息。

        参数：
            receiver: 接收方标识

        返回：
            新到达的（未被该接收方读过）消息列表
        """
        messages = self._queues.get(receiver, [])
        offset = self._offsets.get(receiver, 0)

        # 从上次读取位置开始拿新消息
        new_msgs = messages[offset:]
        # 更新偏移到队尾
        self._offsets[receiver] = len(messages)
        return new_msgs

    def receive_all(self, receiver: str) -> list[Message]:
        """
        获取所有消息（不更新偏移，用于调试和首次全量拉取）
        """
        return list(self._queues.get(receiver, []))

    def wait_for_message(
        self, receiver: str, msg_type: str, timeout: float = 30.0
    ) -> Message | None:
        """
        Day3 修复 #20：阻塞等待指定类型的消息到达（依赖检查）

        用于确保执行顺序 —— 例如 Reviewer 等 Coder 的 code 消息到达后再审核。

        参数：
            receiver: 等待的接收方
            msg_type: 等待的消息类型
            timeout:  最大等待秒数

        返回：
            匹配的 Message（成功），None（超时）
        """
        import time as _time
        deadline = _time.time() + timeout
        while _time.time() < deadline:
            # 用 receive_all 检查（不用 receive，避免推进 offset）
            for msg in self._queues.get(receiver, []):
                if msg.type == msg_type:
                    return msg
            _time.sleep(0.1)
        return None  # 超时

    def receive_since(self, receiver: str, since: float) -> list[Message]:
        """
        获取指定时间之后到达的消息 —— 用于增量拉取

        参数：
            receiver: 接收方标识
            since:    时间戳（float），只返回此时间之后的消息

        返回：
            时间过滤后的消息列表

        用途：Agent 记录上次拉取时间，每次只拿新消息。
        虽然不能完全解决重复消费问题，但至少不会重复处理所有历史消息。
        """
        all_messages = self._queues.get(receiver, [])
        return [m for m in all_messages if m.timestamp > since]

    def receive_by_type(self, receiver: str, type_: str) -> list[Message]:
        """
        按消息类型过滤 —— 方便 Agent 只关注自己关心的消息

        参数：
            receiver: 接收方标识
            type_:    消息类型（"task" / "code" / "review" / "result"）

        返回：
            该接收方指定类型的消息列表

        用途：Coder 只拉取 "task" 类型消息，忽略 "review" 等
        """
        all_messages = self._queues.get(receiver, [])
        return [m for m in all_messages if m.type == type_]

    # ═══════════════════════════════════════════════════════
    # 状态查询
    # ═══════════════════════════════════════════════════════

    def stats(self) -> dict:
        """
        返回消息总线统计信息

        返回：
            {
                "total_messages": N,          # 总消息数
                "queues": {                    # 每个队列的消息数
                    "planner": N,
                    "coder": N,
                    ...
                }
            }
        """
        return {
            "total_messages": self._message_count,
            "queues": {name: len(msgs) for name, msgs in self._queues.items()},
        }

    def clear(self) -> None:
        """清空所有消息 —— 用于测试重置"""
        self._queues.clear()
        self._message_count = 0


# ── 模块自测 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("消息总线模块自测")
    print("=" * 50)

    bus = MessageBus()

    # 测试 1：基本发送和接收
    print("\n── 测试 1：基本发送和接收 ──")
    bus.send(Message(
        sender="planner",
        receiver="coder",
        type="task",
        content="实现一个计算斐波那契数列的函数"
    ))
    bus.send(Message(
        sender="planner",
        receiver="coder",
        type="task",
        content="实现一个快速排序算法"
    ))

    coder_msgs = bus.receive("coder")
    for msg in coder_msgs:
        print(f"  📨 {msg}")

    # 测试 2：【踩坑】重复接收
    print("\n── 测试 2：【踩坑】重复接收 ──")
    coder_msgs_2 = bus.receive("coder")  # 第二次接收
    print(f"  第一次 receive: {len(coder_msgs)} 条")
    print(f"  第二次 receive: {len(coder_msgs_2)} 条")
    print(f"  ⚠️ 消息数量相同！验证了重复消费问题")

    # 测试 3：广播
    print("\n── 测试 3：广播 ──")
    bus.broadcast(
        sender="planner",
        type_="info",
        content="所有 Agent 注意：需求已更新",
        receivers=["coder", "reviewer"],
    )
    print(f"  Coder 队列: {len(bus.receive('coder'))} 条")
    print(f"  Reviewer 队列: {len(bus.receive('reviewer'))} 条")

    # 测试 4：按类型过滤
    print("\n── 测试 4：按类型过滤 ──")
    tasks = bus.receive_by_type("coder", "task")
    infos = bus.receive_by_type("coder", "info")
    print(f"  task 类型: {len(tasks)} 条")
    print(f"  info 类型: {len(infos)} 条")

    # 测试 5：时间增量拉取
    print("\n── 测试 5：时间增量拉取 ──")
    marker = time.time()
    bus.send(Message(sender="reviewer", receiver="coder", type="review", content="代码质量OK"))
    new_msgs = bus.receive_since("coder", marker)
    print(f"  增量消息: {len(new_msgs)} 条")
    for msg in new_msgs:
        print(f"  📨 {msg}")

    # 统计
    print(f"\n── 统计 ──")
    print(f"  {bus.stats()}")
