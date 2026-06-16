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

        _queues 的结构：
            {
                "planner": [msg1, msg2, ...],
                "coder":   [msg3, ...],
                "reviewer": [msg4, ...],
            }
        """
        # dict 的 key 是接收方标识，value 是消息列表
        # 初始为空，第一个 send() 调用时自动创建对应 key
        self._queues: dict[str, list[Message]] = {}
        # 全局消息计数 —— 用于统计
        self._message_count: int = 0

    # ═══════════════════════════════════════════════════════
    # 发送
    # ═══════════════════════════════════════════════════════

    def send(self, message: Message) -> None:
        """
        发送一条消息到指定接收方的队列

        参数：
            message: Message 实例

        行为：
            - 消息追加到 receiver 对应的队列末尾
            - 如果 receiver 队列不存在，自动创建
            - 【踩坑】不做任何去重检查或消息大小限制
        """
        receiver = message.receiver

        # 如果该接收方还没有队列，创建一个
        if receiver not in self._queues:
            self._queues[receiver] = []

        # 追加到队尾 —— FIFO 语义
        self._queues[receiver].append(message)
        self._message_count += 1

    def broadcast(self, sender: str, type_: str, content: str, receivers: list[str]) -> None:
        """
        向多个接收方广播同一条消息

        参数：
            sender:    发送方标识
            type_:     消息类型
            content:   消息内容
            receivers: 接收方标识列表

        内部实现：对每个 receiver 创建独立的 Message 实例并 send()

        用途：Planner 把任务分发给多个 Coder 时用
        """
        for receiver in receivers:
            msg = Message(
                sender=sender,
                receiver=receiver,
                type=type_,
                content=content,
            )
            self.send(msg)

    # ═══════════════════════════════════════════════════════
    # 接收
    # ═══════════════════════════════════════════════════════

    def receive(self, receiver: str) -> list[Message]:
        """
        获取指定接收方的所有消息

        参数：
            receiver: 接收方标识（如 "coder"）

        返回：
            该接收方的消息列表（按到达时间排序）

        【踩坑】只读不删——关键踩坑点：
            - 消息不会被消费，下次 receive() 还能拿到
            - 同一个 receiver 调用 N 次 receive()，每次都拿到全量历史消息
            - 多个 Agent 都能读到发给同一个 receiver 的消息
            - 这会导致重复处理、无限循环
            - Day3 修复：增加 consume() 方法标记已读，或 receive() 直接 pop

        类比 Java：
            相当于只调用了 Queue.peek() 而不调 poll()，
            消息永远留在队列里。
        """
        return self._queues.get(receiver, [])

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
