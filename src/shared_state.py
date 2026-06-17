"""
共享状态管理 —— Day2 阶段四

核心职责：
- 提供一个线程安全的共享状态对象，供所有 Agent 读写
- 追踪任务进度（阶段、完成数、中间结果等）
- 提供乐观锁（版本号）防止并发冲突

刻意踩坑点：
- 【踩坑】不加锁直接改了：提供 bypass_lock() 方法演示直接修改全局 dict 的后果
- 【踩坑】版本号冲突：乐观锁检测到冲突后不做自动重试，直接报错
- 【踩坑】dict 是引用类型：get() 返回的是引用，外部修改会影响内部状态

类比 Java：
    SharedState 相当于一个线程安全的 ConcurrentHashMap + 乐观锁版本控制。
    类似 JPA 的 @Version 注解，但这里是手动实现的。
"""

# ── 导入依赖 ──────────────────────────────────────────────
import threading
import time
from copy import deepcopy


class SharedState:
    """
    多 Agent 共享的状态存储

    使用方式：
        state = SharedState()
        v = state.set("progress", {"phase": "planning"})
        success = state.update("progress", new_val, expected_version=v)

    特性：
    - 线程安全（内部有 threading.Lock）
    - 版本号追踪（乐观锁，类似 JPA @Version）
    - 支持快照（deep copy）
    """

    def __init__(self):
        """初始化空状态"""
        # ── 核心存储 ──
        # 使用普通 dict，线程安全由 _lock 保证
        # 【踩坑】这里不把 _data 暴露出去，防止外部直接改
        self._data: dict = {}

        # ── 版本号 ──
        # 每次 set/update 成功后递增
        # 用于乐观锁：调用方必须提供 expected_version 才能更新
        self._version: int = 0

        # ── 线程锁 ──
        # 保证 set/get/update 操作的原子性
        # 相当于 Java 的 synchronized 块或 ReentrantLock
        self._lock = threading.Lock()

        # ── 统计 ──
        self._set_count: int = 0      # set() 调用次数
        self._conflict_count: int = 0 # 乐观锁冲突次数
        self._bypass_count: int = 0   # bypass_lock() 调用次数

    # ═══════════════════════════════════════════════════════
    # 基础读写
    # ═══════════════════════════════════════════════════════

    def get(self, key: str, default=None):
        """
        安全读取状态值

        参数：
            key:     键名
            default: 键不存在时的返回值

        返回：
            值（深拷贝，防止外部修改影响内部状态）
        """
        with self._lock:
            val = self._data.get(key, default)
            # 深拷贝：防止外部拿到引用后直接修改
            # 【踩坑保护】如果这里是浅拷贝，外部改 dict 会污染内部状态
            return deepcopy(val) if val is not None else default

    def set(self, key: str, value) -> int:
        """
        安全写入状态值，返回新版本号

        参数：
            key:   键名
            value: 值（会被深拷贝存储）

        返回：
            新版本号（供后续 update 的乐观锁使用）
        """
        with self._lock:
            # 深拷贝存储：防止外部后续修改传入的对象影响内部
            self._data[key] = deepcopy(value)
            self._version += 1
            self._set_count += 1
            return self._version

    def update(self, key: str, value, expected_version: int) -> bool:
        """
        乐观锁更新：只有版本号匹配时才写入

        参数：
            key:              键名
            value:            新值
            expected_version: 期望的版本号（之前 get/set 返回的）

        返回：
            True:  更新成功
            False: 版本冲突（被其他线程/Agent 抢先改了）

        Java 类比：JPA 的 @Version 乐观锁
        """
        with self._lock:
            if self._version != expected_version:
                # 版本不匹配 —— 被其他 Agent 抢先修改了
                self._conflict_count += 1
                return False

            # 版本匹配 —— 安全写入
            self._data[key] = deepcopy(value)
            self._version += 1
            self._set_count += 1
            return True

    def update_with_retry(self, key: str, compute_new_value, max_retries: int = 3) -> bool:
        """
        Day3 修复 #13：乐观锁更新 + 自动重试

        三步循环（Read → Compute → Write）：
          ① 读最新值 + 当前版本号（加锁）
          ② 计算新值（无锁，不阻塞读）
          ③ 加锁，检查版本号：匹配→写入成功 / 不匹配→回到①

        参数：
            key:                键名
            compute_new_value:  计算新值的函数。
                               接收当前最新值，返回要写入的新值。
                               例: lambda v: (v or 0) + 1
            max_retries:        最大重试次数（默认 3）

        返回：
            True:  更新成功
            False: max_retries 次后仍然版本冲突

        使用示例：
            # 对 counter 执行 +1 操作（原子安全）
            state.update_with_retry("counter", lambda v: (v or 0) + 1)

        Java 类比：@Version + @Retryable(maxAttempts=3)
        """
        for attempt in range(1, max_retries + 1):
            # ── 步骤①：读最新值 + 版本号（加锁）──
            with self._lock:
                current_value = deepcopy(self._data.get(key))
                current_version = self._version

            # ── 步骤②：计算新值（无锁，不阻塞其他读操作）──
            try:
                new_value = compute_new_value(current_value)
            except Exception as e:
                # 计算函数出错了，直接记录失败
                print(f"  [SharedState] update_with_retry: 计算新值失败: {e}")
                return False

            # ── 步骤③：尝试写入（加锁）──
            with self._lock:
                if self._version == current_version:
                    # 版本没变 → 写入成功！
                    self._data[key] = deepcopy(new_value)
                    self._version += 1
                    self._set_count += 1
                    return True
                else:
                    # 版本变了 → 被其他 Agent 抢先了
                    self._conflict_count += 1
                    if attempt < max_retries:
                        # 等一小段随机时间再重试（避免活锁）
                        time.sleep(0.01 * attempt)
                        # continue 到下一次循环
                    # 最后一次尝试也失败了 → 返回 False

        # 所有重试耗尽
        return False

    # ═══════════════════════════════════════════════════════
    # 【踩坑】不安全操作
    # ═══════════════════════════════════════════════════════

    def bypass_lock(self, key: str, value) -> None:
        """
        【踩坑演示】绕过锁直接修改状态

        这个方法模拟「各 Agent 直接改全局 dict」的场景：
        - 不检查版本号
        - 不做深拷贝
        - 不加锁（故意用了 with self._lock 但马上释放，实际不是真正的锁保护）

        用法：对比 bypass_lock 和 set 的差异，观察并发冲突。

        对应真实场景：多个 Agent 各持一份状态引用，直接改 dict，
        导致 A 的修改被 B 覆盖。
        """
        # ⚠️ 【踩坑】虽然获取了锁，但这里我们故意演示：
        # 如果有 Agent 不通过 set/update，而是持有 _data 引用直接修改...
        # 实际上因为 _data 被封装了，这个方法是唯一能演示的入口
        self._bypass_count += 1

        # 【踩坑】不做版本检查，直接覆盖
        # 这模拟了 Agent 拿到引用后直接 self._data[key] = value
        # 绕过了版本号和深拷贝保护
        self._data[key] = value  # ← 直接赋值，不递增版本号！
        # 注意：这里故意不递增 _version，模拟绕过版本控制

    # ═══════════════════════════════════════════════════════
    # 便利方法
    # ═══════════════════════════════════════════════════════

    def get_version(self) -> int:
        """获取当前版本号（用于乐观锁）"""
        with self._lock:
            return self._version

    def snapshot(self) -> dict:
        """
        返回完整的深拷贝快照

        用于 Orchestrator 查看当前全局状态
        """
        with self._lock:
            return deepcopy(self._data)

    def has(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            return key in self._data

    def keys(self) -> list:
        """列出所有键"""
        with self._lock:
            return list(self._data.keys())

    def clear(self):
        """重置状态"""
        with self._lock:
            self._data.clear()
            self._version = 0
            self._set_count = 0
            self._conflict_count = 0
            self._bypass_count = 0

    # ═══════════════════════════════════════════════════════
    # 任务进度便利方法（多 Agent 协调专用）
    # ═══════════════════════════════════════════════════════

    def init_progress(self, total_tasks: int) -> int:
        """
        初始化任务进度追踪

        参数：
            total_tasks: 总任务数

        返回：
            版本号
        """
        return self.set("progress", {
            "phase": "planning",      # 当前阶段: planning / coding / reviewing / done
            "total": total_tasks,     # 总任务数
            "completed": 0,           # 已完成数
            "results": {},            # 各 Agent 的中间结果
            "errors": [],             # 错误记录
        })

    def mark_task_done(self, agent_name: str, task_id: int, result_summary: str) -> bool:
        """
        标记一个任务完成（带乐观锁）

        参数：
            agent_name:    完成任务的 Agent 名（如 "coder-1"）
            task_id:       任务编号
            result_summary: 结果摘要

        返回：
            True: 成功  /  False: 版本冲突
        """
        with self._lock:
            progress = self._data.get("progress")
            if not progress:
                return False

            progress["completed"] += 1
            progress["results"][f"{agent_name}_task{task_id}"] = {
                "completed_at": time.time(),
                "summary": result_summary[:200],  # 截断，防止内存膨胀
            }
            self._version += 1
            self._set_count += 1
            return True

    def record_error(self, agent_name: str, error_msg: str):
        """记录错误（不加乐观锁，错误记录不应被阻塞）"""
        with self._lock:
            progress = self._data.get("progress")
            if progress is not None:
                progress["errors"].append({
                    "agent": agent_name,
                    "time": time.time(),
                    "error": error_msg[:500],
                })
                self._version += 1

    def get_progress(self) -> dict:
        """获取当前进度快照"""
        return self.get("progress", {})

    # ═══════════════════════════════════════════════════════
    # 信息查看
    # ═══════════════════════════════════════════════════════

    def stats(self) -> dict:
        """返回状态统计信息"""
        with self._lock:
            return {
                "version": self._version,
                "keys": len(self._data),
                "set_count": self._set_count,
                "conflict_count": self._conflict_count,
                "bypass_count": self._bypass_count,
            }

    def __repr__(self) -> str:
        """打印友好信息"""
        s = self.stats()
        return (
            f"SharedState(version={s['version']}, "
            f"keys={s['keys']}, sets={s['set_count']}, "
            f"conflicts={s['conflict_count']}, bypasses={s['bypass_count']})"
        )


# ── 模块自测：纯逻辑测试（不涉及 LLM）─────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("SharedState 自测")
    print("=" * 50)

    state = SharedState()

    # 测试基础读写
    v1 = state.set("name", "test-project")
    print(f"\n1. set 后版本: {v1}")
    print(f"   读取: {state.get('name')}")
    assert state.get("name") == "test-project"
    print("   ✅ 基础读写正常")

    # 测试乐观锁成功
    ok = state.update("name", "updated", expected_version=v1)
    print(f"\n2. 乐观锁更新（版本匹配）: {'✅ 成功' if ok else '❌ 失败'}")
    assert ok
    assert state.get("name") == "updated"

    # 测试乐观锁冲突
    v2 = state.set("counter", 0)
    ok = state.update("counter", 1, expected_version=v2 - 999)  # 错误的版本号
    print(f"\n3. 乐观锁更新（版本冲突）: {'✅ 成功' if ok else '⚠️ 冲突（预期）'}")
    assert not ok, "应该检测到版本冲突"
    assert state.get("counter") == 0, "版本冲突时值不应被修改"
    print("   ✅ 乐观锁冲突检测正常")

    # 测试 bypass_lock 踩坑
    v_before = state.get_version()
    state.bypass_lock("sneaky", "绕过了版本控制")
    v_after = state.get_version()
    print(f"\n4. bypass_lock 后: 值={state.get('sneaky')}, 版本: {v_before}→{v_after}")
    print(f"   ⚠️ 版本号未递增（预期内，踩坑演示）")

    # 测试深拷贝隔离
    data = {"items": [1, 2, 3]}
    state.set("list_data", data)
    retrieved = state.get("list_data")
    retrieved["items"].append(999)  # 修改取出的数据
    original = state.get("list_data")
    assert original["items"] == [1, 2, 3], "内部数据不应被外部修改污染"
    print(f"\n5. 深拷贝隔离: 外部修改不影响内部 ✅")

    # 测试进度追踪
    state.init_progress(5)
    state.mark_task_done("coder-1", 1, "完成了冒泡排序")
    state.mark_task_done("coder-2", 2, "完成了二分查找")
    state.record_error("coder-3", "API 超时")
    progress = state.get_progress()
    print(f"\n6. 进度追踪: completed={progress['completed']}/{progress['total']}")
    print(f"   错误数: {len(progress['errors'])}")
    print(f"   结果数: {len(progress['results'])}")
    assert progress["completed"] == 2
    assert len(progress["errors"]) == 1
    print("   ✅ 进度追踪正常")

    # 测试快照
    snap = state.snapshot()
    print(f"\n7. 快照: {list(snap.keys())}")
    assert "name" in snap
    assert "progress" in snap
    print("   ✅ 快照正常")

    # 测试并发绕过锁的演示
    print(f"\n8. 统计: {state.stats()}")
    print("   ✅ 全部自测通过")

    state.clear()
    print("\n" + "=" * 50)
    print("✅ SharedState 自测全部通过")
    print("=" * 50)
