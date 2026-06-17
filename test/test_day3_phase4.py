"""
Day3 阶段四验证脚本 —— API 层联调 + 全链路测试

运行方式：
    source .venv/bin/activate
    python test/test_day3_phase4.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect
import urllib.request
import urllib.error
import json
import time
import concurrent.futures
import subprocess
import signal


# ═══════════════════════════════════════════════════════════
def start_server():
    """启动测试服务器"""
    import subprocess
    # 用 sys.executable 在当前激活的 venv 下运行 uvicorn
    # 注意：uv venv 下 sys.executable 指向基础 Python，需配合 -m 加载 venv site-packages
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # 等待服务就绪（最多重试 10 次，每次 0.5s）
    for _ in range(10):
        time.sleep(0.5)
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/", timeout=2)
            return proc
        except Exception:
            pass
    proc.terminate()
    raise RuntimeError("服务器启动超时")


def stop_server(proc):
    """停止服务器"""
    proc.terminate()
    proc.wait(timeout=5)


def api_post(path, data):
    """POST 请求"""
    req = urllib.request.Request(
        f"http://127.0.0.1:8000{path}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=180)
    return json.loads(resp.read())


def api_get(path):
    """GET 请求"""
    resp = urllib.request.urlopen(f"http://127.0.0.1:8000{path}")
    return json.loads(resp.read())


# ═══════════════════════════════════════════════════════════
def test_4a_orchestrate_serial():
    """串行编排端到端"""
    print("=" * 60)
    print("验证 4a：串行编排 /chat/orchestrate")
    print("=" * 60)

    result = api_post("/chat/orchestrate", {
        "requirement": "实现判断回文字符串的函数",
        "mode": "serial",
        "with_fix": False,  # 加快测试
    })

    assert result["status"] == "ok", f"状态不对: {result}"
    assert len(result["plan"]) > 50, f"plan 太短: {len(result['plan'])}字"
    assert len(result["code"]) > 100, f"code 太短: {len(result['code'])}字"
    assert len(result["review"]) > 100, f"review 太短"
    print(f"  ✅ plan:{len(result['plan'])}字 code:{len(result['code'])}字 review:{len(result['review'])}字")
    print("  ✅ 验证 4a 通过\n")
    return True


def test_4b_orchestrate_parallel():
    """并行编排端到端"""
    print("=" * 60)
    print("验证 4b：并行编排 /chat/orchestrate")
    print("=" * 60)

    result = api_post("/chat/orchestrate", {
        "requirement": "实现冒泡排序和二分查找",
        "mode": "parallel",
        "n_coders": 2,
    })

    assert result["status"] == "ok"
    assert len(result.get("codes", [])) == 2, f"codes 数量不对: {len(result.get('codes',[]))}"
    assert result["timing"]["coders"] > 0, "缺少耗时统计"
    print(f"  ✅ {len(result['codes'])}个Coder plan:{len(result['plan'])}字 review:{len(result['review'])}字")
    print(f"     耗时: plan={result['timing']['plan']:.1f}s coders={result['timing']['coders']:.1f}s")
    print("  ✅ 验证 4b 通过\n")
    return True


def test_4c_progress():
    """进度端点"""
    print("=" * 60)
    print("验证 4c：进度查询 /chat/progress")
    print("=" * 60)

    # 没有任务时
    result = api_get("/chat/progress")
    assert "count" in result, "缺少 count"
    print(f"  ✅ /chat/progress: count={result['count']}")

    # 不存在的任务
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/chat/progress/nonexistent")
        assert False, "应该返回 404"
    except urllib.error.HTTPError as e:
        assert e.code == 404
        print("  ✅ 不存在任务返回 404")
    print("  ✅ 验证 4c 通过\n")
    return True


def test_4d_concurrent():
    """并发请求"""
    print("=" * 60)
    print("验证 4d：并发 3 个 /chat 请求")
    print("=" * 60)

    def call(i):
        t0 = time.time()
        data = json.dumps({"message": f"计算 {i}*{i}"}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8000/chat", data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=60)
        return time.time() - t0

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        times = list(ex.map(call, [1, 2, 3]))

    for i, t in enumerate(times):
        print(f"  请求{i+1}: {t:.1f}s")
    # 并发下每个请求应该在合理时间内完成
    assert all(t < 30 for t in times), "有请求超时"
    print("  ✅ 3 个并发请求全部完成")
    print("  ✅ 验证 4d 通过\n")
    return True


def test_4e_endpoint_exists():
    """端点结构验证（不调 LLM）"""
    print("=" * 60)
    print("验证 4e：API 端点结构")
    print("=" * 60)

    from src import api

    # /chat/orchestrate 存在
    assert hasattr(api, "orchestrate"), "缺少 orchestrate 端点"
    assert inspect.iscoroutinefunction(api.orchestrate), "orchestrate 不是 async"
    print("  ✅ /chat/orchestrate 是 async 端点")

    # /chat/progress 存在
    assert hasattr(api, "get_progress"), "缺少 get_progress 端点"
    assert hasattr(api, "list_progress"), "缺少 list_progress 端点"
    print("  ✅ /chat/progress/{task_id} 和 /chat/progress 存在")

    # OrchestrateRequest 模型存在
    assert hasattr(api, "OrchestrateRequest"), "缺少 OrchestrateRequest 模型"
    print("  ✅ OrchestrateRequest Pydantic 模型存在")

    print("  ✅ 验证 4e 通过\n")
    return True


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Day3 阶段四：API 联调 + 全链路测试")
    print("=" * 60 + "\n")

    # 启动服务器
    print("启动测试服务器...")
    server = start_server()

    try:
        # 验证服务可用
        health = api_get("/")
        assert health["status"] == "ok"
        print(f"✅ 服务就绪: {health['service']}\n")

        results = {}

        # 结构测试（不调 LLM）
        test_4e_endpoint_exists()

        # 端到端测试（调 LLM）
        tests = [
            ("4a 串行编排", test_4a_orchestrate_serial),
            ("4b 并行编排", test_4b_orchestrate_parallel),
            ("4c 进度查询", test_4c_progress),
            ("4d 并发压测", test_4d_concurrent),
        ]

        for name, fn in tests:
            try:
                fn()
                results[name] = "✅"
            except Exception as e:
                results[name] = f"❌ {e}"

        print("=" * 60)
        print("验证汇总")
        print("=" * 60)
        for name, r in results.items():
            print(f"  {r}  {name}")
        print("=" * 60)

    finally:
        stop_server(server)
