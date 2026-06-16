# Agent From Scratch Plus

> 从零手搓 AI Agent 的学习工程  
> 当前分支：`01-solo-agent-handcode` | Day1 原生单Agent

---

## 项目导航

```
agent-from-scratch-plus/
├── AGENTS.md              ← 本文档（项目导航，AI 执行前必读）
├── 01-计划.md             ← Day1 每日任务清单
├── .env                   ← DeepSeek API Key（不入库）
├── .env.example           ← 环境变量模板
├── .gitignore             ← Git 忽略规则
├── requirements.txt       ← Python 依赖
│
├── src/                   ← 源代码
│   ├── __init__.py
│   ├── llm_client.py      ← LLM 调用封装（DeepSeek / OpenAI 兼容）
│   ├── tools.py           ← 工具注册中心 + 内置工具（计算器、文本查询）
│   └── agent.py           ← ReAct Agent 核心循环（Think→Act→Observe）
│
├── test_phase1.py         ← 阶段一 验证脚本
├── test_phase2.py         ← 阶段二 验证脚本
│
└── docs/
    ├── 计划/计划.md        ← 总学习计划（6天）
    ├── 执行/01-执行.md     ← Day1 执行日志（逐步更新）
    ├── 学习/01-学习.md     ← 阶段一 学习指南（概念+代码讲解）
    └── 学习/02-学习.md     ← 阶段二 学习指南（ReAct循环详解）
```

---

## 当前进度

| 阶段 | 状态 | 产出 |
|------|------|------|
| 环境验证 | ✅ 完成 | .venv, .env, .gitignore |
| 阶段一：LLM调用+工具 | ✅ 完成 | src/llm_client.py, src/tools.py |
| 阶段二：ReAct循环 | ✅ 完成 | src/agent.py |
| 阶段三：对话记忆 | ⬜ 待开始 | src/memory.py |
| 阶段四：FastAPI封装 | ⬜ 待开始 | src/api.py, main.py |

---

## 技术栈

- **语言**: Python 3.11.15（uv 管理）
- **LLM**: DeepSeek（OpenAI 兼容协议，`deepseek-chat`）
- **Web框架**: FastAPI + uvicorn（阶段四）
- **包管理**: uv 0.11.7

---

## 运行方式

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行阶段一测试
python test_phase1.py

# 运行阶段二测试（ReAct Agent）
python test_phase2.py

# 阶段三测试（待创建）
# python test_phase3.py
```

---

## 执行约定

1. 每次执行前读取本文件了解项目状态
2. 每次执行后更新 `docs/执行/01-执行.md`
3. 代码文件使用中文注释，标注刻意踩坑点
4. `.env` 不入库，通过 `.gitignore` 保护
5. 每个代码块必须带注释，说明这段代码的作用——这是教学项目，不是生产代码
