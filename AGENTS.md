# Agent From Scratch Plus

> 从零手搓 AI Agent 的学习工程  
> 当前分支：`02-multi-agent-handcode` | Day2 原生多Agent

## 项目概述

教学项目，用纯 Python 从零实现 AI Agent 系统，不依赖 LangChain 等框架。每天 8 小时，总计 6 天，从单 Agent 到多 Agent 协作再到框架重构+工程化。

## 快速开始

```bash
source .venv/bin/activate          # 激活虚拟环境
python test_day2_phase1.py         # 运行当前阶段测试
```

## 文档导航

| 文档 | 内容 |
|------|------|
| [docs/进度.md](docs/进度.md) | 每日/每阶段完成情况 |
| [docs/技术栈.md](docs/技术栈.md) | Python 版本、依赖、LLM 配置 |
| [docs/运行指南.md](docs/运行指南.md) | 完整命令、环境配置、常见问题 |
| [docs/约定.md](docs/约定.md) | 编码规范、执行约定、Git 规则 |
| [docs/计划/计划.md](docs/计划/计划.md) | 6天总学习计划 |
| [docs/计划/02-计划.md](docs/计划/02-计划.md) | Day2 详细任务清单 |

## 项目结构

```
agent-from-scratch-plus/
├── AGENTS.md              ← AI 执行前必读（本文档）
├── 01-计划.md / docs/计划/02-计划.md ← 每日任务清单
├── src/                   ← 源代码（按阶段组织）
│   ├── llm_client.py      ← LLM 调用封装
│   ├── tools.py           ← 工具注册中心
│   ├── agent.py           ← ReAct Agent 核心循环
│   ├── memory.py          ← 对话记忆管理
│   ├── api.py             ← FastAPI 接口层
│   └── agents/            ← Day2: 多角色 Agent 实现
├── test_*.py              ← 各阶段验证脚本
├── main.py                ← HTTP 服务入口
└── docs/                  ← 项目文档
    ├── 进度.md            ← 当前进度
    ├── 技术栈.md          ← 技术选型说明
    ├── 运行指南.md        ← 运行命令和环境
    ├── 约定.md            ← 编码规范和约定
    ├── 学习/
    │   ├── 01-单智能体/   ← Day1 学习指南
    │   └── 02-原生多agent/ ← Day2 学习指南
```

## 核心约定

- 代码必须带详细中文注释（教学项目）
- `.env` 不入库，通过 `.gitignore` 保护
- 完成阶段后不立即推送，等用户学完统一推
- 刻意踩坑保留，Day3 统一修复——详见 [docs/约定.md](docs/约定.md)
