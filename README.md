# OpticsGPT Lens Design Agent

面向光学镜头设计的垂直领域智能体系统。项目基于 FastAPI、本地 OpticsGPT-v0、2456 条镜头结构数据和领域规则，实现从自然语言光学需求到结构化参数解析、混合检索、硬约束过滤、孔径尺度适配、多目标重排序、ray tracing 风险评估和多轮推荐解释的端到端流程。

> 说明：项目保留了原始轻量 planner-executor agent，同时新增了 LangGraph 编排版本。LangGraph 版本将 load state、intent classification、parse/update、retrieve、rerank、raytrace、explain、save state 显式建模为状态图节点，便于展示标准 Agent workflow runtime 下的可观测链路。

## 核心能力

- 自然语言需求解析：将应用场景、F 数、视场角、焦距、入瞳孔径、总长、畸变和片数偏好解析为结构化参数。
- Schema 与规则校验：对非法数值、缺失字段、场景先验和 hard constraint 进行补全与校验。
- 分层混合检索：融合严格匹配、宽松召回、数值相似度、场景约束和领域规则，从 2456 条镜头数据中召回 Top-K 候选。
- 孔径尺度适配：根据 F 数和 EPD 反推目标焦距，计算 scale factor，并同步换算实际焦距、TTL 和孔径。
- 多目标重排序：综合 FOV、F number、TTL、结构复杂度、畸变风险、ray spread 和可行性等级生成推荐顺序。
- 多轮 Agent：基于 session_id 维护设计状态，支持约束合并、重新检索、重新评价和推荐理由解释。
- LangGraph 编排：提供标准状态图版本 Agent，支持条件路由和节点级工具调用追踪。
- 结构可视化：解析 CODE V 风格 `.seq` 文件，生成镜头结构布局图。

## 系统架构

```text
User requirement
      |
      v
FastAPI endpoints
      |
      +--> OpticsGPT-v0 / rule fallback parser
      |        |
      |        v
      |   structured requirement
      |
      +--> domain rule enhancement
      |        |
      |        v
      |   hybrid retrieval over lens database
      |        |
      |        v
      |   hard constraint check + scale adaptation
      |        |
      |        v
      |   ray tracing / ray spread evaluation
      |        |
      |        v
      |   reranked candidates + explanation + layout image
      |
      +--> session-based workflow agent
```

## 主要接口

| Endpoint | 用途 |
| --- | --- |
| `GET /health` | 服务健康检查 |
| `POST /parse_requirement` | 自然语言需求解析 |
| `POST /search_lens` | 基于结构化需求检索镜头候选 |
| `POST /design_assist` | 单轮完整设计推荐 |
| `POST /design_feasibility` | 设计可行性分析 |
| `POST /agent/chat` | 多轮 Agent 对话 |
| `POST /agent/langgraph/chat` | LangGraph 状态图版本多轮 Agent |
| `POST /layout/optiland/generate` | 生成镜头结构图 |
| `POST /v1/chat/completions` | OpenAI-compatible 本地模型接口 |

## 快速开始

### 1. 创建环境

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 准备模型与数据

完整镜头数据集不包含在公开仓库中。公开版本只保留 `data/` 目录结构；运行完整系统时，请将私有镜头数据放到本地路径，例如 `data/lens_data.csv`，或通过 `LENS_DATA_PATH` 指向你的本地数据文件。

本地模型默认路径为：

```text
/root/.cache/modelscope/hub/models/ckdckd/OpticsGPT-v0
```

Windows 或自定义路径建议通过环境变量指定：

```bash
set MODEL_DIR=C:\path\to\OpticsGPT-v0
set LENS_DATA_PATH=.\data\lens_data.csv
set REALTIME_RAYTRACE=1
```

### 3. 启动服务

```bash
uvicorn app:app --host 0.0.0.0 --port 6006
```

启动后访问：

- API 文档：`http://127.0.0.1:6006/docs`
- 前端页面：`http://127.0.0.1:6006/static/index.html`

## 示例请求

```bash
curl -X POST http://127.0.0.1:6006/design_assist ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"设计一个车载广角镜头，F数2.0，视场角大于120度，孔径4mm，总长尽量短，畸变尽量低\",\"top_k\":9}"
```

多轮 Agent 示例：

```bash
curl -X POST http://127.0.0.1:6006/agent/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"demo-001\",\"message\":\"设计一个手机超广角镜头，F数2.0，视场角130度，孔径3.5mm\",\"top_k\":9}"

curl -X POST http://127.0.0.1:6006/agent/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"demo-001\",\"message\":\"总长再短一点，结构简单一点\",\"top_k\":9}"
```

LangGraph Agent 示例：

```bash
curl -X POST http://127.0.0.1:6006/agent/langgraph/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"lg-demo-001\",\"message\":\"设计一个车载广角镜头，F数2.0，视场角大于120度，孔径4mm\",\"top_k\":9}"
```

## 测试

项目包含一组不依赖本地大模型的轻量单元测试，用于验证数据加载、孔径解析和 session 状态隔离：

```bash
pytest -q
```

## 项目结构

```text
.
├── app.py                         # FastAPI 入口与主要 API
├── hybrid_retrieval_engine.py      # 分层混合检索与候选召回
├── aperture_scale_utils.py         # 孔径尺度适配
├── design_result_optimizer.py      # 多目标重排序与推荐解释
├── realtime_raytrace_engine.py     # ray tracing / ray spread 评价
├── optiland_layout_renderer.py     # seq 解析与镜头布局图生成
├── lens_loader.py                  # 镜头数据读取与字段归一化
├── kg_rules.py                     # 场景规则与领域知识增强
├── agent/                          # 轻量 workflow agent
│   └── langgraph_agent.py           # LangGraph 状态图编排版本
├── data/                           # 公开仓库仅保留目录结构，完整数据不提交
├── static/                         # 简单前端与生成图像
├── tests/                          # 轻量核心工具测试
└── run_api_tests.py                # 批量 API 测试脚本
```

## 当前工程边界

- Agent 目前是单进程内存态 session，不适合直接用于多副本部署；生产环境应接入 Redis/PostgreSQL 等外部状态存储。
- 规则型 intent classifier 可解释性强，但泛化能力有限；可替换为 LLM function calling 或 LangGraph 状态图。
- ray tracing 为轻量实时评价与风险估计，不等价于完整商业光学优化软件。
- 大模型和完整数据集是否公开，需要根据数据来源和授权情况确认；公开仓库可保留样例数据。
- 代码中部分中文注释存在历史编码问题，建议在正式开源前统一整理为 UTF-8。

## 面试讲法

推荐表述为：

> 我没有简单套 LangGraph，而是针对光学镜头设计任务实现了一个轻量级 workflow agent：用 session state 维护多轮约束，用 planner 将用户意图映射为 parse、retrieve、rerank、raytrace、explain 等工具链，再结合领域规则和硬约束保证推荐结果可解释、可控。这个设计更偏工程可控性和领域效果，如果进入更复杂的多工具协作场景，可以迁移到 LangGraph 做状态图编排。

现在也可以补充：

> 后续我将同一套工具链迁移到了 LangGraph：每个阶段被建模为 graph node，并用 conditional edge 做意图路由。这样既保留了原有领域规则的可控性，也能展示标准 Agent workflow 框架下的状态流转、节点追踪和扩展能力。
