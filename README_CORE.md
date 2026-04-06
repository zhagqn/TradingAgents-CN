# TradingAgents-CN Core — 精简核心版

> 用多个 AI 智能体从不同角度分析股票，通过辩论达成共识，输出交易决策。

## 🎯 核心价值链

```
获取数据 → 多角度分析 → 辩论/风控 → 输出决策
```

## ⚡ 快速开始

### 安装

```bash
pip install -e .  # 使用 pyproject_core.toml 重命名为 pyproject.toml
```

### 3 行代码使用

```python
from tradingagents_core import TradingAgents

agent = TradingAgents(llm_provider="deepseek")  # 或 openai/qwen/zhipu
result = agent.analyze("600519")                 # 分析贵州茅台
print(result["decision"]["action"])              # 买入/持有/卖出
```

### CLI 使用

```bash
python cli_core.py 600519                    # 分析贵州茅台
python cli_core.py AAPL --provider openai    # 用 OpenAI 分析苹果
python cli_core.py 00700 --provider qwen     # 用通义千问分析腾讯
python cli_core.py 000001 --json             # 输出 JSON 格式
```

## 📁 项目结构

```
tradingagents_core/
├── __init__.py              # 入口：暴露 TradingAgents, create_llm
├── config.py                # 统一配置 (60行)
├── data/
│   ├── providers.py         # AKShare + YFinance 数据提供商 (350行)
│   ├── tools.py             # 4个核心 @tool 函数 (160行)
│   └── indicators.py        # 技术指标计算 (170行)
├── agents/
│   ├── states.py            # LangGraph 状态定义 (70行)
│   ├── analysts.py          # 4个分析师 - 工厂模式 (190行)
│   ├── researchers.py       # 看多/看空研究员 (90行)
│   ├── risk_debaters.py     # 3个风控辩论者 (110行)
│   ├── managers.py          # 研究经理 + 风控经理 (120行)
│   └── trader.py            # 交易决策 (70行)
├── graph/
│   └── workflow.py          # LangGraph 工作流 + TradingAgents API (350行)
└── llm/
    └── factory.py           # LLM 统一工厂 (100行)
```

**总计: ~1,500 行核心代码** (对比原版 239,000+ 行)

## 🏗 架构

```
                    ┌──────────────────────┐
                    │    TradingAgents      │
                    │   analyze("600519")   │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──┐  ┌─────────▼──┐  ┌─────────▼──┐  ┌─────────────┐
    │ 技术分析师  │  │ 基本面分析师│  │ 新闻分析师  │  │ 情绪分析师   │
    │ (MACD/RSI) │  │ (PE/ROE)   │  │ (东财/YF)  │  │ (社交媒体)   │
    └─────────┬──┘  └─────────┬──┘  └─────────┬──┘  └─────────┬───┘
              │                │                │                │
              └────────────────┼────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                                  │
    ┌─────────▼──────┐              ┌───────────▼─────┐
    │ 🐂 看多研究员   │◄────辩论────►│ 🐻 看空研究员    │
    └────────┬───────┘              └───────┬─────────┘
             │                               │
             └───────────┬───────────────────┘
                         │
               ┌─────────▼──────────┐
               │  📋 研究经理(裁判)  │
               └─────────┬──────────┘
                         │
               ┌─────────▼──────────┐
               │    💰 交易员        │
               └─────────┬──────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼─────┐  ┌──────▼───┐  ┌────────▼──┐
   │ 激进风控  │  │ 保守风控  │  │ 中性风控   │
   └────┬─────┘  └──────┬───┘  └────────┬──┘
        └────────────────┼────────────────┘
                         │
               ┌─────────▼──────────┐
               │  🏦 风控经理(最终)  │
               └────────────────────┘
                         │
                    最终交易决策
```

## 🔧 支持的 LLM 提供商

| 提供商 | 环境变量 | 默认模型 | 价格 |
|--------|----------|----------|------|
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat | ¥0.5/次分析 |
| 通义千问 | `DASHSCOPE_API_KEY` | qwen-plus | ¥1/次分析 |
| 智谱 AI | `ZHIPU_API_KEY` | glm-4 | ¥1/次分析 |
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini | $0.1/次分析 |
| SiliconFlow | `SILICONFLOW_API_KEY` | deepseek-chat | ¥0.3/次分析 |
| OpenRouter | `OPENROUTER_API_KEY` | deepseek-chat | 按模型计费 |

## 🌍 支持的市场

| 市场 | 数据源 | 费用 | 示例代码 |
|------|--------|------|----------|
| A股 | AKShare | 免费 | 600519, 000001 |
| 港股 | AKShare | 免费 | 00700, 09988 |
| 美股 | YFinance | 免费 | AAPL, TSLA |

## 📊 返回数据结构

```python
{
    "symbol": "600519",
    "date": "2025-01-15",
    "decision": {
        "action": "买入",          # 买入/持有/卖出
        "target_price": 1850.0,    # 目标价格
        "confidence": 0.8,         # 置信度 (0-1)
        "risk_score": 0.3,         # 风险 (0-1)
    },
    "reports": {
        "market": "...",           # 技术分析报告
        "fundamentals": "...",     # 基本面报告
        "news": "...",             # 新闻报告
        "sentiment": "...",        # 情绪报告
    },
    "debate": {
        "bull_case": "...",        # 看多论据
        "bear_case": "...",        # 看空论据
        "investment_plan": "...",  # 投资计划
    },
    "risk": {
        "final_decision": "...",   # 风控最终决策
    },
}
```

## 🧪 测试

```bash
pytest tests_core/ -v
```

## 📐 设计原则

1. **单一职责** — 每个文件只做一件事
2. **组合优于继承** — 工厂函数 + 配置字典
3. **最小接口** — 对外只暴露 `TradingAgents.analyze(symbol)`
4. **免费优先** — AKShare + YFinance 覆盖全球主要市场
5. **按需复杂** — 基础零配置，高级通过参数开启
6. **LLM 无关** — 统一接口，一行代码切换提供商
7. **可测试** — 每层可独立测试，不依赖外部服务

## 📖 应用场景

- **个人量化研究** — 替代人工翻阅研报/财报/新闻
- **批量股票筛选** — 快速筛选几十只股票
- **嵌入交易系统** — 作为分析引擎模块
- **投研日报生成** — 自动化每日投研报告
- **教育研究** — AI 多智能体系统演示
- **跨市场对比** — 统一框架分析中美港三市场

## 🆚 精简前后对比

| 维度 | 原版 | Core 版 | 缩减 |
|------|------|---------|------|
| 代码行数 | 239,391 | ~1,500 | 99.4%↓ |
| Python 文件 | 1,197 | 16 | 98.7%↓ |
| 依赖包 | 55+ | ~10 | 82%↓ |
| 基础设施 | MongoDB+Redis | 无 | 100%↓ |
| 核心功能保留 | — | 100% | ✅ |
