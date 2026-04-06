"""
交易员 — 基于所有分析报告做出最终投资决策。
"""

from __future__ import annotations

import logging
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from tradingagents_core.agents.states import AgentState
from tradingagents_core.data.providers import detect_market, get_currency, get_market_name

logger = logging.getLogger(__name__)


def create_trader(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    """创建交易员节点。"""

    def trader_node(state: AgentState) -> dict:
        ticker = state["company_of_interest"]
        investment_plan = state.get("investment_plan", "")

        market = detect_market(ticker)
        currency_name, currency_symbol = get_currency(market)
        market_name = get_market_name(market)

        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        prompt = f"""你是一位专业交易员，基于分析师报告做出投资决策。

⚠️ 重要: 当前分析的股票是 {ticker}（{market_name}），使用{currency_name}（{currency_symbol}）计价。

必须提供的决策要素:
1. 投资建议: 明确的 **买入** / **持有** / **卖出**
2. 目标价格: 具体数值（{currency_symbol}），必须提供
3. 置信度: 0-1 之间
4. 风险评分: 0-1 之间（0=低风险，1=高风险）
5. 详细理由

目标价格参考:
- 基于基本面分析（PE/PB/DCF）
- 参考技术面支撑/阻力位
- 考虑行业平均估值
- 结合市场情绪和新闻
- 必须使用正确的货币单位（{currency_symbol}）

格式要求:
最终交易建议: **买入/持有/卖出**
目标价格: {currency_symbol}XX.XX
置信度: X.X
风险评分: X.X
详细理由: ...

以下是综合分析:
## 投资计划
{investment_plan}

## 市场分析
{market_report}

## 情绪分析
{sentiment_report}

## 新闻分析
{news_report}

## 基本面分析
{fundamentals_report}

请用中文撰写所有分析和建议。绝对禁止使用错误的公司名称或混淆不同的股票。"""

        logger.info("[交易员] 生成交易决策: %s", ticker)

        try:
            response = llm.invoke(prompt)
            result = response.content
        except Exception as e:
            logger.error("[交易员] LLM 调用失败: %s", e)
            result = f"交易决策生成失败: {e}。建议持有 {ticker}，等待进一步分析。"

        return {
            "trader_investment_plan": result,
            "sender": "Trader",
        }

    return trader_node
