"""
看多/看空研究员 — 参数化工厂，消除 90% 重复代码。
"""

from __future__ import annotations

import logging
from typing import Callable, Literal

from langchain_core.language_models.chat_models import BaseChatModel

from tradingagents_core.agents.states import AgentState

logger = logging.getLogger(__name__)

# ─── 研究员提示 ──────────────────────────────────────────────

BULL_PROMPT = """你是一位看多分析师，负责为投资该股票构建有力的买入论据。

要求:
- 强调增长潜力、竞争优势和正面市场指标
- 用具体数据和推理反驳看空论点
- 以对话形式呈现论据，不要只列数据
- 使用公司名称而非股票代码
- 所有分析必须使用中文"""

BEAR_PROMPT = """你是一位看空分析师，负责论证不应投资该股票的理由。

要求:
- 强调风险、挑战和负面指标
- 指出市场饱和、财务不稳定、竞争劣势等问题
- 用具体数据和推理反驳看多论点
- 以对话形式呈现论据
- 使用公司名称而非股票代码
- 所有分析必须使用中文"""


def create_researcher(
    stance: Literal["bullish", "bearish"],
    llm: BaseChatModel,
) -> Callable[[AgentState], dict]:
    """创建看多或看空研究员节点。

    Args:
        stance: "bullish" 或 "bearish"
        llm: LLM 实例
    """
    system_prompt = BULL_PROMPT if stance == "bullish" else BEAR_PROMPT
    label = "看多" if stance == "bullish" else "看空"
    history_key = "bull_history" if stance == "bullish" else "bear_history"

    def researcher_node(state: AgentState) -> dict:
        ticker = state["company_of_interest"]
        debate_state = state["investment_debate_state"]
        history = debate_state.get("history", "")
        current_response = debate_state.get("current_response", "")

        # 收集分析师报告
        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        prompt = f"""{system_prompt}

你正在分析的股票是: {ticker}

以下是各分析师提供的研究数据:

## 市场分析
{market_report}

## 情绪分析
{sentiment_report}

## 新闻分析
{news_report}

## 基本面分析
{fundamentals_report}

## 辩论历史
{history}

对方最新观点:
{current_response}

请基于以上真实数据，从{label}的角度给出你的分析论据。"""

        logger.info("[%s研究员] 分析 %s", label, ticker)

        try:
            response = llm.invoke(prompt)
            response_text = response.content
        except Exception as e:
            logger.error("[%s研究员] LLM 调用失败: %s", label, e)
            response_text = f"{label}分析暂不可用: {e}"

        formatted_response = f"{label}分析师: {response_text}"

        new_debate_state = {
            **debate_state,
            "history": history + "\n\n" + formatted_response,
            history_key: debate_state.get(history_key, "") + "\n\n" + formatted_response,
            "current_response": formatted_response,
            "count": debate_state.get("count", 0) + 1,
        }

        return {"investment_debate_state": new_debate_state}

    return researcher_node
