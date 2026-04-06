"""
研究经理 + 风控经理 — 辩论裁判角色。
"""

from __future__ import annotations

import logging
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from tradingagents_core.agents.states import AgentState

logger = logging.getLogger(__name__)


# ─── 研究经理（投资辩论裁判）────────────────────────────────

def create_research_manager(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    """创建研究经理节点 — 评估看多/看空辩论并做出投资决策。"""

    def research_manager_node(state: AgentState) -> dict:
        debate_state = state["investment_debate_state"]
        history = debate_state.get("history", "")

        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        prompt = f"""作为投资组合经理和辩论主持人，您的职责是批判性地评估这轮辩论并做出明确决策：
支持看跌分析师、看涨分析师，或者在有强有力理由时选择持有。

总结双方关键观点，重点关注最有说服力的证据。
您的建议——买入、卖出或持有——必须明确且可操作。
避免仅因双方都有道理就默认选择持有；要基于最强论点做出决定。

请为交易员制定详细投资计划，包括:
1. 您的建议：基于最有说服力论点的明确立场
2. 理由：为什么这些论点导致您的结论
3. 战略行动：实施建议的具体步骤
4. 目标价格分析：基于基本面、新闻、情绪提供具体目标价格
5. 风险调整价格情景（保守、基准、乐观）

💰 您必须提供具体的目标价格，不要回复"无法确定"。

以下是综合分析报告:
市场研究：{market_report}
情绪分析：{sentiment_report}
新闻分析：{news_report}
基本面分析：{fundamentals_report}

以下是辩论历史：
{history}

请用中文撰写所有分析内容和建议。"""

        logger.info("[研究经理] 评估辩论中...")

        try:
            response = llm.invoke(prompt)
            result = response.content
        except Exception as e:
            logger.error("[研究经理] LLM 调用失败: %s", e)
            result = "基于当前信息，建议持有，等待更明确的市场信号。"

        new_debate_state = {
            **debate_state,
            "judge_decision": result,
            "current_response": result,
        }

        return {
            "investment_debate_state": new_debate_state,
            "investment_plan": result,
        }

    return research_manager_node


# ─── 风控经理（风控辩论裁判）────────────────────────────────

def create_risk_manager(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    """创建风控经理节点 — 评估风控辩论并做出最终交易决策。"""

    def risk_manager_node(state: AgentState) -> dict:
        company = state["company_of_interest"]
        risk_state = state["risk_debate_state"]
        history = risk_state.get("history", "")
        trader_plan = state.get("investment_plan", "")

        prompt = f"""作为风险管理委员会主席和辩论主持人，评估激进、中性和保守三位风险分析师的辩论，
确定最佳行动方案。决策必须产生明确建议：买入、卖出或持有。

决策指导原则:
1. 总结每位分析师的最强观点
2. 用辩论中的直接引用支持您的建议
3. 从交易员原始计划开始调整: {trader_plan}
4. 力求清晰果断，避免模糊

交付成果:
- 明确可操作的建议：买入、卖出或持有
- 基于辩论的详细推理
- 具体目标价格

分析师辩论历史:
{history}

请用中文撰写所有分析内容和建议。"""

        logger.info("[风控经理] 做出最终决策...")

        max_retries = 2
        result = ""

        for attempt in range(max_retries + 1):
            try:
                response = llm.invoke(prompt)
                if response and hasattr(response, "content") and len(response.content.strip()) > 10:
                    result = response.content.strip()
                    break
            except Exception as e:
                logger.warning("[风控经理] 第 %d 次尝试失败: %s", attempt + 1, e)

        if not result:
            result = f"""**默认建议：持有**

由于技术原因无法生成详细分析，基于当前市场状况和风险控制原则，建议对{company}采取持有策略。

建议:
- 密切关注市场动态和公司基本面变化
- 设置合理的止损和止盈位
- 等待更好的入场或出场时机"""

        new_risk_state = {
            **risk_state,
            "judge_decision": result,
            "latest_speaker": "Judge",
        }

        return {
            "risk_debate_state": new_risk_state,
            "final_trade_decision": result,
        }

    return risk_manager_node
