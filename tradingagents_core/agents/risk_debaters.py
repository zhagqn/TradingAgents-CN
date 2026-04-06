"""
风控辩论者 — 3 种风格参数化为一个工厂函数。
"""

from __future__ import annotations

import logging
from typing import Callable, Literal

from langchain_core.language_models.chat_models import BaseChatModel

from tradingagents_core.agents.states import AgentState

logger = logging.getLogger(__name__)

# ─── 风控提示 ────────────────────────────────────────────────

RISK_PROMPTS = {
    "aggressive": """你是一位激进风险分析师，主张追求高回报的投资机会。

- 强调上行潜力、增长机会和创新收益
- 直接回应保守和中性分析师的担忧
- 指出保守策略中的错失机会
- 挑战对方逻辑中的薄弱环节
- 主张承担风险的好处
- 使用中文分析""",

    "conservative": """你是一位保守风险分析师，优先考虑资产保护和风险最小化。

- 优先考虑稳定性、安全性和风险缓解
- 批判性审查高风险因素
- 指出决策中可能导致过度风险的部分
- 展示审慎替代方案如何确保长期收益
- 直接反驳激进和中性分析师的论点
- 使用中文分析""",

    "neutral": """你是一位中性风险分析师，提供平衡的风险评估视角。

- 全面评估上行和下行风险
- 挑战激进和保守分析师的观点
- 指出激进观点过于乐观之处
- 指出保守观点过于谨慎之处
- 展示适度风险策略为何能兼顾两者优势
- 使用中文分析""",
}

HISTORY_KEYS = {
    "aggressive": "risky_history",
    "conservative": "safe_history",
    "neutral": "neutral_history",
}

RESPONSE_KEYS = {
    "aggressive": "current_risky_response",
    "conservative": "current_safe_response",
    "neutral": "current_neutral_response",
}

LABELS = {
    "aggressive": "激进分析师",
    "conservative": "保守分析师",
    "neutral": "中性分析师",
}


def create_risk_debater(
    style: Literal["aggressive", "conservative", "neutral"],
    llm: BaseChatModel,
) -> Callable[[AgentState], dict]:
    """创建风控辩论者节点。"""
    system_prompt = RISK_PROMPTS[style]
    label = LABELS[style]
    history_key = HISTORY_KEYS[style]
    response_key = RESPONSE_KEYS[style]

    def risk_node(state: AgentState) -> dict:
        risk_state = state["risk_debate_state"]
        history = risk_state.get("history", "")
        trader_plan = state.get("trader_investment_plan", "")

        # 各方最新观点
        other_responses = []
        for k, v in RESPONSE_KEYS.items():
            if k != style:
                resp = risk_state.get(v, "")
                if resp:
                    other_responses.append(f"{LABELS[k]}: {resp}")

        prompt = f"""{system_prompt}

## 交易员投资计划
{trader_plan}

## 分析报告
市场分析: {state.get('market_report', '暂无')}
新闻分析: {state.get('news_report', '暂无')}
基本面分析: {state.get('fundamentals_report', '暂无')}
情绪分析: {state.get('sentiment_report', '暂无')}

## 辩论历史
{history}

## 其他分析师观点
{chr(10).join(other_responses) if other_responses else '暂无'}

请从{label}的角度给出你的风险评估和建议。"""

        logger.info("[%s] 风险评估中...", label)

        try:
            response = llm.invoke(prompt)
            response_text = response.content
        except Exception as e:
            logger.error("[%s] LLM 调用失败: %s", label, e)
            response_text = f"{label}评估暂不可用"

        formatted = f"{label}: {response_text}"

        new_risk_state = {
            **risk_state,
            "history": history + "\n\n" + formatted,
            history_key: risk_state.get(history_key, "") + "\n\n" + formatted,
            response_key: response_text,
            "latest_speaker": label,
            "count": risk_state.get("count", 0) + 1,
        }

        return {"risk_debate_state": new_risk_state}

    return risk_node
