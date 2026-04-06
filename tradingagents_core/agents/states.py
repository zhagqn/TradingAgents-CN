"""
LangGraph 状态定义 — 贯穿整个工作流的数据结构。
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


class InvestDebateState(TypedDict):
    """投资辩论状态"""
    history: Annotated[str, "辩论完整历史"]
    bull_history: Annotated[str, "看多方历史"]
    bear_history: Annotated[str, "看空方历史"]
    current_response: Annotated[str, "最新发言"]
    judge_decision: Annotated[str, "裁判决定"]
    count: Annotated[int, "当前发言轮次"]


class RiskDebateState(TypedDict):
    """风控辩论状态"""
    history: Annotated[str, "辩论完整历史"]
    risky_history: Annotated[str, "激进方历史"]
    safe_history: Annotated[str, "保守方历史"]
    neutral_history: Annotated[str, "中性方历史"]
    latest_speaker: Annotated[str, "最后发言者"]
    current_risky_response: Annotated[str, "激进方最新发言"]
    current_safe_response: Annotated[str, "保守方最新发言"]
    current_neutral_response: Annotated[str, "中性方最新发言"]
    judge_decision: Annotated[str, "裁判决定"]
    count: Annotated[int, "当前发言轮次"]


class AgentState(MessagesState):
    """主工作流状态 — 贯穿分析 → 辩论 → 决策全流程。"""

    # 股票信息
    company_of_interest: Annotated[str, "分析目标股票代码"]
    trade_date: Annotated[str, "交易日期"]

    # 消息路由
    sender: Annotated[str, "发送消息的智能体"]

    # 分析师报告
    market_report: Annotated[str, "技术分析报告"]
    sentiment_report: Annotated[str, "情绪分析报告"]
    news_report: Annotated[str, "新闻分析报告"]
    fundamentals_report: Annotated[str, "基本面分析报告"]

    # 工具调用计数（防止死循环）
    market_tool_call_count: Annotated[int, "技术分析工具调用次数"]
    news_tool_call_count: Annotated[int, "新闻分析工具调用次数"]
    sentiment_tool_call_count: Annotated[int, "情绪分析工具调用次数"]
    fundamentals_tool_call_count: Annotated[int, "基本面工具调用次数"]

    # 投资辩论
    investment_debate_state: Annotated[InvestDebateState, "投资辩论状态"]
    investment_plan: Annotated[str, "研究经理的投资计划"]

    # 交易决策
    trader_investment_plan: Annotated[str, "交易员的投资计划"]

    # 风控辩论
    risk_debate_state: Annotated[RiskDebateState, "风控辩论状态"]
    final_trade_decision: Annotated[str, "最终交易决策"]
