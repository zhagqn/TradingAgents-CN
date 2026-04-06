"""
LangGraph 工作流 — 核心编排引擎。

工作流:
  分析师 (串行) → 看多/看空辩论 → 研究经理裁决 → 交易员 → 风控辩论 → 风控经理最终决策

对外只暴露一个接口: TradingAgents.analyze(symbol)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from tradingagents_core.agents.analysts import create_analyst
from tradingagents_core.agents.managers import create_research_manager, create_risk_manager
from tradingagents_core.agents.researchers import create_researcher
from tradingagents_core.agents.risk_debaters import create_risk_debater
from tradingagents_core.agents.states import AgentState, InvestDebateState, RiskDebateState
from tradingagents_core.agents.trader import create_trader
from tradingagents_core.config import CoreConfig
from tradingagents_core.data.providers import UnifiedDataService
from tradingagents_core.llm.factory import create_llm

logger = logging.getLogger(__name__)


# ─── 条件逻辑 ────────────────────────────────────────────────

def _make_debate_router(max_rounds: int):
    """创建投资辩论路由函数。"""
    max_count = 2 * max_rounds

    def router(state: AgentState) -> str:
        count = state["investment_debate_state"].get("count", 0)
        current = state["investment_debate_state"].get("current_response", "")
        if count >= max_count:
            return "Research Manager"
        return "Bear Researcher" if current.startswith("看多") else "Bull Researcher"

    return router


def _make_risk_router(max_rounds: int):
    """创建风控辩论路由函数。"""
    max_count = 3 * max_rounds

    def router(state: AgentState) -> str:
        count = state["risk_debate_state"].get("count", 0)
        speaker = state["risk_debate_state"].get("latest_speaker", "")
        if count >= max_count:
            return "Risk Judge"
        if speaker.startswith("激进"):
            return "Safe Analyst"
        elif speaker.startswith("保守"):
            return "Neutral Analyst"
        else:
            return "Risky Analyst"

    return router


# ─── 构建工作流 ──────────────────────────────────────────────

def build_trading_graph(
    llm: BaseChatModel,
    deep_llm: BaseChatModel | None = None,
    analysts: list[str] | None = None,
    max_debate_rounds: int = 1,
    max_risk_rounds: int = 1,
    max_recur_limit: int = 100,
) -> Any:
    """构建交易分析 LangGraph 工作流。

    Args:
        llm: 快速思考 LLM（用于分析师、研究员、交易员）
        deep_llm: 深度思考 LLM（用于研究经理、风控经理），默认与 llm 相同
        analysts: 选择的分析师列表，默认全部
        max_debate_rounds: 投资辩论轮次
        max_risk_rounds: 风控辩论轮次
        max_recur_limit: LangGraph 递归上限

    Returns:
        编译后的 LangGraph
    """
    if analysts is None:
        analysts = ["market", "fundamentals", "news", "sentiment"]
    if deep_llm is None:
        deep_llm = llm
    if not analysts:
        raise ValueError("至少需要选择一个分析师")

    workflow = StateGraph(AgentState)

    # 1. 添加分析师节点（串行执行）
    for name in analysts:
        workflow.add_node(f"{name.capitalize()} Analyst", create_analyst(name, llm))

    # 2. 添加研究员节点
    workflow.add_node("Bull Researcher", create_researcher("bullish", llm))
    workflow.add_node("Bear Researcher", create_researcher("bearish", llm))
    workflow.add_node("Research Manager", create_research_manager(deep_llm))

    # 3. 添加交易员
    workflow.add_node("Trader", create_trader(llm))

    # 4. 添加风控辩论节点
    workflow.add_node("Risky Analyst", create_risk_debater("aggressive", llm))
    workflow.add_node("Safe Analyst", create_risk_debater("conservative", llm))
    workflow.add_node("Neutral Analyst", create_risk_debater("neutral", llm))
    workflow.add_node("Risk Judge", create_risk_manager(deep_llm))

    # 5. 定义边
    # START → 第一个分析师
    workflow.add_edge(START, f"{analysts[0].capitalize()} Analyst")

    # 分析师串行连接
    for i in range(len(analysts) - 1):
        workflow.add_edge(
            f"{analysts[i].capitalize()} Analyst",
            f"{analysts[i + 1].capitalize()} Analyst",
        )
    # 最后一个分析师 → Bull Researcher
    workflow.add_edge(f"{analysts[-1].capitalize()} Analyst", "Bull Researcher")

    # 辩论循环
    debate_router = _make_debate_router(max_debate_rounds)
    workflow.add_conditional_edges(
        "Bull Researcher",
        debate_router,
        {"Bear Researcher": "Bear Researcher", "Research Manager": "Research Manager"},
    )
    workflow.add_conditional_edges(
        "Bear Researcher",
        debate_router,
        {"Bull Researcher": "Bull Researcher", "Research Manager": "Research Manager"},
    )

    # 研究经理 → 交易员
    workflow.add_edge("Research Manager", "Trader")

    # 交易员 → 风控辩论
    workflow.add_edge("Trader", "Risky Analyst")

    risk_router = _make_risk_router(max_risk_rounds)
    workflow.add_conditional_edges(
        "Risky Analyst",
        risk_router,
        {"Safe Analyst": "Safe Analyst", "Risk Judge": "Risk Judge"},
    )
    workflow.add_conditional_edges(
        "Safe Analyst",
        risk_router,
        {"Neutral Analyst": "Neutral Analyst", "Risk Judge": "Risk Judge"},
    )
    workflow.add_conditional_edges(
        "Neutral Analyst",
        risk_router,
        {"Risky Analyst": "Risky Analyst", "Risk Judge": "Risk Judge"},
    )

    # 风控经理 → 结束
    workflow.add_edge("Risk Judge", END)

    return workflow.compile()


# ─── 信号处理 ────────────────────────────────────────────────

def _extract_decision(text: str) -> dict:
    """从最终交易决策文本中提取结构化信号。"""
    # 提取动作
    action = "持有"
    if re.search(r"买入|BUY", text, re.IGNORECASE):
        action = "买入"
    elif re.search(r"卖出|SELL", text, re.IGNORECASE):
        action = "卖出"

    # 提取目标价格
    target_price = None
    price_patterns = [
        r"目标价[位格]?[：:]?\s*[¥\$HK\$]?(\d+(?:\.\d+)?)",
        r"[¥\$](\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*元",
    ]
    for pattern in price_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                target_price = float(match.group(1))
                break
            except ValueError:
                continue

    # 提取置信度
    confidence = 0.7
    conf_match = re.search(r"置信度[：:]?\s*([\d.]+)", text)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
        except ValueError:
            pass

    # 提取风险评分
    risk_score = 0.5
    risk_match = re.search(r"风险评分[：:]?\s*([\d.]+)", text)
    if risk_match:
        try:
            risk_score = float(risk_match.group(1))
        except ValueError:
            pass

    return {
        "action": action,
        "target_price": target_price,
        "confidence": confidence,
        "risk_score": risk_score,
    }


# ─── 对外 API ────────────────────────────────────────────────

class TradingAgents:
    """多智能体股票分析系统 — 核心 API。

    Example:
        >>> from tradingagents_core import TradingAgents
        >>> agent = TradingAgents(llm_provider="deepseek")
        >>> result = agent.analyze("600519")
        >>> print(result["decision"]["action"])
        '买入'
    """

    def __init__(
        self,
        config: CoreConfig | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        **llm_kwargs: Any,
    ) -> None:
        if config is None:
            config = CoreConfig.from_env()

        # 允许通过构造参数覆盖
        if llm_provider:
            config.llm_provider = llm_provider
        if llm_model:
            config.llm_model = llm_model

        self.config = config

        # 创建 LLM
        self.llm = create_llm(
            provider=config.llm_provider,
            model=config.llm_model,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            timeout=config.llm_timeout,
        )

        # 深度思考 LLM
        if config.deep_llm_provider:
            self.deep_llm = create_llm(
                provider=config.deep_llm_provider,
                model=config.deep_llm_model,
                base_url=config.deep_llm_base_url,
                api_key=config.deep_llm_api_key,
                temperature=config.llm_temperature,
            )
        else:
            self.deep_llm = self.llm

        # 构建工作流
        self.graph = build_trading_graph(
            llm=self.llm,
            deep_llm=self.deep_llm,
            analysts=config.selected_analysts,
            max_debate_rounds=config.max_debate_rounds,
            max_risk_rounds=config.max_risk_discuss_rounds,
            max_recur_limit=config.max_recur_limit,
        )

    def analyze(self, symbol: str, date: str | None = None) -> dict:
        """分析股票并返回结构化结果。

        Args:
            symbol: 股票代码（如 600519、00700、AAPL）
            date: 分析日期（YYYY-MM-DD），默认今天

        Returns:
            dict: {
                "symbol": str,
                "date": str,
                "decision": {"action", "target_price", "confidence", "risk_score"},
                "reports": {"market", "fundamentals", "news", "sentiment"},
                "debate": {"bull_case", "bear_case", "investment_plan"},
                "risk": {"final_decision"},
                "raw_decision": str,
            }
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info("开始分析: symbol=%s, date=%s", symbol, date)

        initial_state = {
            "messages": [HumanMessage(content=f"请对股票 {symbol} 进行全面分析，交易日期为 {date}。")],
            "company_of_interest": symbol,
            "trade_date": date,
            "investment_debate_state": InvestDebateState(
                history="", bull_history="", bear_history="",
                current_response="", judge_decision="", count=0,
            ),
            "risk_debate_state": RiskDebateState(
                history="", risky_history="", safe_history="",
                neutral_history="", latest_speaker="",
                current_risky_response="", current_safe_response="",
                current_neutral_response="", judge_decision="", count=0,
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
            "market_tool_call_count": 0,
            "news_tool_call_count": 0,
            "sentiment_tool_call_count": 0,
            "fundamentals_tool_call_count": 0,
        }

        # 执行工作流
        result = self.graph.invoke(
            initial_state,
            config={"recursion_limit": self.config.max_recur_limit},
        )

        # 提取结构化决策
        raw_decision = result.get("final_trade_decision", "")
        decision = _extract_decision(raw_decision)

        return {
            "symbol": symbol,
            "date": date,
            "decision": decision,
            "reports": {
                "market": result.get("market_report", ""),
                "fundamentals": result.get("fundamentals_report", ""),
                "news": result.get("news_report", ""),
                "sentiment": result.get("sentiment_report", ""),
            },
            "debate": {
                "bull_case": result.get("investment_debate_state", {}).get("bull_history", ""),
                "bear_case": result.get("investment_debate_state", {}).get("bear_history", ""),
                "investment_plan": result.get("investment_plan", ""),
            },
            "risk": {
                "final_decision": raw_decision,
            },
            "raw_decision": raw_decision,
        }
