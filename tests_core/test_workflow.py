"""工作流和智能体测试"""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from tradingagents_core.agents.states import AgentState, InvestDebateState, RiskDebateState
from tradingagents_core.agents.researchers import create_researcher
from tradingagents_core.agents.risk_debaters import create_risk_debater
from tradingagents_core.agents.managers import create_research_manager, create_risk_manager
from tradingagents_core.agents.trader import create_trader
from tradingagents_core.graph.workflow import _extract_decision, build_trading_graph


class TestDecisionExtraction:
    """测试决策信号提取"""

    def test_buy_signal(self):
        text = "最终交易建议: **买入**\n目标价格: ¥45.50\n置信度: 0.8\n风险评分: 0.3"
        result = _extract_decision(text)
        assert result["action"] == "买入"
        assert result["target_price"] == 45.5
        assert result["confidence"] == 0.8
        assert result["risk_score"] == 0.3

    def test_sell_signal(self):
        text = "建议卖出，目标价 $190.00"
        result = _extract_decision(text)
        assert result["action"] == "卖出"
        assert result["target_price"] == 190.0

    def test_hold_default(self):
        text = "目前市场不确定性较高，建议观望"
        result = _extract_decision(text)
        assert result["action"] == "持有"

    def test_price_with_yuan(self):
        text = "目标价位45.50元"
        result = _extract_decision(text)
        assert result["target_price"] == 45.5


class TestResearchers:
    """测试研究员"""

    def test_bull_researcher(self, mock_llm):
        researcher = create_researcher("bullish", mock_llm)
        state = _make_test_state()
        result = researcher(state)
        assert "investment_debate_state" in result
        assert result["investment_debate_state"]["count"] == 1

    def test_bear_researcher(self, mock_llm):
        researcher = create_researcher("bearish", mock_llm)
        state = _make_test_state()
        result = researcher(state)
        assert "investment_debate_state" in result


class TestRiskDebaters:
    """测试风控辩论者"""

    @pytest.mark.parametrize("style", ["aggressive", "conservative", "neutral"])
    def test_risk_debater(self, mock_llm, style):
        debater = create_risk_debater(style, mock_llm)
        state = _make_test_state()
        result = debater(state)
        assert "risk_debate_state" in result
        assert result["risk_debate_state"]["count"] == 1


class TestManagers:
    """测试管理者"""

    def test_research_manager(self, mock_llm):
        manager = create_research_manager(mock_llm)
        state = _make_test_state()
        result = manager(state)
        assert "investment_plan" in result
        assert "investment_debate_state" in result

    def test_risk_manager(self, mock_llm):
        manager = create_risk_manager(mock_llm)
        state = _make_test_state()
        result = manager(state)
        assert "final_trade_decision" in result
        assert "risk_debate_state" in result


class TestTrader:
    """测试交易员"""

    def test_trader(self, mock_llm):
        trader = create_trader(mock_llm)
        state = _make_test_state()
        result = trader(state)
        assert "trader_investment_plan" in result
        assert result["sender"] == "Trader"


class TestBuildGraph:
    """测试工作流构建"""

    def test_build_graph_default(self, mock_llm):
        graph = build_trading_graph(llm=mock_llm)
        assert graph is not None

    def test_build_graph_custom_analysts(self, mock_llm):
        graph = build_trading_graph(
            llm=mock_llm,
            analysts=["market", "fundamentals"],
        )
        assert graph is not None

    def test_build_graph_empty_analysts_raises(self, mock_llm):
        with pytest.raises(ValueError):
            build_trading_graph(llm=mock_llm, analysts=[])


# ─── 辅助函数 ────────────────────────────────────────────────

def _make_test_state() -> dict:
    """创建测试用状态。"""
    return {
        "messages": [],
        "company_of_interest": "600519",
        "trade_date": "2025-01-15",
        "sender": "",
        "market_report": "市场分析: 股价在均线上方，MACD 金叉",
        "sentiment_report": "情绪中性偏积极",
        "news_report": "公司发布年报，业绩超预期",
        "fundamentals_report": "PE 25倍，ROE 30%，估值合理",
        "market_tool_call_count": 0,
        "news_tool_call_count": 0,
        "sentiment_tool_call_count": 0,
        "fundamentals_tool_call_count": 0,
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        },
        "investment_plan": "初步投资计划: 建议关注",
        "trader_investment_plan": "",
        "risk_debate_state": {
            "history": "",
            "risky_history": "",
            "safe_history": "",
            "neutral_history": "",
            "latest_speaker": "",
            "current_risky_response": "",
            "current_safe_response": "",
            "current_neutral_response": "",
            "judge_decision": "",
            "count": 0,
        },
        "final_trade_decision": "",
    }
