"""
TradingAgents-CN Core — 精简版多智能体股票分析系统

核心价值链: 获取数据 → 多角度分析 → 辩论/风控 → 输出决策

支持 A股、港股、美股，使用免费数据源 (AKShare + YFinance)。
"""

__version__ = "0.1.0"

from tradingagents_core.graph.workflow import TradingAgents, build_trading_graph
from tradingagents_core.llm.factory import create_llm
from tradingagents_core.config import CoreConfig

__all__ = [
    "TradingAgents",
    "build_trading_graph",
    "create_llm",
    "CoreConfig",
]
