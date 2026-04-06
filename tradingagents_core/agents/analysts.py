"""
分析师工厂 — 4 个分析师统一为配置驱动的工厂模式。

每个分析师 = 系统提示 + 工具绑定 + 统一执行逻辑。
"""

from __future__ import annotations

import logging
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt import ToolNode

from tradingagents_core.agents.states import AgentState
from tradingagents_core.data.tools import (
    get_stock_market_data,
    get_stock_fundamentals,
    get_stock_news,
    get_stock_sentiment,
)

logger = logging.getLogger(__name__)

# ─── 分析师系统提示 ──────────────────────────────────────────

MARKET_ANALYST_PROMPT = """你是一位专业的股票技术分析师。根据提供的行情数据和技术指标，进行全面的技术分析。

分析要求:
1. 必须调用工具获取真实行情数据，禁止虚构数据
2. 基于 MACD、RSI、均线、布林带、KDJ 等指标判断趋势
3. 识别关键支撑位和阻力位
4. 给出技术面的投资建议（看多/看空/中性）

输出格式要求:
- 股票信息（代码、市场类型、货币）
- 技术指标分析
- 价格趋势判断
- 技术面投资建议

⚠️ 工作流程：先调用工具获取数据，再基于数据撰写报告。一次工具调用即可。"""

FUNDAMENTALS_ANALYST_PROMPT = """你是一位专业的股票基本面分析师。基于公司的财务数据进行价值分析。

分析要求:
1. 必须调用工具获取真实基本面数据，禁止虚构或假设
2. 分析 PE/PB/PEG 等估值指标
3. 评估 ROE、营收增长、利润率等运营指标
4. 判断当前估值是否合理，给出合理价格区间

输出格式要求:
- 公司基本信息
- 核心财务指标分析
- 估值水平判断（低估/合理/高估）
- 基本面投资建议

⚠️ 工作流程：先调用工具获取数据，再基于数据撰写报告。一次工具调用即可。"""

NEWS_ANALYST_PROMPT = """你是一位专业的金融新闻分析师。基于最新新闻评估其对股票的影响。

分析要求:
1. 必须调用工具获取真实新闻数据，禁止虚构
2. 评估新闻事件的紧迫性和市场影响力
3. 分析投资者情绪变化
4. 预判新闻对短期/中期价格的影响

输出格式要求:
- 重要新闻摘要
- 新闻情绪分析（正面/负面/中性）
- 对股价的预期影响
- 基于新闻的投资建议

⚠️ 工作流程：先调用工具获取数据，再基于数据撰写报告。一次工具调用即可。"""

SENTIMENT_ANALYST_PROMPT = """你是一位专业的市场情绪分析师。分析投资者情绪和市场氛围。

分析要求:
1. 必须调用工具获取真实情绪数据，禁止虚构
2. 量化情绪强度（1-10分）
3. 分析散户与机构投资者的观点差异
4. 识别情绪反转信号

输出格式要求:
- 市场情绪概况
- 情绪强度评分
- 预期价格波动范围
- 基于情绪的投资建议

⚠️ 工作流程：先调用工具获取数据，再基于数据撰写报告。一次工具调用即可。"""

# ─── 分析师配置映射 ──────────────────────────────────────────

ANALYST_CONFIGS = {
    "market": {
        "prompt": MARKET_ANALYST_PROMPT,
        "tools": [get_stock_market_data],
        "report_key": "market_report",
        "counter_key": "market_tool_call_count",
    },
    "fundamentals": {
        "prompt": FUNDAMENTALS_ANALYST_PROMPT,
        "tools": [get_stock_fundamentals],
        "report_key": "fundamentals_report",
        "counter_key": "fundamentals_tool_call_count",
    },
    "news": {
        "prompt": NEWS_ANALYST_PROMPT,
        "tools": [get_stock_news],
        "report_key": "news_report",
        "counter_key": "news_tool_call_count",
    },
    "sentiment": {
        "prompt": SENTIMENT_ANALYST_PROMPT,
        "tools": [get_stock_sentiment],
        "report_key": "sentiment_report",
        "counter_key": "sentiment_tool_call_count",
    },
}


# ─── 统一分析师工厂 ──────────────────────────────────────────

def create_analyst(
    analyst_type: str,
    llm: BaseChatModel,
) -> Callable[[AgentState], dict]:
    """创建分析师节点函数。

    Args:
        analyst_type: "market" | "fundamentals" | "news" | "sentiment"
        llm: LLM 实例

    Returns:
        LangGraph 节点函数
    """
    config = ANALYST_CONFIGS[analyst_type]
    system_prompt = config["prompt"]
    tools = config["tools"]
    report_key = config["report_key"]
    counter_key = config["counter_key"]

    MAX_TOOL_CALLS = 3
    max_tool_calls = MAX_TOOL_CALLS

    def analyst_node(state: AgentState) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        tool_call_count = state.get(counter_key, 0)

        logger.info("[%s分析师] 开始分析 %s", analyst_type, ticker)

        # 构建分析请求
        analysis_request = f"请分析股票 {ticker}，交易日期 {trade_date}。请调用工具获取真实数据后再进行分析。"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        chain = prompt | llm.bind_tools(tools)

        # 执行分析
        try:
            result = chain.invoke({"messages": [HumanMessage(content=analysis_request)]})
        except Exception as e:
            logger.error("[%s分析师] LLM 调用失败: %s", analyst_type, e)
            return {
                report_key: f"{analyst_type} 分析失败: {e}",
                counter_key: tool_call_count,
            }

        # 处理工具调用
        if hasattr(result, "tool_calls") and result.tool_calls and tool_call_count < max_tool_calls:
            tool_node = ToolNode(tools)
            tool_messages = tool_node.invoke({"messages": [result]})

            # 用工具结果再次调用 LLM 生成报告
            all_messages = [
                HumanMessage(content=analysis_request),
                result,
            ]
            # tool_messages 可能是 dict 或 list
            if isinstance(tool_messages, dict):
                all_messages.extend(tool_messages.get("messages", []))
            elif isinstance(tool_messages, list):
                all_messages.extend(tool_messages)

            try:
                # 不绑定工具，直接生成报告
                report_chain = prompt | llm
                final_result = report_chain.invoke({"messages": all_messages})
                report_content = final_result.content
            except Exception as e:
                logger.warning("[%s分析师] 报告生成失败，使用工具数据: %s", analyst_type, e)
                # 回退：直接使用工具返回的数据作为报告
                report_content = str(tool_messages)

            return {
                report_key: report_content,
                counter_key: tool_call_count + 1,
            }
        else:
            # LLM 直接生成了报告（未调用工具）
            return {
                report_key: result.content if hasattr(result, "content") else str(result),
                counter_key: tool_call_count,
            }

    return analyst_node


def create_msg_delete() -> Callable[[AgentState], dict]:
    """创建消息清理节点。"""
    def delete_messages(state: AgentState) -> dict:
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]
        placeholder = HumanMessage(content="Continue")
        return {"messages": removal_operations + [placeholder]}
    return delete_messages
