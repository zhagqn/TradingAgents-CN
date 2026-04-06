#!/usr/bin/env python3
"""
TradingAgents-CN Core CLI — 精简命令行入口。

Usage:
    python cli_core.py 600519                         # 分析贵州茅台
    python cli_core.py AAPL --provider openai          # 用 OpenAI 分析苹果
    python cli_core.py 00700 --provider qwen           # 用通义千问分析腾讯
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import logging
from datetime import datetime


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TradingAgents-CN Core — 多智能体股票分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s 600519                          分析贵州茅台 (默认 DeepSeek)
  %(prog)s AAPL --provider openai          用 OpenAI 分析苹果
  %(prog)s 00700 --provider qwen           用通义千问分析腾讯
  %(prog)s 000001 --date 2025-01-15        指定分析日期
  %(prog)s 601318 --json                   输出 JSON 格式
        """,
    )
    parser.add_argument("symbol", help="股票代码（A股如600519、港股如00700、美股如AAPL）")
    parser.add_argument("--provider", default=None, help="LLM 提供商 (deepseek/openai/qwen/zhipu)")
    parser.add_argument("--model", default=None, help="模型名称")
    parser.add_argument("--date", default=None, help="分析日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")

    args = parser.parse_args()

    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # 延迟导入以加速 --help
    from tradingagents_core import TradingAgents
    from tradingagents_core.data.providers import detect_market, get_market_name, get_currency

    market = detect_market(args.symbol)
    market_name = get_market_name(market)
    currency_name, currency_symbol = get_currency(market)

    print(f"\n{'='*60}")
    print(f"📊 TradingAgents-CN Core 分析系统")
    print(f"{'='*60}")
    print(f"股票代码: {args.symbol} ({market_name})")
    print(f"货币: {currency_name} ({currency_symbol})")
    print(f"分析日期: {args.date or datetime.now().strftime('%Y-%m-%d')}")
    print(f"LLM: {args.provider or os.getenv('LLM_PROVIDER', 'deepseek')}")
    print(f"{'='*60}\n")

    kwargs = {}
    if args.provider:
        kwargs["llm_provider"] = args.provider
    if args.model:
        kwargs["llm_model"] = args.model

    try:
        agent = TradingAgents(**kwargs)
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        sys.exit(1)

    print("🔄 正在分析，请稍候...\n")

    try:
        result = agent.analyze(args.symbol, date=args.date)
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 格式化输出
    decision = result["decision"]
    reports = result["reports"]

    print(f"\n{'='*60}")
    print(f"📋 分析结果: {args.symbol}")
    print(f"{'='*60}\n")

    # 决策摘要
    action_emoji = {"买入": "🟢", "卖出": "🔴", "持有": "🟡"}.get(decision["action"], "⚪")
    print(f"{action_emoji} 投资建议: **{decision['action']}**")
    if decision.get("target_price"):
        print(f"🎯 目标价格: {currency_symbol}{decision['target_price']}")
    print(f"📊 置信度: {decision['confidence']:.0%}")
    print(f"⚠️  风险评分: {decision['risk_score']:.0%}")

    # 分析报告摘要
    print(f"\n{'─'*60}")
    print("📈 各维度分析摘要:")
    print(f"{'─'*60}")

    for name, key in [("技术分析", "market"), ("基本面", "fundamentals"), ("新闻", "news"), ("情绪", "sentiment")]:
        report = reports.get(key, "")
        if report:
            # 截取前 200 字符作为摘要
            summary = report[:200].replace("\n", " ").strip()
            if len(report) > 200:
                summary += "..."
            print(f"\n📌 {name}: {summary}")

    # 最终决策
    raw = result.get("raw_decision", "")
    if raw:
        print(f"\n{'─'*60}")
        print("🏦 风控委员会最终决策:")
        print(f"{'─'*60}")
        print(raw[:1000])
        if len(raw) > 1000:
            print(f"\n... (共 {len(raw)} 字符)")

    print(f"\n{'='*60}")
    print("✅ 分析完成")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
