"""统一配置模块 — 零基础设施依赖"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CoreConfig:
    """核心配置 — 所有参数都有合理默认值，可零配置运行。"""

    # ── LLM 设置 ──────────────────────────────────────────────
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4000
    llm_timeout: int = 180

    # ── 深度思考 LLM（用于 judge / risk manager）────────────
    deep_llm_provider: Optional[str] = None  # None = 与 llm_provider 相同
    deep_llm_model: Optional[str] = None
    deep_llm_base_url: Optional[str] = None
    deep_llm_api_key: Optional[str] = None

    # ── 分析师选择 ─────────────────────────────────────────
    selected_analysts: list[str] = field(
        default_factory=lambda: ["market", "fundamentals", "news", "sentiment"]
    )

    # ── 辩论与风控 ─────────────────────────────────────────
    max_debate_rounds: int = 1
    max_risk_discuss_rounds: int = 1
    max_recur_limit: int = 100

    # ── 数据缓存 ──────────────────────────────────────────
    cache_ttl: int = 3600  # 秒

    @classmethod
    def from_env(cls) -> CoreConfig:
        """从环境变量构建配置（优先级：环境变量 > 默认值）"""
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
            llm_model=os.getenv("LLM_MODEL", "deepseek-chat"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4000")),
            llm_timeout=int(os.getenv("LLM_TIMEOUT", "180")),
            deep_llm_provider=os.getenv("DEEP_LLM_PROVIDER"),
            deep_llm_model=os.getenv("DEEP_LLM_MODEL"),
            deep_llm_base_url=os.getenv("DEEP_LLM_BASE_URL"),
            deep_llm_api_key=os.getenv("DEEP_LLM_API_KEY"),
            max_debate_rounds=int(os.getenv("MAX_DEBATE_ROUNDS", "1")),
            max_risk_discuss_rounds=int(os.getenv("MAX_RISK_DISCUSS_ROUNDS", "1")),
        )
