"""
LLM 工厂 — 统一接口，一行代码切换提供商。

关键洞察: DeepSeek、通义千问、智谱 AI、SiliconFlow 等均兼容 OpenAI API，
因此只需 ChatOpenAI 一个类 + 不同 base_url 即可覆盖绝大多数国产 LLM。
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

# ─── 提供商默认配置 ───────────────────────────────────────────
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "env_key": "DASHSCOPE_API_KEY",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "env_key": "DASHSCOPE_API_KEY",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4",
        "env_key": "ZHIPU_API_KEY",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "deepseek-ai/DeepSeek-V2.5",
        "env_key": "SILICONFLOW_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-chat",
        "env_key": "OPENROUTER_API_KEY",
    },
}


def create_llm(
    provider: str = "deepseek",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4000,
    timeout: int = 180,
) -> BaseChatModel:
    """
    创建 LLM 实例 — 所有提供商统一使用 OpenAI 兼容接口。

    Args:
        provider: 提供商名称 (deepseek / openai / qwen / zhipu / siliconflow / openrouter / ...)
        model: 模型名称，不指定则使用提供商默认值
        base_url: API 地址，不指定则使用提供商默认值
        api_key: API 密钥，不指定则从环境变量读取
        temperature: 温度参数
        max_tokens: 最大 token 数
        timeout: 超时时间（秒）

    Returns:
        BaseChatModel 实例

    Example:
        >>> llm = create_llm("deepseek")
        >>> llm = create_llm("qwen", model="qwen-max")
        >>> llm = create_llm("openai", model="gpt-4o", api_key="sk-xxx")
    """
    defaults = PROVIDER_DEFAULTS.get(provider.lower(), {})

    resolved_model = model or defaults.get("model", provider)
    resolved_url = base_url or defaults.get("base_url")
    resolved_key = api_key or os.getenv(
        defaults.get("env_key", f"{provider.upper()}_API_KEY"), ""
    )

    if not resolved_key:
        env_var = defaults.get("env_key", f"{provider.upper()}_API_KEY")
        raise ValueError(
            f"未找到 {provider} 的 API Key。请设置环境变量 {env_var} 或传入 api_key 参数。"
        )

    logger.info("创建 LLM: provider=%s, model=%s, url=%s", provider, resolved_model, resolved_url)

    return ChatOpenAI(
        model=resolved_model,
        base_url=resolved_url,
        api_key=resolved_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
