"""pytest 配置与公共 fixture"""

import pytest


@pytest.fixture
def mock_llm():
    """模拟 LLM 实例，用于不需要真实 API 的测试。"""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage

    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="模拟分析结果: 建议持有，目标价格 ¥50.00")
    llm.bind_tools.return_value = llm
    return llm
