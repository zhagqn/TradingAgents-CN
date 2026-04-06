"""配置模块测试"""

from tradingagents_core.config import CoreConfig


class TestCoreConfig:
    def test_defaults(self):
        config = CoreConfig()
        assert config.llm_provider == "deepseek"
        assert config.llm_model == "deepseek-chat"
        assert config.max_debate_rounds == 1
        assert config.max_risk_discuss_rounds == 1
        assert len(config.selected_analysts) == 4

    def test_custom_config(self):
        config = CoreConfig(
            llm_provider="qwen",
            llm_model="qwen-max",
            selected_analysts=["market", "fundamentals"],
            max_debate_rounds=2,
        )
        assert config.llm_provider == "qwen"
        assert config.llm_model == "qwen-max"
        assert len(config.selected_analysts) == 2
        assert config.max_debate_rounds == 2

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "zhipu")
        monkeypatch.setenv("LLM_MODEL", "glm-4")
        config = CoreConfig.from_env()
        assert config.llm_provider == "zhipu"
        assert config.llm_model == "glm-4"
