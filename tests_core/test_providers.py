"""数据提供商和指标测试"""

import pandas as pd
import numpy as np
import pytest

from tradingagents_core.data.providers import detect_market, get_currency, get_market_name
from tradingagents_core.data.indicators import (
    ma, ema, macd, rsi, boll, kdj, atr, calculate_all_indicators,
)


class TestMarketDetection:
    """测试市场自动识别"""

    def test_china_a_shares(self):
        assert detect_market("600519") == "cn"
        assert detect_market("000001") == "cn"
        assert detect_market("300750") == "cn"
        assert detect_market("688001") == "cn"

    def test_china_with_prefix(self):
        assert detect_market("SH600519") == "cn"
        assert detect_market("SZ000001") == "cn"

    def test_hong_kong(self):
        assert detect_market("00700.HK") == "hk"
        assert detect_market("09988.HK") == "hk"
        assert detect_market("0700") == "hk"

    def test_us_stocks(self):
        assert detect_market("AAPL") == "us"
        assert detect_market("TSLA") == "us"
        assert detect_market("GOOG") == "us"

    def test_currency(self):
        assert get_currency("cn") == ("人民币", "¥")
        assert get_currency("hk") == ("港币", "HK$")
        assert get_currency("us") == ("美元", "$")

    def test_market_name(self):
        assert get_market_name("cn") == "中国A股"
        assert get_market_name("hk") == "港股"
        assert get_market_name("us") == "美股"


class TestIndicators:
    """测试技术指标计算"""

    @pytest.fixture
    def sample_data(self):
        """生成模拟 OHLCV 数据"""
        np.random.seed(42)
        n = 100
        close = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        high = close + abs(np.random.randn(n))
        low = close - abs(np.random.randn(n))
        volume = pd.Series(np.random.randint(1000, 10000, n))
        return pd.DataFrame({
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    def test_ma(self, sample_data):
        result = ma(sample_data["close"], 5)
        assert len(result) == len(sample_data)
        assert not result.isna().all()

    def test_ema(self, sample_data):
        result = ema(sample_data["close"], 12)
        assert len(result) == len(sample_data)

    def test_macd(self, sample_data):
        result = macd(sample_data["close"])
        assert "macd_dif" in result.columns
        assert "macd_dea" in result.columns
        assert "macd_hist" in result.columns

    def test_rsi(self, sample_data):
        result = rsi(sample_data["close"], 14)
        assert len(result) == len(sample_data)
        # RSI 在有效范围内
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_boll(self, sample_data):
        result = boll(sample_data["close"])
        assert "boll_upper" in result.columns
        assert "boll_mid" in result.columns
        assert "boll_lower" in result.columns

    def test_kdj(self, sample_data):
        result = kdj(sample_data["high"], sample_data["low"], sample_data["close"])
        assert "kdj_k" in result.columns
        assert "kdj_d" in result.columns
        assert "kdj_j" in result.columns

    def test_atr(self, sample_data):
        result = atr(sample_data["high"], sample_data["low"], sample_data["close"])
        assert len(result) == len(sample_data)
        assert (result.dropna() >= 0).all()

    def test_calculate_all(self, sample_data):
        result = calculate_all_indicators(sample_data)
        assert "ma5" in result.columns
        assert "macd_dif" in result.columns
        assert "rsi_14" in result.columns
        assert "boll_upper" in result.columns
