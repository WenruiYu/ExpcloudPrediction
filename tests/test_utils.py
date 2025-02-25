"""Tests for the utility functions and IndicatorCalculator class."""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.utils import IndicatorCalculator, calculate_technical_indicators
from src.core.config import TECHNICAL_INDICATORS_CONFIG


class TestIndicatorCalculator:
    """Test cases for the IndicatorCalculator class."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=30)
        data = {
            'open': np.random.uniform(1800, 1900, 30),
            'high': np.random.uniform(1850, 1950, 30),
            'low': np.random.uniform(1750, 1850, 30),
            'close': np.random.uniform(1800, 1900, 30),
            'volume': np.random.randint(5000, 15000, 30),
            'amount': np.random.randint(9000000, 27000000, 30)
        }
        
        # Ensure high is always greater than low
        for i in range(30):
            data['high'][i] = max(data['high'][i], data['low'][i] + 50)
        
        return pd.DataFrame(data, index=dates)
    
    def test_initialization(self):
        """Test initialization with default and custom config."""
        # Test with default config
        calculator = IndicatorCalculator()
        assert calculator.config == TECHNICAL_INDICATORS_CONFIG
        
        # Test with custom config
        custom_config = {'sma_period': 15, 'ema_period': 10}
        calculator = IndicatorCalculator(custom_config)
        assert calculator.config == custom_config
    
    def test_validate_dataframe(self, sample_price_data, caplog):
        """Test DataFrame validation."""
        calculator = IndicatorCalculator()
        
        # Test valid DataFrame
        result = calculator.validate_dataframe(sample_price_data)
        assert len(result) == len(sample_price_data)
        assert 'close' in result.columns
        
        # Test with missing required column
        df_no_close = sample_price_data.drop('close', axis=1)
        with pytest.raises(ValueError, match="Required column 'close' not found"):
            calculator.validate_dataframe(df_no_close)
        
        # Test with NaN values in close
        df_with_nan = sample_price_data.copy()
        df_with_nan.loc[df_with_nan.index[0:3], 'close'] = np.nan
        
        # Use caplog to check the log message instead of pytest.warns
        with caplog.at_level(logging.WARNING):
            result = calculator.validate_dataframe(df_with_nan)
            assert len(result) == len(sample_price_data) - 3
            # Verify the warning was logged
            assert "Dropped" in caplog.text and "NaN close prices" in caplog.text
    
    def test_calculate_sma(self, sample_price_data):
        """Test SMA calculation."""
        calculator = IndicatorCalculator({'sma_period': 5})
        
        # Calculate SMA
        sma = calculator.calculate_sma(sample_price_data)
        
        # Verify result
        assert len(sma) == len(sample_price_data)
        assert isinstance(sma, pd.Series)
        
        # Manual calculation to verify
        expected = sample_price_data['close'].rolling(window=5, min_periods=1).mean()
        pd.testing.assert_series_equal(sma, expected)
    
    def test_calculate_ema(self, sample_price_data):
        """Test EMA calculation."""
        calculator = IndicatorCalculator({'ema_period': 5})
        
        # Calculate EMA
        ema = calculator.calculate_ema(sample_price_data)
        
        # Verify result
        assert len(ema) == len(sample_price_data)
        assert isinstance(ema, pd.Series)
        
        # Manual calculation to verify
        expected = sample_price_data['close'].ewm(span=5, adjust=False, min_periods=1).mean()
        pd.testing.assert_series_equal(ema, expected)
    
    def test_calculate_rsi(self, sample_price_data):
        """Test RSI calculation."""
        calculator = IndicatorCalculator({'rsi_period': 14})
        
        # Calculate RSI
        rsi = calculator.calculate_rsi(sample_price_data)
        
        # Verify result
        assert len(rsi) == len(sample_price_data)
        assert isinstance(rsi, pd.Series)
        assert all(0 <= x <= 100 for x in rsi.dropna())
    
    def test_calculate_macd(self, sample_price_data):
        """Test MACD calculation."""
        calculator = IndicatorCalculator({
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        })
        
        # Calculate MACD
        macd, signal, hist = calculator.calculate_macd(sample_price_data)
        
        # Verify result
        assert len(macd) == len(sample_price_data)
        assert len(signal) == len(sample_price_data)
        assert len(hist) == len(sample_price_data)
        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)
    
    def test_calculate_bollinger_bands(self, sample_price_data):
        """Test Bollinger Bands calculation."""
        calculator = IndicatorCalculator()
        
        # Calculate Bollinger Bands
        upper, middle, lower = calculator.calculate_bollinger_bands(sample_price_data)
        
        # Verify result
        assert len(upper) == len(sample_price_data)
        assert len(middle) == len(sample_price_data)
        assert len(lower) == len(sample_price_data)
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        
        # All upper values should be greater than middle values
        assert all(upper.dropna() >= middle.dropna())
        
        # All lower values should be less than middle values
        assert all(lower.dropna() <= middle.dropna())
    
    def test_calculate_atr(self, sample_price_data):
        """Test ATR calculation."""
        calculator = IndicatorCalculator()
        
        # Calculate ATR
        atr = calculator.calculate_atr(sample_price_data)
        
        # Verify result
        assert len(atr) == len(sample_price_data)
        assert isinstance(atr, pd.Series)
        assert all(x >= 0 for x in atr.dropna())  # ATR values should be positive
    
    def test_calculate_stochastic(self, sample_price_data):
        """Test Stochastic Oscillator calculation."""
        calculator = IndicatorCalculator()
        
        # Calculate Stochastic
        k, d = calculator.calculate_stochastic(sample_price_data)
        
        # Verify result
        assert len(k) == len(sample_price_data)
        assert len(d) == len(sample_price_data)
        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)
        
        # Values should be between 0 and 100
        assert all(0 <= x <= 100 for x in k.dropna())
        assert all(0 <= x <= 100 for x in d.dropna())
    
    def test_get_indicator_function(self, sample_price_data):
        """Test getting indicator function by name."""
        calculator = IndicatorCalculator()
        
        # Test valid indicator names
        assert calculator.get_indicator_function('sma') == calculator.calculate_sma
        assert calculator.get_indicator_function('ema') == calculator.calculate_ema
        assert calculator.get_indicator_function('rsi') == calculator.calculate_rsi
        assert calculator.get_indicator_function('macd') == calculator.calculate_macd
        
        # Test case insensitivity
        assert calculator.get_indicator_function('SMA') == calculator.calculate_sma
        
        # Test invalid indicator name
        invalid_func = calculator.get_indicator_function('invalid_name')
        assert callable(invalid_func)
        
        # The returned function for invalid name should return a Series
        result = invalid_func(sample_price_data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)


class TestCalculateTechnicalIndicators:
    """Test cases for the calculate_technical_indicators function."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=30)
        data = {
            'open': np.random.uniform(1800, 1900, 30),
            'high': np.random.uniform(1850, 1950, 30),
            'low': np.random.uniform(1750, 1850, 30),
            'close': np.random.uniform(1800, 1900, 30),
            'volume': np.random.randint(5000, 15000, 30),
            'amount': np.random.randint(9000000, 27000000, 30)
        }
        return pd.DataFrame(data, index=dates)
    
    def test_calculate_technical_indicators_default(self, sample_price_data):
        """Test calculating technical indicators with default settings."""
        result = calculate_technical_indicators(sample_price_data)
        
        # Verify result structure
        assert len(result) == len(sample_price_data)
        
        # Default indicators should be added
        assert 'SMA_20' in result.columns
        assert 'EMA_20' in result.columns
        assert 'RSI_14' in result.columns
        assert 'MACD' in result.columns
        assert 'MACD_Signal' in result.columns
        assert 'MACD_Hist' in result.columns
    
    def test_calculate_technical_indicators_specific(self, sample_price_data):
        """Test calculating specific technical indicators."""
        result = calculate_technical_indicators(
            sample_price_data,
            indicators=['sma', 'bollinger']
        )
        
        # Verify result
        assert 'SMA_20' in result.columns
        assert 'BB_Upper' in result.columns
        assert 'BB_Middle' in result.columns
        assert 'BB_Lower' in result.columns
        
        # Indicators not specified should not be included
        assert 'RSI_14' not in result.columns
        assert 'MACD' not in result.columns
    
    def test_calculate_technical_indicators_custom_config(self, sample_price_data):
        """Test calculating indicators with custom configuration."""
        custom_config = {
            'sma_period': 5,
            'rsi_period': 7
        }
        
        result = calculate_technical_indicators(
            sample_price_data,
            indicators=['sma', 'rsi'],
            config=custom_config
        )
        
        # Verify result uses custom periods
        assert 'SMA_5' in result.columns
        assert 'RSI_7' in result.columns
    
    def test_calculate_technical_indicators_error_handling(self):
        """Test error handling in calculate_technical_indicators."""
        # Test with None input
        result = calculate_technical_indicators(None)
        assert result.empty
        
        # Test with empty DataFrame
        result = calculate_technical_indicators(pd.DataFrame())
        assert result.empty
    
    def test_calculate_technical_indicators_handles_exceptions(self, sample_price_data):
        """Test that the function handles exceptions properly."""
        with patch('src.core.utils.IndicatorCalculator.calculate_sma', side_effect=Exception('Test error')):
            # Should not raise exception
            result = calculate_technical_indicators(sample_price_data, indicators=['sma'])
            assert 'SMA_20' not in result.columns


if __name__ == '__main__':
    pytest.main(['-xvs', __file__]) 