# src/utils.py

from typing import Optional, List, Dict, Tuple, Any, Union, Callable
import pandas as pd
import numpy as np
import ta
import logging
import logging.config
from functools import lru_cache

# Try to import using relative imports if running as a module
try:
    from src.config import TECHNICAL_INDICATORS_CONFIG, LOGGING_CONFIG
except ImportError:
    # Fall back to direct imports if running the file directly
    from config import TECHNICAL_INDICATORS_CONFIG, LOGGING_CONFIG

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class IndicatorCalculator:
    """Class for calculating various technical indicators."""
    
    def __init__(self, config: Dict[str, int] = TECHNICAL_INDICATORS_CONFIG):
        """
        Initialize the indicator calculator with configuration.
        
        Args:
            config: Dictionary containing parameters for technical indicators
        """
        self.config = config
        logger.info("Initialized IndicatorCalculator")
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare DataFrame for indicator calculation.
        
        Args:
            df: DataFrame containing price data
            
        Returns:
            Validated DataFrame
        """
        if df is None or df.empty:
            raise ValueError("Invalid or empty DataFrame")
            
        df = df.copy()
        
        # Ensure required columns exist
        required_columns = ['close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        # Convert price columns to numeric
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN in close prices
        if df['close'].isna().any():
            original_len = len(df)
            df = df.dropna(subset=['close'])
            dropped_rows = original_len - len(df)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows with NaN close prices")
                
        return df
    
    def calculate_sma(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            df: DataFrame containing price data
            period: Period for SMA calculation
            
        Returns:
            Series containing SMA values
        """
        period = period or self.config['sma_period']
        try:
            return df['close'].rolling(window=period, min_periods=1).mean()
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}", exc_info=True)
            return pd.Series(index=df.index)
    
    def calculate_ema(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            df: DataFrame containing price data
            period: Period for EMA calculation
            
        Returns:
            Series containing EMA values
        """
        period = period or self.config['ema_period']
        try:
            return df['close'].ewm(span=period, adjust=False, min_periods=1).mean()
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}", exc_info=True)
            return pd.Series(index=df.index)
    
    def calculate_rsi(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            df: DataFrame containing price data
            period: Period for RSI calculation
            
        Returns:
            Series containing RSI values
        """
        period = period or self.config['rsi_period']
        try:
            return ta.momentum.RSIIndicator(df['close'], window=period).rsi()
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}", exc_info=True)
            return pd.Series(index=df.index)
    
    def calculate_macd(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            df: DataFrame containing price data
            
        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        try:
            macd = ta.trend.MACD(
                df['close'],
                window_fast=self.config['macd_fast'],
                window_slow=self.config['macd_slow'],
                window_sign=self.config['macd_signal']
            )
            return macd.macd(), macd.macd_signal(), macd.macd_diff()
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}", exc_info=True)
            empty_series = pd.Series(index=df.index)
            return empty_series, empty_series, empty_series
    
    def calculate_bollinger_bands(
        self, 
        df: pd.DataFrame, 
        window: int = 20, 
        window_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame containing price data
            window: Window for moving average calculation
            window_dev: Number of standard deviations
            
        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        try:
            indicator = ta.volatility.BollingerBands(
                df['close'],
                window=window,
                window_dev=window_dev
            )
            return (
                indicator.bollinger_hband(), 
                indicator.bollinger_mavg(),
                indicator.bollinger_lband()
            )
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}", exc_info=True)
            empty_series = pd.Series(index=df.index)
            return empty_series, empty_series, empty_series
    
    def calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame containing price data
            window: Window for ATR calculation
            
        Returns:
            Series containing ATR values
        """
        try:
            # Ensure high, low, close columns exist
            required = ['high', 'low', 'close']
            if not all(col in df.columns for col in required):
                logger.warning("Missing required columns for ATR calculation")
                return pd.Series(index=df.index)
                
            return ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=window
            ).average_true_range()
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}", exc_info=True)
            return pd.Series(index=df.index)
    
    def calculate_stochastic(
        self, 
        df: pd.DataFrame, 
        window: int = 14, 
        smooth_window: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            df: DataFrame containing price data
            window: Window for stochastic calculation
            smooth_window: Window for smoothing
            
        Returns:
            Tuple of (Stochastic %K, Stochastic %D)
        """
        try:
            # Ensure high, low, close columns exist
            required = ['high', 'low', 'close']
            if not all(col in df.columns for col in required):
                logger.warning("Missing required columns for Stochastic calculation")
                empty_series = pd.Series(index=df.index)
                return empty_series, empty_series
                
            indicator = ta.momentum.StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=window,
                smooth_window=smooth_window
            )
            return indicator.stoch(), indicator.stoch_signal()
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}", exc_info=True)
            empty_series = pd.Series(index=df.index)
            return empty_series, empty_series

    @lru_cache(maxsize=32)
    def get_indicator_function(self, indicator_name: str) -> Callable:
        """
        Get the function to calculate a specific indicator.
        
        Args:
            indicator_name: Name of the indicator
            
        Returns:
            Function to calculate the indicator
        """
        indicator_functions = {
            'sma': self.calculate_sma,
            'ema': self.calculate_ema,
            'rsi': self.calculate_rsi,
            'macd': self.calculate_macd,
            'bollinger': self.calculate_bollinger_bands,
            'atr': self.calculate_atr,
            'stochastic': self.calculate_stochastic
        }
        
        indicator_name = indicator_name.lower()
        if indicator_name not in indicator_functions:
            logger.warning(f"Unknown indicator: {indicator_name}")
            return lambda df: pd.Series(index=df.index)
            
        return indicator_functions[indicator_name]


def calculate_technical_indicators(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None,
    config: Dict[str, int] = TECHNICAL_INDICATORS_CONFIG
) -> pd.DataFrame:
    """
    Calculate technical indicators and add them to the DataFrame.
    
    Args:
        df: DataFrame containing price data
        indicators: List of indicators to calculate. If None, calculates all.
                   Options: ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'stochastic']
        config: Dictionary containing parameters for technical indicators
    
    Returns:
        DataFrame with added technical indicators
    """
    try:
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_technical_indicators")
            return pd.DataFrame()
            
        calculator = IndicatorCalculator(config)
        
        # Validate and prepare DataFrame
        df = calculator.validate_dataframe(df)
        
        # Define indicators to calculate
        indicators = [ind.lower() for ind in (indicators or ['sma', 'ema', 'rsi', 'macd'])]
        
        # Calculate indicators
        result_df = df.copy()
        
        # Standard indicators (returning a single Series)
        standard_indicators = {
            'sma': ('SMA_{period}', calculator.calculate_sma),
            'ema': ('EMA_{period}', calculator.calculate_ema),
            'rsi': ('RSI_{period}', calculator.calculate_rsi),
            'atr': ('ATR_14', lambda df: calculator.calculate_atr(df))
        }
        
        for ind_name, (column_pattern, func) in standard_indicators.items():
            if ind_name in indicators:
                try:
                    period = config.get(f'{ind_name}_period', 14)
                    column_name = column_pattern.format(period=period)
                    result_df[column_name] = func(df)
                except Exception as e:
                    logger.error(f"Error adding {ind_name} indicator: {e}", exc_info=True)
        
        # Complex indicators (returning multiple Series)
        if 'macd' in indicators:
            try:
                macd, signal, hist = calculator.calculate_macd(df)
                result_df['MACD'] = macd
                result_df['MACD_Signal'] = signal
                result_df['MACD_Hist'] = hist
            except Exception as e:
                logger.error(f"Error adding MACD indicator: {e}", exc_info=True)
                
        if 'bollinger' in indicators:
            try:
                upper, middle, lower = calculator.calculate_bollinger_bands(df)
                result_df['BB_Upper'] = upper
                result_df['BB_Middle'] = middle
                result_df['BB_Lower'] = lower
            except Exception as e:
                logger.error(f"Error adding Bollinger Bands indicator: {e}", exc_info=True)
                
        if 'stochastic' in indicators:
            try:
                k, d = calculator.calculate_stochastic(df)
                result_df['Stoch_K'] = k
                result_df['Stoch_D'] = d
            except Exception as e:
                logger.error(f"Error adding Stochastic indicator: {e}", exc_info=True)
        
        # Fill any NaN values with appropriate methods
        result_df = result_df.fillna(method='bfill').fillna(method='ffill')
        
        # Log successful calculation
        logger.info(f"Successfully calculated {len(indicators)} technical indicators")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in calculate_technical_indicators: {e}", exc_info=True)
        return df
