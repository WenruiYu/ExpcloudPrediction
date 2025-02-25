# src\core\utils.py

from typing import Optional, List, Dict, Tuple, Any, Union, Callable
import pandas as pd
import numpy as np
import ta
import logging
import logging.config
from functools import lru_cache

# Try to import using relative imports if running as a module
try:
    from src.core.config import TECHNICAL_INDICATORS_CONFIG, LOGGING_CONFIG
except ImportError:
    # Fall back to direct imports if running the file directly
    from ..core.config import TECHNICAL_INDICATORS_CONFIG, LOGGING_CONFIG

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

    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        Args:
            df: DataFrame containing price and volume data
            
        Returns:
            Series containing OBV values
        """
        try:
            if 'volume' not in df.columns:
                logger.warning("Volume data not available for OBV calculation")
                return pd.Series(index=df.index)
                
            return ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        except Exception as e:
            logger.error(f"Error calculating OBV: {e}", exc_info=True)
            return pd.Series(index=df.index)
    
    def calculate_adx(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            df: DataFrame containing price data
            period: Period for ADX calculation
            
        Returns:
            Series containing ADX values
        """
        period = period or self.config.get('adx_period', 14)
        try:
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logger.warning("Required columns missing for ADX calculation")
                return pd.Series(index=df.index)
                
            return ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=period).adx()
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}", exc_info=True)
            return pd.Series(index=df.index)
    
    def calculate_cci(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            df: DataFrame containing price data
            period: Period for CCI calculation
            
        Returns:
            Series containing CCI values
        """
        period = period or self.config.get('cci_period', 20)
        try:
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logger.warning("Required columns missing for CCI calculation")
                return pd.Series(index=df.index)
                
            return ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=period).cci()
        except Exception as e:
            logger.error(f"Error calculating CCI: {e}", exc_info=True)
            return pd.Series(index=df.index)
    
    def calculate_mfi(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Money Flow Index (MFI).
        
        Args:
            df: DataFrame containing price and volume data
            period: Period for MFI calculation
            
        Returns:
            Series containing MFI values
        """
        period = period or self.config.get('mfi_period', 14)
        try:
            if not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
                logger.warning("Required columns missing for MFI calculation")
                return pd.Series(index=df.index)
                
            return ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=period).money_flow_index()
        except Exception as e:
            logger.error(f"Error calculating MFI: {e}", exc_info=True)
            return pd.Series(index=df.index)
            
    def calculate_williams_r(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Williams %R.
        
        Args:
            df: DataFrame containing price data
            period: Period for Williams %R calculation
            
        Returns:
            Series containing Williams %R values
        """
        period = period or self.config.get('williams_r_period', 14)
        try:
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logger.warning("Required columns missing for Williams %R calculation")
                return pd.Series(index=df.index)
                
            return ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=period).williams_r()
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}", exc_info=True)
            return pd.Series(index=df.index)
    
    def calculate_psar(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Parabolic SAR.
        
        Args:
            df: DataFrame containing price data
            
        Returns:
            Series containing Parabolic SAR values
        """
        try:
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logger.warning("Required columns missing for PSAR calculation")
                return pd.Series(index=df.index)
                
            step = self.config.get('psar_step', 0.02)
            max_step = self.config.get('psar_max_step', 0.2)
            return ta.trend.PSARIndicator(df['high'], df['low'], df['close'], step=step, max_step=max_step).psar()
        except Exception as e:
            logger.error(f"Error calculating Parabolic SAR: {e}", exc_info=True)
            return pd.Series(index=df.index)
    
    def calculate_ichimoku(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate Ichimoku Cloud components.
        
        Args:
            df: DataFrame containing price data
            
        Returns:
            Tuple of Series containing Ichimoku components
        """
        try:
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logger.warning("Required columns missing for Ichimoku calculation")
                empty_series = pd.Series(index=df.index)
                return empty_series, empty_series, empty_series, empty_series, empty_series
            
            conv_window = self.config.get('ichimoku_conv_window', 9)
            base_window = self.config.get('ichimoku_base_window', 26)
            span_window = self.config.get('ichimoku_span_window', 52)
            
            indicator = ta.trend.IchimokuIndicator(
                df['high'], df['low'], 
                window1=conv_window, 
                window2=base_window, 
                window3=span_window
            )
            
            return (
                indicator.ichimoku_conversion_line(), 
                indicator.ichimoku_base_line(),
                indicator.ichimoku_a(), 
                indicator.ichimoku_b(),
                indicator.ichimoku_conversion_line() - indicator.ichimoku_base_line()  # Custom signal line
            )
        except Exception as e:
            logger.error(f"Error calculating Ichimoku Cloud: {e}", exc_info=True)
            empty_series = pd.Series(index=df.index)
            return empty_series, empty_series, empty_series, empty_series, empty_series

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
            'stochastic': self.calculate_stochastic,
            'obv': self.calculate_obv,
            'adx': self.calculate_adx,
            'cci': self.calculate_cci,
            'mfi': self.calculate_mfi,
            'williams_r': self.calculate_williams_r,
            'psar': self.calculate_psar,
            'ichimoku': self.calculate_ichimoku
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
                   Options: ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'stochastic', 
                             'obv', 'adx', 'cci', 'mfi', 'williams_r', 'psar', 'ichimoku']
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
            'atr': ('ATR_14', lambda df: calculator.calculate_atr(df)),
            'obv': ('OBV', calculator.calculate_obv),
            'adx': ('ADX_{period}', calculator.calculate_adx),
            'cci': ('CCI_{period}', calculator.calculate_cci),
            'mfi': ('MFI_{period}', calculator.calculate_mfi),
            'williams_r': ('Williams_R_{period}', calculator.calculate_williams_r),
            'psar': ('PSAR', calculator.calculate_psar)
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
        
        # Add Ichimoku Cloud components
        if 'ichimoku' in indicators:
            try:
                conv, base, span_a, span_b, signal = calculator.calculate_ichimoku(df)
                result_df['Ichimoku_Conversion'] = conv
                result_df['Ichimoku_Base'] = base
                result_df['Ichimoku_SpanA'] = span_a
                result_df['Ichimoku_SpanB'] = span_b
                result_df['Ichimoku_Signal'] = signal
            except Exception as e:
                logger.error(f"Error adding Ichimoku indicator: {e}", exc_info=True)
        
        # Fill any NaN values with appropriate methods (updated to use bfill() and ffill())
        result_df = result_df.bfill().ffill()
        
        # Log successful calculation
        logger.info(f"Successfully calculated {len(indicators)} technical indicators")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in calculate_technical_indicators: {e}", exc_info=True)
        return df
