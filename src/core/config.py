# src\core\config.py

import os
from pathlib import Path
from typing import Dict, List, Union, Optional

# Environment variable configuration
def get_env_var(var_name: str, default_value: str) -> str:
    """Get environment variable with a default value if not set."""
    return os.environ.get(var_name, default_value)

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent  # src directory
DATA_DIR = Path(get_env_var("DATA_DIR", str(PROJECT_ROOT / 'data')))  # data directory at project root
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
CACHE_DIR = DATA_DIR / 'cache'
MACRO_DATA_DIR = DATA_DIR / 'macro'

# Create all directories
REQUIRED_DIRS: List[Path] = [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, MACRO_DATA_DIR]
for directory in REQUIRED_DIRS:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Warning: Permission denied when creating directory {directory}")

# Date range for data collection
START_DATE = get_env_var('START_DATE', '2001-08-31')
END_DATE = get_env_var('END_DATE', '2025-02-21')

# Stock ticker symbols
MOUTAI_TICKER = get_env_var('MOUTAI_TICKER', '600519')  # Kweichow Moutai
MOUTAI_TICKER_BS = get_env_var('MOUTAI_TICKER_BS', 'sh.600519')

# ICBC ticker symbols (for reference)
ICBC_TICKER = get_env_var('ICBC_TICKER', '601398')  # Industrial and Commercial Bank of China
ICBC_TICKER_BS = get_env_var('ICBC_TICKER_BS', 'sh.601398')

# Default ticker to use for the application
DEFAULT_TICKER = get_env_var('DEFAULT_TICKER', MOUTAI_TICKER)
DEFAULT_TICKER_BS = get_env_var('DEFAULT_TICKER_BS', MOUTAI_TICKER_BS)

# Data collection settings
BATCH_SIZE = int(get_env_var('BATCH_SIZE', '1000'))
MAX_WORKERS = int(get_env_var('MAX_WORKERS', '4'))
CACHE_EXPIRY_DAYS = int(get_env_var('CACHE_EXPIRY_DAYS', '1'))

# Technical indicators default parameters
TECHNICAL_INDICATORS_CONFIG: Dict[str, int] = {
    'sma_period': int(get_env_var('SMA_PERIOD', '20')),
    'ema_period': int(get_env_var('EMA_PERIOD', '20')),
    'rsi_period': int(get_env_var('RSI_PERIOD', '14')),
    'macd_fast': int(get_env_var('MACD_FAST', '12')),
    'macd_slow': int(get_env_var('MACD_SLOW', '26')),
    'macd_signal': int(get_env_var('MACD_SIGNAL', '9')),
    # New indicator parameters
    'adx_period': int(get_env_var('ADX_PERIOD', '14')),
    'cci_period': int(get_env_var('CCI_PERIOD', '20')),
    'mfi_period': int(get_env_var('MFI_PERIOD', '14')),
    'williams_r_period': int(get_env_var('WILLIAMS_R_PERIOD', '14')),
    'psar_step': float(get_env_var('PSAR_STEP', '0.02')),
    'psar_max_step': float(get_env_var('PSAR_MAX_STEP', '0.2')),
    'ichimoku_conv_window': int(get_env_var('ICHIMOKU_CONV_WINDOW', '9')),
    'ichimoku_base_window': int(get_env_var('ICHIMOKU_BASE_WINDOW', '26')),
    'ichimoku_span_window': int(get_env_var('ICHIMOKU_SPAN_WINDOW', '52'))
}

# Macro data sources configuration
MACRO_DATA_SOURCES: Dict[str, Dict[str, str]] = {
    'gdp': {
        'function': 'ak.macro_china_gdp_yearly',
        'filename': 'china_gdp_yearly.csv',
        'description': "China's annual GDP growth rate"
    },
    'cpi': {
        'function': 'ak.macro_china_cpi_monthly',
        'filename': 'china_cpi_monthly.csv',
        'description': "China's monthly CPI data"
    },
    'm2': {
        'function': 'ak.macro_china_supply_of_money',
        'filename': 'china_supply_of_money.csv',
        'description': "China's monthly Money Supply data"
    }
}

# Logging configuration
LOGGING_CONFIG: Dict[str, Union[str, Dict]] = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'filename': str(BASE_DIR / 'logs' / 'app.log'),
            'mode': 'a',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
}

# Ensure log directory exists
LOG_DIR = BASE_DIR / 'logs'
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    print(f"Warning: Permission denied when creating log directory {LOG_DIR}")
