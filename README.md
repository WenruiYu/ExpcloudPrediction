# ExpcloudPrediction

A Python-based application for collecting, processing, and analyzing financial stock data and macroeconomic indicators focused on Chinese markets.

## Features

- **Stock Data Collection**: Fetch historical stock data from BaoStock API
- **Technical Indicators**: Calculate various technical indicators including:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Average True Range (ATR)
  - Stochastic Oscillator
- **Macroeconomic Data**: Collect and process various Chinese macroeconomic indicators
- **Data Processing**: Efficient parallel processing of financial data
- **Caching System**: Smart caching to reduce redundant API calls

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ExpcloudPrediction.git
cd ExpcloudPrediction
```

2. Create a virtual environment and activate it:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/MacOS
python -m venv .venv
source .venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
ExpcloudPrediction/
├── data/                  # Data storage directory
│   ├── raw/               # Raw data from API
│   ├── processed/         # Processed data
│   ├── cache/             # Cached data to reduce API calls
│   ├── macro/             # Macroeconomic data
│   └── external/          # External data sources
├── src/                   # Source code
│   ├── config.py          # Configuration settings
│   ├── data_collection.py # Stock data collection
│   ├── utils.py           # Utility functions
│   └── macro_data_collection.py # Macroeconomic data collection
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Test directory
├── docs/                  # Documentation
└── requirements.txt       # Project dependencies
```

## Usage

### Collecting Stock Data

```python
from src.data_collection import StockDataCollector

# Initialize collector with default settings
collector = StockDataCollector()

# Or specify parameters
collector = StockDataCollector(
    symbol='sh.600519',          # Stock symbol
    start_date='2020-01-01',     # Start date
    end_date='2023-12-31',       # End date
    batch_size=1000,             # Batch size for processing
    max_workers=4                # Number of worker threads
)

# Using with context manager
with StockDataCollector('sh.600519') as collector:
    # Get processed data with technical indicators
    data = collector.get_data()
    
    # Or force refresh (ignore cache)
    data = collector.get_data(force_refresh=True)
    
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    print(f"Data sample:\n{data.head()}")
```

### Collecting Macroeconomic Data

```python
from src.macro_data_collection import MacroDataCollector

# Initialize with all sources
collector = MacroDataCollector()

# Or specify specific sources
collector = MacroDataCollector(sources=['gdp', 'cpi', 'm2'])

# Get all macro data
data_dict = collector.get_data()

# Get specific macro data
gdp_data = collector.get_data(source='gdp')

# Force refresh of data
cpi_data = collector.get_data(source='cpi', force_refresh=True)
```

### Command-line Usage for Macro Data

You can also collect macroeconomic data using the command-line interface:

```bash
# Collect all macro data
python -m src.macro_data_collection

# Collect specific sources
python -m src.macro_data_collection --sources gdp cpi

# Force refresh all data
python -m src.macro_data_collection --overwrite
```

### Configuration

The project uses environment variables for configuration. You can override the default settings by setting the following environment variables:

- `DATA_DIR`: Base directory for data storage
- `START_DATE`: Start date for data collection (YYYY-MM-DD)
- `END_DATE`: End date for data collection (YYYY-MM-DD)
- `ICBC_TICKER`: ICBC ticker symbol
- `ICBC_TICKER_BS`: ICBC ticker symbol in BaoStock format
- `BATCH_SIZE`: Batch size for parallel processing
- `MAX_WORKERS`: Maximum number of worker threads
- `CACHE_EXPIRY_DAYS`: Number of days before cache expires
- Technical indicator parameters:
  - `SMA_PERIOD`: Period for SMA calculation
  - `EMA_PERIOD`: Period for EMA calculation
  - `RSI_PERIOD`: Period for RSI calculation
  - `MACD_FAST`: Fast period for MACD calculation
  - `MACD_SLOW`: Slow period for MACD calculation
  - `MACD_SIGNAL`: Signal period for MACD calculation

## License

[MIT License](LICENSE)

## Acknowledgements

- [BaoStock](http://baostock.com/) for providing financial data API
- [TA-Lib](https://ta-lib.org/) for technical analysis functions
- [akshare](https://akshare.akfamily.xyz/) for macroeconomic data
