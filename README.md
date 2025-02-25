# ExpcloudPrediction

A Python-based application for collecting, processing, and analyzing financial stock data and macroeconomic indicators focused on Chinese markets.

## Features

- **Stock Data Collection**: Fetch historical stock data (ICBC and Moutai) from BaoStock API
- **Technical Indicators**: Calculate SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and Stochastic Oscillator
- **Macroeconomic Data**: Collect Chinese GDP, CPI, and M2 data using akshare
- **Data Processing**: Efficient parallel processing with smart caching

## Quick Start

1. **Setup Environment**

```bash
# Clone the repository
git clone https://github.com/WenruiYu/ExpcloudPrediction.git
cd ExpcloudPrediction

# Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment (optional)
cp .env.example .env
# Edit .env file as needed
```

2. **Collect Stock Data**

```bash
# Run from command line
python -m src.main --stock --symbol sh.600519

# For ICBC stock data
python -m src.main --stock --symbol sh.601398
```

3. **Collect Macroeconomic Data**

```bash
# Run from command line
python -m src.main --macro

# Specific sources
python -m src.main --macro --macro-sources gdp cpi
```

## Using the Python API

### Stock Data Collection

```python
from src.data_collection import StockDataCollector

# Using Moutai stock (default)
with StockDataCollector() as collector:
    data = collector.get_data()
    print(f"Data columns: {data.columns.tolist()}")
    print(data.head())

# Using ICBC stock
with StockDataCollector(symbol="sh.601398") as collector:
    data = collector.get_data()
```

### Macroeconomic Data Collection

```python
from src.macro_data_collection import MacroDataCollector

# Get all macro data
collector = MacroDataCollector()
data_dict = collector.get_data()

# Get specific data
gdp_data = collector.get_data(source='gdp')
```

### Calculate Technical Indicators

```python
from src.utils import calculate_technical_indicators
import pandas as pd

# Load your stock data
data = pd.read_csv('your_stock_data.csv', index_col='date', parse_dates=True)

# Calculate indicators
result = calculate_technical_indicators(
    data,
    indicators=['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'stochastic']
)
```

## Configuration

The project uses environment variables for configuration:

- **Stock Tickers**:
  - `DEFAULT_TICKER_BS=sh.600519` (Moutai)
  - `DEFAULT_TICKER_BS=sh.601398` (ICBC)
  
- **Date Range**:
  - `START_DATE=2001-08-31`
  - `END_DATE=2025-02-21`

See `.env.example` for all configuration options.

## Running Tests

The project includes a comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run specific tests
pytest tests/test_stock_data.py
pytest tests/test_macro_data.py
pytest tests/test_utils.py

# Run with verbosity
pytest -v tests/
```

## Project Structure

```
ExpcloudPrediction/
├── data/                   # Data storage
│   ├── raw/                # Raw data
│   ├── processed/          # Processed data
│   ├── cache/              # Cache files
│   └── macro/              # Macroeconomic data
├── src/                    # Source code
│   ├── config.py           # Configuration
│   ├── data_collection.py  # Stock data collection
│   ├── macro_data_collection.py  # Macro data
│   ├── utils.py            # Technical indicators
│   └── main.py             # Command-line interface
├── tests/                  # Test suite
└── requirements.txt        # Dependencies
```

## License

[MIT License](LICENSE)

## Acknowledgements

- [BaoStock](http://baostock.com/) for providing financial data API
- [TA-Lib](https://ta-lib.org/) for technical analysis functions
- [akshare](https://akshare.akfamily.xyz/) for macroeconomic data
