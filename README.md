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

2. **Run the Application**

```bash
# Run with default settings (collects both stock and macro data)
python -m src.main

# Run with specific options
python -m src.main --stock --macro
```

3. **Command-line Options**

```bash
# Stock Data Options:
# ------------------
# Collect stock data for Moutai (default)
python -m src.main --stock

# Collect stock data for ICBC
python -m src.main --stock --symbol sh.601398

# Specify date range
python -m src.main --stock --start-date 2020-01-01 --end-date 2023-01-01

# Calculate specific technical indicators
python -m src.main --stock --indicators sma ema rsi macd bollinger atr stochastic

# Macro Data Options:
# ------------------
# Collect all macroeconomic data
python -m src.main --macro

# Collect specific macro data sources
python -m src.main --macro --macro-sources gdp cpi

# General Options:
# --------------
# Force refresh (ignore cache)
python -m src.main --force-refresh

# Hide data preview
python -m src.main --no-preview

# Combined Options:
# ---------------
# Collect both stock and macro data with custom settings
python -m src.main --stock --symbol sh.601398 --macro --macro-sources gdp m2 --force-refresh
```

4. **Cleaning Data Files**

The project includes a utility to easily clean data files:

```bash
# Clean all data files (with confirmation prompt)
python -m src.data.cleaner --all

# Clean specific data directories
python -m src.data.cleaner --raw --processed

# Simulate cleaning (dry run without actual deletion)
python -m src.data.cleaner --all --dry-run

# Clean cache files without confirmation
python -m src.data.cleaner --cache --force

# Advanced options
python -m src.data.cleaner --all --no-gitkeep  # Also remove .gitkeep files
```

## Using the Python API

### Stock Data Collection

```python
from src.data.stock_collection import StockDataCollector

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
from src.data.macro_collection import MacroDataCollector

# Get all macro data
collector = MacroDataCollector()
data_dict = collector.get_data()

# Get specific data
gdp_data = collector.get_data(source='gdp')
```

### Calculate Technical Indicators

```python
from src.core.utils import calculate_technical_indicators
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

The project is organized into the following directory structure:

```
ExpcloudPrediction/
├── data/                 # Data storage directory
│   ├── raw/              # Raw collected data
│   ├── processed/        # Processed and aggregated data
│   ├── cache/            # Cached data to avoid redundant API calls
│   ├── macro/            # Macroeconomic data
│   ├── feature_store/    # Processed features with metadata
│   └── model_ready/      # Feature sets ready for model training
├── docs/                 # Documentation files
├── models/               # Trained models and model artifacts
├── notebooks/            # Jupyter notebooks for exploration and analysis
├── src/                  # Source code
│   ├── core/             # Core functionality and configuration
│   ├── data/             # Data collection and processing
│   ├── features/         # Feature engineering and management
│   └── cli/              # Command-line interfaces
└── tests/                # Unit and integration tests
```

## Usage

### Command-line Interface

The project now provides a unified command-line interface with the following commands:

```
python -m src.main [command] [options]
```

Available commands:

- `data`: Collect and process stock and macro data
- `features`: Generate features and prepare model inputs
- `clean`: Clean data files
- `all`: Run the complete pipeline (data collection + feature generation)

For detailed help on specific commands:

```
python -m src.main data --help
python -m src.main features --help
python -m src.main clean --help
```

### Examples

1. Collect stock data for a specific symbol:
```
python -m src.main data --stock --symbol sh.600519
```

2. Generate features for a specific symbol:
```
python -m src.main features --symbol sh.600519 --include-time --include-lag
```

3. Clean all data directories:
```
python -m src.main clean --all
```

4. Run the complete pipeline for a specific symbol:
```
python -m src.main all --symbol sh.600519
```

## License

[MIT License](LICENSE)

## Acknowledgements

- [BaoStock](http://baostock.com/) for providing financial data API
- [TA-Lib](https://ta-lib.org/) for technical analysis functions
- [akshare](https://akshare.akfamily.xyz/) for macroeconomic data

## Data Organization System

This project includes a comprehensive data organization system that handles diverse data sources with different frequencies and availability periods. The system is designed to streamline the process from data collection to model-ready inputs.

### System Components

#### 1. Metadata Registry (`src/data_metadata.py`)

The metadata registry provides a centralized system for defining data sources with:
- Different time frequencies (daily, weekly, monthly, quarterly, yearly)
- Different availability periods (historical data that starts/ends at various dates)
- Various data categories (stock, macro, company, industry)
- Rules for frequency alignment (how to handle conversion between frequencies)

```python
# Example: Registering a data source
from src.data_metadata import register_data_source, FREQUENCY_DAILY, CATEGORY_STOCK

register_data_source('new_data_source', {
    'category': CATEGORY_STOCK,
    'frequency': FREQUENCY_DAILY,
    'function': 'module.function_name',
    'filename_pattern': '{symbol}_data.csv',
    'availability_start': '2010-01-01',
    'date_column': 'date',
    'required_fields': ['field1', 'field2']
})
```

#### 2. Data Transformation (`src/data_transformation.py`)

This module handles the alignment and transformation of data with different frequencies:
- Loads data sources with consistent interfaces
- Resamples data to target frequencies (e.g., converting monthly GDP to daily)
- Creates aligned datasets from multiple sources
- Prepares data for model inputs with proper sequence construction

```python
# Example: Creating aligned data from multiple sources
from src.data_transformation import DataTransformer

transformer = DataTransformer(target_frequency='daily')
aligned_data = transformer.create_aligned_dataset(
    sources=['stock_price', 'gdp', 'cpi'],
    start_date='2010-01-01',
    end_date='2022-12-31',
    symbol='sh.600519'  # For stock_price
)
```

#### 3. Feature Store (`src/feature_store.py`)

The feature store provides a centralized repository for processed features:
- Stores processed features with versioning
- Provides efficient access to features for model training
- Manages metadata about features and their lineage
- Handles feature generation from raw data sources

```python
# Example: Storing and retrieving features
from src.feature_store import FeatureStore

store = FeatureStore()
store.store_feature(
    feature_id='stock_feature',
    data=stock_data,
    metadata={
        'frequency': 'daily',
        'source_ids': ['stock_price'],
        'description': 'Stock price data with indicators'
    }
)

# Later retrieve it
feature_data = store.get_feature('stock_feature')
```

#### 4. Data Integration (`src/data_integration.py`)

The data integration module serves as an adapter between the existing data collection components and the new data organization system:
- Connects the existing data collectors to the feature store
- Registers data sources from existing collections to the metadata registry
- Creates a unified workflow from data collection to feature generation

```python
# Example: Using the integrator
from src.data_integration import DataIntegrator

integrator = DataIntegrator()
aligned_data = integrator.create_aligned_features(
    feature_id="moutai_with_macro",
    symbol="sh.600519",
    macro_sources=['gdp', 'cpi', 'm2'],
    start_date='2019-01-01',
    end_date='2023-12-31'
)
```

### Feature Pipeline

The feature pipeline (`src/feature_pipeline.py`) demonstrates how to use all these components together to create a complete workflow from data collection to model-ready inputs:

```bash
# Example: Running the feature pipeline
python -m src.feature_pipeline --symbol sh.600519 --start-date 2010-01-01 --end-date 2023-12-31 --include-time --include-lag --include-rolling
```

The pipeline performs the following steps:
1. Collects and registers stock and macro data
2. Creates aligned features from multiple data sources
3. Generates advanced features (time-based, lag, rolling window)
4. Prepares model inputs with proper sequence construction
5. Saves model-ready data in a standardized format

### Data Directory Structure

The data organization system works with the following directory structure:

- `data/raw`: Raw data files from different sources
- `data/processed`: Processed data files with cleaned and transformed data
- `data/cache`: Cached data to avoid redundant API calls
- `data/macro`: Macroeconomic data files
- `data/feature_store`: Stored features with versioning (created automatically)
- `data/model_ready`: Model-ready data in standardized format (created by the feature pipeline)
