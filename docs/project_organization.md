# ExpcloudPrediction Project Organization

This document explains the organization of the ExpcloudPrediction project, focusing on the modular architecture and workflow.

## Overall Architecture

The project follows a modular architecture that separates concerns into distinct components:

1. **Data Collection**: Obtaining raw data from various sources (stock data, macroeconomic indicators)
2. **Data Processing**: Cleaning and preparing raw data for feature extraction
3. **Feature Engineering**: Creating features from processed data
4. **Feature Store**: Managing and versioning features
5. **Model Ready Data**: Preparing aligned and normalized data for model training

## Directory Structure

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
│   │   ├── __init__.py
│   │   └── config.py     # Configuration parameters
│   ├── data/             # Data collection and processing
│   │   ├── __init__.py
│   │   ├── collector.py  # Stock data collection
│   │   ├── macro.py      # Macroeconomic data collection
│   │   ├── cleaner.py    # Data cleaning utilities
│   │   └── utils.py      # Data processing utilities
│   ├── features/         # Feature engineering and management
│   │   ├── __init__.py
│   │   ├── metadata.py   # Feature metadata management
│   │   ├── store.py      # Feature storage and retrieval
│   │   ├── generator.py  # Feature generation
│   │   └── transform.py  # Feature transformation
│   ├── cli/              # Command-line interfaces
│   │   ├── __init__.py
│   │   ├── data_cli.py   # CLI for data collection
│   │   └── feature_cli.py # CLI for feature generation
│   └── main.py           # Main entry point
└── tests/                # Unit and integration tests
    ├── test_data/        # Test data and fixtures
    ├── test_features/    # Tests for feature components
    └── test_utils.py     # Tests for utility functions
```

## Module Descriptions

### Core Module (`src/core/`)

Contains configuration and shared utilities that are used throughout the application:

- `config.py`: Configuration parameters, paths, API keys, default values
- Other core utilities as needed

### Data Module (`src/data/`)

Responsible for collecting, cleaning, and processing raw data:

- `collector.py`: Stock data collection from APIs or local sources
- `macro.py`: Macroeconomic data collection from various sources
- `cleaner.py`: Utility to clean data directories
- `utils.py`: Data processing utilities

### Features Module (`src/features/`)

Implements feature engineering, transformation, and storage:

- `metadata.py`: Manages metadata for data sources and features
- `store.py`: Implements the feature store for versioned feature management
- `generator.py`: Generates different types of features
- `transform.py`: Handles frequency alignment and transformation of features

### CLI Module (`src/cli/`)

Command-line interfaces for different functionalities:

- `data_cli.py`: CLI for data collection operations
- `feature_cli.py`: CLI for feature generation and management

### Main Entry Point (`src/main.py`)

The unified entry point that provides a consistent interface to access all functionality.

## Data Flow

The typical data flow through the system is:

1. Raw data collection (Stock, Macro) → `data/raw/`, `data/macro/`
2. Data processing → `data/processed/`
3. Feature generation → `data/feature_store/`
4. Model-ready data preparation → `data/model_ready/`

## Usage Examples

### Command-line Interface

The project provides a unified command-line interface:

```
python -m src.main [command] [options]
```

Available commands:

- `data`: Collect and process stock and macro data
- `features`: Generate features and prepare model inputs
- `clean`: Clean data files
- `all`: Run the complete pipeline (data collection + feature generation)

### Python API Usage

```python
# Using the data collector
from src.data.collector import StockDataCollector

collector = StockDataCollector(symbol="sh.600519")
data = collector.get_data()

# Using the feature store
from src.features.store import FeatureStore

store = FeatureStore()
features = store.get_features("sh.600519", ["price", "volume", "macd"])
```

## Future Development

Areas for future development include:

1. Model training and evaluation modules
2. Prediction pipeline
3. Backtesting framework
4. Web dashboard for monitoring and visualization
5. Automated data quality monitoring 