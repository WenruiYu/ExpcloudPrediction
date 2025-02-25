# ExpCloud Stock Prediction System

A comprehensive system for collecting stock and macroeconomic data, generating features, and preparing inputs for time series prediction models.

## Overview

This project provides an end-to-end pipeline for stock price prediction:

1. **Data Collection**: Collects stock data from BaoStock and macroeconomic indicators (GDP, CPI, M2)
2. **Feature Engineering**: Creates aligned features from different data sources and generates time-based, lag, and rolling window features
3. **Model Input Preparation**: Prepares sequence-based inputs for time series prediction models

## Prerequisites

- Python 3.8+
- Core dependencies:
  - pandas, numpy
  - PyTables (tables package) for HDF5 storage
- Data collection:
  - BaoStock API (for stock data)
  - AkShare (for macroeconomic indicators)
- Technical analysis:
  - ta (Technical Analysis library)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ExpcloudPrediction.git
cd ExpcloudPrediction
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
ExpcloudPrediction/
├── data/                  # Data directory (ignored by Git)
│   ├── cache/             # Cached data files
│   ├── feature_store/     # HDF5 feature store
│   ├── macro/             # Macroeconomic data
│   ├── model_ready/       # Prepared model inputs
│   ├── processed/         # Processed stock data
│   └── raw/               # Raw data files
├── src/                   # Source code
│   ├── cli/               # Command-line interfaces
│   ├── core/              # Core utilities and configurations
│   ├── data/              # Data collection modules
│   ├── features/          # Feature engineering
│   ├── models/            # Prediction models
│   └── main.py            # Main entry point
└── README.md              # This file
```

## Usage

The main interface is through the command-line. The system offers several commands:

### Collecting Data

To collect stock and macroeconomic data:

```bash
# Collect both stock and macro data (default behavior)
python -m src.main data

# Collect only stock data for a specific symbol
python -m src.main data --stock --symbol sh.600519

# Collect only macroeconomic data
python -m src.main data --macro
```

### Generating Features

To generate features from collected data:

```bash
# Generate all feature types (time-based, lag, rolling)
python -m src.main features --symbol sh.600519

# Generate specific feature types
python -m src.main features --symbol sh.600519 --include-time --include-lag
```

### Running Complete Pipeline

To run the entire pipeline (data collection, feature generation, and model input preparation):

```bash
python -m src.main all --symbol sh.600519
```

### Command-line Options

#### For Data Collection:
- `--stock`: Collect stock data
- `--symbol`: Stock symbol (default: sh.600519)
- `--macro`: Collect macroeconomic data
- `--force-refresh`: Force refresh data (ignore cache)

#### For Feature Generation:
- `--symbol`: Stock symbol (default: sh.600519)
- `--start-date`: Start date for data collection (YYYY-MM-DD)
- `--end-date`: End date for data collection (YYYY-MM-DD)
- `--macro-sources`: Macroeconomic data sources to include (gdp, cpi, m2)
- `--frequency`: Target frequency for features (daily, weekly, monthly)
- `--include-time`: Include time-based features
- `--include-lag`: Include lag features
- `--include-rolling`: Include rolling window features
- `--sequence-length`: Length of input sequences (default: 30)
- `--forecast-horizon`: Number of days to forecast (default: 5)
- `--target-column`: Target column for prediction (default: close_stock_price)
- `--force-refresh`: Force refresh data (ignore cache)
- `--output-dir`: Output directory for model-ready data

### Cleaning Data Directories

To clean up data directories:

```bash
# Clean all data directories with confirmation prompt
python -m src.main clean

# Preview cleaning without removing files (dry run)
python -m src.main clean --dry-run
```

The cleaner automatically cleans all data directories while preserving the directory structure. It removes all files except directory placeholders.

## Feature Pipeline Details

The feature pipeline handles:

1. **Data Alignment**: Aligns data from different sources with different frequencies (daily, monthly, quarterly, yearly)
2. **Time Features**: Extracts time-based features (day of week, month, quarter, etc.)
3. **Lag Features**: Creates lagged versions of features
4. **Rolling Features**: Computes rolling statistics (mean, std, min, max)

## Data Storage

The system stores data in multiple formats:

1. **Raw Data**: Original data saved in the `data/raw/` directory with naming convention `raw_macro_{source_id}_{date}.csv`
2. **Processed Data**: Cleaned and processed data stored in `data/processed/` and `data/macro/`
3. **Feature Store**: Features saved in HDF5 format in `data/feature_store/`
4. **Model-Ready Data**: Prepared sequences stored in `data/model_ready/`

## Output

The pipeline generates model-ready data in the `data/model_ready/` directory:

- `x_train.npy`, `y_train.npy`: Training data and labels
- `x_val.npy`, `y_val.npy`: Validation data and labels
- `x_test.npy`, `y_test.npy`: Test data and labels
- `metadata.json`: Metadata including column information, normalization parameters, and date ranges

The prepared data consists of sequences suitable for LSTM or other time series models, with shape (samples, sequence_length, features).

## Macroeconomic Data

The system collects and processes several macroeconomic indicators:

1. **GDP**: Quarterly Gross Domestic Product and year-over-year growth rate
2. **CPI**: Monthly Consumer Price Index
3. **M2**: Monthly money supply metrics (M2, M1, M0) and their year-over-year growth rates

All date data is stored in the "yyyy-mm-dd" format for consistency.

## Troubleshooting

- **BaoStock Connection Issues**: Ensure you have internet access and BaoStock services are available
- **Empty Datasets**: Verify that the date range contains trading days
- **PyTables Errors**: Make sure you have the tables package installed
- **Sequence Generation Failures**: Ensure your date range is long enough to create sequences of the desired length

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
