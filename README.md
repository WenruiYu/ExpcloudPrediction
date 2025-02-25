# ExpCloud Stock Prediction System

A comprehensive system for collecting stock and macroeconomic data, generating features, and preparing inputs for time series prediction models.

## Overview

This project provides an end-to-end pipeline for stock price prediction:

1. **Data Collection**: Collects stock data from BaoStock and macroeconomic indicators (GDP, CPI, M2)
2. **Feature Engineering**: Creates aligned features from different data sources and generates time-based, lag, and rolling window features
3. **Model Input Preparation**: Prepares sequence-based inputs for time series prediction models

## Prerequisites

- Python 3.8+
- PyTables (tables package)
- pandas, numpy
- BaoStock API access

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
├── data/                  # Data directory
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
python -m src.main data --symbol sh.600519 --start 2022-01-01 --end 2023-01-15
```

### Generating Features

To generate features from collected data:

```bash
python -m src.main features --symbol sh.600519 --start 2022-01-01 --end 2023-01-15 --include-time --include-lag --include-rolling
```

### Running Complete Pipeline

To run the entire pipeline (data collection, feature generation, and model input preparation):

```bash
python -m src.main all --symbol sh.600519 --start 2022-01-01 --end 2023-01-15
```

### Command-line Options

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

## Feature Pipeline Details

The feature pipeline handles:

1. **Data Alignment**: Aligns data from different sources with different frequencies (daily, monthly, quarterly, yearly)
2. **Time Features**: Extracts time-based features (day of week, month, quarter, etc.)
3. **Lag Features**: Creates lagged versions of features
4. **Rolling Features**: Computes rolling statistics (mean, std, min, max)

## Output

The pipeline generates model-ready data in the `data/model_ready/` directory:

- `x_train.npy`, `y_train.npy`: Training data and labels
- `x_val.npy`, `y_val.npy`: Validation data and labels
- `x_test.npy`, `y_test.npy`: Test data and labels
- `metadata.json`: Metadata including column information, normalization parameters, and date ranges

The prepared data consists of sequences suitable for LSTM or other time series models, with shape (samples, sequence_length, features).

## Troubleshooting

- **BaoStock Connection Issues**: Ensure you have internet access and BaoStock services are available
- **Empty Datasets**: Verify that the date range contains trading days
- **PyTables Errors**: Make sure you have the tables package installed
- **Sequence Generation Failures**: Ensure your date range is long enough to create sequences of the desired length

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
