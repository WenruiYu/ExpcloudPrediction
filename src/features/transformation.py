"""
Data transformation module for handling data with different frequencies.

This module provides functionality for:
- Loading data sources with different frequencies
- Aligning data to common frequencies
- Transforming data for model input
- Handling missing data and different availability periods
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import logging

# Try to import using relative imports if running as a module
try:
    from src.features.metadata import (
        get_source_metadata, list_data_sources, get_sources_by_availability,
        get_common_date_range, VALID_FREQUENCIES, FREQUENCY_DAILY, 
        FREQUENCY_WEEKLY, FREQUENCY_MONTHLY, FREQUENCY_QUARTERLY, FREQUENCY_YEARLY
    )
    from src.core.config import DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, RAW_DATA_DIR, MACRO_DATA_DIR
except ImportError:
    # Fall back to direct imports if running the file directly
    from ..features.metadata import (
        get_source_metadata, list_data_sources, get_sources_by_availability,
        get_common_date_range, VALID_FREQUENCIES, FREQUENCY_DAILY,
        FREQUENCY_WEEKLY, FREQUENCY_MONTHLY, FREQUENCY_QUARTERLY, FREQUENCY_YEARLY
    )
    from ..core.config import DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, RAW_DATA_DIR, MACRO_DATA_DIR

# Configure logging
logger = logging.getLogger(__name__)

# Frequency mapping to pandas frequency strings
FREQUENCY_MAP = {
    FREQUENCY_DAILY: 'D',
    FREQUENCY_WEEKLY: 'W',
    FREQUENCY_MONTHLY: 'M',
    FREQUENCY_QUARTERLY: 'Q',
    FREQUENCY_YEARLY: 'Y'
}

class MetadataHandler:
    """Simple handler for source metadata."""
    
    def __init__(self):
        """Initialize with basic source metadata."""
        # Source type mapping
        self.source_types = {
            'stock_price': 'stock_price',
            'gdp': 'macro',
            'cpi': 'macro',
            'm2': 'macro'
        }
        
        # Source frequency mapping
        self.source_frequencies = {
            'stock_price': FREQUENCY_DAILY,
            'gdp': FREQUENCY_QUARTERLY,
            'cpi': FREQUENCY_MONTHLY,
            'm2': FREQUENCY_MONTHLY
        }
    
    def get_source_type(self, source_id: str) -> str:
        """Get the type of a source."""
        return self.source_types.get(source_id, 'unknown')
    
    def get_source_frequency(self, source_id: str) -> str:
        """Get the frequency of a source."""
        return self.source_frequencies.get(source_id)

class DataTransformer:
    """Handles transformation and alignment of data with different frequencies."""
    
    def __init__(self, target_frequency: str = FREQUENCY_DAILY):
        """
        Initialize the data transformer.
        
        Args:
            target_frequency: Target frequency for alignment (default: daily)
        """
        if target_frequency not in VALID_FREQUENCIES:
            raise ValueError(f"Invalid target frequency: {target_frequency}")
            
        self.target_frequency = target_frequency
        self.data_registry = {}  # Store loaded dataframes
        
        # Initialize the metadata handler
        self.metadata_handler = MetadataHandler()
        
        logger.info(f"Initialized DataTransformer with target frequency: {target_frequency}")
    
    def _get_data_path(self, source_id: str, **kwargs) -> Path:
        """
        Get the path to a data source file.
        
        Args:
            source_id: Identifier for the data source
            **kwargs: Additional parameters for file path construction
            
        Returns:
            Path object for the data file
        """
        metadata = get_source_metadata(source_id)
        category = metadata['category']
        filename = metadata['filename_pattern']
        
        # Replace placeholders in filename
        for key, value in kwargs.items():
            if '{' + key + '}' in filename:
                filename = filename.replace('{' + key + '}', value)
        
        # Determine directory based on category
        if category == 'stock':
            base_dir = PROCESSED_DATA_DIR
        elif category == 'macro':
            base_dir = MACRO_DATA_DIR
        elif category == 'company':
            base_dir = DATA_DIR / 'company'
        elif category == 'industry':
            base_dir = DATA_DIR / 'industry'
        else:
            base_dir = DATA_DIR
        
        return base_dir / filename
    
    def load_data_source(self, source_id: str, **kwargs) -> pd.DataFrame:
        """
        Load data for a specific source.
        
        Args:
            source_id: Source identifier
            **kwargs: Additional arguments for data source parameters
            
        Returns:
            DataFrame with source data
        """
        # Get source type
        source_type = self.metadata_handler.get_source_type(source_id)
        
        if source_type == 'stock_price':
            # Load processed stock data
            symbol = kwargs.get('symbol')
            if not symbol:
                logger.error("Symbol required for stock data")
                return pd.DataFrame()
            
            filepath = os.path.join(PROCESSED_DATA_DIR, f"{symbol}_processed_stock_data.csv")
            logger.info(f"Loading data from {filepath}")
            
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return pd.DataFrame()
            
            try:
                data = pd.read_csv(filepath)
                # Set date as index
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    data.set_index('date', inplace=True)
                return data
            except Exception as e:
                logger.error(f"Error loading stock data: {e}")
                return pd.DataFrame()
        
        elif source_id in ['gdp', 'cpi', 'm2']:
            # Load macro data
            file_type = 'yearly' if source_id == 'gdp' else 'monthly'
            filepath = os.path.join(MACRO_DATA_DIR, f"china_{source_id}_{file_type}.csv")
            logger.info(f"Loading data from {filepath}")
            
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return pd.DataFrame()
            
            try:
                data = pd.read_csv(filepath)
                
                # Set date as index
                for date_col in ['date', 'year', 'month']:
                    if date_col in data.columns:
                        data[date_col] = pd.to_datetime(data[date_col])
                        data.set_index(date_col, inplace=True)
                        break
                
                # Create date index if none found
                if not isinstance(data.index, pd.DatetimeIndex):
                    logger.warning(f"No date column found in {filepath}, creating date index")
                    freq = 'Q' if source_id == 'gdp' else 'M'
                    date_range = pd.date_range(start='2000-01-01', periods=len(data), freq=freq)
                    data.index = date_range
                
                return data
            except Exception as e:
                logger.error(f"Error loading macro data: {e}")
                return pd.DataFrame()
        
        else:
            logger.error(f"Unknown source type: {source_type}")
            return pd.DataFrame()
    
    def align_to_frequency(self, data: pd.DataFrame, source_id: str) -> pd.DataFrame:
        """
        Align data to target frequency.
        
        Args:
            data: Input DataFrame with datetime index
            source_id: Source identifier
            
        Returns:
            DataFrame with aligned data
        """
        # Ensure the index is a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning(f"Converting index to DatetimeIndex for source '{source_id}'")
            data.index = pd.to_datetime(data.index)
        
        # Get source and target frequencies
        source_freq = self.metadata_handler.get_source_frequency(source_id)
        if not source_freq:
            logger.warning(f"No frequency information for source '{source_id}'. Assuming no alignment needed.")
            return data
        
        if source_freq == self.target_frequency:
            # No frequency alignment needed
            logger.info(f"No frequency alignment needed for source '{source_id}'")
            return data
        
        logger.info(f"Aligning {source_id} from {source_freq} to {self.target_frequency}")
        
        # Convert to pandas frequency string
        source_pd_freq = self._to_pandas_freq(source_freq)
        pd_target_freq = self._to_pandas_freq(self.target_frequency)
        
        # Resample based on relative frequencies
        if self._is_higher_frequency(self.target_frequency, source_freq):
            # Upsampling (e.g., monthly to daily)
            logger.info(f"Upsampling {source_id} data to {self.target_frequency}")
            resampled = data.resample(pd_target_freq).asfreq()
            resampled = resampled.ffill()  # Forward fill missing values
        else:
            # Downsampling (e.g., daily to monthly)
            logger.info(f"Downsampling {source_id} data to {self.target_frequency}")
            resampled = data.resample(pd_target_freq).last()  # Use last value in period
        
        return resampled
    
    def create_aligned_dataset(
        self, 
        sources: List[str], 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Create a dataset with multiple sources aligned to the same frequency.
        
        Args:
            sources: List of source IDs to include
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            **kwargs: Additional parameters for file path construction
            
        Returns:
            DataFrame with aligned data sources
        """
        if not sources:
            logger.warning("No sources specified for aligned dataset")
            return pd.DataFrame()
        
        # Find common date range if not specified
        if not start_date or not end_date:
            common_start, common_end = get_common_date_range(sources)
            start_date = start_date or common_start
            end_date = end_date or common_end
        
        logger.info(f"Creating aligned dataset for {len(sources)} sources from {start_date} to {end_date}")
        
        # Load and align each source
        aligned_data = {}
        for source_id in sources:
            try:
                # Load data
                data = self.load_data_source(source_id, **kwargs)
                
                if data.empty:
                    logger.warning(f"Empty data for source '{source_id}', skipping")
                    continue
                
                # Align to target frequency
                aligned = self.align_to_frequency(data, source_id)
                
                if aligned.empty:
                    logger.warning(f"Empty aligned data for source '{source_id}', skipping")
                    continue
                
                # Filter to date range
                if start_date:
                    aligned = aligned[aligned.index >= start_date]
                if end_date:
                    aligned = aligned[aligned.index <= end_date]
                
                # Prepare for merge
                aligned_data[source_id] = aligned
                
                logger.info(f"Prepared {len(aligned)} rows for source '{source_id}'")
                
            except Exception as e:
                logger.error(f"Error processing source '{source_id}': {e}", exc_info=True)
        
        # Merge all data sources
        if not aligned_data:
            logger.warning("No data available for aligned dataset")
            return pd.DataFrame()
        
        # Start with the first source
        first_source = list(aligned_data.keys())[0]
        result = aligned_data[first_source].copy()
        
        # Add suffix to columns to identify the source
        result.columns = [f"{col}_{first_source}" for col in result.columns]
        
        # Merge with remaining sources
        for source_id, data in list(aligned_data.items())[1:]:
            # Add suffix to columns
            data = data.copy()
            data.columns = [f"{col}_{source_id}" for col in data.columns]
            
            # Merge on index
            result = result.join(data, how='outer')
        
        # Fill missing values
        result = result.ffill().bfill()
        
        logger.info(f"Created aligned dataset with {len(result)} rows and {len(result.columns)} columns")
        
        return result
    
    def prepare_model_inputs(
        self, 
        aligned_data: pd.DataFrame, 
        target_column: str = 'close_stock_price',
        feature_columns: List[str] = None,
        sequence_length: int = 30,
        forecast_horizon: int = 5,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        normalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Prepare aligned data for model training.
        
        Args:
            aligned_data: Aligned feature data
            target_column: Target column name
            feature_columns: List of feature columns (None for all)
            sequence_length: Input sequence length
            forecast_horizon: Forecast horizon
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            normalize: Whether to normalize data
            
        Returns:
            Dictionary with model inputs
        """
        if aligned_data.empty:
            logger.warning("Empty aligned data, cannot prepare model inputs")
            return {}
        
        logger.info(f"Preparing model inputs with sequence length {sequence_length} and forecast horizon {forecast_horizon}")
        
        # Prepare data
        data = aligned_data.copy()
        
        # Identify numeric columns only
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        
        # Find the target column with potential suffixes
        actual_target_column = None
        target_candidates = [
            target_column,
            f"{target_column}_primary",
            f"{target_column}_secondary"
        ]
        
        for candidate in target_candidates:
            if candidate in numeric_columns:
                actual_target_column = candidate
                break
        
        if actual_target_column is None:
            logger.error(f"Target column '{target_column}' or its variants not found or not numeric")
            return {}
        
        logger.info(f"Using '{actual_target_column}' as the target column")
        
        # Select feature columns (numeric only)
        if feature_columns is None:
            feature_columns = [col for col in numeric_columns if col != actual_target_column]
        else:
            # Filter to keep only numeric columns that exist
            feature_columns = [col for col in feature_columns if col in numeric_columns]
            
        logger.info(f"Using {len(feature_columns)} numeric feature columns")
        
        # Shift target for forecasting
        if forecast_horizon > 0:
            data[f'{actual_target_column}_target'] = data[actual_target_column].shift(-forecast_horizon)
            data = data.dropna(subset=[f'{actual_target_column}_target'])
        else:
            data[f'{actual_target_column}_target'] = data[actual_target_column]
        
        # Normalize data
        if normalize:
            scaler_dict = {}
            for col in feature_columns + [f'{actual_target_column}_target']:
                if col in numeric_columns:  # Only normalize numeric columns
                    mean = data[col].mean()
                    std = data[col].std()
                    if std == 0:
                        std = 1  # Avoid division by zero
                    data[col] = (data[col] - mean) / std
                    scaler_dict[col] = {'mean': mean, 'std': std}
        else:
            scaler_dict = None
        
        # Create sequences for time series modeling
        x_sequences = []
        y_sequences = []
        dates = []
        
        for i in range(len(data) - sequence_length):
            x_seq = data[feature_columns].iloc[i:i+sequence_length].values
            y_seq = data[f'{actual_target_column}_target'].iloc[i+sequence_length-1]
            
            x_sequences.append(x_seq)
            y_sequences.append(y_seq)
            dates.append(data.index[i+sequence_length-1])
        
        if not x_sequences:
            logger.warning("No sequences created, dataset may be too small")
            return {}
        
        # Convert to numpy arrays
        x = np.array(x_sequences)
        y = np.array(y_sequences)
        
        # Split into train, validation, and test sets
        total_samples = len(x)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_samples))
        
        # Create splits
        result = {
            'x_train': x[train_indices],
            'y_train': y[train_indices],
            'x_val': x[val_indices],
            'y_val': y[val_indices],
            'x_test': x[test_indices],
            'y_test': y[test_indices],
            'train_dates': [dates[i] for i in train_indices],
            'val_dates': [dates[i] for i in val_indices],
            'test_dates': [dates[i] for i in test_indices],
            'feature_columns': feature_columns,
            'target_column': actual_target_column,
            'scaler_dict': scaler_dict,
            'sequence_length': sequence_length,
            'forecast_horizon': forecast_horizon
        }
        
        logger.info(f"Created model inputs with {len(train_indices)} training samples, "
                   f"{len(val_indices)} validation samples, and {len(test_indices)} test samples")
        
        return result

    def _to_pandas_freq(self, freq: str) -> str:
        """Convert frequency string to pandas frequency string."""
        return FREQUENCY_MAP.get(freq)
        
    def _is_higher_frequency(self, freq1: str, freq2: str) -> bool:
        """Check if freq1 is higher frequency than freq2."""
        if freq1 not in VALID_FREQUENCIES or freq2 not in VALID_FREQUENCIES:
            return False
        return VALID_FREQUENCIES.index(freq1) < VALID_FREQUENCIES.index(freq2)

# Utility functions
def create_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from datetime index.
    
    Args:
        data: DataFrame with datetime index
        
    Returns:
        DataFrame with added time features
    """
    df = data.copy()
    
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not DatetimeIndex, attempting to convert")
        df.index = pd.to_datetime(df.index)
    
    # Extract datetime components and explicitly convert to int64 to avoid UInt32/UInt type issues
    df['dayofweek'] = df.index.dayofweek.astype('int64')
    df['quarter'] = df.index.quarter.astype('int64')
    df['month'] = df.index.month.astype('int64')
    df['year'] = df.index.year.astype('int64')
    df['dayofyear'] = df.index.dayofyear.astype('int64')
    df['dayofmonth'] = df.index.day.astype('int64')
    df['weekofyear'] = df.index.isocalendar().week.astype('int64')
    
    # Cyclical encoding for periodic features
    df['dayofweek_sin'] = np.sin(df['dayofweek'] * (2 * np.pi / 7))
    df['dayofweek_cos'] = np.cos(df['dayofweek'] * (2 * np.pi / 7))
    df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
    df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))
    df['quarter_sin'] = np.sin((df['quarter'] - 1) * (2 * np.pi / 4))
    df['quarter_cos'] = np.cos((df['quarter'] - 1) * (2 * np.pi / 4))
    
    return df

def create_lag_features(data: pd.DataFrame, columns: List[str], lag_periods: List[int]) -> pd.DataFrame:
    """
    Create lag features for specified columns.
    
    Args:
        data: DataFrame to add lag features to
        columns: Columns to create lag features for
        lag_periods: List of lag periods
        
    Returns:
        DataFrame with added lag features
    """
    df = data.copy()
    
    for col in columns:
        for lag in lag_periods:
            if col in df.columns:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df

def create_rolling_features(
    data: pd.DataFrame, 
    columns: List[str], 
    windows: List[int],
    functions: List[str] = ['mean', 'std', 'min', 'max']
) -> pd.DataFrame:
    """
    Create rolling window features for specified columns.
    
    Args:
        data: DataFrame to add rolling features to
        columns: Columns to create rolling features for
        windows: List of window sizes
        functions: List of functions to apply to rolling windows
        
    Returns:
        DataFrame with added rolling features
    """
    df = data.copy()
    
    for col in columns:
        for window in windows:
            for func in functions:
                if col in df.columns:
                    if func == 'mean':
                        df[f'{col}_roll_{window}_{func}'] = df[col].rolling(window=window, min_periods=1).mean()
                    elif func == 'std':
                        df[f'{col}_roll_{window}_{func}'] = df[col].rolling(window=window, min_periods=1).std()
                    elif func == 'min':
                        df[f'{col}_roll_{window}_{func}'] = df[col].rolling(window=window, min_periods=1).min()
                    elif func == 'max':
                        df[f'{col}_roll_{window}_{func}'] = df[col].rolling(window=window, min_periods=1).max()
    
    return df

def create_ewm_features(
    data: pd.DataFrame, 
    columns: List[str], 
    alphas: List[float]
) -> pd.DataFrame:
    """
    Create exponentially weighted moving average features.
    
    Args:
        data: DataFrame to add EWM features to
        columns: Columns to create EWM features for
        alphas: List of alpha values for EWM
        
    Returns:
        DataFrame with added EWM features
    """
    df = data.copy()
    
    for col in columns:
        for alpha in alphas:
            if col in df.columns:
                df[f'{col}_ewm_{alpha}'] = df[col].ewm(alpha=alpha, min_periods=1).mean()
    
    return df

if __name__ == "__main__":
    # Example usage
    transformer = DataTransformer(target_frequency=FREQUENCY_DAILY)
    
    # Example 1: Load and align a macro data source
    try:
        data = transformer.load_data_source('gdp')
        aligned_data = transformer.align_to_frequency(data, 'gdp')
        print(f"Original GDP data shape: {data.shape}")
        print(f"Aligned GDP data shape: {aligned_data.shape}")
    except Exception as e:
        print(f"Error loading GDP data: {e}")
    
    # Example 2: Create aligned dataset with multiple sources
    try:
        sources = ['stock_price', 'gdp', 'cpi']
        aligned_dataset = transformer.create_aligned_dataset(
            sources, 
            start_date='2010-01-01', 
            end_date='2022-12-31',
            symbol='sh.600519'  # For stock_price
        )
        print(f"Aligned dataset shape: {aligned_dataset.shape}")
        print(f"Aligned dataset columns: {aligned_dataset.columns.tolist()}")
    except Exception as e:
        print(f"Error creating aligned dataset: {e}")
        
    # Example 3: Prepare model inputs
    try:
        if 'aligned_dataset' in locals() and not aligned_dataset.empty:
            target_column = 'close_stock_price'
            model_inputs = transformer.prepare_model_inputs(
                aligned_dataset,
                target_column=target_column,
                sequence_length=30,
                forecast_horizon=5
            )
            print(f"Model inputs created with {model_inputs['x_train'].shape[0]} training samples")
    except Exception as e:
        print(f"Error preparing model inputs: {e}") 