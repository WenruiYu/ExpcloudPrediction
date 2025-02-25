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
        
        # Create a mapping from frequency to pandas frequency string
        self.freq_map = {
            FREQUENCY_DAILY: 'D',
            FREQUENCY_WEEKLY: 'W',
            FREQUENCY_MONTHLY: 'M',
            FREQUENCY_QUARTERLY: 'Q',
            FREQUENCY_YEARLY: 'Y'
        }
        
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
        Load a data source into the registry.
        
        Args:
            source_id: Identifier for the data source
            **kwargs: Additional parameters for file path construction
            
        Returns:
            DataFrame containing the data
        """
        # Check if already loaded
        if source_id in self.data_registry:
            logger.info(f"Using cached data for source '{source_id}'")
            return self.data_registry[source_id]
        
        # Get metadata
        metadata = get_source_metadata(source_id)
        
        # Get file path
        file_path = self._get_data_path(source_id, **kwargs)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        
        # Load data
        try:
            # Determine date column
            date_column = metadata.get('date_column')
            
            # Read CSV
            data = pd.read_csv(file_path)
            
            # Convert date column to datetime and set as index
            if date_column and date_column in data.columns:
                data[date_column] = pd.to_datetime(data[date_column])
                data.set_index(date_column, inplace=True)
            elif date_column:
                logger.warning(f"Date column '{date_column}' not found in {file_path}")
            
            # Basic data validation
            if data.empty:
                logger.warning(f"Empty dataset loaded from {file_path}")
                return pd.DataFrame()
            
            # Check required fields
            required_fields = metadata.get('required_fields', [])
            missing_fields = [field for field in required_fields if field not in data.columns]
            if missing_fields:
                logger.warning(f"Missing required fields in {file_path}: {missing_fields}")
            
            # Store in registry
            self.data_registry[source_id] = data
            logger.info(f"Loaded {len(data)} rows for source '{source_id}'")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data for source '{source_id}': {e}", exc_info=True)
            return pd.DataFrame()
    
    def align_to_frequency(self, data: pd.DataFrame, source_id: str) -> pd.DataFrame:
        """
        Align data to target frequency based on metadata rules.
        
        Args:
            data: DataFrame to align
            source_id: Identifier for the data source
            
        Returns:
            Aligned DataFrame
        """
        if data.empty:
            return pd.DataFrame()
        
        metadata = get_source_metadata(source_id)
        source_freq = metadata['frequency']
        
        # If frequencies match, no resampling needed
        if source_freq == self.target_frequency:
            logger.info(f"No frequency alignment needed for source '{source_id}'")
            return data
        
        logger.info(f"Aligning {source_id} from {source_freq} to {self.target_frequency}")
        
        # Get resampling method
        from src.features.metadata import metadata_registry
        method = metadata_registry.get_frequency_mapping(source_id, self.target_frequency)
        
        if not method:
            logger.warning(f"No mapping method specified for {source_id} to {self.target_frequency}")
            method = 'ffill'  # Default to forward fill
        
        # Convert frequencies to pandas frequency strings
        pd_target_freq = self.freq_map[self.target_frequency]
        
        # Handle upsampling (e.g., monthly to daily)
        if VALID_FREQUENCIES.index(self.target_frequency) > VALID_FREQUENCIES.index(source_freq):
            logger.info(f"Upsampling {source_id} data to {self.target_frequency}")
            
            # Create date range for the index
            start_date = data.index.min()
            end_date = data.index.max()
            
            # Create a new index with the target frequency
            new_index = pd.date_range(start=start_date, end=end_date, freq=pd_target_freq)
            
            # Reindex the data
            resampled = data.reindex(new_index)
            
            # Apply the specified method to fill gaps
            if method == 'ffill':
                resampled = resampled.ffill()
            elif method == 'bfill':
                resampled = resampled.bfill()
            else:
                logger.warning(f"Unsupported upsampling method '{method}' for source '{source_id}'")
                resampled = resampled.ffill()
                
        # Handle downsampling (e.g., daily to monthly)
        else:
            logger.info(f"Downsampling {source_id} data to {self.target_frequency}")
            
            if method == 'first':
                resampled = data.resample(pd_target_freq).first()
            elif method == 'last':
                resampled = data.resample(pd_target_freq).last()
            elif method == 'mean':
                resampled = data.resample(pd_target_freq).mean()
            elif method == 'sum':
                resampled = data.resample(pd_target_freq).sum()
            elif method == 'min':
                resampled = data.resample(pd_target_freq).min()
            elif method == 'max':
                resampled = data.resample(pd_target_freq).max()
            else:
                logger.warning(f"Unsupported downsampling method '{method}' for source '{source_id}'")
                resampled = data.resample(pd_target_freq).last()
        
        logger.info(f"Aligned {len(data)} rows to {len(resampled)} rows for source '{source_id}'")
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
        target_column: str, 
        feature_columns: Optional[List[str]] = None,
        sequence_length: int = 30,
        forecast_horizon: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        normalize: bool = True
    ) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
        """
        Prepare model inputs from aligned data.
        
        Args:
            aligned_data: DataFrame with aligned data sources
            target_column: Column to use as prediction target
            feature_columns: Columns to use as features (if None, use all except target)
            sequence_length: Number of time steps to use for sequence inputs
            forecast_horizon: Number of time steps to forecast
            train_ratio: Proportion of data to use for training
            val_ratio: Proportion of data to use for validation
            test_ratio: Proportion of data to use for testing
            normalize: Whether to normalize data
            
        Returns:
            Dictionary with model inputs
        """
        if aligned_data.empty:
            logger.warning("Empty aligned dataset, cannot prepare model inputs")
            return {}
        
        if target_column not in aligned_data.columns:
            logger.error(f"Target column '{target_column}' not found in aligned dataset")
            return {}
        
        logger.info(f"Preparing model inputs with sequence length {sequence_length} and forecast horizon {forecast_horizon}")
        
        # Select feature columns
        if feature_columns is None:
            feature_columns = [col for col in aligned_data.columns if col != target_column]
        else:
            # Validate feature columns
            missing_cols = [col for col in feature_columns if col not in aligned_data.columns]
            if missing_cols:
                logger.warning(f"Missing feature columns: {missing_cols}")
                feature_columns = [col for col in feature_columns if col in aligned_data.columns]
        
        # Prepare data
        data = aligned_data.copy()
        
        # Shift target for forecasting
        if forecast_horizon > 0:
            data[f'{target_column}_target'] = data[target_column].shift(-forecast_horizon)
            data = data.dropna(subset=[f'{target_column}_target'])
        else:
            data[f'{target_column}_target'] = data[target_column]
        
        # Normalize data
        if normalize:
            scaler_dict = {}
            for col in feature_columns + [f'{target_column}_target']:
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
            y_seq = data[f'{target_column}_target'].iloc[i+sequence_length-1]
            
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
            'target_column': target_column,
            'scaler_dict': scaler_dict,
            'sequence_length': sequence_length,
            'forecast_horizon': forecast_horizon
        }
        
        logger.info(f"Created model inputs with {len(train_indices)} training samples, "
                   f"{len(val_indices)} validation samples, and {len(test_indices)} test samples")
        
        return result

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
    
    # Extract datetime components
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    
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