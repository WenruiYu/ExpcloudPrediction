"""
Feature store for managing processed features from different data sources.

This module provides a centralized feature store that:
- Stores processed features from different data sources
- Handles versioning of features
- Provides efficient access to features for model training and inference
- Manages metadata about features
"""

import os
import json
import hashlib
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import logging
import shutil

# Try to import using relative imports if running as a module
try:
    from src.features.metadata import (
        get_source_metadata, list_data_sources,
        VALID_FREQUENCIES, FREQUENCY_DAILY
    )
    from src.features.transformation import (
        DataTransformer, create_time_features, create_lag_features,
        create_rolling_features, create_ewm_features
    )
    from src.core.config import DATA_DIR
except ImportError:
    # Fall back to direct imports if running the file directly
    from ..features.metadata import (
        get_source_metadata, list_data_sources,
        VALID_FREQUENCIES, FREQUENCY_DAILY
    )
    from ..features.transformation import (
        DataTransformer, create_time_features, create_lag_features,
        create_rolling_features, create_ewm_features
    )
    from ..core.config import DATA_DIR

# Configure logging
logger = logging.getLogger(__name__)

# Define feature store base directory
FEATURE_STORE_DIR = DATA_DIR / 'feature_store'
FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Feature metadata file
FEATURE_METADATA_FILE = FEATURE_STORE_DIR / 'feature_metadata.json'


class FeatureStore:
    """Central repository for features with different frequencies."""
    
    def __init__(self, base_dir: Path = FEATURE_STORE_DIR):
        """
        Initialize the feature store.
        
        Args:
            base_dir: Base directory for the feature store
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.feature_registry = self._load_feature_registry()
        
        # Create subdirectories for different frequencies
        for freq in VALID_FREQUENCIES:
            (self.base_dir / freq).mkdir(exist_ok=True)
        
        logger.info(f"Initialized FeatureStore at {self.base_dir}")
        logger.info(f"Found {len(self.feature_registry)} registered features")
    
    def _load_feature_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the feature registry from disk.
        
        Returns:
            Dictionary mapping feature IDs to metadata
        """
        if FEATURE_METADATA_FILE.exists():
            try:
                with open(FEATURE_METADATA_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading feature metadata: {e}")
                return {}
        else:
            logger.info("Feature metadata file not found, creating new registry")
            return {}
    
    def _save_feature_registry(self) -> None:
        """Save the feature registry to disk."""
        try:
            with open(FEATURE_METADATA_FILE, 'w') as f:
                json.dump(self.feature_registry, f, indent=2)
            logger.info(f"Saved feature registry with {len(self.feature_registry)} features")
        except Exception as e:
            logger.error(f"Error saving feature metadata: {e}")
    
    def _get_feature_path(self, feature_id: str, version: Optional[str] = None) -> Path:
        """
        Get the path to a feature file.
        
        Args:
            feature_id: Identifier for the feature
            version: Specific version to retrieve (if None, use latest)
            
        Returns:
            Path object for the feature file
        """
        if feature_id not in self.feature_registry:
            raise KeyError(f"Feature '{feature_id}' not found in registry")
        
        metadata = self.feature_registry[feature_id]
        frequency = metadata['frequency']
        
        # Determine version
        if version is None:
            version = metadata['latest_version']
        
        # Construct file path
        file_path = self.base_dir / frequency / f"{feature_id}_v{version}.h5"
        
        return file_path
    
    def register_feature(
        self, 
        feature_id: str, 
        metadata: Dict[str, Any],
        overwrite: bool = False
    ) -> None:
        """
        Register a new feature in the store.
        
        Args:
            feature_id: Identifier for the feature
            metadata: Metadata dictionary for the feature
            overwrite: Whether to overwrite existing feature metadata
            
        Returns:
            None
        """
        # Validate required fields
        required_fields = ['frequency', 'description', 'source_ids', 'creation_date']
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required field '{field}' in feature metadata")
        
        # Check if frequency is valid
        if metadata['frequency'] not in VALID_FREQUENCIES:
            raise ValueError(f"Invalid frequency '{metadata['frequency']}' for feature '{feature_id}'")
        
        # Check if already exists
        if feature_id in self.feature_registry and not overwrite:
            raise ValueError(f"Feature '{feature_id}' already exists in registry")
        
        # Set creation timestamp if not provided
        if 'creation_date' not in metadata:
            metadata['creation_date'] = datetime.now().isoformat()
        
        # Initialize version
        if 'latest_version' not in metadata:
            metadata['latest_version'] = '1.0.0'
        
        # Add to registry
        self.feature_registry[feature_id] = metadata
        self._save_feature_registry()
        
        logger.info(f"Registered feature '{feature_id}' with frequency '{metadata['frequency']}'")
    
    def update_feature_metadata(
        self, 
        feature_id: str, 
        metadata_updates: Dict[str, Any]
    ) -> None:
        """
        Update metadata for an existing feature.
        
        Args:
            feature_id: Identifier for the feature
            metadata_updates: Dictionary with metadata fields to update
            
        Returns:
            None
        """
        if feature_id not in self.feature_registry:
            raise KeyError(f"Feature '{feature_id}' not found in registry")
        
        # Update metadata
        self.feature_registry[feature_id].update(metadata_updates)
        self._save_feature_registry()
        
        logger.info(f"Updated metadata for feature '{feature_id}'")
    
    def store_feature(
        self, 
        feature_id: str, 
        data: pd.DataFrame, 
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store a feature in the feature store.
        
        Args:
            feature_id: Identifier for the feature
            data: DataFrame containing the feature data
            version: Version string for the feature (if None, increment latest)
            metadata: Additional metadata to update
            
        Returns:
            None
        """
        if data.empty:
            raise ValueError("Cannot store empty feature data")
        
        # If feature doesn't exist, register it first
        if feature_id not in self.feature_registry:
            if metadata is None:
                raise ValueError(f"Feature '{feature_id}' not found and no metadata provided for registration")
            
            self.register_feature(feature_id, metadata)
        
        # Get metadata
        feature_metadata = self.feature_registry[feature_id]
        
        # Determine version
        if version is None:
            # Parse current version
            current_version = feature_metadata['latest_version']
            major, minor, patch = map(int, current_version.split('.'))
            
            # Increment patch version
            new_version = f"{major}.{minor}.{patch + 1}"
            version = new_version
            
            # Update latest version in metadata
            feature_metadata['latest_version'] = new_version
            self.feature_registry[feature_id] = feature_metadata
            self._save_feature_registry()
        
        # Ensure data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(f"Feature data index could not be converted to datetime: {e}")
        
        # Get storage path
        frequency = feature_metadata['frequency']
        file_path = self.base_dir / frequency / f"{feature_id}_v{version}.h5"
        
        # Create parent directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Store data in HDF5 format
        try:
            # Store feature data using HDF5 format
            with pd.HDFStore(file_path, mode='w') as store:
                store.put('data', data, format='table')
            
            # Update file info in metadata
            feature_metadata['file_info'] = {
                'path': str(file_path),
                'version': version,
                'rows': len(data),
                'columns': len(data.columns),
                'start_date': data.index.min().isoformat(),
                'end_date': data.index.max().isoformat(),
                'stored_date': datetime.now().isoformat()
            }
            
            # Update any additional metadata
            if metadata:
                feature_metadata.update(metadata)
            
            # Save updated metadata
            self.feature_registry[feature_id] = feature_metadata
            self._save_feature_registry()
            
            logger.info(f"Stored feature '{feature_id}' version {version} with {len(data)} rows in {file_path}")
            
        except Exception as e:
            logger.error(f"Error storing feature '{feature_id}': {e}", exc_info=True)
            if file_path.exists():
                try:
                    file_path.unlink()  # Remove partial file
                except Exception:
                    pass
            raise
    
    def get_feature(
        self, 
        feature_id: str, 
        version: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve a feature from the store.
        
        Args:
            feature_id: Identifier for the feature
            version: Specific version to retrieve (if None, use latest)
            start_date: Start date for filtering (YYYY-MM-DD)
            end_date: End date for filtering (YYYY-MM-DD)
            
        Returns:
            DataFrame containing the feature data
        """
        # Get file path
        try:
            file_path = self._get_feature_path(feature_id, version)
        except KeyError as e:
            logger.error(f"Feature not found: {e}")
            return pd.DataFrame()
        
        if not file_path.exists():
            logger.error(f"Feature file not found at {file_path}")
            return pd.DataFrame()
        
        # Load data
        try:
            with pd.HDFStore(file_path, mode='r') as store:
                data = store['data']
            
            # Filter by date if specified
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            logger.info(f"Retrieved feature '{feature_id}' with {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving feature '{feature_id}': {e}", exc_info=True)
            return pd.DataFrame()
    
    def list_features(
        self, 
        frequency: Optional[str] = None,
        source_id: Optional[str] = None
    ) -> List[str]:
        """
        List available features, optionally filtered by frequency or source.
        
        Args:
            frequency: Filter by data frequency
            source_id: Filter by source ID
            
        Returns:
            List of feature IDs
        """
        features = list(self.feature_registry.keys())
        
        # Apply filters
        if frequency:
            features = [
                f for f in features 
                if self.feature_registry[f]['frequency'] == frequency
            ]
        
        if source_id:
            features = [
                f for f in features 
                if source_id in self.feature_registry[f].get('source_ids', [])
            ]
        
        return features
    
    def get_feature_metadata(self, feature_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific feature.
        
        Args:
            feature_id: Identifier for the feature
            
        Returns:
            Metadata dictionary
        """
        if feature_id not in self.feature_registry:
            raise KeyError(f"Feature '{feature_id}' not found in registry")
        
        return self.feature_registry[feature_id].copy()
    
    def delete_feature(self, feature_id: str, delete_files: bool = True) -> None:
        """
        Delete a feature from the store.
        
        Args:
            feature_id: Identifier for the feature
            delete_files: Whether to delete associated files
            
        Returns:
            None
        """
        if feature_id not in self.feature_registry:
            raise KeyError(f"Feature '{feature_id}' not found in registry")
        
        if delete_files:
            # Delete all versions of the feature
            feature_metadata = self.feature_registry[feature_id]
            frequency = feature_metadata['frequency']
            
            # Get directory for this frequency
            freq_dir = self.base_dir / frequency
            
            # Delete all files for this feature
            for file_path in freq_dir.glob(f"{feature_id}_v*.h5"):
                try:
                    file_path.unlink()
                    logger.info(f"Deleted feature file {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting feature file {file_path}: {e}")
        
        # Remove from registry
        del self.feature_registry[feature_id]
        self._save_feature_registry()
        
        logger.info(f"Deleted feature '{feature_id}' from registry")
    
    def create_feature_set(
        self, 
        feature_ids: List[str],
        target_date: Optional[str] = None,
        lookback_window: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create a set of features for a specific prediction target.
        
        Args:
            feature_ids: List of feature IDs to include
            target_date: Target date for prediction (if None, use all available dates)
            lookback_window: Number of days to look back from target_date
            start_date: Start date for feature set (overrides lookback_window)
            end_date: End date for feature set (overrides target_date)
            
        Returns:
            DataFrame with combined features
        """
        if not feature_ids:
            logger.warning("No features specified for feature set")
            return pd.DataFrame()
        
        # Determine date range
        if target_date and lookback_window:
            end_date = target_date
            target_dt = pd.to_datetime(target_date)
            start_date = (target_dt - pd.Timedelta(days=lookback_window)).strftime('%Y-%m-%d')
        
        # Load and merge features
        feature_dfs = []
        for feature_id in feature_ids:
            try:
                feature_data = self.get_feature(feature_id, start_date=start_date, end_date=end_date)
                if feature_data.empty:
                    logger.warning(f"Empty data for feature '{feature_id}', skipping")
                    continue
                feature_dfs.append(feature_data)
            except Exception as e:
                logger.error(f"Error loading feature '{feature_id}': {e}")
        
        if not feature_dfs:
            logger.warning("No valid features found for feature set")
            return pd.DataFrame()
        
        # Merge all features
        result = feature_dfs[0].copy()
        for df in feature_dfs[1:]:
            result = result.join(df, how='outer')
        
        # Fill missing values
        result = result.ffill().bfill()
        
        logger.info(f"Created feature set with {len(result)} rows and {len(result.columns)} columns")
        return result

# FeatureGenerator class for creating features from raw data
class FeatureGenerator:
    """Generate features from raw data sources and store them in the feature store."""
    
    def __init__(self, feature_store: FeatureStore = None):
        """
        Initialize the feature generator.
        
        Args:
            feature_store: FeatureStore instance to use (creates new one if None)
        """
        self.feature_store = feature_store or FeatureStore()
        self.transformer = DataTransformer()
        logger.info("Initialized FeatureGenerator")
    
    def generate_aligned_feature(
        self,
        feature_id: str,
        source_ids: List[str],
        frequency: str = FREQUENCY_DAILY,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        description: str = "",
        **kwargs
    ) -> None:
        """
        Generate and store an aligned feature from multiple data sources.
        
        Args:
            feature_id: Identifier for the feature
            source_ids: List of source IDs to include
            frequency: Frequency for the feature
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            description: Description of the feature
            **kwargs: Additional parameters for data loading
            
        Returns:
            None
        """
        # Set transformer target frequency
        self.transformer.target_frequency = frequency
        
        # Create aligned dataset
        aligned_data = self.transformer.create_aligned_dataset(
            source_ids,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
        
        if aligned_data.empty:
            logger.warning(f"Empty aligned data for feature '{feature_id}', skipping")
            return
        
        # Create metadata
        metadata = {
            'frequency': frequency,
            'source_ids': source_ids,
            'description': description or f"Aligned feature from {', '.join(source_ids)}",
            'creation_date': datetime.now().isoformat(),
            'parameters': {
                'start_date': start_date,
                'end_date': end_date,
                **kwargs
            }
        }
        
        # Store feature
        self.feature_store.store_feature(feature_id, aligned_data, metadata=metadata)
        
        logger.info(f"Generated and stored aligned feature '{feature_id}' with {len(aligned_data)} rows")
    
    def generate_time_features(
        self,
        feature_id: str,
        base_feature_id: str,
        frequency: str = FREQUENCY_DAILY,
        description: str = "",
    ) -> None:
        """
        Generate and store time-based features from a base feature.
        
        Args:
            feature_id: Identifier for the new feature
            base_feature_id: Identifier for the base feature
            frequency: Frequency for the feature
            description: Description of the feature
            
        Returns:
            None
        """
        # Get base feature
        base_data = self.feature_store.get_feature(base_feature_id)
        
        if base_data.empty:
            logger.warning(f"Empty base feature '{base_feature_id}' for time features, skipping")
            return
        
        # Create time features
        feature_data = create_time_features(base_data)
        
        # Get base feature metadata
        base_metadata = self.feature_store.get_feature_metadata(base_feature_id)
        
        # Create metadata
        metadata = {
            'frequency': frequency,
            'source_ids': base_metadata.get('source_ids', []),
            'description': description or f"Time features derived from {base_feature_id}",
            'creation_date': datetime.now().isoformat(),
            'derived_from': base_feature_id,
            'feature_type': 'time_features'
        }
        
        # Store feature
        self.feature_store.store_feature(feature_id, feature_data, metadata=metadata)
        
        logger.info(f"Generated and stored time features '{feature_id}' with {len(feature_data)} rows")
    
    def generate_lag_features(
        self,
        feature_id: str,
        base_feature_id: str,
        columns: List[str],
        lag_periods: List[int],
        frequency: str = FREQUENCY_DAILY,
        description: str = "",
    ) -> None:
        """
        Generate and store lag features from a base feature.
        
        Args:
            feature_id: Identifier for the new feature
            base_feature_id: Identifier for the base feature
            columns: Columns to create lag features for
            lag_periods: List of lag periods
            frequency: Frequency for the feature
            description: Description of the feature
            
        Returns:
            None
        """
        # Get base feature
        base_data = self.feature_store.get_feature(base_feature_id)
        
        if base_data.empty:
            logger.warning(f"Empty base feature '{base_feature_id}' for lag features, skipping")
            return
        
        # Create lag features
        feature_data = create_lag_features(base_data, columns, lag_periods)
        
        # Get base feature metadata
        base_metadata = self.feature_store.get_feature_metadata(base_feature_id)
        
        # Create metadata
        metadata = {
            'frequency': frequency,
            'source_ids': base_metadata.get('source_ids', []),
            'description': description or f"Lag features derived from {base_feature_id}",
            'creation_date': datetime.now().isoformat(),
            'derived_from': base_feature_id,
            'feature_type': 'lag_features',
            'parameters': {
                'columns': columns,
                'lag_periods': lag_periods
            }
        }
        
        # Store feature
        self.feature_store.store_feature(feature_id, feature_data, metadata=metadata)
        
        logger.info(f"Generated and stored lag features '{feature_id}' with {len(feature_data)} rows")
    
    def generate_rolling_features(
        self,
        feature_id: str,
        base_feature_id: str,
        columns: List[str],
        windows: List[int],
        functions: List[str] = ['mean', 'std', 'min', 'max'],
        frequency: str = FREQUENCY_DAILY,
        description: str = "",
    ) -> None:
        """
        Generate and store rolling window features from a base feature.
        
        Args:
            feature_id: Identifier for the new feature
            base_feature_id: Identifier for the base feature
            columns: Columns to create rolling features for
            windows: List of window sizes
            functions: List of functions to apply to rolling windows
            frequency: Frequency for the feature
            description: Description of the feature
            
        Returns:
            None
        """
        # Get base feature
        base_data = self.feature_store.get_feature(base_feature_id)
        
        if base_data.empty:
            logger.warning(f"Empty base feature '{base_feature_id}' for rolling features, skipping")
            return
        
        # Create rolling features
        feature_data = create_rolling_features(base_data, columns, windows, functions)
        
        # Get base feature metadata
        base_metadata = self.feature_store.get_feature_metadata(base_feature_id)
        
        # Create metadata
        metadata = {
            'frequency': frequency,
            'source_ids': base_metadata.get('source_ids', []),
            'description': description or f"Rolling features derived from {base_feature_id}",
            'creation_date': datetime.now().isoformat(),
            'derived_from': base_feature_id,
            'feature_type': 'rolling_features',
            'parameters': {
                'columns': columns,
                'windows': windows,
                'functions': functions
            }
        }
        
        # Store feature
        self.feature_store.store_feature(feature_id, feature_data, metadata=metadata)
        
        logger.info(f"Generated and stored rolling features '{feature_id}' with {len(feature_data)} rows")

if __name__ == "__main__":
    # Example usage
    feature_store = FeatureStore()
    generator = FeatureGenerator(feature_store)
    
    # Example 1: List registered features
    try:
        features = feature_store.list_features()
        print(f"Registered features: {features}")
    except Exception as e:
        print(f"Error listing features: {e}")
    
    # Example 2: Generate aligned feature from multiple sources
    try:
        generator.generate_aligned_feature(
            feature_id="moutai_with_macro",
            source_ids=['stock_price', 'gdp', 'cpi'],
            start_date='2010-01-01',
            end_date='2022-12-31',
            description="Moutai stock with macroeconomic indicators",
            symbol='sh.600519'
        )
    except Exception as e:
        print(f"Error generating aligned feature: {e}")
    
    # Example 3: Generate time features
    try:
        if "moutai_with_macro" in feature_store.list_features():
            generator.generate_time_features(
                feature_id="moutai_time_features",
                base_feature_id="moutai_with_macro",
                description="Time-based features for Moutai stock"
            )
    except Exception as e:
        print(f"Error generating time features: {e}") 