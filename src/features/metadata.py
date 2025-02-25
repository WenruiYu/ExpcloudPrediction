"""
Metadata system for defining and managing diverse data sources.

This module provides a flexible metadata system for defining data sources with different:
- Time frequencies (daily, weekly, monthly, quarterly, yearly)
- Availability periods (start/end dates)
- Data categories (stock, macro, company, industry)
- Transformation rules for frequency alignment
"""

import importlib
import inspect
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
import pandas as pd
import logging

# Try to import using relative imports if running as a module
try:
    from src.core.config import LOGGING_CONFIG
except ImportError:
    # Fall back to direct imports if running the file directly
    from ..core.config import LOGGING_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

# Define constants for frequencies
FREQUENCY_DAILY = 'daily'
FREQUENCY_WEEKLY = 'weekly'
FREQUENCY_MONTHLY = 'monthly'
FREQUENCY_QUARTERLY = 'quarterly'
FREQUENCY_YEARLY = 'yearly'

VALID_FREQUENCIES = [
    FREQUENCY_DAILY,
    FREQUENCY_WEEKLY,
    FREQUENCY_MONTHLY,
    FREQUENCY_QUARTERLY,
    FREQUENCY_YEARLY
]

# Define constants for data categories
CATEGORY_STOCK = 'stock'
CATEGORY_MACRO = 'macro'
CATEGORY_COMPANY = 'company'
CATEGORY_INDUSTRY = 'industry'

VALID_CATEGORIES = [
    CATEGORY_STOCK,
    CATEGORY_MACRO,
    CATEGORY_COMPANY,
    CATEGORY_INDUSTRY
]

# Sample data sources with enhanced metadata
DATA_SOURCES_METADATA = {
    'stock_price': {
        'category': CATEGORY_STOCK,
        'frequency': FREQUENCY_DAILY,
        'function': 'baostock.query_history_k_data_plus',
        'function_args': {
            'code': '{symbol}',
            'fields': 'date,code,open,high,low,close,volume,amount',
            'frequency': 'd',
            'adjustflag': '3'
        },
        'collector_class': 'src.data_collection.StockDataCollector',
        'filename_pattern': '{symbol}_price_data.csv',
        'availability_start': '2001-01-01',
        'availability_end': None,  # Still ongoing
        'description': "Daily stock price data from BaoStock",
        'required_fields': ['open', 'high', 'low', 'close', 'volume'],
        'date_column': 'date',
        'id_column': 'code'
    },
    'gdp': {
        'category': CATEGORY_MACRO,
        'frequency': FREQUENCY_QUARTERLY,
        'function': 'ak.macro_china_gdp_yearly',
        'function_args': {},
        'collector_class': 'src.macro_data_collection.MacroDataCollector',
        'filename_pattern': 'china_gdp.csv',
        'availability_start': '1990-01-01',
        'availability_end': None,  # Still ongoing
        'description': "China's GDP growth rate",
        'required_fields': ['gdp', 'gdp_yoy'],
        'date_column': 'year',
        'frequency_mapping': {
            FREQUENCY_DAILY: 'ffill',      # Forward fill for daily alignment
            FREQUENCY_MONTHLY: 'ffill'     # Forward fill for monthly alignment
        }
    },
    'cpi': {
        'category': CATEGORY_MACRO,
        'frequency': FREQUENCY_MONTHLY,
        'function': 'ak.macro_china_cpi_monthly',
        'function_args': {},
        'collector_class': 'src.macro_data_collection.MacroDataCollector',
        'filename_pattern': 'china_cpi_monthly.csv',
        'availability_start': '1990-01-01',
        'availability_end': None,  # Still ongoing
        'description': "China's Consumer Price Index",
        'required_fields': ['cpi'],
        'date_column': 'month',
        'frequency_mapping': {
            FREQUENCY_DAILY: 'ffill',        # Forward fill for daily alignment
            FREQUENCY_WEEKLY: 'ffill',       # Forward fill for weekly alignment
            FREQUENCY_QUARTERLY: 'mean'      # Average for quarterly alignment
        }
    },
    'm2': {
        'category': CATEGORY_MACRO,
        'frequency': FREQUENCY_MONTHLY,
        'function': 'ak.macro_china_supply_of_money',
        'function_args': {},
        'collector_class': 'src.macro_data_collection.MacroDataCollector',
        'filename_pattern': 'china_supply_of_money.csv',
        'availability_start': '1990-01-01', 
        'availability_end': None,  # Still ongoing
        'description': "China's M2 money supply",
        'required_fields': ['m2'],
        'date_column': 'month',
        'frequency_mapping': {
            FREQUENCY_DAILY: 'ffill',        # Forward fill for daily alignment
            FREQUENCY_WEEKLY: 'ffill',       # Forward fill for weekly alignment
            FREQUENCY_QUARTERLY: 'last'      # Last value for quarterly alignment
        }
    }
}

class DataSourceMetadata:
    """Class for managing and validating data source metadata."""
    
    def __init__(self, metadata_registry: Dict[str, Dict[str, Any]] = None):
        """
        Initialize with an optional metadata registry.
        
        Args:
            metadata_registry: Dictionary containing source_id to metadata mappings
        """
        self.registry = metadata_registry or {}
        self._validate_registry()
        
    def _validate_registry(self) -> None:
        """Validate all entries in the metadata registry."""
        for source_id, metadata in self.registry.items():
            self._validate_metadata(source_id, metadata)
    
    def _validate_metadata(self, source_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Validate a single metadata entry.
        
        Args:
            source_id: Identifier for the data source
            metadata: Metadata dictionary for the source
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check required fields
        required_fields = [
            'category', 'frequency', 'function', 'filename_pattern',
            'availability_start', 'description', 'date_column'
        ]
        
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required field '{field}' in metadata for source '{source_id}'")
        
        # Validate category
        if metadata['category'] not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{metadata['category']}' for source '{source_id}'. "
                f"Must be one of: {VALID_CATEGORIES}"
            )
        
        # Validate frequency
        if metadata['frequency'] not in VALID_FREQUENCIES:
            raise ValueError(
                f"Invalid frequency '{metadata['frequency']}' for source '{source_id}'. "
                f"Must be one of: {VALID_FREQUENCIES}"
            )
        
        # Validate date formats
        try:
            datetime.strptime(metadata['availability_start'], '%Y-%m-%d')
            if metadata.get('availability_end'):
                datetime.strptime(metadata['availability_end'], '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format in metadata for source '{source_id}': {e}")
        
        # Validate frequency mappings if present
        if 'frequency_mapping' in metadata:
            valid_methods = ['ffill', 'bfill', 'mean', 'sum', 'min', 'max', 'first', 'last']
            for freq, method in metadata['frequency_mapping'].items():
                if freq not in VALID_FREQUENCIES:
                    raise ValueError(
                        f"Invalid target frequency '{freq}' in frequency_mapping for source '{source_id}'. "
                        f"Must be one of: {VALID_FREQUENCIES}"
                    )
                if method not in valid_methods:
                    raise ValueError(
                        f"Invalid mapping method '{method}' in frequency_mapping for source '{source_id}'. "
                        f"Must be one of: {valid_methods}"
                    )
        
        return True
    
    def register_source(self, source_id: str, metadata: Dict[str, Any]) -> None:
        """
        Register a new data source.
        
        Args:
            source_id: Identifier for the data source
            metadata: Metadata dictionary for the source
            
        Returns:
            None
        """
        # Validate the metadata
        self._validate_metadata(source_id, metadata)
        
        # Check if already exists
        if source_id in self.registry:
            logger.warning(f"Overwriting existing metadata for source '{source_id}'")
        
        # Add to registry
        self.registry[source_id] = metadata
        logger.info(f"Registered data source '{source_id}'")
    
    def get_metadata(self, source_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific source.
        
        Args:
            source_id: Identifier for the data source
            
        Returns:
            Metadata dictionary
        """
        if source_id not in self.registry:
            raise KeyError(f"Data source '{source_id}' not found in registry")
        
        return self.registry[source_id]
    
    def list_sources(self, 
                    category: Optional[str] = None, 
                    frequency: Optional[str] = None) -> List[str]:
        """
        List available data sources, optionally filtered by category and frequency.
        
        Args:
            category: Filter by data category
            frequency: Filter by data frequency
            
        Returns:
            List of source IDs
        """
        sources = list(self.registry.keys())
        
        # Apply filters
        if category:
            sources = [s for s in sources if self.registry[s]['category'] == category]
        
        if frequency:
            sources = [s for s in sources if self.registry[s]['frequency'] == frequency]
        
        return sources
    
    def get_collector_class(self, source_id: str) -> Optional[Any]:
        """
        Get the collector class for a data source.
        
        Args:
            source_id: Identifier for the data source
            
        Returns:
            Collector class object
        """
        metadata = self.get_metadata(source_id)
        collector_path = metadata.get('collector_class')
        
        if not collector_path:
            return None
        
        try:
            module_path, class_name = collector_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import collector class for source '{source_id}': {e}")
            return None
    
    def get_function(self, source_id: str) -> Optional[Callable]:
        """
        Get the data collection function for a source.
        
        Args:
            source_id: Identifier for the data source
            
        Returns:
            Function object
        """
        metadata = self.get_metadata(source_id)
        function_path = metadata.get('function')
        
        if not function_path:
            return None
        
        try:
            module_path, func_name = function_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, func_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import function for source '{source_id}': {e}")
            return None
    
    def get_frequency_mapping(self, source_id: str, target_frequency: str) -> Optional[str]:
        """
        Get the frequency mapping method for a source and target frequency.
        
        Args:
            source_id: Identifier for the data source
            target_frequency: Target frequency to map to
            
        Returns:
            Mapping method name or None if not specified
        """
        metadata = self.get_metadata(source_id)
        mappings = metadata.get('frequency_mapping', {})
        
        # If no mapping specified and frequencies match, no mapping needed
        if not mappings and metadata['frequency'] == target_frequency:
            return None
        
        # If no mapping specified for this target frequency, use a default
        if target_frequency not in mappings:
            # Default mapping strategy based on frequencies
            source_freq = metadata['frequency']
            freq_hierarchy = [
                FREQUENCY_YEARLY,
                FREQUENCY_QUARTERLY,
                FREQUENCY_MONTHLY,
                FREQUENCY_WEEKLY,
                FREQUENCY_DAILY
            ]
            
            # Get indices in hierarchy
            source_idx = freq_hierarchy.index(source_freq)
            target_idx = freq_hierarchy.index(target_frequency)
            
            # If target is more granular than source (e.g., monthly to daily)
            if target_idx > source_idx:
                return 'ffill'  # Forward fill is usually appropriate
            # If target is less granular than source (e.g., daily to monthly)
            elif target_idx < source_idx:
                # For price data, last value is often appropriate
                if 'price' in source_id or metadata['category'] == CATEGORY_STOCK:
                    return 'last'
                # For other data, mean is often appropriate
                else:
                    return 'mean'
            else:
                return None  # Same frequency, no mapping needed
        
        return mappings[target_frequency]

# Initialize the metadata registry with the sample data
metadata_registry = DataSourceMetadata(DATA_SOURCES_METADATA)

def get_source_metadata(source_id: str) -> Dict[str, Any]:
    """
    Get metadata for a specific data source.
    
    Args:
        source_id: Identifier for the data source
        
    Returns:
        Metadata dictionary
    """
    return metadata_registry.get_metadata(source_id)

def list_data_sources(category: Optional[str] = None, 
                     frequency: Optional[str] = None) -> List[str]:
    """
    List available data sources, optionally filtered by category and frequency.
    
    Args:
        category: Filter by data category
        frequency: Filter by data frequency
        
    Returns:
        List of source IDs
    """
    return metadata_registry.list_sources(category, frequency)

def register_data_source(source_id: str, metadata: Dict[str, Any]) -> None:
    """
    Register a new data source.
    
    Args:
        source_id: Identifier for the data source
        metadata: Metadata dictionary for the source
        
    Returns:
        None
    """
    metadata_registry.register_source(source_id, metadata)

# Helper functions for working with the metadata registry
def get_sources_by_availability(min_date: str, max_date: Optional[str] = None) -> List[str]:
    """
    Get sources available for a date range.
    
    Args:
        min_date: Minimum date in YYYY-MM-DD format
        max_date: Maximum date in YYYY-MM-DD format (None for no upper limit)
        
    Returns:
        List of source IDs available for the entire date range
    """
    min_date_obj = datetime.strptime(min_date, '%Y-%m-%d')
    max_date_obj = datetime.strptime(max_date, '%Y-%m-%d') if max_date else datetime.now()
    
    available_sources = []
    
    for source_id, metadata in metadata_registry.registry.items():
        source_start = datetime.strptime(metadata['availability_start'], '%Y-%m-%d')
        
        # Check if source ends before our date range
        if metadata.get('availability_end'):
            source_end = datetime.strptime(metadata['availability_end'], '%Y-%m-%d')
            if source_end < min_date_obj:
                continue
        
        # Check if source starts after our date range
        if source_start > max_date_obj:
            continue
        
        # Source is at least partially available in our date range
        available_sources.append(source_id)
    
    return available_sources

def get_common_date_range(source_ids: List[str]) -> tuple:
    """
    Find the common date range for a list of data sources.
    
    Args:
        source_ids: List of source IDs
        
    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    if not source_ids:
        return None, None
    
    # Start with extreme dates
    min_start = '1000-01-01'
    max_end = '9999-12-31'
    
    for source_id in source_ids:
        try:
            metadata = metadata_registry.get_metadata(source_id)
            
            # Update start date (latest of all start dates)
            source_start = metadata['availability_start']
            if source_start > min_start:
                min_start = source_start
            
            # Update end date (earliest of all end dates)
            source_end = metadata.get('availability_end')
            if source_end and source_end < max_end:
                max_end = source_end
                
        except KeyError:
            logger.warning(f"Source '{source_id}' not found in registry")
    
    # If max_end is still the default, return None (no upper limit)
    if max_end == '9999-12-31':
        max_end = None
    
    return min_start, max_end

if __name__ == "__main__":
    # Example usage
    print("Available data sources:")
    for source_id in list_data_sources():
        metadata = get_source_metadata(source_id)
        print(f" - {source_id}: {metadata['description']} ({metadata['frequency']})")
    
    print("\nMacro data sources:")
    for source_id in list_data_sources(category=CATEGORY_MACRO):
        metadata = get_source_metadata(source_id)
        print(f" - {source_id}: {metadata['description']}")
    
    print("\nData sources available from 2010-01-01:")
    for source_id in get_sources_by_availability("2010-01-01"):
        print(f" - {source_id}")
    
    # Example of finding common date range
    sources = ['stock_price', 'gdp', 'cpi']
    start_date, end_date = get_common_date_range(sources)
    print(f"\nCommon date range for {sources}: {start_date} to {end_date}") 