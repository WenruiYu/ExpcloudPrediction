"""
Data integration module to connect the new feature systems with existing data workflows.

This module serves as an adapter between:
1. The existing data collection classes (StockDataCollector, MacroDataCollector)
2. The new metadata registry, data transformation, and feature store systems

It provides utilities for:
- Converting existing data to the format required by the feature store
- Registering data sources from existing collections to the metadata registry
- Creating a unified workflow from data collection to feature generation
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging

# Try to import using relative imports if running as a module
try:
    from src.core.config import (
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, MACRO_DATA_DIR,
        DEFAULT_TICKER_BS, START_DATE, END_DATE
    )
    from src.data.stock_collection import StockDataCollector
    from src.data.macro_collection import MacroDataCollector
    from src.features.metadata import (
        register_data_source, get_source_metadata, list_data_sources,
        FREQUENCY_DAILY, FREQUENCY_MONTHLY, FREQUENCY_QUARTERLY, FREQUENCY_YEARLY,
        CATEGORY_STOCK, CATEGORY_MACRO
    )
    from src.features.transformation import DataTransformer
    from src.features.store import FeatureStore, FeatureGenerator
except ImportError:
    # Fall back to direct imports if running the file directly
    from ..core.config import (
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, MACRO_DATA_DIR,
        DEFAULT_TICKER_BS, START_DATE, END_DATE
    )
    from ..data.stock_collection import StockDataCollector
    from ..data.macro_collection import MacroDataCollector
    from ..features.metadata import (
        register_data_source, get_source_metadata, list_data_sources,
        FREQUENCY_DAILY, FREQUENCY_MONTHLY, FREQUENCY_QUARTERLY, FREQUENCY_YEARLY,
        CATEGORY_STOCK, CATEGORY_MACRO
    )
    from ..features.transformation import DataTransformer
    from ..features.store import FeatureStore, FeatureGenerator

# Configure logging
logger = logging.getLogger(__name__)

class DataIntegrator:
    """Class for integrating existing data collection with the new feature system."""
    
    def __init__(self):
        """Initialize the data integrator."""
        self.feature_store = FeatureStore()
        self.feature_generator = FeatureGenerator(self.feature_store)
        self.transformer = DataTransformer()
        
        # Ensure stock data sources are registered
        self._register_stock_data_sources()
        
        # Ensure macro data sources are registered
        self._register_macro_data_sources()
        
        logger.info("Initialized DataIntegrator")
    
    def _register_stock_data_sources(self) -> None:
        """Register stock data sources in the metadata registry."""
        try:
            # Basic stock price data
            register_data_source('stock_price', {
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
                'filename_pattern': '{symbol}_processed_stock_data.csv',
                'availability_start': '2001-01-01',
                'availability_end': None,  # Still ongoing
                'description': "Daily stock price data with technical indicators",
                'required_fields': ['open', 'high', 'low', 'close', 'volume'],
                'date_column': 'date',
                'id_column': 'code'
            })
            
            logger.info("Registered stock data sources")
        except Exception as e:
            logger.error(f"Error registering stock data sources: {e}")
    
    def _register_macro_data_sources(self) -> None:
        """Register macro data sources in the metadata registry."""
        try:
            # GDP data
            register_data_source('gdp', {
                'category': CATEGORY_MACRO,
                'frequency': FREQUENCY_QUARTERLY,
                'function': 'ak.macro_china_gdp_yearly',
                'function_args': {},
                'collector_class': 'src.macro_data_collection.MacroDataCollector',
                'filename_pattern': 'china_gdp_yearly.csv',
                'availability_start': '1990-01-01',
                'availability_end': None,  # Still ongoing
                'description': "China's GDP growth rate",
                'required_fields': ['gdp', 'gdp_yoy'],
                'date_column': 'year',
                'frequency_mapping': {
                    FREQUENCY_DAILY: 'ffill',  # Forward fill for daily alignment
                    FREQUENCY_MONTHLY: 'ffill'  # Forward fill for monthly alignment
                }
            })
            
            # CPI data
            register_data_source('cpi', {
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
                    FREQUENCY_DAILY: 'ffill',  # Forward fill for daily alignment
                    FREQUENCY_QUARTERLY: 'mean'  # Average for quarterly alignment
                }
            })
            
            # M2 supply data
            register_data_source('m2', {
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
                    FREQUENCY_DAILY: 'ffill',  # Forward fill for daily alignment
                    FREQUENCY_QUARTERLY: 'last'  # Last value for quarterly alignment
                }
            })
            
            logger.info("Registered macro data sources")
        except Exception as e:
            logger.error(f"Error registering macro data sources: {e}")
    
    def collect_and_register_stock_data(
        self,
        symbol: str = DEFAULT_TICKER_BS,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Collect stock data and register it in the feature store.
        
        Args:
            symbol: Stock symbol to collect
            start_date: Start date for data collection
            end_date: End date for data collection
            force_refresh: Whether to force refresh data
            
        Returns:
            Processed DataFrame or None on failure
        """
        logger.info(f"Collecting and registering stock data for {symbol}")
        
        try:
            # Collect data using existing collector
            collector = StockDataCollector(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # Get processed data
            data = collector.get_data(force_refresh=force_refresh)
            
            if data is None or data.empty:
                logger.error(f"Failed to collect stock data for {symbol}")
                return None
            
            # Create feature ID
            feature_id = f"stock_{symbol.replace('.', '_')}"
            
            # Store in feature store
            self.feature_store.store_feature(
                feature_id=feature_id,
                data=data,
                metadata={
                    'frequency': FREQUENCY_DAILY,
                    'source_ids': ['stock_price'],
                    'description': f"Stock data for {symbol}",
                    'creation_date': datetime.now().isoformat(),
                    'parameters': {
                        'symbol': symbol,
                        'start_date': start_date,
                        'end_date': end_date
                    }
                }
            )
            
            logger.info(f"Registered stock data as feature '{feature_id}'")
            
            return data
            
        except Exception as e:
            logger.error(f"Error collecting and registering stock data: {e}", exc_info=True)
            return None
    
    def collect_and_register_macro_data(
        self,
        sources: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Collect macroeconomic data and register it in the feature store.
        
        Args:
            sources: List of macro data sources to collect
            force_refresh: Whether to force refresh data
            
        Returns:
            Dictionary of source names to their DataFrames
        """
        logger.info(f"Collecting and registering macro data for sources: {sources or 'all'}")
        
        try:
            # Collect data using existing collector
            collector = MacroDataCollector(sources=sources)
            results = collector.fetch_all_sources(overwrite=force_refresh)
            
            if not results:
                logger.error("Failed to collect any macro data")
                return {}
            
            # Register each macro data source in the feature store
            for source_id, data in results.items():
                if data is None or data.empty:
                    logger.warning(f"Empty data for source '{source_id}', skipping")
                    continue
                
                # Create feature ID
                feature_id = f"macro_{source_id}"
                
                # Get source metadata
                try:
                    source_metadata = get_source_metadata(source_id)
                    frequency = source_metadata['frequency']
                except Exception:
                    logger.warning(f"No metadata for source '{source_id}', assuming monthly")
                    frequency = FREQUENCY_MONTHLY
                
                # Store in feature store
                self.feature_store.store_feature(
                    feature_id=feature_id,
                    data=data,
                    metadata={
                        'frequency': frequency,
                        'source_ids': [source_id],
                        'description': f"Macro data for {source_id}",
                        'creation_date': datetime.now().isoformat(),
                        'parameters': {
                            'force_refresh': force_refresh
                        }
                    }
                )
                
                logger.info(f"Registered macro data as feature '{feature_id}'")
            
            return results
            
        except Exception as e:
            logger.error(f"Error collecting and registering macro data: {e}", exc_info=True)
            return {}
    
    def create_aligned_features(
        self,
        feature_id: str,
        symbol: str = DEFAULT_TICKER_BS,
        macro_sources: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = FREQUENCY_DAILY,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Create aligned features from stock and macro data.
        
        Args:
            feature_id: Identifier for the new feature
            symbol: Stock symbol to include
            macro_sources: List of macro sources to include
            start_date: Start date for data
            end_date: End date for data
            frequency: Target frequency for alignment
            force_refresh: Whether to force refresh data
            
        Returns:
            Aligned DataFrame or None on failure
        """
        logger.info(f"Creating aligned features for {symbol} with {macro_sources}")
        
        try:
            # Collect stock data if needed
            stock_feature_id = f"stock_{symbol.replace('.', '_')}"
            if force_refresh or stock_feature_id not in self.feature_store.list_features():
                stock_data = self.collect_and_register_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    force_refresh=force_refresh
                )
                if stock_data is None:
                    logger.error(f"Failed to collect stock data for {symbol}")
                    return None
            
            # Collect macro data if needed
            macro_data_available = False
            if macro_sources:
                if force_refresh or not all(f"macro_{s}" in self.feature_store.list_features() for s in macro_sources):
                    macro_results = self.collect_and_register_macro_data(
                        sources=macro_sources,
                        force_refresh=force_refresh
                    )
                    if macro_results and any(data is not None for data in macro_results.values()):
                        macro_data_available = True
                        # Filter out failed sources
                        macro_sources = [s for s, data in macro_results.items() if data is not None]
                        if not macro_sources:
                            logger.warning("All macro data collection failed. Using stock data only.")
                else:
                    macro_data_available = True
            
            # Prepare source IDs
            source_ids = ['stock_price']
            if macro_sources and macro_data_available:
                source_ids.extend(macro_sources)
            
            # Generate aligned feature
            self.feature_generator.generate_aligned_feature(
                feature_id=feature_id,
                source_ids=source_ids,
                frequency=frequency,
                start_date=start_date,
                end_date=end_date,
                description=f"Aligned feature for {symbol}" + 
                            (f" with macro data ({', '.join(macro_sources)})" if macro_sources and macro_data_available else " without macro data"),
                symbol=symbol
            )
            
            # Get the feature
            aligned_data = self.feature_store.get_feature(feature_id)
            
            if aligned_data is not None:
                logger.info(f"Created aligned feature '{feature_id}' with {len(aligned_data)} rows")
            else:
                logger.error(f"Failed to create aligned feature '{feature_id}'")
            
            return aligned_data
            
        except Exception as e:
            logger.error(f"Error creating aligned features: {e}", exc_info=True)
            return None
    
    def generate_advanced_features(
        self,
        base_feature_id: str,
        include_time_features: bool = True,
        include_lag_features: bool = True,
        include_rolling_features: bool = True,
        price_columns: Optional[List[str]] = None,
        lag_periods: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Generate advanced features from a base feature.
        
        Args:
            base_feature_id: Identifier for the base feature
            include_time_features: Whether to include time features
            include_lag_features: Whether to include lag features
            include_rolling_features: Whether to include rolling features
            price_columns: Columns to use for lag and rolling features
            lag_periods: Lag periods to use
            rolling_windows: Rolling windows to use
            
        Returns:
            Dictionary of feature names to DataFrames
        """
        logger.info(f"Generating advanced features from '{base_feature_id}'")
        
        features = {}
        
        try:
            # Get base feature metadata
            base_metadata = self.feature_store.get_feature_metadata(base_feature_id)
            frequency = base_metadata['frequency']
            
            # Set default parameters
            if price_columns is None:
                # Look for typical price columns
                base_data = self.feature_store.get_feature(base_feature_id)
                if base_data is None or base_data.empty:
                    logger.error(f"Empty base feature '{base_feature_id}'")
                    return None
                
                # Try to find price columns
                potential_cols = [
                    col for col in base_data.columns 
                    if any(price_term in col.lower() for price_term in 
                         ['close', 'open', 'high', 'low', 'price', 'volume'])
                ]
                
                price_columns = potential_cols if potential_cols else base_data.columns[:5]
            
            if lag_periods is None:
                # Default lag periods based on frequency
                if frequency == FREQUENCY_DAILY:
                    lag_periods = [1, 5, 10, 20, 60]
                elif frequency == FREQUENCY_WEEKLY:
                    lag_periods = [1, 4, 8, 12, 24]
                elif frequency == FREQUENCY_MONTHLY:
                    lag_periods = [1, 3, 6, 12, 24]
                else:
                    lag_periods = [1, 2, 4, 8]
            
            if rolling_windows is None:
                # Default rolling windows based on frequency
                if frequency == FREQUENCY_DAILY:
                    rolling_windows = [5, 10, 20, 60]
                elif frequency == FREQUENCY_WEEKLY:
                    rolling_windows = [4, 8, 12, 24]
                elif frequency == FREQUENCY_MONTHLY:
                    rolling_windows = [3, 6, 12, 24]
                else:
                    rolling_windows = [2, 4, 8]
            
            # Generate time features
            if include_time_features:
                time_feature_id = f"{base_feature_id}_time"
                self.feature_generator.generate_time_features(
                    feature_id=time_feature_id,
                    base_feature_id=base_feature_id,
                    frequency=frequency,
                    description=f"Time features derived from {base_feature_id}"
                )
                features['time'] = self.feature_store.get_feature(time_feature_id)
                logger.info(f"Generated time features as '{time_feature_id}'")
            
            # Generate lag features
            if include_lag_features:
                lag_feature_id = f"{base_feature_id}_lag"
                self.feature_generator.generate_lag_features(
                    feature_id=lag_feature_id,
                    base_feature_id=base_feature_id,
                    columns=price_columns,
                    lag_periods=lag_periods,
                    frequency=frequency,
                    description=f"Lag features derived from {base_feature_id}"
                )
                features['lag'] = self.feature_store.get_feature(lag_feature_id)
                logger.info(f"Generated lag features as '{lag_feature_id}'")
            
            # Generate rolling features
            if include_rolling_features:
                rolling_feature_id = f"{base_feature_id}_rolling"
                self.feature_generator.generate_rolling_features(
                    feature_id=rolling_feature_id,
                    base_feature_id=base_feature_id,
                    columns=price_columns,
                    windows=rolling_windows,
                    functions=['mean', 'std', 'min', 'max'],
                    frequency=frequency,
                    description=f"Rolling features derived from {base_feature_id}"
                )
                features['rolling'] = self.feature_store.get_feature(rolling_feature_id)
                logger.info(f"Generated rolling features as '{rolling_feature_id}'")
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating advanced features: {e}", exc_info=True)
            return None
    
    def create_model_inputs(
        self,
        feature_ids: List[str],
        target_column: str,
        sequence_length: int = 30,
        forecast_horizon: int = 5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create model inputs from features.
        
        Args:
            feature_ids: List of feature IDs to include
            target_column: Column to use as prediction target
            sequence_length: Length of input sequences
            forecast_horizon: Number of days to forecast
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary with model inputs
        """
        logger.info(f"Creating model inputs from {feature_ids}")
        
        try:
            # Create feature set
            feature_set = self.feature_store.create_feature_set(
                feature_ids=feature_ids,
                start_date=start_date,
                end_date=end_date
            )
            
            if feature_set.empty:
                logger.error("Empty feature set, cannot create model inputs")
                return None
            
            # Prepare data for model
            model_inputs = self.transformer.prepare_model_inputs(
                aligned_data=feature_set,
                target_column=target_column,
                sequence_length=sequence_length,
                forecast_horizon=forecast_horizon
            )
            
            if not model_inputs:
                logger.error("Failed to create model inputs")
                return None
            
            logger.info(f"Created model inputs with {model_inputs['x_train'].shape[0]} training samples")
            
            return model_inputs
            
        except Exception as e:
            logger.error(f"Error creating model inputs: {e}", exc_info=True)
            return None

if __name__ == "__main__":
    # Example usage
    integrator = DataIntegrator()
    
    # Collect and register stock data
    try:
        stock_data = integrator.collect_and_register_stock_data(
            symbol=DEFAULT_TICKER_BS,
            start_date='2019-01-01',
            end_date='2023-12-31'
        )
        print(f"Collected stock data with {len(stock_data) if stock_data is not None else 0} rows")
    except Exception as e:
        print(f"Error in stock data collection: {e}")
    
    # Collect and register macro data
    try:
        macro_data = integrator.collect_and_register_macro_data(
            sources=['gdp', 'cpi', 'm2']
        )
        print("Collected macro data:")
        for source, data in macro_data.items():
            print(f" - {source}: {len(data) if data is not None else 0} rows")
    except Exception as e:
        print(f"Error in macro data collection: {e}")
    
    # Create aligned features
    try:
        aligned_data = integrator.create_aligned_features(
            feature_id="moutai_with_macro",
            symbol=DEFAULT_TICKER_BS,
            macro_sources=['gdp', 'cpi', 'm2'],
            start_date='2019-01-01',
            end_date='2023-12-31'
        )
        print(f"Created aligned feature with {len(aligned_data) if aligned_data is not None else 0} rows")
    except Exception as e:
        print(f"Error creating aligned features: {e}") 