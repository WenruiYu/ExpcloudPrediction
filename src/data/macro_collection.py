# src/data/macro_collection.py

import argparse
import importlib
import os
import time
import pandas as pd
import logging
import logging.config
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from datetime import datetime, timedelta

# Try to import using relative imports if running as a module
try:
    from src.core.config import MACRO_DATA_DIR, MACRO_DATA_SOURCES, LOGGING_CONFIG, CACHE_EXPIRY_DAYS
except ImportError:
    # Fall back to direct imports if running the file directly
    from ..core.config import MACRO_DATA_DIR, MACRO_DATA_SOURCES, LOGGING_CONFIG, CACHE_EXPIRY_DAYS

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Check if akshare is available
AKSHARE_AVAILABLE = False
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    logger.warning("akshare package is not installed. Macroeconomic data collection will not work. Install with: pip install akshare")

class MacroDataCollector:
    """Class for collecting and managing macroeconomic data."""
    
    def __init__(
        self, 
        sources: Optional[List[str]] = None,
        cache_expiry_days: int = CACHE_EXPIRY_DAYS
    ):
        """
        Initialize the macro data collector.
        
        Args:
            sources: List of data sources to fetch. If None, uses all sources.
            cache_expiry_days: Number of days before cache expires
        """
        self.sources = sources or list(MACRO_DATA_SOURCES.keys())
        self.cache_expiry_days = cache_expiry_days
        
        # Validate sources
        invalid_sources = [src for src in self.sources if src not in MACRO_DATA_SOURCES]
        if invalid_sources:
            logger.warning(f"Invalid sources: {invalid_sources}. These will be ignored.")
            self.sources = [src for src in self.sources if src in MACRO_DATA_SOURCES]
            
        logger.info(f"Initialized MacroDataCollector with sources: {self.sources}")
        
        # Warn if akshare is not available
        if not AKSHARE_AVAILABLE:
            logger.warning("akshare package is not installed. Macroeconomic data collection will not work. Install with: pip install akshare")
    
    def is_cache_valid(self, file_path: Path) -> bool:
        """
        Check if a cached file is valid and not expired.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if valid, False otherwise
        """
        if not file_path.exists():
            return False
        
        # Check file modification time
        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        expiry_time = datetime.now() - timedelta(days=self.cache_expiry_days)
        
        if file_mtime < expiry_time:
            logger.info(f"Cache expired: {file_path}")
            return False
            
        return True
    
    def fetch_single_source(
        self, 
        source_name: str, 
        config: Dict[str, str], 
        overwrite: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single source.
        
        Args:
            source_name: Name of the data source
            config: Configuration dictionary for the source
            overwrite: Whether to overwrite existing files
            
        Returns:
            DataFrame with fetched data or None on failure
        """
        file_path = MACRO_DATA_DIR / config['filename']
        
        # Check cache unless overwrite is specified
        if not overwrite and self.is_cache_valid(file_path):
            try:
                logger.info(f"Using cached data for {source_name} from {file_path}")
                return pd.read_csv(file_path)
            except Exception as e:
                logger.warning(f"Error reading cached file {file_path}: {e}")
        
        # Check if akshare is available before attempting to fetch data
        if not AKSHARE_AVAILABLE:
            logger.error(f"akshare package is not installed. Cannot fetch {source_name} data.")
            return None
            
        logger.info(f"Fetching {config['description']}...")
        
        try:
            # Dynamically import and call the function
            module_name, func_name = config['function'].split('.')
            
            # Use the global ak module if available, otherwise try to import
            if module_name == 'ak' and AKSHARE_AVAILABLE:
                module = ak
            else:
                try:
                    module = importlib.import_module(module_name)
                except ImportError as e:
                    logger.error(f"Could not import module {module_name}: {e}")
                    return None
                    
            func = getattr(module, func_name)
            
            # Implement retry mechanism
            max_retries = 3
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    data = func()
                    
                    # Validate data
                    if data is None or data.empty:
                        raise ValueError(f"No data returned for {source_name}")
                    
                    # Basic data cleaning and validation
                    # Convert date columns to datetime if they exist
                    date_columns = [col for col in data.columns if 'date' in col.lower()]
                    for col in date_columns:
                        try:
                            data[col] = pd.to_datetime(data[col], errors='coerce')
                        except Exception as e:
                            logger.warning(f"Failed to convert column {col} to datetime: {e}")
                    
                    # Save to file
                    data.to_csv(file_path, index=False)
                    logger.info(f"Data saved to {file_path}")
                    
                    return data
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt+1} failed for {source_name}: {e}. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        raise
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {source_name}: {e}", exc_info=True)
            return None
    
    def fetch_all_sources(self, overwrite: bool = False) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetch data for all specified sources.
        
        Args:
            overwrite: Whether to overwrite existing files
            
        Returns:
            Dictionary mapping source names to their DataFrames
        """
        results = {}
        
        for source in self.sources:
            if source in MACRO_DATA_SOURCES:
                logger.info(f"Processing {source}...")
                data = self.fetch_single_source(source, MACRO_DATA_SOURCES[source], overwrite)
                results[source] = data
            else:
                logger.warning(f"Unknown source: {source}")
                
        # Log summary
        success_count = sum(1 for data in results.values() if data is not None)
        logger.info(f"Successfully fetched {success_count}/{len(self.sources)} data sources")
        
        return results
    
    def get_data(
        self, 
        source: Optional[str] = None, 
        force_refresh: bool = False
    ) -> Union[Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Main method to get data for one or all sources.
        
        Args:
            source: Specific source to get. If None, gets all sources.
            force_refresh: Whether to force refresh the data
            
        Returns:
            Dictionary of DataFrames or a single DataFrame if source is specified
        """
        try:
            if source is not None:
                # Get single source
                if source not in MACRO_DATA_SOURCES:
                    logger.error(f"Unknown source: {source}")
                    return None
                    
                return self.fetch_single_source(source, MACRO_DATA_SOURCES[source], force_refresh)
            else:
                # Get all sources
                return self.fetch_all_sources(force_refresh)
                
        except Exception as e:
            logger.error(f"Error in get_data: {e}", exc_info=True)
            return {} if source is None else None


def main():
    """Command-line entry point for the macro data collector."""
    parser = argparse.ArgumentParser(description="Fetch and save macroeconomic data.")
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help="Fetch and save all data, overwriting existing files."
    )
    parser.add_argument(
        '--sources',
        nargs='+',
        choices=list(MACRO_DATA_SOURCES.keys()),
        help="Specific data sources to fetch (default: all)",
        default=list(MACRO_DATA_SOURCES.keys())
    )
    args = parser.parse_args()

    # Create collector and fetch data
    collector = MacroDataCollector(sources=args.sources)
    results = collector.fetch_all_sources(overwrite=args.overwrite)
    
    # Print results summary
    print("\nResults Summary:")
    for source, data in results.items():
        status = "Success" if data is not None else "Failed"
        rows = len(data) if data is not None else 0
        print(f"{source}: {status} ({rows} rows)")

if __name__ == "__main__":
    main()
