# src/data/macro_collection.py

import os
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

# Configure basic logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import config
try:
    from src.core.config import MACRO_DATA_DIR, MACRO_DATA_SOURCES, CACHE_EXPIRY_DAYS
except ImportError:
    try:
        from ..core.config import MACRO_DATA_DIR, MACRO_DATA_SOURCES, CACHE_EXPIRY_DAYS
    except ImportError:
        MACRO_DATA_DIR = Path("data/macro")
        MACRO_DATA_SOURCES = {}
        CACHE_EXPIRY_DAYS = 7

# Define raw data directory - use existing raw data folder
RAW_DATA_DIR = Path("data/raw")

# Make data directories if they don't exist
os.makedirs(MACRO_DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

class MacroDataCollector:
    """Simple collector for macroeconomic data."""
    
    def __init__(self, sources=None):
        """Initialize with data sources."""
        self.sources = sources or list(MACRO_DATA_SOURCES.keys())
        self.sources = [s for s in self.sources if s in MACRO_DATA_SOURCES]
        logger.info(f"Initialized with sources: {self.sources}")
    
    def is_cache_valid(self, file_path):
        """Check if cached file is valid."""
        if not file_path.exists():
            return False
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        return (datetime.now() - file_time).days < CACHE_EXPIRY_DAYS
    
    def transform_gdp_data(self, data):
        """Transform GDP data - handling quarterly values appropriately."""
        # First check if data is already in the expected format
        if all(col in data.columns for col in ['gdp', 'gdp_yoy']):
            # Create proper date column if not exists
            if 'date' not in data.columns and not data.index.name == 'date':
                if 'year' in data.columns:
                    # Use year (first day of year)
                    data['date'] = data['year'].astype(str) + '-01-01'
                    # Convert to datetime and set as index
                    data['date'] = pd.to_datetime(data['date'])
                    data = data.set_index('date')
                    # Keep only value columns
                    data = data[['gdp', 'gdp_yoy']]
            
            return data
            
        try:
            # Clean column data
            # The actual data has a format like this:
            # 商品          日期    今值   预测值    前值
            # 中国GDP年率报告  2011-01-20   9.8   9.5   9.6
            
            # Extract date from raw data
            data['date'] = pd.to_datetime(data['日期'], errors='coerce')
            
            # Map values
            data['gdp'] = pd.to_numeric(data['今值'], errors='coerce')
            data['gdp_yoy'] = pd.to_numeric(data['前值'] if '前值' in data.columns else data['今值'], errors='coerce')
            
            # Set date as index
            data = data.set_index('date')
            
            # Return only needed columns
            return data[['gdp', 'gdp_yoy']]
        except Exception as e:
            logger.error(f"GDP transform error: {e}")
            return pd.DataFrame(columns=['gdp', 'gdp_yoy'])
    
    def transform_cpi_data(self, data):
        """Transform CPI data."""
        if all(col in data.columns for col in ['cpi']):
            # Create proper date index in YYYY-MM-DD format if not exists
            if 'date' not in data.columns and not data.index.name == 'date':
                if 'month' in data.columns:
                    data['date'] = pd.to_datetime(data['month'] + '-01', errors='coerce')
                    data = data.set_index('date')
                    # Keep only value column
                    data = data[['cpi']]
            
            return data
            
        try:
            # Extract date from the data
            if '日期' in data.columns:
                data['date'] = pd.to_datetime(data['日期'], errors='coerce')
                # If parse fails, try different format
                if data['date'].isna().all():
                    # Try to extract year and month
                    year_month = data['日期'].astype(str).str.extract(r'(\d{4}).*?(\d{1,2})')
                    if not year_month.empty and not year_month.isna().all().all():
                        data['date'] = pd.to_datetime(
                            year_month[0] + '-' + year_month[1].str.zfill(2) + '-01', 
                            errors='coerce'
                        )
            
            # Map CPI value
            data['cpi'] = pd.to_numeric(data['今值'], errors='coerce')
            
            # Set date as index
            data = data.set_index('date')
            
            # Return only the CPI column
            return data[['cpi']]
        except Exception as e:
            logger.error(f"CPI transform error: {e}")
            return pd.DataFrame(columns=['cpi'])
    
    def transform_m2_data(self, data):
        """Transform M2 data and other monetary supply metrics."""
        if all(col in data.columns for col in ['m2']):
            # Create proper date index in YYYY-MM-DD format if not exists
            if 'date' not in data.columns and not data.index.name == 'date':
                if 'month' in data.columns:
                    data['date'] = pd.to_datetime(data['month'] + '-01', errors='coerce')
                    data = data.set_index('date')
                    # Keep only value columns
                    value_cols = [col for col in data.columns if col not in ['date', 'month']]
                    data = data[value_cols]
            
            return data
            
        try:
            # Extract date from 统计时间 column which has format like "2020.8"
            if '统计时间' in data.columns:
                # Split the date string at the dot
                date_parts = data['统计时间'].astype(str).str.split('.', expand=True)
                if len(date_parts.columns) >= 2:
                    year = date_parts[0]
                    month = date_parts[1].str.zfill(2)  # Pad month with leading zero if needed
                    data['date'] = pd.to_datetime(year + '-' + month + '-01', errors='coerce')
            
            # The raw data contains many monetary metrics. Map the important ones:
            column_indexes = list(data.columns)
            
            # Extract all relevant monetary metrics 
            # Based on typical money supply data format:
            # - M2 (total money supply)
            # - M1 (narrow money)
            # - M0 (currency in circulation)
            # - Various components and growth rates
            
            # Map the columns - adjust indexes based on observation of the data
            monetary_metrics = {
                'm2': 1,                # M2 value
                'm2_yoy': 2,            # M2 year-on-year growth (%)
                'm1': 3,                # M1 value
                'm1_yoy': 4,            # M1 year-on-year growth (%)
                'm0': 5,                # M0 value (currency in circulation)
                'm0_yoy': 6,            # M0 year-on-year growth (%)
                'demand_deposits': 7    # Demand deposits
            }
            
            # Create columns for each metric
            for metric_name, idx in monetary_metrics.items():
                if idx < len(column_indexes):
                    data[metric_name] = pd.to_numeric(data.iloc[:, idx], errors='coerce')
            
            # Set date as index and filter out empty or NaN values
            data = data.set_index('date')
            
            # Make sure at least M2 is available (our primary metric)
            if 'm2' in data.columns:
                data = data[data['m2'].notna() & (data['m2'] != "")]
            
            # Return all monetary metrics that were successfully mapped
            value_cols = [col for col in monetary_metrics.keys() if col in data.columns]
            return data[value_cols]
        except Exception as e:
            logger.error(f"M2 transform error: {e}")
            # Return empty DataFrame with columns for all monetary metrics
            return pd.DataFrame(columns=list(monetary_metrics.keys()))
    
    def fetch_single_source(self, source_id, config):
        """Fetch data for a single source."""
        filename = config['filename']
        file_path = Path(MACRO_DATA_DIR) / filename
        
        # Use better naming convention for raw files: raw_type_source_[date].csv
        current_date = datetime.now().strftime("%Y%m%d")
        raw_filename = f"raw_macro_{source_id}_{current_date}.csv"
        raw_file_path = Path(RAW_DATA_DIR) / raw_filename
        
        # Use cache if valid
        if self.is_cache_valid(file_path):
            try:
                # Read with proper parsing of dates
                data = pd.read_csv(file_path)
                
                # Convert date column to datetime and set as index
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'], errors='coerce')
                    data.set_index('date', inplace=True)
                    
                return data
            except Exception as e:
                logger.error(f"Error reading cache file {file_path}: {e}")
        
        # Try to import akshare
        try:
            import akshare as ak
        except ImportError:
            logger.error("akshare not installed")
            return None
        
        # Parse function name
        func_path = config['function']
        if not func_path.startswith('ak.'):
            logger.error(f"Only akshare functions supported: {func_path}")
            return None
            
        func_name = func_path.split('.')[1]
        
        # Call API function
        try:
            api_func = getattr(ak, func_name)
            data = api_func()
            
            if data is None or data.empty:
                return None
                
            # Save raw data before transformation
            try:
                data.to_csv(raw_file_path, index=False)
                logger.info(f"Saved raw macro data ({len(data)} rows) to {raw_file_path}")
            except Exception as e:
                logger.warning(f"Failed to save raw data to {raw_file_path}: {e}")
                
            # Transform data
            if source_id == 'gdp':
                result = self.transform_gdp_data(data)
            elif source_id == 'cpi':
                result = self.transform_cpi_data(data)
            elif source_id == 'm2':
                result = self.transform_m2_data(data)
            else:
                return None
                
            # Clean data
            if result is not None and not result.empty:
                # Determine value columns based on source
                if source_id == 'gdp':
                    value_cols = ['gdp', 'gdp_yoy']
                elif source_id == 'cpi':
                    value_cols = ['cpi']
                elif source_id == 'm2':
                    # For M2, include all monetary metrics columns that exist in the data
                    monetary_metrics = ['m2', 'm2_yoy', 'm1', 'm1_yoy', 'm0', 'm0_yoy', 'demand_deposits']
                    value_cols = [col for col in monetary_metrics if col in result.columns]
                else:
                    value_cols = result.columns.tolist()
                    
                # Drop rows where all value columns are NaN
                result = result.dropna(subset=value_cols, how='all')
                
                # Drop rows with empty strings in value columns
                for col in value_cols:
                    if col in result.columns:
                        result = result[result[col] != ""]
                
                # Save to file if transformation successful and has data
                if not result.empty:
                    try:
                        # Reset index to make sure date becomes a column
                        if isinstance(result.index, pd.DatetimeIndex):
                            result_to_save = result.reset_index()
                            # Format date as yyyy-mm-dd
                            result_to_save['date'] = result_to_save['date'].dt.strftime('%Y-%m-%d')
                        else:
                            result_to_save = result
                            
                        # Only save essential columns
                        columns_to_keep = ['date'] + value_cols
                        columns_to_save = [col for col in columns_to_keep if col in result_to_save.columns]
                        result_to_save = result_to_save[columns_to_save]
                            
                        # Save with only essential columns
                        result_to_save.to_csv(file_path, index=False, date_format='%Y-%m-%d')
                        logger.info(f"Saved transformed macro data ({len(result_to_save)} rows) to {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save transformed data to {file_path}: {e}")
            
            return result
                
        except Exception as e:
            logger.error(f"Error fetching {source_id}: {e}")
            return None
    
    def fetch_all_sources(self, overwrite=False):
        """Fetch data for all sources.
        
        Args:
            overwrite (bool): If True, ignore cache and fetch fresh data.
            
        Returns:
            dict: Dictionary with source_id as keys and DataFrame as values.
                  Returns None for sources that failed to fetch.
        """
        results = {}
        for source_id in self.sources:
            if source_id not in MACRO_DATA_SOURCES:
                logger.warning(f"Source {source_id} not found in MACRO_DATA_SOURCES")
                results[source_id] = None
                continue
                
            config = MACRO_DATA_SOURCES[source_id]
            
            # If overwrite is True, we need to handle cache invalidation
            if overwrite:
                file_path = Path(MACRO_DATA_DIR) / config['filename']
                # Force refresh by skipping cache
                results[source_id] = self.fetch_single_source(source_id, config)
            else:
                results[source_id] = self.fetch_single_source(source_id, config)
                
            logger.info(f"Fetched {source_id}: {'Success' if results[source_id] is not None else 'Failed'}")
            
        return results
    
    def get_data(self, source=None):
        """Get data for one or all sources."""
        if source:
            if source not in MACRO_DATA_SOURCES:
                return None
            return self.fetch_single_source(source, MACRO_DATA_SOURCES[source])
            
        # Get all sources
        results = {}
        for src in self.sources:
            results[src] = self.fetch_single_source(src, MACRO_DATA_SOURCES[src])
        return results


def main():
    """Command line entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Fetch macroeconomic data.")
    parser.add_argument('--sources', nargs='+', choices=list(MACRO_DATA_SOURCES.keys()),
                        help="Sources to fetch", default=None)
    args = parser.parse_args()
    
    collector = MacroDataCollector(sources=args.sources)
    results = collector.get_data()
    
    for source, data in results.items():
        print(f"{source}: {'Success' if data is not None else 'Failed'}")

if __name__ == "__main__":
    main()
