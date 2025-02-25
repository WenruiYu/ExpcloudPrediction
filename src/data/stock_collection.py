# src/data/stock_collection.py

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import baostock as bs
import logging
import logging.config
from typing import Optional, List, Dict, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# Try to import using relative imports if running as a module
try:
    from src.core.utils import calculate_technical_indicators
    from src.core.config import (
        START_DATE, END_DATE, DEFAULT_TICKER_BS,
        RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR,
        BATCH_SIZE, MAX_WORKERS, LOGGING_CONFIG,
        CACHE_EXPIRY_DAYS
    )
except ImportError:
    # Fall back to direct imports if running the file directly
    from ..core.utils import calculate_technical_indicators
    from ..core.config import (
        START_DATE, END_DATE, DEFAULT_TICKER_BS,
        RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR,
        BATCH_SIZE, MAX_WORKERS, LOGGING_CONFIG,
        CACHE_EXPIRY_DAYS
    )

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class StockDataCollector:
    """Class for collecting and processing stock data."""
    
    def __init__(
        self,
        symbol: str = DEFAULT_TICKER_BS,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        batch_size: int = BATCH_SIZE,
        max_workers: int = MAX_WORKERS,
        cache_expiry_days: int = CACHE_EXPIRY_DAYS,
        indicators: Optional[List[str]] = None
    ):
        """
        Initialize the stock data collector.
        
        Args:
            symbol: Stock symbol in BaoStock format
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            batch_size: Size of data chunks for parallel processing
            max_workers: Maximum number of worker threads
            cache_expiry_days: Number of days before cache expires
            indicators: List of technical indicators to calculate
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.cache_expiry_days = cache_expiry_days
        self.indicators = indicators
        
        # Initialize BaoStock session
        self.bs_session = None
        
        logger.info(f"Initialized StockDataCollector for {symbol} from {start_date} to {end_date}")
    
    def __enter__(self):
        """Context manager entry."""
        self.login()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.logout()
        if exc_type:
            logger.error(f"An error occurred: {exc_val}")
            return False
        return True
    
    def login(self) -> bool:
        """Login to BaoStock API."""
        try:
            self.bs_session = bs.login()
            if self.bs_session.error_code != '0':
                logger.error(f"Login failed: {self.bs_session.error_msg}")
                return False
            logger.info("Successfully logged in to BaoStock")
            return True
        except Exception as e:
            logger.error(f"Login failed with exception: {str(e)}", exc_info=True)
            return False
    
    def logout(self) -> None:
        """Logout from BaoStock API."""
        if self.bs_session:
            bs.logout()
            logger.info("Logged out from BaoStock")
    
    @staticmethod
    def process_data_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single chunk of data.
        
        Args:
            chunk: DataFrame containing stock data
            
        Returns:
            Processed DataFrame
        """
        if chunk.empty:
            return pd.DataFrame()
        
        chunk = chunk.copy()
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        
        # Vectorized operations instead of loop
        chunk[numeric_columns] = chunk[numeric_columns].apply(pd.to_numeric, errors='coerce')
        chunk['date'] = pd.to_datetime(chunk['date'])
        
        return chunk
    
    def is_cache_valid(self, cache_file: Path) -> bool:
        """
        Check if cache file is valid and not expired.
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_file.exists():
            return False
        
        # Check if cache is expired
        file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        expiry_time = datetime.now() - timedelta(days=self.cache_expiry_days)
        
        if file_mtime < expiry_time:
            logger.info(f"Cache expired: {cache_file}")
            return False
        
        return True
    
    def fetch_stock_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data with parallel processing and caching.
        
        Returns:
            DataFrame containing stock data or None on failure
        """
        cache_file = CACHE_DIR / f'{self.symbol}_{self.start_date}_{self.end_date}_cache.csv'
        
        # Check cache first
        if self.is_cache_valid(cache_file):
            try:
                logger.info(f"Using cached data from {cache_file}")
                return pd.read_csv(
                    cache_file,
                    parse_dates=['date'],
                    index_col='date'
                )
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        try:
            # Validate inputs
            if not all([self.symbol, self.start_date, self.end_date]):
                raise ValueError("Missing required parameters")

            # Ensure logged in
            if not self.bs_session or self.bs_session.error_code != '0':
                if not self.login():
                    raise ConnectionError("Could not establish connection to BaoStock")
                
            # Query historical K-line data with retry mechanism
            for attempt in range(3):  # Retry up to 3 times
                try:
                    rs = bs.query_history_k_data_plus(
                        self.symbol,
                        "date,code,open,high,low,close,volume,amount",
                        start_date=self.start_date,
                        end_date=self.end_date,
                        frequency="d",
                        adjustflag="3"
                    )
                    
                    if rs.error_code != '0':
                        raise Exception(f"Query failed: {rs.error_msg}")
                    
                    break  # Break the retry loop if successful
                except Exception as e:
                    if attempt < 2:  # If not the last attempt
                        logger.warning(f"Query attempt {attempt+1} failed: {e}. Retrying...")
                        time.sleep(2)  # Wait before retrying
                    else:
                        raise  # Re-raise the exception on the last attempt
            
            # Get all data at once and convert to DataFrame
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if not data_list:
                raise ValueError("No data retrieved")

            # Create DataFrame from all data
            columns = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
            final_data = pd.DataFrame(data_list, columns=columns)

            # Process the data in parallel chunks
            chunk_size = self.batch_size
            chunks = [final_data[i:i + chunk_size] for i in range(0, len(final_data), chunk_size)]
            
            processed_chunks = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.process_data_chunk, chunk) for chunk in chunks]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        processed_chunks.append(result)
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}", exc_info=True)

            # Combine processed chunks
            final_data = pd.concat(processed_chunks, ignore_index=True)
            
            # Sort by date before setting index
            final_data['date'] = pd.to_datetime(final_data['date'])
            final_data.sort_values('date', ascending=True, inplace=True)
            
            # Basic data validation
            if final_data.isnull().sum().sum() > 0:
                logger.warning(f"Dataset contains {final_data.isnull().sum().sum()} missing values")
            
            # Save raw data
            raw_file_path = RAW_DATA_DIR / f'{self.symbol}_stock_data.csv'
            final_data.to_csv(raw_file_path, index=False)
            logger.info(f"Raw data saved to {raw_file_path}")
            
            # Set date as index after sorting
            final_data.set_index('date', inplace=True)
            
            # Save to cache
            final_data.to_csv(cache_file)
            logger.info(f"Data cached to {cache_file}")

            return final_data

        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}", exc_info=True)
            return None
    
    def process_stock_data(self, data: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """
        Process raw stock data with optimized memory usage.
        
        Args:
            data: DataFrame containing stock data. If None, will fetch data first.
            
        Returns:
            Processed DataFrame or None on failure
        """
        try:
            # If no data provided, fetch it first
            if data is None:
                data = self.fetch_stock_data()
                
            if data is None or data.empty:
                raise ValueError("Invalid input data")

            # Make a copy to avoid modifying the original data
            data = data.copy()

            # Convert date to datetime and set as index if not already
            if 'date' in data.columns:  # If date is still a column
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
            elif not isinstance(data.index, pd.DatetimeIndex):  # If date is index but not datetime
                data.index = pd.to_datetime(data.index)

            # Convert numeric columns efficiently
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_columns:
                if col in data.columns:  # Check if column exists
                    data[col] = pd.to_numeric(data[col], errors='coerce')

            # Calculate technical indicators
            processed_data = calculate_technical_indicators(data, indicators=self.indicators)

            # Validate processed data
            if processed_data.isnull().sum().sum() > 0:
                logger.warning(f"Processed data contains {processed_data.isnull().sum().sum()} missing values")
                
                # Fill NaN values for technical indicators (updated to use bfill() and ffill())
                processed_data = processed_data.bfill().ffill()

            # Save processed data efficiently
            processed_file_path = PROCESSED_DATA_DIR / f'{self.symbol}_processed_stock_data.csv'
            processed_data.to_csv(processed_file_path)
            logger.info(f"Processed data saved to {processed_file_path}")

            return processed_data

        except Exception as e:
            logger.error(f"Error processing stock data: {str(e)}", exc_info=True)
            return None

    def get_data(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Main method to fetch and process stock data.
        
        Args:
            force_refresh: If True, ignore cache and fetch fresh data
            
        Returns:
            Processed DataFrame or None on failure
        """
        try:
            if force_refresh:
                # Delete cache if forcing refresh
                cache_file = CACHE_DIR / f'{self.symbol}_{self.start_date}_{self.end_date}_cache.csv'
                if cache_file.exists():
                    os.remove(cache_file)
                    logger.info(f"Deleted cache file {cache_file}")
            
            # Fetch raw data
            raw_data = self.fetch_stock_data()
            if raw_data is None:
                logger.error("Failed to fetch stock data")
                return None
                
            # Process data
            processed_data = self.process_stock_data(raw_data)
            if processed_data is None:
                logger.error("Failed to process stock data")
                return None
                
            logger.info(f"Successfully retrieved and processed data for {self.symbol}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in get_data: {str(e)}", exc_info=True)
            return None


if __name__ == "__main__":
    # Use as a context manager
    with StockDataCollector(DEFAULT_TICKER_BS, START_DATE, END_DATE) as collector:
        processed_data = collector.get_data()
        if processed_data is not None:
            logger.info(f"Data shape: {processed_data.shape}")
            logger.info(f"Data columns: {processed_data.columns.tolist()}")
            logger.info(f"Data sample:\n{processed_data.head()}")
        else:
            logger.error("Failed to get processed data")