"""Tests for the StockDataCollector class."""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_collection import StockDataCollector
from src.config import DEFAULT_TICKER_BS


class TestStockDataCollector:
    """Test cases for the StockDataCollector class."""
    
    @pytest.fixture
    def mock_bs_login(self):
        """Create a mock for BaoStock login function."""
        with patch('baostock.login') as mock:
            mock_result = MagicMock()
            mock_result.error_code = '0'
            mock.return_value = mock_result
            yield mock
    
    @pytest.fixture
    def mock_bs_logout(self):
        """Create a mock for BaoStock logout function."""
        with patch('baostock.logout') as mock:
            yield mock
    
    @pytest.fixture
    def mock_bs_query(self):
        """Create a mock for BaoStock query_history_k_data_plus function."""
        with patch('baostock.query_history_k_data_plus') as mock:
            # Create a mock result object
            mock_result = MagicMock()
            mock_result.error_code = '0'
            
            # Sample data for Moutai (SH.600519)
            data = [
                ['2023-01-01', 'sh.600519', '1800.0', '1850.0', '1790.0', '1830.0', '10000', '18300000'],
                ['2023-01-02', 'sh.600519', '1830.0', '1880.0', '1820.0', '1860.0', '12000', '22320000'],
                ['2023-01-03', 'sh.600519', '1860.0', '1900.0', '1850.0', '1890.0', '11000', '20790000'],
            ]
            
            # Setup mock to return sample data
            def next_side_effect():
                if hasattr(mock_result, '_idx'):
                    mock_result._idx += 1
                else:
                    mock_result._idx = 0
                return mock_result._idx < len(data)
                
            def get_row_data_side_effect():
                return data[mock_result._idx - 1]
                
            mock_result.next = MagicMock(side_effect=next_side_effect)
            mock_result.get_row_data = MagicMock(side_effect=get_row_data_side_effect)
            mock.return_value = mock_result
            yield mock
    
    @pytest.fixture
    def sample_stock_data(self):
        """Create a sample stock DataFrame."""
        dates = pd.date_range(start='2023-01-01', periods=3)
        data = {
            'open': [1800.0, 1830.0, 1860.0],
            'high': [1850.0, 1880.0, 1900.0],
            'low': [1790.0, 1820.0, 1850.0],
            'close': [1830.0, 1860.0, 1890.0],
            'volume': [10000, 12000, 11000],
            'amount': [18300000, 22320000, 20790000],
            'code': ['sh.600519'] * 3
        }
        return pd.DataFrame(data, index=dates)
    
    def test_initialization(self):
        """Test that StockDataCollector initializes correctly."""
        collector = StockDataCollector(
            symbol='sh.600519',
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        assert collector.symbol == 'sh.600519'
        assert collector.start_date == '2023-01-01'
        assert collector.end_date == '2023-01-31'
        assert collector.batch_size == 1000  # Default value
        assert collector.max_workers == 4  # Default value
        assert collector.bs_session is None  # Not logged in yet
    
    def test_login_successful(self, mock_bs_login):
        """Test successful login to BaoStock."""
        collector = StockDataCollector()
        result = collector.login()
        
        assert result is True
        mock_bs_login.assert_called_once()
        assert collector.bs_session is not None
    
    def test_login_failed(self, mock_bs_login):
        """Test failed login to BaoStock."""
        mock_bs_login.return_value.error_code = '1'
        mock_bs_login.return_value.error_msg = 'Login failed'
        
        collector = StockDataCollector()
        result = collector.login()
        
        assert result is False
        mock_bs_login.assert_called_once()
    
    def test_context_manager(self, mock_bs_login, mock_bs_logout):
        """Test that StockDataCollector works as a context manager."""
        with StockDataCollector() as collector:
            assert collector.bs_session is not None
        
        mock_bs_login.assert_called_once()
        mock_bs_logout.assert_called_once()
    
    def test_fetch_stock_data(self, mock_bs_login, mock_bs_logout, mock_bs_query, monkeypatch):
        """Test fetching stock data from BaoStock."""
        # Mock cache handling
        monkeypatch.setattr(StockDataCollector, 'is_cache_valid', lambda self, path: False)
        
        with StockDataCollector(
            symbol='sh.600519',
            start_date='2023-01-01',
            end_date='2023-01-03'
        ) as collector:
            data = collector.fetch_stock_data()
            
            # Verify API calls
            mock_bs_login.assert_called_once()
            mock_bs_query.assert_called_once_with(
                'sh.600519',
                'date,code,open,high,low,close,volume,amount',
                start_date='2023-01-01', 
                end_date='2023-01-03',
                frequency='d',
                adjustflag='3'
            )
            
            # Verify data
            assert data is not None
            assert len(data) == 3
            assert 'close' in data.columns
            assert 'open' in data.columns
            assert isinstance(data.index, pd.DatetimeIndex)
            assert data['close'].iloc[0] == 1830.0
    
    def test_process_data_chunk(self):
        """Test processing a chunk of data."""
        # Create test data
        chunk = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'code': ['sh.600519', 'sh.600519'],
            'open': ['1800.0', '1830.0'],
            'high': ['1850.0', '1880.0'],
            'low': ['1790.0', '1820.0'],
            'close': ['1830.0', '1860.0'],
            'volume': ['10000', '12000'],
            'amount': ['18300000', '22320000']
        })
        
        # Process chunk
        result = StockDataCollector.process_data_chunk(chunk)
        
        # Verify result
        assert result is not None
        assert len(result) == 2
        assert pd.api.types.is_numeric_dtype(result['close'])
        assert pd.api.types.is_datetime64_dtype(result['date'])
        assert result['close'].iloc[0] == 1830.0
    
    def test_is_cache_valid(self, tmp_path):
        """Test cache validation."""
        collector = StockDataCollector()
        
        # Create a dummy cache file
        cache_file = tmp_path / 'test_cache.csv'
        with open(cache_file, 'w') as f:
            f.write('dummy content')
        
        # Test with fresh file
        assert collector.is_cache_valid(cache_file) is True
        
        # Test with expired file (modify time)
        old_time = datetime.now() - timedelta(days=2)
        os.utime(cache_file, (old_time.timestamp(), old_time.timestamp()))
        assert collector.is_cache_valid(cache_file) is False
        
        # Test with non-existent file
        non_existent = tmp_path / 'non_existent.csv'
        assert collector.is_cache_valid(non_existent) is False
    
    def test_process_stock_data(self, sample_stock_data):
        """Test processing stock data with technical indicators."""
        collector = StockDataCollector()
        
        # Mock fetch_stock_data to return our sample data
        with patch.object(collector, 'fetch_stock_data', return_value=sample_stock_data):
            processed_data = collector.process_stock_data()
            
            # Verify that technical indicators were added
            assert processed_data is not None
            assert len(processed_data) == len(sample_stock_data)
            assert any(col.startswith('SMA_') for col in processed_data.columns)
            assert any(col.startswith('EMA_') for col in processed_data.columns)
            assert any(col.startswith('RSI_') for col in processed_data.columns)
            assert 'MACD' in processed_data.columns
    
    def test_get_data_with_cache(self, monkeypatch, sample_stock_data):
        """Test getting data with cache."""
        collector = StockDataCollector()
        
        # Mock methods
        monkeypatch.setattr(collector, 'fetch_stock_data', lambda: sample_stock_data)
        monkeypatch.setattr(collector, 'process_stock_data', lambda data=None: sample_stock_data)
        
        # Test normal data fetch
        result = collector.get_data()
        assert result is not None
        assert len(result) == 3
        
        # Test with force refresh
        with patch.object(os, 'remove') as mock_remove:
            result = collector.get_data(force_refresh=True)
            assert result is not None
            assert len(result) == 3
            mock_remove.assert_called_once()
    
    def test_get_data_error_handling(self, monkeypatch):
        """Test error handling in get_data."""
        collector = StockDataCollector()
        
        # Mock fetch_stock_data to return None (simulating error)
        monkeypatch.setattr(collector, 'fetch_stock_data', lambda: None)
        
        # Test fetch failure
        result = collector.get_data()
        assert result is None


if __name__ == '__main__':
    pytest.main(['-xvs', __file__]) 