"""Tests for the MacroDataCollector class."""

import os
import sys
import pytest
import pandas as pd
import importlib
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.macro_collection import MacroDataCollector
from src.core.config import MACRO_DATA_SOURCES


class TestMacroDataCollector:
    """Test cases for the MacroDataCollector class."""
    
    @pytest.fixture
    def sample_gdp_data(self):
        """Create sample GDP data in expected format."""
        dates = pd.date_range(start='2022-01-01', periods=3, freq='Y')
        return pd.DataFrame({
            'gdp': [5.0, 4.5, 5.2],
            'gdp_yoy': [4.8, 4.0, 4.5]
        }, index=dates)
    
    @pytest.fixture
    def sample_raw_gdp_data(self):
        """Create sample raw GDP data in Chinese format."""
        df = pd.DataFrame({
            '商品': ['GDP', 'GDP', 'GDP'],
            '日期': ['2022-01-20', '2023-01-20', '2024-01-20'],
            '今值': [5.0, 4.5, 5.2],
            '预测值': [4.9, 4.3, 5.0],
            '前值': [4.8, 4.0, 4.5]
        })
        # Add date column for newer implementation
        df['date'] = pd.to_datetime(df['日期'])
        return df
    
    @pytest.fixture
    def mock_ak(self, sample_raw_gdp_data):
        """Create a mock for the akshare module."""
        mock = MagicMock()
        mock.macro_china_gdp_yearly.return_value = sample_raw_gdp_data
        return mock
    
    def test_transform_gdp_data(self, sample_raw_gdp_data):
        """Test GDP data transformation."""
        collector = MacroDataCollector()
        transformed = collector.transform_gdp_data(sample_raw_gdp_data)
        
        # Check that the result has a DatetimeIndex
        assert isinstance(transformed.index, pd.DatetimeIndex)
        assert transformed.index.name == 'date'
        
        # Check that it has the right columns
        assert 'gdp' in transformed.columns
        assert 'gdp_yoy' in transformed.columns
        
        # Check row count and values
        assert len(transformed) == 3
        assert transformed['gdp'].iloc[0] == 5.0
        assert transformed['gdp_yoy'].iloc[0] == 4.8
    
    def test_is_cache_valid(self, tmp_path):
        """Test cache validation."""
        collector = MacroDataCollector()
        
        # Create a test file
        test_file = tmp_path / "test_cache.csv"
        test_file.write_text("test")
        
        # Test with recent file
        assert collector.is_cache_valid(test_file) == True
        
        # Modify the file time to be old
        old_time = datetime.now() - timedelta(days=30)
        os.utime(test_file, (old_time.timestamp(), old_time.timestamp()))
        
        # Test with old file
        assert collector.is_cache_valid(test_file) == False
    
    def test_fetch_single_source_success(self, mock_ak, sample_raw_gdp_data, tmp_path):
        """Test successful data fetching."""
        with patch('src.data.macro_collection.MACRO_DATA_DIR', tmp_path):
            with patch('src.data.macro_collection.RAW_DATA_DIR', tmp_path), patch.dict('sys.modules', {'akshare': mock_ak}):
                collector = MacroDataCollector()
                
                # Test config
                config = {
                    'function': 'ak.macro_china_gdp_yearly',
                    'filename': 'gdp_test.csv'
                }
                
                # Ensure cache is not used
                with patch.object(collector, 'is_cache_valid', return_value=False):
                    result = collector.fetch_single_source('gdp', config)
                
                # Verify result
                assert result is not None
                
                # Check that result has the right columns
                assert 'gdp' in result.columns
                assert 'gdp_yoy' in result.columns
                
                # Check that the index is a DatetimeIndex
                assert isinstance(result.index, pd.DatetimeIndex)
                
                # Check row count
                assert len(result) == 3
                
                # Verify file was created
                assert (tmp_path / 'gdp_test.csv').exists()
    
    def test_fetch_single_source_error(self, mock_ak, tmp_path):
        """Test error handling in data fetching."""
        with patch('src.data.macro_collection.MACRO_DATA_DIR', tmp_path):
            with patch('src.data.macro_collection.RAW_DATA_DIR', tmp_path), patch.dict('sys.modules', {'akshare': mock_ak}):
                collector = MacroDataCollector()
                
                # Set up error
                mock_ak.macro_china_gdp_yearly.side_effect = Exception("API error")
                
                # Test config
                config = {
                    'function': 'ak.macro_china_gdp_yearly',
                    'filename': 'gdp_test.csv'
                }
                
                # Ensure cache is not used
                with patch.object(collector, 'is_cache_valid', return_value=False):
                    result = collector.fetch_single_source('gdp', config)
                
                # Verify result is None on error
                assert result is None


if __name__ == '__main__':
    pytest.main(['-xvs', __file__]) 