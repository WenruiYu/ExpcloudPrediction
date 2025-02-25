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
        """Create sample GDP data."""
        return pd.DataFrame({
            'year': ['2020', '2021', '2022'],
            'gdp': [101.95, 114.37, 121.02],
            'gdp_yoy': [2.3, 8.4, 3.0]
        })
    
    @pytest.fixture
    def sample_cpi_data(self):
        """Create sample CPI data."""
        return pd.DataFrame({
            'month': ['2023-01', '2023-02', '2023-03'],
            'cpi': [101.0, 101.5, 102.1]
        })
    
    @pytest.fixture
    def mock_ak_module(self, sample_gdp_data, sample_cpi_data):
        """Create a mock for the akshare module."""
        mock_ak = MagicMock()
        mock_ak.macro_china_gdp_yearly.return_value = sample_gdp_data
        mock_ak.macro_china_cpi_monthly.return_value = sample_cpi_data
        
        with patch.dict('sys.modules', {'ak': mock_ak}):
            yield mock_ak
    
    def test_initialization_with_defaults(self):
        """Test that MacroDataCollector initializes with default values."""
        collector = MacroDataCollector()
        assert set(collector.sources) == set(MACRO_DATA_SOURCES.keys())
        assert collector.cache_expiry_days == 1  # Default value
    
    def test_initialization_with_specific_sources(self):
        """Test initialization with specific sources."""
        collector = MacroDataCollector(sources=['gdp', 'cpi'])
        assert set(collector.sources) == {'gdp', 'cpi'}
    
    def test_initialization_with_invalid_sources(self, caplog):
        """Test that invalid sources are filtered out with warning."""
        collector = MacroDataCollector(sources=['gdp', 'invalid_source'])
        assert set(collector.sources) == {'gdp'}
        assert "Invalid sources" in caplog.text
    
    def test_is_cache_valid(self, tmp_path):
        """Test cache validation logic."""
        collector = MacroDataCollector()
        
        # Create a test file
        cache_file = tmp_path / 'test_cache.csv'
        with open(cache_file, 'w') as f:
            f.write('test data')
        
        # Test valid cache
        assert collector.is_cache_valid(cache_file) is True
        
        # Test expired cache
        old_time = datetime.now() - timedelta(days=2)
        os.utime(cache_file, (old_time.timestamp(), old_time.timestamp()))
        assert collector.is_cache_valid(cache_file) is False
        
        # Test non-existent file
        non_existent = tmp_path / 'non_existent.csv'
        assert collector.is_cache_valid(non_existent) is False
    
    def test_fetch_single_source_using_cache(self, tmp_path, sample_gdp_data):
        """Test fetching a single source with cache."""
        # Setup
        collector = MacroDataCollector()
        cache_file = tmp_path / 'china_gdp_yearly.csv'
        sample_gdp_data.to_csv(cache_file, index=False)
        
        # Create mock config
        config = {
            'function': 'ak.macro_china_gdp_yearly',
            'filename': 'china_gdp_yearly.csv',  # Just the filename, not the full path
            'description': "Test GDP data"
        }
        
        # Mock is_cache_valid to return True and patch MACRO_DATA_DIR
        with patch.object(collector, 'is_cache_valid', return_value=True), \
             patch('src.data.macro_collection.MACRO_DATA_DIR', tmp_path):
            # Mock pd.read_csv to return our sample data
            with patch('pandas.read_csv', return_value=sample_gdp_data) as mock_read_csv:
                result = collector.fetch_single_source('gdp', config)
                
                # Verify result
                assert result is not None
                assert result.equals(sample_gdp_data)
                mock_read_csv.assert_called_once_with(tmp_path / 'china_gdp_yearly.csv')
    
    def test_fetch_single_source_fresh_data(self, tmp_path, mock_ak_module, sample_gdp_data):
        """Test fetching fresh data for a single source."""
        # Setup
        collector = MacroDataCollector()
        cache_dir = tmp_path / 'macro'
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / 'china_gdp_yearly.csv'
        
        # Create mock config
        config = {
            'function': 'ak.macro_china_gdp_yearly',
            'filename': str(cache_file.name),
            'description': "Test GDP data"
        }
        
        # Ensure we don't use cache
        with patch.object(collector, 'is_cache_valid', return_value=False):
            # Mock Path location
            with patch('src.data.macro_collection.MACRO_DATA_DIR', cache_dir):
                result = collector.fetch_single_source('gdp', config)
                
                # Verify result
                assert result is not None
                assert result.equals(sample_gdp_data)
                mock_ak_module.macro_china_gdp_yearly.assert_called_once()
                assert cache_file.exists()
    
    def test_fetch_single_source_error_handling(self, mock_ak_module):
        """Test error handling in fetch_single_source."""
        # Setup
        collector = MacroDataCollector()
        mock_ak_module.macro_china_gdp_yearly.side_effect = Exception("API error")
        
        # Create mock config
        config = {
            'function': 'ak.macro_china_gdp_yearly',
            'filename': 'china_gdp_yearly.csv',
            'description': "Test GDP data"
        }
        
        # Ensure we don't use cache
        with patch.object(collector, 'is_cache_valid', return_value=False):
            result = collector.fetch_single_source('gdp', config)
            
            # Verify result is None on error
            assert result is None
            mock_ak_module.macro_china_gdp_yearly.assert_called()
    
    def test_fetch_single_source_retry_mechanism(self, mock_ak_module, sample_gdp_data):
        """Test that fetch_single_source retries on failure."""
        # Setup
        collector = MacroDataCollector()
        
        # Make the API call fail once, then succeed
        mock_ak_module.macro_china_gdp_yearly.side_effect = [
            Exception("API error"),
            sample_gdp_data
        ]
        
        # Create mock config
        config = {
            'function': 'ak.macro_china_gdp_yearly',
            'filename': 'china_gdp_yearly.csv',
            'description': "Test GDP data"
        }
        
        # Mock time.sleep to speed up test
        with patch('time.sleep'):
            # Ensure we don't use cache
            with patch.object(collector, 'is_cache_valid', return_value=False):
                # Mock file operations
                with patch('pandas.DataFrame.to_csv'):
                    result = collector.fetch_single_source('gdp', config)
                    
                    # Verify result
                    assert result is not None
                    assert result.equals(sample_gdp_data)
                    assert mock_ak_module.macro_china_gdp_yearly.call_count == 2
    
    def test_fetch_all_sources(self, mock_ak_module, sample_gdp_data, sample_cpi_data):
        """Test fetching all data sources."""
        # Setup
        collector = MacroDataCollector(sources=['gdp', 'cpi'])
        
        # Mock fetch_single_source to return our sample data
        with patch.object(collector, 'fetch_single_source') as mock_fetch:
            mock_fetch.side_effect = [sample_gdp_data, sample_cpi_data]
            
            results = collector.fetch_all_sources()
            
            # Verify results
            assert len(results) == 2
            assert results['gdp'].equals(sample_gdp_data)
            assert results['cpi'].equals(sample_cpi_data)
            assert mock_fetch.call_count == 2
    
    def test_get_data_all_sources(self, mock_ak_module):
        """Test get_data for all sources."""
        # Setup
        collector = MacroDataCollector()

        # Mock fetch_all_sources
        mock_results = {'gdp': pd.DataFrame({'test': [1, 2, 3]})}
        with patch.object(collector, 'fetch_all_sources', return_value=mock_results) as mock_fetch:
            results = collector.get_data()

            # Verify results
            assert results == mock_results
            # More flexible assertion that works with both positional and keyword arguments
            assert mock_fetch.call_count == 1
            # Verify the argument value without being strict about positional vs. keyword
            call_args = mock_fetch.call_args
            assert call_args[0] == (False,) or call_args[1] == {'force_refresh': False}
    
    def test_get_data_single_source(self, mock_ak_module, sample_gdp_data):
        """Test get_data for a single source."""
        # Setup
        collector = MacroDataCollector()
        
        # Mock fetch_single_source
        with patch.object(collector, 'fetch_single_source', return_value=sample_gdp_data) as mock_fetch:
            result = collector.get_data(source='gdp')
            
            # Verify result
            assert result.equals(sample_gdp_data)
            mock_fetch.assert_called_once()
    
    def test_get_data_unknown_source(self, caplog):
        """Test get_data with an unknown source."""
        collector = MacroDataCollector()
        result = collector.get_data(source='unknown_source')
        
        assert result is None
        assert "Unknown source" in caplog.text
    
    def test_get_data_error_handling(self):
        """Test error handling in get_data."""
        # Setup
        collector = MacroDataCollector()
        
        # Mock fetch_all_sources to raise exception
        with patch.object(collector, 'fetch_all_sources', side_effect=Exception("Test error")):
            result = collector.get_data()
            
            # Verify empty dict on error
            assert result == {}


if __name__ == '__main__':
    pytest.main(['-xvs', __file__]) 