"""
Tests for Temporal Correctness

Validates that temporal index and feature computation prevent lookahead bias.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.temporal_index import TemporalIndex
from core.feature_computer import FeatureComputer


class TestTemporalIndex:
    """Test cases for TemporalIndex."""
    
    def test_initialization(self):
        """Test temporal index initialization."""
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        temporal_index = TemporalIndex(dates)
        
        assert len(temporal_index.timestamps) == len(dates)
        assert temporal_index.timestamps.is_monotonic_increasing
    
    def test_duplicate_timestamps_rejected(self):
        """Test that duplicate timestamps are rejected."""
        dates = pd.DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-02'])
        
        with pytest.raises(ValueError, match="Timestamps must be unique"):
            TemporalIndex(dates)
    
    def test_unsorted_timestamps_sorted(self):
        """Test that unsorted timestamps are automatically sorted."""
        dates = pd.DatetimeIndex(['2020-01-03', '2020-01-01', '2020-01-02'])
        temporal_index = TemporalIndex(dates)
        
        assert temporal_index.timestamps.is_monotonic_increasing
    
    def test_set_current_time(self):
        """Test setting current time."""
        dates = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
        temporal_index = TemporalIndex(dates)
        
        temporal_index.set_current_time(pd.Timestamp('2020-01-05'))
        assert temporal_index._current_idx == 4  # 0-indexed
    
    def test_get_available_data_mask(self):
        """Test available data mask."""
        dates = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
        temporal_index = TemporalIndex(dates)
        temporal_index.set_current_time(pd.Timestamp('2020-01-05'))
        
        mask = temporal_index.get_available_data_mask()
        
        # Should have access to first 5 days (indices 0-4)
        assert mask.sum() == 5
        assert all(mask.iloc[:5])
        assert not any(mask.iloc[5:])
    
    def test_lookback_window(self):
        """Test lookback window retrieval."""
        dates = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
        temporal_index = TemporalIndex(dates)
        temporal_index.set_current_time(pd.Timestamp('2020-01-06'))
        
        # Get 3-day lookback window with 1-day lag
        window = temporal_index.get_lookback_window(window=3, lag=1)
        
        # Should get days 2, 3, 4 (indices 1, 2, 3)
        assert len(window) == 3
        assert window[0] == pd.Timestamp('2020-01-02')
        assert window[-1] == pd.Timestamp('2020-01-04')
    
    def test_validate_no_lookahead(self):
        """Test lookahead validation."""
        dates = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
        temporal_index = TemporalIndex(dates)
        temporal_index.set_current_time(pd.Timestamp('2020-01-05'))
        
        # Valid access (past dates)
        past_dates = [pd.Timestamp('2020-01-03'), pd.Timestamp('2020-01-04')]
        assert temporal_index.validate_no_lookahead(past_dates)
        
        # Invalid access (future date)
        future_dates = [pd.Timestamp('2020-01-07')]
        with pytest.raises(ValueError, match="Lookahead detected"):
            temporal_index.validate_no_lookahead(future_dates)


class TestFeatureComputer:
    """Test cases for FeatureComputer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        self.temporal_index = TemporalIndex(dates)
        self.feature_computer = FeatureComputer(self.temporal_index)
        
        # Sample price data
        np.random.seed(42)
        self.prices = pd.Series(
            100 + np.cumsum(np.random.randn(len(dates))),
            index=dates
        )
    
    def test_simple_feature_with_lag(self):
        """Test that simple features respect lag."""
        result = self.feature_computer.compute_simple_feature(
            data=self.prices,
            operation='pct_change',
            lag=1
        )
        
        # First value should be NaN due to pct_change
        # Second value should also be NaN due to lag
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        
        # Third value should be computed
        assert not pd.isna(result.iloc[2])
    
    def test_rolling_feature_with_lag(self):
        """Test that rolling features respect lag."""
        result = self.feature_computer.compute_rolling_feature(
            data=self.prices,
            window=5,
            operation='mean',
            lag=1
        )
        
        # First 5 values should be NaN (window requirement)
        # Plus 1 more due to lag
        assert pd.isna(result.iloc[:6]).all()
        
        # After that, should have values
        assert not pd.isna(result.iloc[6])
    
    def test_feature_registration_and_retrieval(self):
        """Test feature caching."""
        feature_data = pd.Series([1, 2, 3, 4, 5])
        
        self.feature_computer.register_feature('test_feature', feature_data)
        
        retrieved = self.feature_computer.get_feature('test_feature')
        
        pd.testing.assert_series_equal(retrieved, feature_data)
    
    def test_derived_feature_computation(self):
        """Test derived feature computation."""
        # Create dependency features
        feature1 = pd.Series([1, 2, 3, 4, 5], index=range(5))
        feature2 = pd.Series([5, 4, 3, 2, 1], index=range(5))
        
        self.feature_computer.register_feature('feat1', feature1)
        self.feature_computer.register_feature('feat2', feature2)
        
        # Compute derived feature (sum of dependencies)
        def sum_func(feat1, feat2):
            return feat1 + feat2
        
        result = self.feature_computer.compute_derived_feature(
            dependencies=['feat1', 'feat2'],
            computation_func=sum_func,
            lag=1
        )
        
        # First value should be NaN due to lag
        assert pd.isna(result.iloc[0])
        
        # Subsequent values should be sum (shifted by 1)
        assert result.iloc[1] == 6  # 1 + 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
