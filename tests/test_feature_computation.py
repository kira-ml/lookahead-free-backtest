"""
Tests for Feature Computation

Validates feature definitions and computation logic.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import pandas as pd
import numpy as np
from features.definitions import (
    momentum_signal,
    volatility_adjusted_return,
    trend_strength,
    mean_reversion_signal,
    relative_strength
)


class TestFeatureDefinitions:
    """Test cases for custom feature definitions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        self.price_returns = pd.Series(
            np.random.randn(100) * 0.02,
            index=dates
        )
        
        self.volume = pd.Series(
            np.random.randint(1000, 10000, 100),
            index=dates
        )
        
        self.prices = pd.Series(
            100 * np.exp(np.cumsum(self.price_returns)),
            index=dates
        )
    
    def test_momentum_signal(self):
        """Test momentum signal computation."""
        volume_ma = self.volume.rolling(5).mean()
        
        signal = momentum_signal(self.price_returns, volume_ma)
        
        # Signal should be 0 or 1
        assert set(signal.dropna().unique()).issubset({0.0, 1.0})
        
        # Should have some signal activations
        assert signal.sum() > 0
    
    def test_volatility_adjusted_return(self):
        """Test volatility-adjusted return computation."""
        volatility = self.price_returns.rolling(10).std()
        
        adjusted = volatility_adjusted_return(self.price_returns, volatility)
        
        # Should have same length
        assert len(adjusted) == len(self.price_returns)
        
        # Adjusted returns should be normalized by volatility
        # (lower magnitude on average)
        assert adjusted.abs().mean() < self.price_returns.abs().mean() * 2
    
    def test_trend_strength(self):
        """Test trend strength indicator."""
        trend = trend_strength(self.prices, window=20)
        
        # Should have NaN for first window
        assert pd.isna(trend.iloc[:19]).all()
        
        # Should have values after window
        assert not pd.isna(trend.iloc[20])
        
        # Trend should be numeric
        assert pd.api.types.is_numeric_dtype(trend)
    
    def test_mean_reversion_signal(self):
        """Test mean reversion signal."""
        signal = mean_reversion_signal(self.prices, lookback=20, threshold=2.0)
        
        # Signal should be -1, 0, or 1
        assert set(signal.dropna().unique()).issubset({-1.0, 0.0, 1.0})
        
        # Most values should be neutral (0)
        assert (signal == 0).sum() > (signal != 0).sum()
    
    def test_relative_strength(self):
        """Test relative strength computation."""
        # Create a benchmark series
        benchmark = pd.Series(
            100 * np.exp(np.cumsum(np.random.randn(100) * 0.01)),
            index=self.prices.index
        )
        
        rel_strength = relative_strength(self.prices, benchmark, window=20)
        
        # Should have NaN for first window
        assert pd.isna(rel_strength.iloc[:19]).all()
        
        # Should have values after window
        assert not pd.isna(rel_strength.iloc[20])
        
        # Relative strength should be numeric
        assert pd.api.types.is_numeric_dtype(rel_strength)


class TestFeatureRegistry:
    """Test cases for feature registry."""
    
    def test_load_from_config(self):
        """Test loading feature specifications from config."""
        from features.registry import FeatureRegistry
        
        config_path = Path(__file__).parent.parent / 'config' / 'feature_specs.yaml'
        registry = FeatureRegistry(str(config_path))
        
        # Should have loaded features
        assert len(registry.list_features()) > 0
        
        # Check specific feature
        if 'price_return_1d' in registry.list_features():
            feature = registry.get_feature('price_return_1d')
            assert feature['type'] == 'simple'
            assert feature['lag'] >= 1
    
    def test_feature_dependencies(self):
        """Test feature dependency resolution."""
        from features.registry import FeatureRegistry
        
        registry = FeatureRegistry()
        
        # Register features with dependencies
        registry.register_feature(
            name='feature_a',
            type='simple',
            computation='pct_change',
            lag=1,
            dependencies=[]
        )
        
        registry.register_feature(
            name='feature_b',
            type='derived',
            computation='custom',
            lag=1,
            dependencies=['feature_a']
        )
        
        # Check dependencies
        deps = registry.get_dependencies('feature_b')
        assert 'feature_a' in deps


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
