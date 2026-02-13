"""
Feature Computer Module

Computes features with strict temporal constraints and validation.
"""

from typing import Dict, Any, List, Callable
import pandas as pd
import numpy as np
from .temporal_index import TemporalIndex


class FeatureComputer:
    """
    Computes features while maintaining temporal correctness.
    All computations respect the temporal index and specified lags.
    """
    
    def __init__(self, temporal_index: TemporalIndex):
        """
        Initialize feature computer with temporal index.
        
        Args:
            temporal_index: TemporalIndex instance for time management
        """
        self.temporal_index = temporal_index
        self.feature_cache: Dict[str, pd.Series] = {}
    
    def compute_simple_feature(
        self,
        data: pd.Series,
        operation: str,
        lag: int = 1,
        **kwargs
    ) -> pd.Series:
        """
        Compute a simple feature with specified lag.
        
        Args:
            data: Input time series
            operation: Operation to apply (e.g., 'pct_change', 'diff')
            lag: Minimum lag to prevent lookahead
            **kwargs: Additional arguments for the operation
        
        Returns:
            Computed feature series with proper temporal alignment
        """
        if operation == 'pct_change':
            result = data.pct_change(periods=lag)
        elif operation == 'diff':
            result = data.diff(periods=lag)
        elif operation == 'log_return':
            result = np.log(data / data.shift(lag))
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Apply lag to ensure no lookahead
        result = result.shift(lag)
        
        return result
    
    def compute_rolling_feature(
        self,
        data: pd.Series,
        window: int,
        operation: str,
        lag: int = 1,
        **kwargs
    ) -> pd.Series:
        """
        Compute a rolling window feature with specified lag.
        
        Args:
            data: Input time series
            window: Rolling window size
            operation: Aggregation operation (e.g., 'mean', 'std', 'sum')
            lag: Minimum lag to prevent lookahead
            **kwargs: Additional arguments for the operation
        
        Returns:
            Computed rolling feature with proper temporal alignment
        """
        rolling = data.rolling(window=window, min_periods=window)
        
        if operation == 'mean':
            result = rolling.mean()
        elif operation == 'std':
            result = rolling.std()
        elif operation == 'sum':
            result = rolling.sum()
        elif operation == 'min':
            result = rolling.min()
        elif operation == 'max':
            result = rolling.max()
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Apply lag to ensure no lookahead
        result = result.shift(lag)
        
        return result
    
    def compute_derived_feature(
        self,
        dependencies: List[str],
        computation_func: Callable,
        lag: int = 1
    ) -> pd.Series:
        """
        Compute a derived feature from other features.
        
        Args:
            dependencies: List of feature names this feature depends on
            computation_func: Function that takes dependency features and returns result
            lag: Additional lag to apply
        
        Returns:
            Computed derived feature
        """
        # Retrieve dependency features
        dep_features = {
            name: self.feature_cache.get(name)
            for name in dependencies
        }
        
        # Check all dependencies exist
        missing = [name for name, feat in dep_features.items() if feat is None]
        if missing:
            raise ValueError(f"Missing dependencies: {missing}")
        
        # Compute derived feature
        result = computation_func(**dep_features)
        
        # Apply additional lag
        if lag > 0:
            result = result.shift(lag)
        
        return result
    
    def register_feature(self, name: str, feature_data: pd.Series) -> None:
        """
        Register a computed feature in the cache.
        
        Args:
            name: Feature name
            feature_data: Computed feature series
        """
        self.feature_cache[name] = feature_data
    
    def get_feature(self, name: str) -> pd.Series:
        """
        Retrieve a computed feature from cache.
        
        Args:
            name: Feature name
        
        Returns:
            Cached feature series
        """
        if name not in self.feature_cache:
            raise KeyError(f"Feature '{name}' not found in cache")
        
        return self.feature_cache[name]
    
    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self.feature_cache.clear()
