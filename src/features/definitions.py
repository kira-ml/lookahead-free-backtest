"""
Feature Definitions Module

Contains custom feature computation functions.
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict


def momentum_signal(price_return_1d: pd.Series, volume_ma_5d: pd.Series) -> pd.Series:
    """
    Compute momentum signal from price returns and volume.
    
    Args:
        price_return_1d: Daily price returns
        volume_ma_5d: 5-day moving average of volume
    
    Returns:
        Momentum signal series
    """
    # Simple momentum signal: positive return with above-average volume
    signal = (price_return_1d > 0) & (volume_ma_5d > volume_ma_5d.median())
    return signal.astype(float)


def volatility_adjusted_return(returns: pd.Series, volatility: pd.Series) -> pd.Series:
    """
    Compute volatility-adjusted returns.
    
    Args:
        returns: Return series
        volatility: Volatility series
    
    Returns:
        Adjusted return series
    """
    # Avoid division by zero
    vol_safe = volatility.replace(0, np.nan)
    return returns / vol_safe


def trend_strength(price: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute trend strength indicator.
    
    Args:
        price: Price series
        window: Lookback window
    
    Returns:
        Trend strength series
    """
    # Linear regression slope as trend indicator
    def compute_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        y = series.values
        if np.all(np.isnan(y)):
            return np.nan
        coef = np.polyfit(x, y, 1)[0]
        return coef
    
    return price.rolling(window=window).apply(compute_slope, raw=False)


def mean_reversion_signal(
    price: pd.Series,
    lookback: int = 20,
    threshold: float = 2.0
) -> pd.Series:
    """
    Compute mean reversion signal.
    
    Args:
        price: Price series
        lookback: Lookback period for mean and std
        threshold: Number of standard deviations for signal
    
    Returns:
        Mean reversion signal (-1, 0, 1)
    """
    rolling_mean = price.rolling(window=lookback).mean()
    rolling_std = price.rolling(window=lookback).std()
    
    z_score = (price - rolling_mean) / rolling_std
    
    # Signal: -1 (overbought), 0 (neutral), 1 (oversold)
    signal = pd.Series(0, index=price.index)
    signal[z_score > threshold] = -1  # Overbought -> sell
    signal[z_score < -threshold] = 1   # Oversold -> buy
    
    return signal


def relative_strength(
    price: pd.Series,
    benchmark: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Compute relative strength vs benchmark.
    
    Args:
        price: Asset price series
        benchmark: Benchmark price series
        window: Rolling window for returns
    
    Returns:
        Relative strength series
    """
    asset_return = price.pct_change(window)
    benchmark_return = benchmark.pct_change(window)
    
    relative_return = asset_return - benchmark_return
    
    return relative_return


# Registry of custom feature functions
CUSTOM_FEATURES: Dict[str, Callable] = {
    'momentum_signal': momentum_signal,
    'volatility_adjusted_return': volatility_adjusted_return,
    'trend_strength': trend_strength,
    'mean_reversion_signal': mean_reversion_signal,
    'relative_strength': relative_strength,
}


def get_feature_function(name: str) -> Callable:
    """
    Retrieve a custom feature function by name.
    
    Args:
        name: Name of the feature function
    
    Returns:
        Feature computation function
    """
    if name not in CUSTOM_FEATURES:
        raise ValueError(f"Unknown custom feature: {name}")
    
    return CUSTOM_FEATURES[name]
