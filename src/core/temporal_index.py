"""
Temporal Index Module

Maintains strict temporal ordering and provides safe time-aware indexing.
Prevents any accidental access to future data.
"""

from datetime import datetime
from typing import Optional, List
import pandas as pd


class TemporalIndex:
    """
    Manages temporal indexing to ensure no lookahead bias.
    All data access is constrained by the current point-in-time.
    """
    
    def __init__(self, timestamps: pd.DatetimeIndex):
        """
        Initialize temporal index with sorted timestamps.
        
        Args:
            timestamps: DatetimeIndex that will be validated and sorted
        """
        if not isinstance(timestamps, pd.DatetimeIndex):
            raise TypeError("timestamps must be a pandas DatetimeIndex")
        
        self.timestamps = timestamps.sort_values()
        self._current_idx: Optional[int] = None
        self._validate_timestamps()
    
    def _validate_timestamps(self):
        """Validate that timestamps are unique and properly ordered."""
        if self.timestamps.has_duplicates:
            raise ValueError("Timestamps must be unique")
        
        if not self.timestamps.is_monotonic_increasing:
            raise ValueError("Timestamps must be monotonically increasing")
    
    def set_current_time(self, timestamp: datetime) -> None:
        """
        Set the current point-in-time for data access.
        
        Args:
            timestamp: Current timestamp (all future data will be inaccessible)
        """
        if timestamp not in self.timestamps:
            raise ValueError(f"Timestamp {timestamp} not in index")
        
        self._current_idx = self.timestamps.get_loc(timestamp)
    
    def get_available_data_mask(self) -> pd.Series:
        """
        Returns a boolean mask indicating which data points are available
        at the current point-in-time.
        
        Returns:
            Boolean Series where True indicates data is accessible
        """
        if self._current_idx is None:
            raise RuntimeError("Current time not set. Call set_current_time first.")
        
        mask = pd.Series(False, index=self.timestamps)
        mask.iloc[:self._current_idx + 1] = True
        return mask
    
    def get_lookback_window(self, window: int, lag: int = 1) -> pd.DatetimeIndex:
        """
        Get timestamps for a lookback window with specified lag.
        
        Args:
            window: Number of periods to look back
            lag: Lag to avoid lookahead (default: 1)
        
        Returns:
            DatetimeIndex of available historical timestamps
        """
        if self._current_idx is None:
            raise RuntimeError("Current time not set")
        
        end_idx = self._current_idx - lag + 1
        start_idx = max(0, end_idx - window)
        
        if end_idx <= 0:
            return pd.DatetimeIndex([])
        
        return self.timestamps[start_idx:end_idx]
    
    def validate_no_lookahead(self, access_times: List[datetime]) -> bool:
        """
        Validate that all access times are in the past relative to current time.
        
        Args:
            access_times: List of timestamps being accessed
        
        Returns:
            True if no lookahead, raises ValueError otherwise
        """
        if self._current_idx is None:
            raise RuntimeError("Current time not set")
        
        current_time = self.timestamps[self._current_idx]
        
        for access_time in access_times:
            if access_time > current_time:
                raise ValueError(
                    f"Lookahead detected: accessing {access_time} "
                    f"from current time {current_time}"
                )
        
        return True
