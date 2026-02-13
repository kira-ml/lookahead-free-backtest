"""
Data Storage Module

Manages efficient storage and retrieval of time-series data and features.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Efficiently stores and retrieves computed features.
    Supports incremental updates and versioning.
    """
    
    def __init__(self, storage_path: str):
        """
        Initialize feature store.
        
        Args:
            storage_path: Path to feature storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def save_feature(
        self,
        feature_name: str,
        data: pd.Series,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a feature to storage.
        
        Args:
            feature_name: Name of the feature
            data: Feature data as Series
            metadata: Optional metadata dictionary
        """
        feature_path = self.storage_path / f"{feature_name}.parquet"
        
        # Convert Series to DataFrame for parquet storage
        df = data.to_frame(name='value')
        df.to_parquet(feature_path)
        
        # Save metadata
        if metadata:
            self.metadata[feature_name] = metadata
        
        logger.info(f"Saved feature '{feature_name}' to {feature_path}")
    
    def load_feature(self, feature_name: str) -> pd.Series:
        """
        Load a feature from storage.
        
        Args:
            feature_name: Name of the feature to load
        
        Returns:
            Feature data as Series
        """
        feature_path = self.storage_path / f"{feature_name}.parquet"
        
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature '{feature_name}' not found in storage")
        
        df = pd.read_parquet(feature_path)
        
        logger.info(f"Loaded feature '{feature_name}' from {feature_path}")
        
        return df['value']
    
    def feature_exists(self, feature_name: str) -> bool:
        """
        Check if a feature exists in storage.
        
        Args:
            feature_name: Name of the feature
        
        Returns:
            True if feature exists
        """
        feature_path = self.storage_path / f"{feature_name}.parquet"
        return feature_path.exists()
    
    def list_features(self) -> list:
        """
        List all available features in storage.
        
        Returns:
            List of feature names
        """
        feature_files = self.storage_path.glob("*.parquet")
        return [f.stem for f in feature_files]
    
    def delete_feature(self, feature_name: str) -> None:
        """
        Delete a feature from storage.
        
        Args:
            feature_name: Name of the feature to delete
        """
        feature_path = self.storage_path / f"{feature_name}.parquet"
        
        if feature_path.exists():
            feature_path.unlink()
            logger.info(f"Deleted feature '{feature_name}'")
        
        if feature_name in self.metadata:
            del self.metadata[feature_name]


class TimeSeriesStore:
    """
    Manages storage of raw time-series data.
    """
    
    def __init__(self, storage_path: str):
        """
        Initialize time-series store.
        
        Args:
            storage_path: Path to time-series storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def save_timeseries(
        self,
        name: str,
        data: pd.DataFrame,
        partition_by: Optional[str] = None
    ) -> None:
        """
        Save time-series data.
        
        Args:
            name: Name of the time series
            data: DataFrame with time-series data
            partition_by: Optional column to partition by
        """
        output_path = self.storage_path / f"{name}.parquet"
        
        if partition_by and partition_by in data.columns:
            # Save with partitioning
            data.to_parquet(
                output_path,
                partition_cols=[partition_by],
                index=False
            )
        else:
            # Save without partitioning
            data.to_parquet(output_path, index=False)
        
        logger.info(f"Saved time series '{name}' to {output_path}")
    
    def load_timeseries(self, name: str) -> pd.DataFrame:
        """
        Load time-series data.
        
        Args:
            name: Name of the time series
        
        Returns:
            DataFrame with time-series data
        """
        input_path = self.storage_path / f"{name}.parquet"
        
        if not input_path.exists():
            raise FileNotFoundError(f"Time series '{name}' not found")
        
        df = pd.read_parquet(input_path)
        
        logger.info(f"Loaded time series '{name}' from {input_path}")
        
        return df
