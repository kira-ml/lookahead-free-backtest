"""
Data Ingestion Module

Handles data loading, validation, and temporal sorting.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Manages data ingestion with temporal validation.
    Ensures data is properly sorted and validated before processing.
    """
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        """
        Initialize data ingestion.
        
        Args:
            raw_data_path: Path to raw data directory
            processed_data_path: Path to processed data directory
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        
        # Create directories if they don't exist
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(
        self,
        filename: str,
        timestamp_column: str = 'timestamp',
        parse_dates: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load raw data from file.
        
        Args:
            filename: Name of the data file
            timestamp_column: Name of the timestamp column
            parse_dates: List of columns to parse as dates
        
        Returns:
            Loaded DataFrame
        """
        filepath = self.raw_data_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading data from {filepath}")
        
        # Determine file format and load
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath, parse_dates=parse_dates or [timestamp_column])
        elif filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Loaded {len(df)} rows")
        
        return df
    
    def validate_and_sort(
        self,
        df: pd.DataFrame,
        timestamp_column: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Validate and sort data by timestamp.
        
        Args:
            df: Input DataFrame
            timestamp_column: Name of timestamp column
        
        Returns:
            Validated and sorted DataFrame
        """
        if timestamp_column not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found")
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            logger.info(f"Converting {timestamp_column} to datetime")
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Check for missing timestamps
        missing_count = df[timestamp_column].isna().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} rows with missing timestamps")
            df = df.dropna(subset=[timestamp_column])
        
        # Sort by timestamp
        logger.info("Sorting by timestamp")
        df = df.sort_values(timestamp_column).reset_index(drop=True)
        
        # Check for duplicates
        duplicate_count = df[timestamp_column].duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate timestamps")
        
        logger.info(f"✓ Data validated and sorted: {len(df)} rows")
        logger.info(f"  Time range: {df[timestamp_column].min()} to {df[timestamp_column].max()}")
        
        return df
    
    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename: str,
        format: str = 'parquet'
    ) -> None:
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename (without extension)
            format: File format ('parquet' or 'csv')
        """
        if format == 'parquet':
            filepath = self.processed_data_path / f"{filename}.parquet"
            df.to_parquet(filepath, index=False)
        elif format == 'csv':
            filepath = self.processed_data_path / f"{filename}.csv"
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved processed data to {filepath}")
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        Load processed data.
        
        Args:
            filename: Name of processed data file
        
        Returns:
            Loaded DataFrame
        """
        # Try parquet first, then csv
        parquet_path = self.processed_data_path / f"{filename}.parquet"
        csv_path = self.processed_data_path / f"{filename}.csv"
        
        if parquet_path.exists():
            logger.info(f"Loading processed data from {parquet_path}")
            return pd.read_parquet(parquet_path)
        elif csv_path.exists():
            logger.info(f"Loading processed data from {csv_path}")
            return pd.read_csv(csv_path, parse_dates=['timestamp'])
        else:
            raise FileNotFoundError(
                f"Processed data not found: {filename} "
                f"(checked .parquet and .csv)"
            )
