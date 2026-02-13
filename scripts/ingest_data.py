"""
Data Ingestion Script

Loads raw data, validates temporal ordering, and saves processed data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import yaml
from data.ingestion import DataIngestion
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main data ingestion workflow."""
    # Load pipeline configuration
    config_path = Path(__file__).parent.parent / 'config' / 'pipeline_config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize data ingestion
    ingestion = DataIngestion(
        raw_data_path=config['data']['raw_path'],
        processed_data_path=config['data']['processed_path']
    )
    
    # Example: Load a raw data file
    # Replace 'example_data.csv' with your actual data file
    try:
        logger.info("Starting data ingestion...")
        
        # Load raw data
        # df = ingestion.load_raw_data(
        #     filename='example_data.csv',
        #     timestamp_column='timestamp'
        # )
        
        # For demonstration, create sample data
        import pandas as pd
        import numpy as np
        
        logger.info("Creating sample data for demonstration...")
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'price': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        # Validate and sort
        df = ingestion.validate_and_sort(df, timestamp_column='timestamp')
        
        # Save processed data
        ingestion.save_processed_data(
            df=df,
            filename='market_data',
            format=config['data']['format']
        )
        
        logger.info("✓ Data ingestion completed successfully")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise


if __name__ == '__main__':
    main()
