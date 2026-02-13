"""
Feature Computation Script

Computes features according to specifications with temporal validation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import yaml
import pandas as pd
from data.ingestion import DataIngestion
from data.storage import FeatureStore
from core.temporal_index import TemporalIndex
from core.feature_computer import FeatureComputer
from core.causality_validator import CausalityValidator
from features.registry import FeatureRegistry
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main feature computation workflow."""
    project_root = Path(__file__).parent.parent
    
    # Load pipeline configuration
    config_path = project_root / 'config' / 'pipeline_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load feature specifications
    feature_config_path = project_root / 'config' / 'feature_specs.yaml'
    registry = FeatureRegistry(str(feature_config_path))
    
    # Validate registry
    if not registry.validate_registry():
        logger.error("Feature registry validation failed")
        return
    
    # Initialize causality validator
    validator = CausalityValidator()
    
    # Register features for causality validation
    for feature_name in registry.list_features():
        feature_spec = registry.get_feature(feature_name)
        validator.register_feature(
            name=feature_name,
            dependencies=feature_spec['dependencies'],
            lag=feature_spec['lag']
        )
    
    # Validate causality
    is_valid, errors = validator.validate_causality()
    if not is_valid:
        logger.error("Causality validation failed")
        return
    
    logger.info("Causality validation passed")
    logger.info(validator.visualize_graph())
    
    # Load processed data
    ingestion = DataIngestion(
        raw_data_path=config['data']['raw_path'],
        processed_data_path=config['data']['processed_path']
    )
    
    try:
        df = ingestion.load_processed_data('market_data')
        logger.info(f"Loaded {len(df)} rows of market data")
        
        # Create temporal index
        temporal_index = TemporalIndex(pd.DatetimeIndex(df['timestamp']))
        
        # Initialize feature computer
        feature_computer = FeatureComputer(temporal_index)
        
        # Initialize feature store
        feature_store = FeatureStore(config['data']['processed_path'])
        
        # Get computation order
        computation_order = validator.get_computation_order()
        logger.info(f"Computing features in order: {computation_order}")
        
        # Compute features
        for feature_name in computation_order:
            feature_spec = registry.get_feature(feature_name)
            logger.info(f"Computing feature: {feature_name}")
            
            if feature_spec['type'] == 'simple':
                result = feature_computer.compute_simple_feature(
                    data=df['price'],
                    operation=feature_spec['computation'],
                    lag=feature_spec['lag']
                )
            
            elif feature_spec['type'] == 'rolling':
                result = feature_computer.compute_rolling_feature(
                    data=df['volume'] if 'volume' in feature_name else df['price'],
                    window=feature_spec['window'],
                    operation=feature_spec['computation'],
                    lag=feature_spec['lag']
                )
            
            elif feature_spec['type'] == 'derived':
                # For derived features, use custom function
                from features.definitions import get_feature_function
                
                custom_func = get_feature_function(feature_spec['computation'])
                
                # Get dependencies
                dep_data = {}
                for dep_name in feature_spec['dependencies']:
                    dep_data[dep_name] = feature_computer.get_feature(dep_name)
                
                result = custom_func(**dep_data)
                
                # Apply lag
                result = result.shift(feature_spec['lag'])
            
            else:
                logger.warning(f"Unknown feature type: {feature_spec['type']}")
                continue
            
            # Register and save feature
            feature_computer.register_feature(feature_name, result)
            feature_store.save_feature(feature_name, result)
            
            logger.info(f"✓ Computed and saved feature: {feature_name}")
        
        logger.info("✓ Feature computation completed successfully")
        
    except Exception as e:
        logger.error(f"Feature computation failed: {e}")
        raise


if __name__ == '__main__':
    main()
