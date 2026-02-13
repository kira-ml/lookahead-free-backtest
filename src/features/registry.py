"""
Feature Registry Module

Manages feature definitions and metadata.
"""

from typing import Dict, List, Any, Optional
import yaml
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    Centralized registry for feature definitions and metadata.
    Loads feature specifications from configuration files.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize feature registry.
        
        Args:
            config_path: Path to feature specifications YAML file
        """
        self.features: Dict[str, Dict[str, Any]] = {}
        self.temporal_constraints: Dict[str, Any] = {}
        
        if config_path:
            self.load_from_config(config_path)
    
    def load_from_config(self, config_path: str) -> None:
        """
        Load feature specifications from YAML configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        logger.info(f"Loading feature specifications from {config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load feature definitions
        if 'features' in config:
            for feature_spec in config['features']:
                self.register_feature(**feature_spec)
        
        # Load temporal constraints
        if 'temporal_constraints' in config:
            self.temporal_constraints = config['temporal_constraints']
            logger.info(f"Loaded temporal constraints: {self.temporal_constraints}")
    
    def register_feature(
        self,
        name: str,
        type: str,
        computation: str,
        lag: int,
        dependencies: Optional[List[str]] = None,
        window: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Register a feature in the registry.
        
        Args:
            name: Feature name
            type: Feature type (simple, rolling, derived)
            computation: Computation method
            lag: Temporal lag
            dependencies: List of dependent features
            window: Window size for rolling features
            **kwargs: Additional feature parameters
        """
        feature_def = {
            'name': name,
            'type': type,
            'computation': computation,
            'lag': lag,
            'dependencies': dependencies or [],
            'window': window,
            **kwargs
        }
        
        self.features[name] = feature_def
        
        logger.info(f"Registered feature: {name} (type={type}, lag={lag})")
    
    def get_feature(self, name: str) -> Dict[str, Any]:
        """
        Get feature definition by name.
        
        Args:
            name: Feature name
        
        Returns:
            Feature definition dictionary
        """
        if name not in self.features:
            raise KeyError(f"Feature '{name}' not found in registry")
        
        return self.features[name]
    
    def list_features(self) -> List[str]:
        """
        List all registered feature names.
        
        Returns:
            List of feature names
        """
        return list(self.features.keys())
    
    def get_features_by_type(self, feature_type: str) -> List[str]:
        """
        Get all features of a specific type.
        
        Args:
            feature_type: Type of features to retrieve
        
        Returns:
            List of feature names
        """
        return [
            name for name, spec in self.features.items()
            if spec['type'] == feature_type
        ]
    
    def get_dependencies(self, feature_name: str) -> List[str]:
        """
        Get dependencies for a feature.
        
        Args:
            feature_name: Name of the feature
        
        Returns:
            List of dependency feature names
        """
        feature = self.get_feature(feature_name)
        return feature.get('dependencies', [])
    
    def validate_registry(self) -> bool:
        """
        Validate all feature definitions in the registry.
        
        Returns:
            True if all features are valid
        """
        errors = []
        
        for name, spec in self.features.items():
            # Check required fields
            required_fields = ['type', 'computation', 'lag']
            for field in required_fields:
                if field not in spec:
                    errors.append(f"Feature '{name}' missing required field: {field}")
            
            # Check lag constraint
            min_lag = self.temporal_constraints.get('min_lag', 1)
            if spec.get('lag', 0) < min_lag:
                errors.append(
                    f"Feature '{name}' lag ({spec['lag']}) below minimum ({min_lag})"
                )
            
            # Check dependencies exist
            for dep in spec.get('dependencies', []):
                if dep not in self.features:
                    errors.append(
                        f"Feature '{name}' depends on undefined feature '{dep}'"
                    )
        
        if errors:
            for error in errors:
                logger.error(error)
            return False
        
        logger.info(f"✓ Registry validation passed for {len(self.features)} features")
        return True
