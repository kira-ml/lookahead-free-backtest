"""
Causality Validator Module

Validates causal relationships and prevents lookahead bias in feature dependencies.
"""

from typing import Dict, List, Set, Tuple
from datetime import datetime
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalityValidator:
    """
    Validates that feature dependencies respect causality and temporal ordering.
    Detects circular dependencies and lookahead bias.
    """
    
    def __init__(self):
        """Initialize the causality validator."""
        self.feature_graph: Dict[str, List[str]] = {}
        self.feature_lags: Dict[str, int] = {}
        self.validation_errors: List[str] = []
    
    def register_feature(
        self,
        name: str,
        dependencies: List[str],
        lag: int
    ) -> None:
        """
        Register a feature and its dependencies.
        
        Args:
            name: Feature name
            dependencies: List of features this feature depends on
            lag: Temporal lag applied to this feature
        """
        if lag < 1:
            self.validation_errors.append(
                f"Feature '{name}' has insufficient lag ({lag}). "
                f"Minimum lag of 1 required to prevent lookahead."
            )
        
        self.feature_graph[name] = dependencies
        self.feature_lags[name] = lag
        
        logger.info(f"Registered feature '{name}' with lag={lag}, deps={dependencies}")
    
    def validate_causality(self) -> Tuple[bool, List[str]]:
        """
        Validate all registered features for causality violations.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        self.validation_errors.clear()
        
        # Check for circular dependencies
        self._check_circular_dependencies()
        
        # Check for invalid lags
        self._check_lag_consistency()
        
        # Check dependency availability
        self._check_dependency_availability()
        
        is_valid = len(self.validation_errors) == 0
        
        if is_valid:
            logger.info("✓ All causality checks passed")
        else:
            logger.error(f"✗ Found {len(self.validation_errors)} causality violations")
            for error in self.validation_errors:
                logger.error(f"  - {error}")
        
        return is_valid, self.validation_errors
    
    def _check_circular_dependencies(self) -> None:
        """Detect circular dependencies in the feature graph."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        
        def has_cycle(feature: str, path: List[str]) -> bool:
            visited.add(feature)
            rec_stack.add(feature)
            path.append(feature)
            
            for dependency in self.feature_graph.get(feature, []):
                if dependency not in visited:
                    if has_cycle(dependency, path.copy()):
                        return True
                elif dependency in rec_stack:
                    cycle_path = path[path.index(dependency):] + [dependency]
                    self.validation_errors.append(
                        f"Circular dependency detected: {' -> '.join(cycle_path)}"
                    )
                    return True
            
            rec_stack.remove(feature)
            return False
        
        for feature in self.feature_graph:
            if feature not in visited:
                has_cycle(feature, [])
    
    def _check_lag_consistency(self) -> None:
        """Check that dependent features have sufficient lags."""
        for feature, dependencies in self.feature_graph.items():
            feature_lag = self.feature_lags.get(feature, 0)
            
            for dep in dependencies:
                dep_lag = self.feature_lags.get(dep, 0)
                
                # Dependent feature should not have a larger lag than the feature using it
                # This ensures temporal consistency
                if dep_lag > feature_lag:
                    logger.warning(
                        f"Feature '{feature}' (lag={feature_lag}) depends on "
                        f"'{dep}' (lag={dep_lag}) with larger lag"
                    )
    
    def _check_dependency_availability(self) -> None:
        """Check that all dependencies are registered."""
        for feature, dependencies in self.feature_graph.items():
            for dep in dependencies:
                if dep not in self.feature_graph:
                    self.validation_errors.append(
                        f"Feature '{feature}' depends on unregistered feature '{dep}'"
                    )
    
    def get_computation_order(self) -> List[str]:
        """
        Get topologically sorted order for feature computation.
        
        Returns:
            List of feature names in valid computation order
        """
        visited: Set[str] = set()
        order: List[str] = []
        
        def visit(feature: str) -> None:
            if feature in visited:
                return
            
            visited.add(feature)
            
            for dep in self.feature_graph.get(feature, []):
                visit(dep)
            
            order.append(feature)
        
        for feature in self.feature_graph:
            visit(feature)
        
        return order
    
    def visualize_graph(self) -> str:
        """
        Create a text representation of the feature dependency graph.
        
        Returns:
            String representation of the graph
        """
        lines = ["Feature Dependency Graph:", "=" * 40]
        
        for feature in self.get_computation_order():
            deps = self.feature_graph.get(feature, [])
            lag = self.feature_lags.get(feature, 0)
            
            if deps:
                lines.append(f"{feature} (lag={lag}) <- {', '.join(deps)}")
            else:
                lines.append(f"{feature} (lag={lag}) [no dependencies]")
        
        return "\n".join(lines)
