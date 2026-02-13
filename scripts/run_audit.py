"""
Audit Script

Runs comprehensive validation and generates audit reports.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import yaml
from validation.audit import AuditLogger
from features.registry import FeatureRegistry
from core.causality_validator import CausalityValidator
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main audit workflow."""
    project_root = Path(__file__).parent.parent
    
    # Load pipeline configuration
    config_path = project_root / 'config' / 'pipeline_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize audit logger
    audit_logger = AuditLogger(
        audit_log_path=config['validation']['audit_log_path'],
        strict_mode=config['validation']['strict_mode']
    )
    
    logger.info("Starting validation audit...")
    
    # Load feature specifications
    feature_config_path = project_root / 'config' / 'feature_specs.yaml'
    registry = FeatureRegistry(str(feature_config_path))
    
    # Validate feature registry
    audit_logger.log_event(
        event_type='validation',
        description='Validating feature registry'
    )
    
    if not registry.validate_registry():
        audit_logger.log_violation(
            violation_type='invalid_registry',
            description='Feature registry validation failed',
            severity='high'
        )
    else:
        audit_logger.log_event(
            event_type='validation',
            description='Feature registry validation passed'
        )
    
    # Validate causality
    validator = CausalityValidator()
    
    for feature_name in registry.list_features():
        feature_spec = registry.get_feature(feature_name)
        validator.register_feature(
            name=feature_name,
            dependencies=feature_spec['dependencies'],
            lag=feature_spec['lag']
        )
    
    audit_logger.log_event(
        event_type='validation',
        description='Validating causality constraints'
    )
    
    is_valid, errors = validator.validate_causality()
    
    if not is_valid:
        for error in errors:
            audit_logger.log_violation(
                violation_type='causality_violation',
                description=error,
                severity='high'
            )
    else:
        audit_logger.log_event(
            event_type='validation',
            description='Causality validation passed'
        )
    
    # Generate and save report
    audit_logger.save_audit_log()
    
    report = audit_logger.generate_report()
    print("\n" + report)
    
    # Print violation summary
    violation_summary = audit_logger.get_violation_summary()
    if violation_summary:
        print("\nViolation Summary:")
        for vtype, count in violation_summary.items():
            print(f"  {vtype}: {count}")
    
    logger.info("✓ Audit completed")


if __name__ == '__main__':
    main()
