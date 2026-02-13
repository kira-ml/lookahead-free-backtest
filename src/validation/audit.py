"""
Audit Module

Provides comprehensive auditing and logging for temporal correctness validation.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Logs all operations related to feature computation and data access
    to ensure reproducibility and detect temporal violations.
    """
    
    def __init__(self, audit_log_path: str, strict_mode: bool = True):
        """
        Initialize audit logger.
        
        Args:
            audit_log_path: Path to audit log file
            strict_mode: If True, raise exceptions on violations
        """
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.strict_mode = strict_mode
        self.violations: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
    
    def log_event(
        self,
        event_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event (e.g., 'feature_compute', 'data_access')
            description: Human-readable description
            metadata: Additional metadata
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'description': description,
            'metadata': metadata or {}
        }
        
        self.events.append(event)
        
        logger.debug(f"[AUDIT] {event_type}: {description}")
    
    def log_violation(
        self,
        violation_type: str,
        description: str,
        severity: str = 'high',
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a temporal violation.
        
        Args:
            violation_type: Type of violation (e.g., 'lookahead', 'data_leakage')
            description: Description of the violation
            severity: Severity level ('low', 'medium', 'high')
            metadata: Additional metadata
        """
        violation = {
            'timestamp': datetime.now().isoformat(),
            'violation_type': violation_type,
            'severity': severity,
            'description': description,
            'metadata': metadata or {}
        }
        
        self.violations.append(violation)
        
        logger.error(f"[VIOLATION] {violation_type} ({severity}): {description}")
        
        if self.strict_mode and severity == 'high':
            raise ValueError(f"Temporal violation detected: {description}")
    
    def log_feature_computation(
        self,
        feature_name: str,
        computation_time: datetime,
        dependencies: List[str],
        lag: int
    ) -> None:
        """
        Log a feature computation event.
        
        Args:
            feature_name: Name of computed feature
            computation_time: Time at which computation occurred
            dependencies: List of dependency features
            lag: Applied lag
        """
        self.log_event(
            event_type='feature_computation',
            description=f"Computed feature '{feature_name}'",
            metadata={
                'feature_name': feature_name,
                'computation_time': computation_time.isoformat(),
                'dependencies': dependencies,
                'lag': lag
            }
        )
    
    def log_data_access(
        self,
        requested_time: datetime,
        accessed_times: List[datetime],
        current_time: datetime
    ) -> None:
        """
        Log a data access event and check for lookahead.
        
        Args:
            requested_time: Requested timestamp
            accessed_times: List of accessed timestamps
            current_time: Current point-in-time
        """
        # Check for lookahead
        future_access = [t for t in accessed_times if t > current_time]
        
        if future_access:
            self.log_violation(
                violation_type='lookahead',
                description=(
                    f"Accessed future data at {current_time}: "
                    f"{[t.isoformat() for t in future_access]}"
                ),
                severity='high',
                metadata={
                    'current_time': current_time.isoformat(),
                    'future_times': [t.isoformat() for t in future_access]
                }
            )
        else:
            self.log_event(
                event_type='data_access',
                description=f"Valid data access at {current_time}",
                metadata={
                    'requested_time': requested_time.isoformat(),
                    'accessed_count': len(accessed_times),
                    'current_time': current_time.isoformat()
                }
            )
    
    def save_audit_log(self) -> None:
        """Save audit log to file."""
        audit_data = {
            'generated_at': datetime.now().isoformat(),
            'strict_mode': self.strict_mode,
            'total_events': len(self.events),
            'total_violations': len(self.violations),
            'events': self.events,
            'violations': self.violations
        }
        
        with open(self.audit_log_path, 'w') as f:
            json.dump(audit_data, f, indent=2)
        
        logger.info(f"Saved audit log to {self.audit_log_path}")
    
    def generate_report(self) -> str:
        """
        Generate a human-readable audit report.
        
        Returns:
            Formatted audit report string
        """
        lines = [
            "=" * 60,
            "AUDIT REPORT",
            "=" * 60,
            f"Generated at: {datetime.now().isoformat()}",
            f"Strict mode: {self.strict_mode}",
            f"Total events: {len(self.events)}",
            f"Total violations: {len(self.violations)}",
            ""
        ]
        
        if self.violations:
            lines.append("VIOLATIONS:")
            lines.append("-" * 60)
            
            for violation in self.violations:
                lines.append(
                    f"[{violation['severity'].upper()}] {violation['violation_type']}: "
                    f"{violation['description']}"
                )
                lines.append(f"  Time: {violation['timestamp']}")
                lines.append("")
        else:
            lines.append("✓ No violations detected")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_violation_summary(self) -> Dict[str, int]:
        """
        Get summary of violations by type.
        
        Returns:
            Dictionary mapping violation types to counts
        """
        summary: Dict[str, int] = {}
        
        for violation in self.violations:
            vtype = violation['violation_type']
            summary[vtype] = summary.get(vtype, 0) + 1
        
        return summary
