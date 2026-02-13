# Project Structure Documentation

This document provides comprehensive documentation of the lookahead-free-backtest project structure, including component justifications, module specifications, and design decisions.

---

# Directory Tree

```
lookahead-free-backtest/
├── config/
│   ├── feature_specs.yaml
│   └── pipeline_config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── core/
│   │   ├── temporal_index.py
│   │   ├── feature_computer.py
│   │   └── causality_validator.py
│   ├── data/
│   │   ├── ingestion.py
│   │   └── storage.py
│   ├── features/
│   │   ├── registry.py
│   │   └── definitions.py
│   └── validation/
│       └── audit.py
├── scripts/
│   ├── ingest_data.py
│   ├── compute_features.py
│   └── run_audit.py
├── tests/
│   ├── test_temporal_correctness.py
│   └── test_feature_computation.py
├── experiments/
├── requirements.txt
└── README.md
```

---

# Component Justification

## `/config`
**Purpose:** Declarative feature definitions and pipeline parameters that drive system behavior.

**Justification:** 
- `feature_specs.yaml`: Centralized feature definitions (window sizes, lags) enable config-driven experimentation without code changes
- `pipeline_config.yaml`: Data paths, date ranges, validation thresholds control execution without hardcoded values
- Configuration-as-code is essential for reproducibility and auditing temporal parameters

**Non-responsibilities:**
- Not for ML model hyperparameters (out of MVP scope)
- Not for infrastructure/deployment configs (local execution only)
- Not for environment-specific settings (single environment)

## `/data`
**Purpose:** Physical storage for raw market data and processed features with temporal partitioning.

**Justification:**
- `raw/`: Immutable source data partitioned by ingestion date for versioning
- `processed/`: Computed features partitioned by date for efficient temporal queries
- Separation enforces read-only raw data and distinguishes source-of-truth from derived artifacts

**Non-responsibilities:**
- Not a database (uses file-based Parquet)
- Not versioned via Git (data tracked via content hashes)
- Not for model artifacts (experiments/ handles that)

## `/src/core`
**Purpose:** Domain-agnostic temporal infrastructure that enforces causality across all operations.

**Justification:**
- Contains the three foundational abstractions (temporal indexing, feature computation, validation) that distinguish this system from standard backtesting
- These modules have zero finance-specific logic—they implement time-aware computation primitives
- Separation enables testing temporal correctness independent of feature definitions

**Non-responsibilities:**
- Not finance domain logic (that's in features/)
- Not I/O operations (that's in data/)
- Not execution orchestration (that's in scripts/)

## `/src/data`
**Purpose:** Concrete I/O operations for market data with timestamp validation.

**Justification:**
- `ingestion.py`: Single module owns loading raw OHLCV, validating timestamps, normalizing formats
- `storage.py`: Single module owns Parquet read/write with partitioning logic
- Clear boundary between "how we compute" (core/) and "how we persist" (data/)

**Non-responsibilities:**
- Not generic file utilities (specific to OHLCV schema)
- Not data transformation (that's feature_computer.py)
- Not external API calls (assumes local CSV/Parquet files)

## `/src/features`
**Purpose:** Feature engineering domain logic and registration system.

**Justification:**
- `registry.py`: Runtime catalog of available features with their temporal metadata (lookback, lag)
- `definitions.py`: Actual feature computation functions (rolling z-score, volatility, etc.)
- Separation allows adding features without modifying core infrastructure

**Non-responsibilities:**
- Not feature storage (storage.py handles that)
- Not feature validation (causality_validator.py does that)
- Not model-specific features (MVP is model-agnostic)

## `/src/validation`
**Purpose:** Causality audit implementation that verifies temporal correctness.

**Justification:**
- Single module implements the critical 1,000-point random audit logic
- Isolated to enable running audits independently of feature computation
- Contains all timestamp comparison and recomputation verification logic

**Non-responsibilities:**
- Not data quality validation (ingestion.py handles that)
- Not model evaluation (out of scope)
- Not continuous monitoring (batch audit tool)

## `/scripts`
**Purpose:** Executable entry points for core pipeline operations.

**Justification:**
- `ingest_data.py`: CLI for loading new market data with date range parameters
- `compute_features.py`: CLI for running feature pipeline over specified periods
- `run_audit.py`: CLI for executing causality audits with configurable sample size
- Each script is a thin orchestration layer calling src/ modules in the correct sequence

**Non-responsibilities:**
- Not reusable library code (that's in src/)
- Not Airflow/scheduling (manual execution for MVP)
- Not parameter tuning/training (model work is post-MVP)

## `/tests`
**Purpose:** Verification of temporal correctness and feature computation accuracy.

**Justification:**
- `test_temporal_correctness.py`: Unit tests for TimeIndex and causality enforcement
- `test_feature_computation.py`: Known-answer tests for feature functions
- Only tests that verify the core value proposition (lookahead prevention)

**Non-responsibilities:**
- Not integration tests (manual end-to-end validation)
- Not performance benchmarks (profiling is ad-hoc)
- Not exhaustive coverage (tests critical paths only)

## `/experiments`
**Purpose:** Timestamped snapshots of feature computation runs and audit results.

**Justification:**
- Each run creates a subdirectory with config, metrics, and audit logs
- Enables comparing different feature configurations without formal experiment tracking
- Minimal file-based versioning sufficient for solo engineer

**Non-responsibilities:**
- Not MLflow/W&B integration (over-engineering)
- Not Git-tracked (experiments are ephemeral)
- Not model registry (models aren't part of MVP)

---

# Python Module Specifications

## temporal_index.py

### Module Contract
- **Module name:** temporal_index.py
- **System role:** Provides time-boundary enforcement primitives for all temporal operations
- **Dependency direction:** No dependencies (foundational) | Depended on by feature_computer.py, causality_validator.py, audit.py

### Interface Protocol

#### Exposed Entry Points
- **Primary interface:**
  ```python
  class TimeIndex:
      def __init__(self, timestamps: pd.DatetimeIndex)
      def get_lookback_window(self, as_of_date: pd.Timestamp, lookback_days: int) -> pd.DatetimeIndex
      def validate_monotonicity(self) -> bool
      def get_valid_computation_range(self, min_lookback: int) -> pd.DatetimeIndex
  ```
- **Return contract:** Returns filtered timestamp indices or raises ValueError on monotonicity violations
- **Lifecycle hooks:** None (stateless after initialization)

#### Consumption Requirements
- **Required from callers:** Sorted pd.DatetimeIndex with timezone-aware timestamps
- **Assumed system state:** Input timestamps are validated (no nulls, no duplicates)

### Data Flow Specification

#### Input Schema
- **Data structure:** `pd.DatetimeIndex` with UTC timezone
- **Data origin:** ingestion.py (validated timestamps from raw data)
- **Preconditions:** `timestamps.is_monotonic_increasing == True`, `timestamps.tz == 'UTC'`

#### Output Schema
- **Data structure:** Filtered `pd.DatetimeIndex` representing valid time windows
- **Data destination:** feature_computer.py (defines computation boundaries)
- **Postconditions:** All returned timestamps ≤ as_of_date; window size matches requested lookback

### Connection Points

#### Upstream Dependencies
- **Modules called:** None (pure computation)
- **Data passed upstream:** N/A
- **Failure propagation:** N/A

#### Downstream Dependencies
- **Modules using this:** feature_computer.py calls `get_lookback_window()` before every feature computation
- **Caller responsibilities:** Must handle ValueError if requested date is before minimum lookback period

### Composition Rules

#### Wiring Instructions
- **Initialization sequence:** `time_idx = TimeIndex(validated_timestamps)`
- **Integration pattern:** Utility (pure functions for temporal slicing)
- **Shared state:** None (immutable after construction)

#### Substitution Constraints
- **Replaceable components:** None (foundational)
- **Fixed interfaces:** All public methods are stable APIs

### Error Boundaries

#### Failure Isolation
- **Failure scope:** Raises ValueError immediately on invalid inputs; does not retry
- **Retry semantics:** Not applicable (deterministic pure functions)
- **Fallback behavior:** No fallbacks; callers must validate inputs

### System Integration

#### Configuration Requirements
- **Parameters affecting connections:** None (behavior is deterministic)
- **Environment assumptions:** Pandas installed, timezone database available

#### Observability
- **Integration checkpoints:** Logs warning if lookback window < 50% requested size (sparse data)
- **Health signals:** Boolean return from `validate_monotonicity()`

### Evolution Boundaries

#### Stable Interface
- **What will not change:** Method signatures, UTC timezone requirement

#### Future Connections
- **Planned integration points:** Could add `get_expanding_window()` for cumulative features
- **Extension constraints:** Cannot relax monotonicity requirement without breaking causality guarantees

---

## feature_computer.py

### Module Contract
- **Module name:** feature_computer.py
- **System role:** Executes feature computations with enforced temporal boundaries
- **Dependency direction:** Depends on temporal_index.py | Depended on by compute_features.py script, causality_validator.py

### Interface Protocol

#### Exposed Entry Points
- **Primary interface:**
  ```python
  class FeatureComputer:
      def __init__(self, time_index: TimeIndex, feature_registry: FeatureRegistry)
      def compute_point_in_time(self, data: pd.DataFrame, as_of_date: pd.Timestamp, feature_names: List[str]) -> pd.Series
      def compute_batch(self, data: pd.DataFrame, date_range: pd.DatetimeIndex, feature_names: List[str]) -> pd.DataFrame
  ```
- **Return contract:** Returns feature values as pd.Series (single date) or pd.DataFrame (batch); raises ValueError if lookback insufficient
- **Lifecycle hooks:** None

#### Consumption Requirements
- **Required from callers:** Raw OHLCV DataFrame with timestamp index, valid feature names from registry
- **Assumed system state:** TimeIndex initialized, FeatureRegistry populated

### Data Flow Specification

#### Input Schema
- **Data structure:** `pd.DataFrame[timestamp, open, high, low, close, volume]` (timestamp as index)
- **Data origin:** storage.py loads raw data
- **Preconditions:** Data sorted by timestamp, no missing required columns

#### Output Schema
- **Data structure:** `pd.DataFrame[timestamp, feature_1, feature_2, ...]` (timestamp as index)
- **Data destination:** storage.py saves to processed/; causality_validator.py recomputes for audits
- **Postconditions:** Every feature at time T computed using only data from ≤ T-lag

### Connection Points

#### Upstream Dependencies
- **Modules called:** 
  - `temporal_index.get_lookback_window()` to slice data
  - `registry.get_feature_spec()` to retrieve function and metadata
- **Data passed upstream:** as_of_date, lookback_days
- **Failure propagation:** ValueError from TimeIndex stops computation; logged and re-raised

#### Downstream Dependencies
- **Modules using this:** 
  - `compute_features.py` calls `compute_batch()` for full dataset
  - `audit.py` calls `compute_point_in_time()` for verification
- **Caller responsibilities:** Must provide date range within data coverage; handle partial failures

### Composition Rules

#### Wiring Instructions
- **Initialization sequence:**
  ```python
  time_idx = TimeIndex(data.index)
  registry = FeatureRegistry.from_config('config/feature_specs.yaml')
  computer = FeatureComputer(time_idx, registry)
  ```
- **Integration pattern:** Pipeline stage (transforms raw data → features)
- **Shared state:** Holds references to TimeIndex and FeatureRegistry (read-only)

#### Substitution Constraints
- **Replaceable components:** FeatureRegistry (could use different feature definitions)
- **Fixed interfaces:** Must always enforce lookback/lag constraints via TimeIndex

### Error Boundaries

#### Failure Isolation
- **Failure scope:** Per-feature failure logged but doesn't halt batch; returns NaN for failed features
- **Retry semantics:** Idempotent (same inputs → same outputs); safe to retry
- **Fallback behavior:** NaN for features that fail computation (e.g., insufficient data)

### System Integration

#### Configuration Requirements
- **Parameters affecting connections:** `feature_specs.yaml` defines which features to compute and their temporal parameters
- **Environment assumptions:** Sufficient memory to hold lookback window in RAM

#### Observability
- **Integration checkpoints:** Logs start/end of batch computation, per-feature timing
- **Health signals:** Returns non-empty DataFrame on success; empty means no valid dates

### Evolution Boundaries

#### Stable Interface
- **What will not change:** Temporal enforcement logic, batch vs. point-in-time distinction

#### Future Connections
- **Planned integration points:** Could add `compute_incremental()` for online serving
- **Extension constraints:** Cannot remove temporal validation without breaking system guarantee

---

## causality_validator.py

### Module Contract
- **Module name:** causality_validator.py
- **System role:** Verifies no feature uses future information via recomputation comparison
- **Dependency direction:** Depends on temporal_index.py, feature_computer.py | Depended on by audit.py

### Interface Protocol

#### Exposed Entry Points
- **Primary interface:**
  ```python
  class CausalityValidator:
      def __init__(self, feature_computer: FeatureComputer)
      def validate_single_point(self, raw_data: pd.DataFrame, features: pd.DataFrame, timestamp: pd.Timestamp) -> ValidationResult
      def validate_batch(self, raw_data: pd.DataFrame, features: pd.DataFrame, sample_size: int) -> List[ValidationResult]
  ```
- **Return contract:** `ValidationResult(timestamp, feature_name, is_valid, stored_value, recomputed_value, diff)`
- **Lifecycle hooks:** None

#### Consumption Requirements
- **Required from callers:** Both raw data and computed features covering the same time period
- **Assumed system state:** FeatureComputer configured identically to original computation

### Data Flow Specification

#### Input Schema
- **Data structure:** 
  - `raw_data`: Original OHLCV DataFrame
  - `features`: Computed features from feature_computer.py
- **Data origin:** 
  - `raw_data` from storage.py
  - `features` from storage.py (processed/ directory)
- **Preconditions:** Timestamp indices must align between raw_data and features

#### Output Schema
- **Data structure:** `List[ValidationResult]` with fields: timestamp, feature_name, is_valid, stored_value, recomputed_value, abs_diff
- **Data destination:** audit.py aggregates results; prints violations
- **Postconditions:** is_valid == True if |stored - recomputed| < 1e-5

### Connection Points

#### Upstream Dependencies
- **Modules called:** `feature_computer.compute_point_in_time()` to recompute features
- **Data passed upstream:** Subset of raw_data up to validation timestamp
- **Failure propagation:** Recomputation errors marked as validation failures (not exceptions)

#### Downstream Dependencies
- **Modules using this:** audit.py calls `validate_batch()` with random sample
- **Caller responsibilities:** Must ensure raw_data contains sufficient history for all requested timestamps

### Composition Rules

#### Wiring Instructions
- **Initialization sequence:**
  ```python
  computer = FeatureComputer(time_idx, registry)
  validator = CausalityValidator(computer)
  ```
- **Integration pattern:** Service (provides validation on-demand)
- **Shared state:** Holds reference to FeatureComputer

#### Substitution Constraints
- **Replaceable components:** None (tightly coupled to FeatureComputer contract)
- **Fixed interfaces:** ValidationResult schema must remain stable

### Error Boundaries

#### Failure Isolation
- **Failure scope:** Per-timestamp validation failures don't stop batch validation
- **Retry semantics:** Idempotent; safe to retry entire batch
- **Fallback behavior:** Marks timestamp as invalid if recomputation raises exception

### System Integration

#### Configuration Requirements
- **Parameters affecting connections:** Tolerance threshold for floating-point comparison (default 1e-5)
- **Environment assumptions:** Same FeatureComputer config as original computation

#### Observability
- **Integration checkpoints:** Logs each validation comparison (timestamp, feature, pass/fail)
- **Health signals:** Returns empty violations list if all checks pass

### Evolution Boundaries

#### Stable Interface
- **What will not change:** Recomputation-based validation approach

#### Future Connections
- **Planned integration points:** Could add statistical tests for distribution drift
- **Extension constraints:** Must always use exact recomputation (no approximations)

---

## ingestion.py

### Module Contract
- **Module name:** ingestion.py
- **System role:** Loads raw market data and enforces timestamp validity
- **Dependency direction:** Depends on storage.py (for write) | Depended on by ingest_data.py script

### Interface Protocol

#### Exposed Entry Points
- **Primary interface:**
  ```python
  class DataIngestionPipeline:
      def ingest_from_csv(self, filepath: str, start_date: str, end_date: str) -> pd.DataFrame
      def validate_schema(self, df: pd.DataFrame) -> bool
      def normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame
  ```
- **Return contract:** Returns validated, timestamp-indexed DataFrame or raises SchemaValidationError
- **Lifecycle hooks:** None

#### Consumption Requirements
- **Required from callers:** Valid file path to CSV with OHLCV columns, date strings in ISO format
- **Assumed system state:** File exists and is readable

### Data Flow Specification

#### Input Schema
- **Data structure:** CSV with columns: `date,open,high,low,close,volume` (optional: adj_close, asset_id)
- **Data origin:** External data vendor files or manual downloads
- **Preconditions:** File exists, has header row, dates parseable

#### Output Schema
- **Data structure:** `pd.DataFrame[timestamp (index), open, high, low, close, volume]` with UTC timezone
- **Data destination:** storage.py writes to data/raw/
- **Postconditions:** Timestamps monotonic, no nulls in required columns, UTC timezone set

### Connection Points

#### Upstream Dependencies
- **Modules called:** `storage.save_raw_data()` to persist
- **Data passed upstream:** Validated DataFrame with content hash
- **Failure propagation:** CSV parse errors raised as DataIngestionError; logged and halts

#### Downstream Dependencies
- **Modules using this:** ingest_data.py script orchestrates ingestion
- **Caller responsibilities:** Must provide valid date range; handle ingestion errors

### Composition Rules

#### Wiring Instructions
- **Initialization sequence:** `pipeline = DataIngestionPipeline()`
- **Integration pattern:** Adapter (converts external CSV → internal schema)
- **Shared state:** None (stateless)

#### Substitution Constraints
- **Replaceable components:** Could swap CSV reader for Parquet/API without changing interface
- **Fixed interfaces:** Output schema (columns, types, timezone) is fixed

### Error Boundaries

#### Failure Isolation
- **Failure scope:** Entire file ingestion fails atomically (no partial writes)
- **Retry semantics:** Idempotent (same file → same output); safe to retry
- **Fallback behavior:** No fallbacks; raises exception on validation failure

### System Integration

#### Configuration Requirements
- **Parameters affecting connections:** Expected columns (configurable in pipeline_config.yaml)
- **Environment assumptions:** Pandas installed, file system writable

#### Observability
- **Integration checkpoints:** Logs row count, date range, content hash after validation
- **Health signals:** Boolean return from `validate_schema()`

### Evolution Boundaries

#### Stable Interface
- **What will not change:** Output DataFrame schema (OHLCV + timestamp index)

#### Future Connections
- **Planned integration points:** Could add `ingest_from_api()` for live data feeds
- **Extension constraints:** Must always validate timestamps and timezone

---

## storage.py

### Module Contract
- **Module name:** storage.py
- **System role:** Abstracts Parquet read/write with date-partitioning for raw and processed data
- **Dependency direction:** No dependencies | Depended on by ingestion.py, feature_computer.py, audit.py

### Interface Protocol

#### Exposed Entry Points
- **Primary interface:**
  ```python
  class DataStore:
      def save_raw_data(self, df: pd.DataFrame, ingestion_date: str) -> str
      def load_raw_data(self, start_date: str, end_date: str) -> pd.DataFrame
      def save_features(self, df: pd.DataFrame, feature_version: str) -> str
      def load_features(self, start_date: str, end_date: str, feature_version: str) -> pd.DataFrame
  ```
- **Return contract:** Returns file path on save; returns DataFrame on load; raises FileNotFoundError if data missing
- **Lifecycle hooks:** None

#### Consumption Requirements
- **Required from callers:** DataFrame with timestamp index for save; valid date strings for load
- **Assumed system state:** data/ directory exists and is writable

### Data Flow Specification

#### Input Schema
- **Data structure:** 
  - Save: `pd.DataFrame` with timestamp index
  - Load: `start_date`, `end_date` as ISO strings
- **Data origin:** 
  - Raw data from ingestion.py
  - Features from feature_computer.py
- **Preconditions:** DataFrame index is sorted timestamp

#### Output Schema
- **Data structure:** 
  - Save: File path string
  - Load: `pd.DataFrame` with timestamp index
- **Data destination:** 
  - Loaded data goes to feature_computer.py, causality_validator.py
- **Postconditions:** Saved data is Parquet-compressed; loaded data covers requested date range

### Connection Points

#### Upstream Dependencies
- **Modules called:** None (pure I/O)
- **Data passed upstream:** N/A
- **Failure propagation:** N/A

#### Downstream Dependencies
- **Modules using this:** All modules that need persistent data
- **Caller responsibilities:** Must handle FileNotFoundError; provide valid date ranges

### Composition Rules

#### Wiring Instructions
- **Initialization sequence:** `store = DataStore(base_path='./data')`
- **Integration pattern:** Utility (I/O abstraction layer)
- **Shared state:** None (stateless except for base path)

#### Substitution Constraints
- **Replaceable components:** Could swap Parquet for CSV or database without changing interface
- **Fixed interfaces:** save/load method signatures are stable

### Error Boundaries

#### Failure Isolation
- **Failure scope:** Per-file failures isolated; one partition failure doesn't affect others
- **Retry semantics:** Idempotent writes (overwrites on conflict); safe to retry
- **Fallback behavior:** Returns empty DataFrame if no files found in date range

### System Integration

#### Configuration Requirements
- **Parameters affecting connections:** `base_path` in pipeline_config.yaml
- **Environment assumptions:** Disk space available, Parquet codec (snappy) installed

#### Observability
- **Integration checkpoints:** Logs file paths and row counts on save/load
- **Health signals:** File existence check via `os.path.exists()`

### Evolution Boundaries

#### Stable Interface
- **What will not change:** Method signatures, Parquet format

#### Future Connections
- **Planned integration points:** Could add `load_from_s3()` for cloud storage
- **Extension constraints:** Must preserve date-partitioning for temporal queries

---

## registry.py

### Module Contract
- **Module name:** registry.py
- **System role:** Maintains catalog of feature definitions with temporal metadata
- **Dependency direction:** Depends on definitions.py | Depended on by feature_computer.py

### Interface Protocol

#### Exposed Entry Points
- **Primary interface:**
  ```python
  class FeatureRegistry:
      @classmethod
      def from_config(cls, config_path: str) -> 'FeatureRegistry'
      def register(self, name: str, func: Callable, lookback: int, lag: int)
      def get_feature_spec(self, name: str) -> FeatureSpec
      def list_features(self) -> List[str]
  ```
- **Return contract:** Returns FeatureSpec namedtuple or raises KeyError if feature not found
- **Lifecycle hooks:** None

#### Consumption Requirements
- **Required from callers:** Valid config YAML path or manual registration of functions
- **Assumed system state:** definitions.py imported and feature functions available

### Data Flow Specification

#### Input Schema
- **Data structure:** YAML config with structure:
  ```yaml
  features:
    - name: rolling_zscore
      function: definitions.rolling_zscore
      lookback_days: 20
      lag: 1
  ```
- **Data origin:** config/feature_specs.yaml (human-authored)
- **Preconditions:** Referenced functions exist in definitions.py

#### Output Schema
- **Data structure:** `FeatureSpec(name, func, lookback, lag)`
- **Data destination:** feature_computer.py uses specs to drive computation
- **Postconditions:** All registered features have valid lookback/lag values (>= 1)

### Connection Points

#### Upstream Dependencies
- **Modules called:** `importlib.import_module()` to load feature functions from definitions.py
- **Data passed upstream:** N/A
- **Failure propagation:** ImportError if function doesn't exist; raises RegistrationError

#### Downstream Dependencies
- **Modules using this:** feature_computer.py calls `get_feature_spec()` for each feature
- **Caller responsibilities:** Must handle KeyError for invalid feature names

### Composition Rules

#### Wiring Instructions
- **Initialization sequence:** `registry = FeatureRegistry.from_config('config/feature_specs.yaml')`
- **Integration pattern:** Service (provides feature metadata on-demand)
- **Shared state:** In-memory dict of name → FeatureSpec

#### Substitution Constraints
- **Replaceable components:** Could load from database instead of YAML
- **Fixed interfaces:** FeatureSpec schema must remain stable

### Error Boundaries

#### Failure Isolation
- **Failure scope:** Invalid feature registration fails immediately at startup
- **Retry semantics:** Not applicable (deterministic config parsing)
- **Fallback behavior:** No fallbacks; raises exception on invalid config

### System Integration

#### Configuration Requirements
- **Parameters affecting connections:** feature_specs.yaml path
- **Environment assumptions:** YAML parser available

#### Observability
- **Integration checkpoints:** Logs number of features registered at initialization
- **Health signals:** `list_features()` returns non-empty list

### Evolution Boundaries

#### Stable Interface
- **What will not change:** FeatureSpec schema (name, func, lookback, lag)

#### Future Connections
- **Planned integration points:** Could add dependency tracking between features
- **Extension constraints:** Cannot change temporal metadata format without updating feature_computer.py

---

## definitions.py

### Module Contract
- **Module name:** definitions.py
- **System role:** Implements concrete feature computation functions
- **Dependency direction:** No dependencies (pure NumPy/Pandas) | Depended on by registry.py

### Interface Protocol

#### Exposed Entry Points
- **Primary interface:**
  ```python
  def rolling_zscore(prices: pd.Series, window: int) -> pd.Series
  def ewm_returns(prices: pd.Series, halflife: int) -> pd.Series
  def volatility_normalized_momentum(prices: pd.Series, return_window: int, vol_window: int) -> pd.Series
  ```
- **Return contract:** Returns pd.Series of same length as input; NaN for insufficient data points
- **Lifecycle hooks:** None

#### Consumption Requirements
- **Required from callers:** pd.Series with numeric values, no nulls (or caller handles nulls)
- **Assumed system state:** Input data is temporally ordered

### Data Flow Specification

#### Input Schema
- **Data structure:** `pd.Series[float]` (prices or returns)
- **Data origin:** feature_computer.py passes windowed data
- **Preconditions:** Series length >= window size (enforced by feature_computer.py)

#### Output Schema
- **Data structure:** `pd.Series[float]` (transformed values)
- **Data destination:** feature_computer.py collects into feature matrix
- **Postconditions:** Output length == input length; early values NaN if insufficient lookback

### Connection Points

#### Upstream Dependencies
- **Modules called:** None (pure computation using NumPy/Pandas)
- **Data passed upstream:** N/A
- **Failure propagation:** N/A

#### Downstream Dependencies
- **Modules using this:** registry.py stores function references; feature_computer.py invokes functions
- **Caller responsibilities:** Must provide sufficient data for window size

### Composition Rules

#### Wiring Instructions
- **Initialization sequence:** Functions are stateless; no initialization needed
- **Integration pattern:** Utility (library of pure functions)
- **Shared state:** None (all functions are pure)

#### Substitution Constraints
- **Replaceable components:** Any function can be swapped if signature matches
- **Fixed interfaces:** Function signatures (inputs/outputs) must remain stable

### Error Boundaries

#### Failure Isolation
- **Failure scope:** Function returns NaN for invalid inputs; does not raise exceptions
- **Retry semantics:** Idempotent (same inputs → same outputs)
- **Fallback behavior:** Returns NaN for edge cases (e.g., zero standard deviation)

### System Integration

#### Configuration Requirements
- **Parameters affecting connections:** Window sizes passed as arguments (defined in feature_specs.yaml)
- **Environment assumptions:** NumPy/Pandas installed

#### Observability
- **Integration checkpoints:** None (pure computation, no side effects)
- **Health signals:** Non-NaN return for valid inputs

### Evolution Boundaries

#### Stable Interface
- **What will not change:** Function signatures (name, params, return type)

#### Future Connections
- **Planned integration points:** Could add Numba JIT decorators for performance
- **Extension constraints:** Must remain pure functions (no side effects)

---

## audit.py

### Module Contract
- **Module name:** audit.py
- **System role:** Orchestrates random-sample causality validation and reports violations
- **Dependency direction:** Depends on causality_validator.py, storage.py | Depended on by run_audit.py script

### Interface Protocol

#### Exposed Entry Points
- **Primary interface:**
  ```python
  class CausalityAuditor:
      def __init__(self, validator: CausalityValidator, store: DataStore)
      def run_audit(self, start_date: str, end_date: str, feature_version: str, sample_size: int) -> AuditReport
      def generate_report(self, results: List[ValidationResult]) -> AuditReport
  ```
- **Return contract:** `AuditReport(total_checks, violations, pass_rate, violation_details)`
- **Lifecycle hooks:** None

#### Consumption Requirements
- **Required from callers:** Date range, feature version, sample size
- **Assumed system state:** Raw data and features exist in storage for specified range

### Data Flow Specification

#### Input Schema
- **Data structure:** Config params: `start_date`, `end_date`, `feature_version`, `sample_size`
- **Data origin:** Command-line arguments from run_audit.py
- **Preconditions:** sample_size <= available timestamps in date range

#### Output Schema
- **Data structure:** `AuditReport(total_checks, violations, pass_rate, violation_details: List[ValidationResult])`
- **Data destination:** run_audit.py prints report; could save to experiments/
- **Postconditions:** pass_rate == 1.0 implies zero lookahead violations

### Connection Points

#### Upstream Dependencies
- **Modules called:** 
  - `storage.load_raw_data()` and `storage.load_features()`
  - `validator.validate_batch()`
- **Data passed upstream:** Loaded DataFrames and sample size
- **Failure propagation:** Storage errors halt audit; validation errors recorded in report

#### Downstream Dependencies
- **Modules using this:** run_audit.py calls `run_audit()` and prints results
- **Caller responsibilities:** Must interpret pass_rate and investigate violations

### Composition Rules

#### Wiring Instructions
- **Initialization sequence:**
  ```python
  store = DataStore()
  validator = CausalityValidator(feature_computer)
  auditor = CausalityAuditor(validator, store)
  ```
- **Integration pattern:** Service (provides validation as a service)
- **Shared state:** Holds references to validator and store

#### Substitution Constraints
- **Replaceable components:** None (tightly coupled to validator contract)
- **Fixed interfaces:** AuditReport schema must remain stable

### Error Boundaries

#### Failure Isolation
- **Failure scope:** Missing data files halt audit; individual validation failures recorded but don't stop audit
- **Retry semantics:** Idempotent (same params → same results for deterministic sample)
- **Fallback behavior:** Partial results returned if some validations fail

### System Integration

#### Configuration Requirements
- **Parameters affecting connections:** Validation tolerance threshold from config
- **Environment assumptions:** Sufficient data coverage for requested date range

#### Observability
- **Integration checkpoints:** Logs progress every 100 validations
- **Health signals:** AuditReport.pass_rate indicates health (1.0 = healthy)

### Evolution Boundaries

#### Stable Interface
- **What will not change:** Random sampling approach, AuditReport schema

#### Future Connections
- **Planned integration points:** Could add continuous monitoring mode (scheduled audits)
- **Extension constraints:** Must always use random sampling (not sequential)

---

# Script Specifications

## ingest_data.py

### Module Contract
- **Module name:** ingest_data.py
- **System role:** CLI entry point for data ingestion workflow
- **Dependency direction:** Depends on ingestion.py, storage.py | No dependencies

### Interface Protocol

#### Exposed Entry Points
- **Primary interface:**
  ```python
  if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--filepath', required=True)
      parser.add_argument('--start-date', required=True)
      parser.add_argument('--end-date', required=True)
  ```
- **Return contract:** Exits with code 0 on success, 1 on failure
- **Lifecycle hooks:** None

#### Consumption Requirements
- **Required from callers:** Command-line arguments for file and date range
- **Assumed system state:** CSV file exists and is readable

### Data Flow Specification

#### Input Schema
- **Data structure:** CLI args: filepath (str), start_date (str), end_date (str)
- **Data origin:** User command-line input
- **Preconditions:** File path is valid, dates are ISO format

#### Output Schema
- **Data structure:** Side effect: writes Parquet to data/raw/; prints confirmation message
- **Data destination:** storage.py persists data
- **Postconditions:** Data ingested and saved with content hash logged

### Connection Points

#### Upstream Dependencies
- **Modules called:** 
  - `ingestion.DataIngestionPipeline.ingest_from_csv()`
  - `storage.DataStore.save_raw_data()`
- **Data passed upstream:** File path, date range, validated DataFrame
- **Failure propagation:** Exceptions printed to stderr; script exits with code 1

#### Downstream Dependencies
- **Modules using this:** None (top-level script)
- **Caller responsibilities:** N/A

### Composition Rules

#### Wiring Instructions
- **Initialization sequence:**
  ```python
  pipeline = DataIngestionPipeline()
  store = DataStore()
  df = pipeline.ingest_from_csv(args.filepath, args.start_date, args.end_date)
  path = store.save_raw_data(df, datetime.now().strftime('%Y-%m-%d'))
  ```
- **Integration pattern:** Orchestrator (coordinates ingestion and storage)
- **Shared state:** None

#### Substitution Constraints
- **Replaceable components:** Could add Parquet/API ingestion variants
- **Fixed interfaces:** CLI arguments are stable

### Error Boundaries

#### Failure Isolation
- **Failure scope:** Entire ingestion fails atomically
- **Retry semantics:** Safe to retry (idempotent if same input)
- **Fallback behavior:** Prints error message and exits

### System Integration

#### Configuration Requirements
- **Parameters affecting connections:** None (all via CLI args)
- **Environment assumptions:** File system writable, Pandas installed

#### Observability
- **Integration checkpoints:** Prints ingestion summary (row count, date range, file path)
- **Health signals:** Exit code 0 = success

### Evolution Boundaries

#### Stable Interface
- **What will not change:** CLI argument names

#### Future Connections
- **Planned integration points:** Could add `--format` flag for Parquet input
- **Extension constraints:** Must always validate data before saving

---

## compute_features.py

### Module Contract
- **Module name:** compute_features.py
- **System role:** CLI entry point for feature computation workflow
- **Dependency direction:** Depends on feature_computer.py, storage.py, registry.py, temporal_index.py | No dependencies

### Interface Protocol

#### Exposed Entry Points
- **Primary interface:**
  ```python
  if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--start-date', required=True)
      parser.add_argument('--end-date', required=True)
      parser.add_argument('--feature-version', default='v1')
  ```
- **Return contract:** Exits with code 0 on success, 1 on failure
- **Lifecycle hooks:** None

#### Consumption Requirements
- **Required from callers:** Command-line arguments for date range and version
- **Assumed system state:** Raw data exists for specified date range; feature_specs.yaml valid

### Data Flow Specification

#### Input Schema
- **Data structure:** CLI args: start_date (str), end_date (str), feature_version (str)
- **Data origin:** User command-line input
- **Preconditions:** Dates are ISO format, raw data exists

#### Output Schema
- **Data structure:** Side effect: writes feature Parquet to data/processed/; prints execution time
- **Data destination:** storage.py persists features
- **Postconditions:** Features computed for all dates in range, saved with version tag

### Connection Points

#### Upstream Dependencies
- **Modules called:**
  - `storage.DataStore.load_raw_data()`
  - `registry.FeatureRegistry.from_config()`
  - `temporal_index.TimeIndex()`
  - `feature_computer.FeatureComputer.compute_batch()`
  - `storage.DataStore.save_features()`
- **Data passed upstream:** Date range, feature names, raw DataFrame
- **Failure propagation:** Exceptions caught, logged, script exits with code 1

#### Downstream Dependencies
- **Modules using this:** None (top-level script)
- **Caller responsibilities:** N/A

### Composition Rules

#### Wiring Instructions
- **Initialization sequence:**
  ```python
  store = DataStore()
  data = store.load_raw_data(args.start_date, args.end_date)
  time_idx = TimeIndex(data.index)
  registry = FeatureRegistry.from_config('config/feature_specs.yaml')
  computer = FeatureComputer(time_idx, registry)
  features = computer.compute_batch(data, data.index, registry.list_features())
  store.save_features(features, args.feature_version)
  ```
- **Integration pattern:** Orchestrator (coordinates feature pipeline)
- **Shared state:** None

#### Substitution Constraints
- **Replaceable components:** Could use different FeatureRegistry source
- **Fixed interfaces:** CLI arguments are stable

### Error Boundaries

#### Failure Isolation
- **Failure scope:** Entire batch fails if any critical error (data missing, invalid config)
- **Retry semantics:** Idempotent (same inputs → same outputs)
- **Fallback behavior:** Prints error traceback and exits

### System Integration

#### Configuration Requirements
- **Parameters affecting connections:** feature_specs.yaml defines which features to compute
- **Environment assumptions:** Sufficient memory for full date range, config file exists

#### Observability
- **Integration checkpoints:** Prints start/end times, feature count, output path
- **Health signals:** Exit code 0 = success

### Evolution Boundaries

#### Stable Interface
- **What will not change:** CLI argument names, feature_specs.yaml schema

#### Future Connections
- **Planned integration points:** Could add `--parallel` flag for multi-asset parallelism
- **Extension constraints:** Must always enforce temporal correctness via FeatureComputer

---

## run_audit.py

### Module Contract
- **Module name:** run_audit.py
- **System role:** CLI entry point for causality audit workflow
- **Dependency direction:** Depends on audit.py, causality_validator.py, feature_computer.py, storage.py, registry.py, temporal_index.py | No dependencies

### Interface Protocol

#### Exposed Entry Points
- **Primary interface:**
  ```python
  if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--start-date', required=True)
      parser.add_argument('--end-date', required=True)
      parser.add_argument('--feature-version', default='v1')
      parser.add_argument('--sample-size', type=int, default=1000)
  ```
- **Return contract:** Exits with code 0 if pass_rate == 1.0, else code 1
- **Lifecycle hooks:** None

#### Consumption Requirements
- **Required from callers:** Command-line arguments for date range, version, sample size
- **Assumed system state:** Raw data and features exist for specified range

### Data Flow Specification

#### Input Schema
- **Data structure:** CLI args: start_date (str), end_date (str), feature_version (str), sample_size (int)
- **Data origin:** User command-line input
- **Preconditions:** sample_size > 0, data and features exist

#### Output Schema
- **Data structure:** Prints AuditReport to stdout; exits with appropriate code
- **Data destination:** Terminal output; optionally save JSON report to experiments/
- **Postconditions:** User knows if causality violations exist

### Connection Points

#### Upstream Dependencies
- **Modules called:**
  - All modules in compute_features.py chain (to initialize FeatureComputer)
  - `causality_validator.CausalityValidator()`
  - `audit.CausalityAuditor.run_audit()`
- **Data passed upstream:** Date range, feature version, sample size
- **Failure propagation:** Exceptions caught, logged, script exits with code 1

#### Downstream Dependencies
- **Modules using this:** None (top-level script)
- **Caller responsibilities:** N/A

### Composition Rules

#### Wiring Instructions
- **Initialization sequence:**
  ```python
  store = DataStore()
  # (initialize feature computer same as compute_features.py)
  validator = CausalityValidator(computer)
  auditor = CausalityAuditor(validator, store)
  report = auditor.run_audit(args.start_date, args.end_date, args.feature_version, args.sample_size)
  print(report)
  sys.exit(0 if report.pass_rate == 1.0 else 1)
  ```
- **Integration pattern:** Orchestrator (coordinates audit workflow)
- **Shared state:** None

#### Substitution Constraints
- **Replaceable components:** Could use different validation strategies
- **Fixed interfaces:** CLI arguments are stable

### Error Boundaries

#### Failure Isolation
- **Failure scope:** Missing data halts audit; validation failures recorded in report
- **Retry semantics:** Idempotent for deterministic random seed
- **Fallback behavior:** Prints partial results if some validations fail

### System Integration

#### Configuration Requirements
- **Parameters affecting connections:** Same as compute_features.py (feature_specs.yaml)
- **Environment assumptions:** Data and features exist, sufficient memory

#### Observability
- **Integration checkpoints:** Prints progress bar for audit samples
- **Health signals:** Exit code 0 = 100% pass rate

### Evolution Boundaries

#### Stable Interface
- **What will not change:** CLI arguments, AuditReport schema

#### Future Connections
- **Planned integration points:** Could add `--save-report` flag for JSON output
- **Extension constraints:** Must always use CausalityValidator (no shortcuts)

---

# Explicit Omissions

## Components Intentionally Excluded

### 1. **Model Training Infrastructure**
- **Excluded:** `models/`, `train.py`, `hyperparameter_tuning.py`
- **Justification:** MVP scope is feature engineering correctness, not modeling. Model work happens post-validation in separate experiments. Including model code would blur focus and add complexity without supporting the core value proposition (lookahead prevention).

### 2. **Inference/Serving Layer**
- **Excluded:** `serve.py`, `api/`, Dockerfile, `inference.py`
- **Justification:** MVP is a backtest validation tool, not a production trading system. Serving infrastructure is premature and would require additional dependencies (Flask, Redis) that don't support the temporal correctness goal.

### 3. **MLOps Orchestration**
- **Excluded:** Airflow DAGs, Kubernetes configs, CI/CD pipelines, `Makefile`
- **Justification:** Single engineer running scripts manually has perfect visibility. Orchestration adds operational complexity without improving causality verification. Scripts are sufficient for MVP validation.

### 4. **Experiment Tracking Framework**
- **Excluded:** MLflow integration, W&B config, DVC setup
- **Justification:** File-based experiment snapshots (experiments/ directory with timestamped subdirs) provide sufficient reproducibility for MVP. Full tracking frameworks add dependencies and learning curve without materially improving temporal correctness validation.

### 5. **Data Versioning System**
- **Excluded:** DVC, Delta Lake, custom versioning layer
- **Justification:** Financial data is immutable post-ingestion. Content hashing at ingestion provides sufficient versioning. Row-level provenance is overkill for static historical prices and adds storage/complexity overhead.

### 6. **Monitoring Infrastructure**
- **Excluded:** Prometheus exporters, Grafana dashboards, alerting configs
- **Justification:** MVP runs on-demand, not continuously. Monitoring makes sense for production serving but not for batch validation tool. Audit script provides sufficient observability via stdout.

### 7. **Feature Store**
- **Excluded:** Feast/Tecton integration, online store (Redis), feature serving API
- **Justification:** On-demand computation with Parquet caching achieves 90% of value at 10% complexity. Feature store adds deployment dependencies and operational burden without improving temporal correctness—the core goal.

### 8. **Data Quality Framework**
- **Excluded:** Great Expectations configs, extensive validation suite, data profiling tools
- **Justification:** ingestion.py performs essential validations (schema, timestamps, nulls). Exhaustive data quality checks are orthogonal to causality verification and add framework dependencies.

### 9. **Distributed Computing Layer**
- **Excluded:** Dask/Spark configs, cluster management, parallel processing utilities
- **Justification:** MVP targets i5 laptop with <1M samples. Single-machine Pandas is sufficient and simpler. Distributed infrastructure is a scaling concern, not an MVP correctness concern.

### 10. **Utilities/Helpers Module**
- **Excluded:** `utils.py`, `helpers.py`, `common.py`
- **Justification:** These become dumping grounds for miscellaneous code. Every function in this codebase has a specific domain home (temporal logic → temporal_index.py, I/O → storage.py, etc.). Utilities dilute cohesion.

### 11. **Configuration Management Framework**
- **Excluded:** Hydra, OmegaConf, centralized config registry
- **Justification:** Two YAML files (feature_specs, pipeline_config) are sufficient. Config frameworks add indirection and learning curve without improving clarity for this small system.

### 12. **Logging Framework**
- **Excluded:** Custom logging config, structured logging (JSON logs), log aggregation
- **Justification:** Standard Python logging with print statements provides adequate visibility for MVP. Structured logging is valuable in production but premature for validation tool.

### 13. **Documentation Generator**
- **Excluded:** Sphinx setup, auto-generated API docs, `docs/` directory
- **Justification:** Code is the documentation for MVP. Module contracts (specified above) are embedded in code. Separate docs add maintenance burden without improving understanding for solo engineer.

### 14. **Testing Infrastructure**
- **Excluded:** pytest fixtures, mocking frameworks, integration tests, test data generators
- **Justification:** Two focused test files (temporal correctness, feature computation) validate core logic. Comprehensive test infrastructure is valuable at scale but premature for MVP where manual validation is faster.

### 15. **Notebooks**
- **Excluded:** `notebooks/`, Jupyter analysis notebooks, exploratory data analysis
- **Justification:** Notebooks are excellent for exploration but terrible for production code. MVP is about correct implementation, not exploration. Notebooks would create parallel codebases and drift from source of truth.

---

## Quality Validation

This structure reflects how a top 1% engineer would build an MVP focused on a single hard problem (temporal correctness) without enterprise ceremony. Every component directly supports the core value proposition. Every omission eliminates complexity that doesn't materially improve causality verification.

---

**Document Version:** 1.0  
**Last Updated:** February 14, 2026  
**Maintainer:** ML Engineering Student
