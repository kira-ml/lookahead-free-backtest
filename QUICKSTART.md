# Quick Start Guide

This guide will get you up and running with the Lookahead-Free Backtest Framework in 5 minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lookahead-free-backtest.git
cd lookahead-free-backtest

# Install dependencies
pip install -r requirements.txt
```

## Quick Test Run

### Step 1: Verify Installation

```bash
pytest tests/test_temporal_correctness.py::TestTemporalIndex -v
```

Expected output:
```
test_initialization PASSED
test_set_current_time PASSED
test_get_available_data_mask PASSED
test_validate_no_lookahead PASSED
✓ All tests passed
```

### Step 2: Run the Pipeline

```bash
# Option A: Run all steps individually
python scripts/ingest_data.py
python scripts/compute_features.py
python scripts/run_audit.py

# Option B: Use Makefile (if on Unix/Mac)
make run-pipeline
```

### Step 3: Verify Temporal Correctness

Check the audit output:
```
✓ All causality checks passed
✓ Feature registry validation passed
✓ 0 violations detected
```

## Understanding the Output

### Data Files Created

```
data/
├── processed/
│   └── market_data.parquet      # Validated, sorted time-series data
└── raw/
    └── (your raw data files)
```

### Feature Files

Features are cached in `data/processed/` as separate Parquet files:
- `price_return_1d.parquet`
- `volume_ma_5d.parquet`
- `volatility_10d.parquet`
- `momentum_signal.parquet`

### Audit Logs

Check `experiments/audit.log` for detailed temporal validation results.

## First Customization: Add Your Own Feature

### 1. Edit Feature Configuration

Open `config/feature_specs.yaml` and add:

```yaml
features:
  # ... existing features ...
  
  - name: "my_custom_zscore"
    type: "rolling"
    computation: "zscore"
    window: 30
    lag: 1
    dependencies: []
```

### 2. Run Feature Computation

```bash
python scripts/compute_features.py
```

### 3. Verify Temporal Correctness

```bash
python scripts/run_audit.py
```

## Python API Usage

Create a new file `my_analysis.py`:

```python
import pandas as pd
from src.data.ingestion import DataIngestion
from src.core.temporal_index import TemporalIndex
from src.core.feature_computer import FeatureComputer

# Load processed data
ingestion = DataIngestion(
    raw_data_path='data/raw',
    processed_data_path='data/processed'
)
df = ingestion.load_processed_data('market_data')

# Create temporal index
temporal_index = TemporalIndex(pd.DatetimeIndex(df['timestamp']))

# Initialize feature computer
feature_computer = FeatureComputer(temporal_index)

# Compute a simple feature
returns = feature_computer.compute_simple_feature(
    data=df['price'],
    operation='pct_change',
    lag=1
)

print(f"Returns computed for {len(returns)} timestamps")
print(f"First non-NaN value at index: {returns.first_valid_index()}")
```

Run it:
```bash
python my_analysis.py
```

## Common Issues

### Issue: "No module named 'src'"

**Solution**: Add project root to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Unix/Mac
$env:PYTHONPATH += ";$(pwd)"              # Windows PowerShell
```

### Issue: "Timestamps must be unique"

**Solution**: Your data has duplicate timestamps. Check and clean:
```python
df = df.drop_duplicates(subset=['timestamp'], keep='first')
```

### Issue: "Lookahead detected"

**Solution**: A feature is accessing future data. Check that:
1. All features have `lag >= 1` in `feature_specs.yaml`
2. No manual data indexing bypasses the temporal index
3. No sorting/aggregation breaks temporal order

## Next Steps

1. **Read the Architecture**: Check [ARCHITECTURE.md](ARCHITECTURE.md) for system design details
2. **Explore Features**: See `src/features/definitions.py` for custom feature examples
3. **Add Your Data**: Replace synthetic data in `scripts/ingest_data.py` with your CSV/Parquet files
4. **Customize Config**: Modify `config/pipeline_config.yaml` for your use case
5. **Run Experiments**: Track results using the experiments tracking system

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/lookahead-free-backtest/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/lookahead-free-backtest/discussions)
- **Documentation**: See README.md and ARCHITECTURE.md

---

**You're all set!** 🚀
