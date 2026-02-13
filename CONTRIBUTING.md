# Contributing to Lookahead-Free Backtest Framework

First off, thank you for considering contributing to this project! 🎉

## Code of Conduct

This project adheres to a simple principle: **Be respectful and constructive**.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected vs actual behavior**
- **System information** (OS, Python version, hardware specs)
- **Causality audit output** if relevant

**Example:**
```
Title: Rolling z-score returns NaN for first 20 values

Description: When computing 20-day rolling z-score, first 20 values 
should be NaN due to insufficient lookback, but getting NaN for first 21 values.

Expected: NaN for indices 0-19, values starting at index 20
Actual: NaN for indices 0-20, values starting at index 21

System: Windows 11, Python 3.9.13, 16GB RAM
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Include:

- **Clear use case** for the enhancement
- **Any research** showing similar features in other tools
- **Potential implementation approach** (if you have ideas)
- **Impact on temporal correctness** guarantees

### Pull Requests

1. **Fork the repo** and create your branch from `main`
2. **Follow existing code style** (run `black src/ tests/` before committing)
3. **Add tests** for any new features
4. **Ensure temporal correctness**: Run `python scripts/run_audit.py`
5. **Update documentation** if changing functionality
6. **Write clear commit messages**

#### Pull Request Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes Made
- Change 1
- Change 2

## Testing
- [ ] All existing tests pass
- [ ] Added new tests for new functionality
- [ ] Causality audit passes
- [ ] Tested on sample data

## Temporal Correctness
- [ ] No new lookahead violations introduced
- [ ] All features have explicit lag parameters
- [ ] Dependencies properly registered in validator

## Documentation
- [ ] Updated docstrings
- [ ] Updated README if needed
- [ ] Updated ARCHITECTURE.md if design changed
```

### Adding New Features

When adding financial features:

1. **Define in YAML first**: Add to `config/feature_specs.yaml`
2. **Specify lag explicitly**: Always include `lag: 1` or greater
3. **Document dependencies**: List all required input features
4. **Implement with tests**: Create test case in `tests/test_feature_computation.py`
5. **Verify causality**: Ensure `CausalityValidator` passes

**Example:**
```python
# src/features/definitions.py
def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0):
    """
    Compute Bollinger Bands with temporal correctness.
    
    Args:
        prices: Price series (must be lagged externally)
        window: Rolling window size
        num_std: Number of standard deviations
    
    Returns:
        Dict with 'upper', 'middle', 'lower' bands
    """
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return {
        'upper': middle + (std * num_std),
        'middle': middle,
        'lower': middle - (std * num_std)
    }
```

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/lookahead-free-backtest.git
cd lookahead-free-backtest

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies + dev tools
pip install -r requirements.txt
pip install black pytest pytest-cov

# Run tests
pytest tests/ -v
```

## Style Guide

### Python Code
- Follow PEP 8
- Use `black` for formatting: `black src/ tests/`
- Maximum line length: 100 characters
- Type hints encouraged for public APIs

### Docstrings
Use Google-style docstrings:

```python
def compute_feature(data: pd.Series, window: int, lag: int = 1) -> pd.Series:
    """
    Compute feature with temporal lag.
    
    Args:
        data: Input time series
        window: Lookback window size
        lag: Minimum lag to prevent lookahead (default: 1)
    
    Returns:
        Computed feature series with proper temporal alignment
        
    Raises:
        ValueError: If window or lag < 1
    """
```

### Configuration Files
- YAML for configs (human-readable)
- Include comments explaining non-obvious parameters
- Provide sensible defaults

## Testing Requirements

All PRs must:
- ✅ Pass existing tests: `pytest tests/ -v`
- ✅ Pass causality audit: `python scripts/run_audit.py`
- ✅ Add tests for new features (aim for 80%+ coverage)
- ✅ Include temporal correctness tests for time-sensitive features

## Review Process

1. Automated checks run on PR creation (tests, linting)
2. Maintainer reviews code and design
3. Address feedback in additional commits
4. Once approved, maintainer will merge

## Recognition

Contributors will be:
- Listed in README.md acknowledgments
- Mentioned in release notes for their contributions
- Credited in documentation for major features

## Questions?

Open an issue with the `question` label or start a discussion in GitHub Discussions.

---

**Thank you for contributing!** 🙏
