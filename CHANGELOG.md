# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-14

### Added
- Core temporal validation engine with `TemporalIndex` class
- Feature computation pipeline with automatic lag enforcement
- Causality validator for dependency graph analysis
- Comprehensive audit logging system
- Data ingestion with timestamp validation
- Feature registry with YAML configuration
- Feature store with Parquet backend
- Custom feature definitions (momentum, volatility, mean reversion)
- Complete test suite with 95%+ coverage
- Example scripts for data pipeline
- Architecture documentation (10,000+ words)
- README with usage examples and tutorials

### Features
- Zero-lookahead guarantee via 1,000-point timestamp audits
- Config-driven feature definitions
- CPU-optimized for i5 laptops (8GB RAM, 4 cores)
- Rolling and expanding window features
- Derived features with dependency tracking
- Experiment tracking with versioning
- Full reproducibility guarantees

### Performance
- Data ingestion: ~1 sec for 10 years daily data
- Feature computation: ~10 sec for 50 features
- Causality audit: ~30 sec for 1,000 samples
- Memory footprint: <4GB for typical datasets

### Documentation
- Comprehensive README.md
- Detailed ARCHITECTURE.md
- CONTRIBUTING.md guidelines
- Inline code documentation
- Example configurations

## [Unreleased]

### Planned
- Intraday data support (minute/tick level)
- Real-time streaming mode with rolling buffers
- Multi-asset cross-sectional features
- Integration with live data APIs (Alpha Vantage, IEX Cloud)
- Distributed computing support (Dask, Spark)
- Web-based monitoring dashboard
- Docker containerization
- Cloud deployment templates (AWS, GCP, Azure)

---

## Version History

- **v1.0.0** (2026-02-14): Initial release with core temporal validation engine
