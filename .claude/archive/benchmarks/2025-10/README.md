# Archive: October 2025 - Benchmark Reports

## Contents

This archive contains preliminary benchmark reports from early development phases.

---

#### `PHASE1_BENCHMARK.md`
- **Archived**: 2025-10-26
- **Size**: 4.9 KB
- **Purpose**: Initial benchmark results for 5-model framework on sample data
- **Data**: Sample dataset (~1,500 records)
- **Models Tested**:
  - Cox PH
  - Coxnet (elastic net Cox)
  - Weibull AFT
  - GBSA (Gradient Boosting)
  - RSF (Random Survival Forest) - later removed
- **Key Metrics**:
  - Training time per model
  - C-index performance
  - Memory usage
- **Status**: ✅ **Baseline established, superseded by production benchmarks**
- **Note**: This was before run-type system implementation

---

#### `PRODUCTION_BENCHMARK_PRELIMINARY.md`
- **Archived**: 2025-10-26
- **Size**: 3.4 KB
- **Purpose**: Early production data benchmark (pre-optimization)
- **Data**: Full production dataset (~287,000 records)
- **Findings**:
  - RSF extremely slow (>10 hours per model)
  - GBSA with default parameters: ~6 hours
  - Cox models: <30 minutes
- **Impact**: Led to performance optimization decisions
- **Optimizations Implemented**:
  - Reduced GBSA n_estimators: 100 → 50 for production
  - Removed RSF from pipeline entirely
  - Added multiprocessing support
- **Status**: ✅ **Superseded by optimized pipeline**

---

## Evolution

### Phase 1: Initial Implementation
- 5 models on sample data
- Basic benchmarking
- Identified performance bottlenecks

### Phase 2: Production Scaling
- Full dataset benchmarking
- Performance optimization
- Configuration system for environment-specific tuning

### Phase 3: Current State (Post-Optimization)
- 4 models (RSF removed)
- Optimized hyperparameters for production
- Multiprocessing support
- Expected runtime: ~2-4 hours for full pipeline

## Performance Improvements

**Before Optimization:**
- Full pipeline: ~31 hours
- RSF: ~10+ hours alone
- GBSA: ~6 hours

**After Optimization:**
- Full pipeline: ~2-4 hours (estimated)
- GBSA: ~1.5 hours (50 estimators)
- Cox models: <30 minutes each
- RSF: Removed

## Related Changes

**Configuration:**
- `src/survival_framework/config.py` - ModelHyperparameters.for_environment("production")
- Production settings: GBSA(n_estimators=50), faster parameters

**Execution:**
- `--execution-mode mp` for multiprocessing
- `--n-jobs 8` for parallel CV folds

**Removed:**
- RSF model (commit daa6298)

## Current Benchmarking

For current performance metrics, see:
- `data/outputs/production/logs/` - Runtime logs with timing
- `data/outputs/production/artifacts/model_metrics.csv` - Per-fold metrics
- MLflow experiments - Detailed tracking

## Retrieval

To view archived benchmarks:
```bash
cat .claude/archive/benchmarks/2025-10/PHASE1_BENCHMARK.md
cat .claude/archive/benchmarks/2025-10/PRODUCTION_BENCHMARK_PRELIMINARY.md
```

These reports are preserved for:
- Historical performance comparison
- Understanding optimization journey
- Documenting decision rationale (e.g., why RSF was removed)
- Baseline for future improvements

## References

- **CHANGELOG.md** - v0.2.0 configuration system, RSF removal
- **src/survival_framework/config.py** - Production optimizations
