# PySpark Integration Plan

**Branch**: `feature/pyspark-integration`  
**Created**: 2025-10-24  
**Purpose**: Enable distributed computing for large-scale survival analysis

---

## Table of Contents

1. [Current Bottlenecks](#current-bottlenecks)
2. [Parallelization Opportunities](#parallelization-opportunities)
3. [Architecture Design](#architecture-design)
4. [Implementation Strategy](#implementation-strategy)
5. [Configuration System](#configuration-system)
6. [Backwards Compatibility](#backwards-compatibility)
7. [Performance Targets](#performance-targets)
8. [Implementation Phases](#implementation-phases)

---

## Current Bottlenecks

### Identified Bottlenecks in Production Pipeline (287K records)

#### 1. **Cross-Validation Loop** (Largest Bottleneck)
- **Current**: Sequential execution of 5 folds × 5 models = 25 training runs
- **Time Estimate**: ~30-60 minutes for full pipeline
- **CPU Utilization**: Single core during model training
- **Memory**: ~1GB peak for production data

**Code Location**: `train.py:154-177`
```python
for name, model in models.items():
    for fold_idx, (tr, te) in enumerate(splits):
        # Sequential training - NO PARALLELISM
        res = evaluate_model(...)
```

#### 2. **Preprocessing Pipeline**
- **Current**: Pandas-based preprocessing (single-threaded)
- **Operations**: Imputation, scaling, one-hot encoding
- **Bottleneck**: Limited for 287K records, but will scale poorly to millions

#### 3. **Prediction Generation**
- **Current**: Sequential prediction for all samples
- **Operations**: Survival function evaluation at multiple time points
- **Bottleneck**: O(n_samples × n_time_points) complexity

#### 4. **Proportional Hazards Testing**
- **Current**: Single-threaded Schoenfeld residuals calculation
- **Operations**: Per-covariate hypothesis testing
- **Minor bottleneck** for current data size

---

## Parallelization Opportunities

### High-Impact Opportunities

#### A. **Parallel Cross-Validation** (HIGHEST PRIORITY)
- **Strategy**: Train multiple folds simultaneously
- **Expected Speedup**: Near-linear with number of cores (5× on 8-core machine)
- **Complexity**: Medium
- **Benefit**: Massive time savings for production runs

#### B. **Parallel Model Training**
- **Strategy**: Train different models in parallel
- **Expected Speedup**: 5× for 5 models on 8-core machine
- **Complexity**: Low
- **Benefit**: Reduces total training time significantly

#### C. **Distributed Data Preprocessing**
- **Strategy**: Use PySpark DataFrame operations for ETL
- **Expected Speedup**: 2-4× for large datasets (>1M records)
- **Complexity**: High
- **Benefit**: Scalability to very large datasets

### Medium-Impact Opportunities

#### D. **Parallel Prediction Generation**
- **Strategy**: Batch predictions across multiple cores
- **Expected Speedup**: Linear with cores for large datasets
- **Complexity**: Low
- **Benefit**: Faster prediction for deployment

#### E. **Parallel Metric Computation**
- **Strategy**: Compute C-index, IBS, AUC in parallel
- **Expected Speedup**: 3× for 3 metrics
- **Complexity**: Low
- **Benefit**: Marginal (metrics are fast)

---

## Architecture Design

### Hybrid Pandas/PySpark Approach

**Rationale**: Not all operations benefit from Spark overhead. Use Spark only where it helps.

### Design Principles

1. **Transparent**: Code should work with or without Spark
2. **Configurable**: Enable/disable Spark via configuration
3. **Backward Compatible**: Existing pandas code continues to work
4. **Smart Switching**: Auto-detect when to use Spark based on data size
5. **Local-First**: Optimize for single-machine parallelism before true distribution

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   User Entry Point                       │
│              (main.py or run_pipeline)                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Execution Strategy Selector                 │
│   - Check dataset size                                   │
│   - Check available cores                                │
│   - Check Spark availability                             │
│   - Select: pandas, multiprocessing, or PySpark          │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────┐          ┌──────────────────┐
│ Pandas Path  │          │   Spark Path     │
│ (Default)    │          │   (Large Data)   │
└──────────────┘          └──────────────────┘
        │                         │
        │                         ▼
        │              ┌──────────────────────┐
        │              │  Spark DataFrame ETL  │
        │              │  - Distributed filter │
        │              │  - Distributed impute │
        │              └──────────┬───────────┘
        │                         │
        │                         ▼
        │              ┌──────────────────────┐
        │              │ Parallel CV Training  │
        │              │ - Spark ThreadPool    │
        │              │ - Model per executor  │
        │              └──────────┬───────────┘
        │                         │
        └────────┬────────────────┘
                 ▼
        ┌─────────────────┐
        │ Collect Results │
        │  (Pandas DF)    │
        └─────────────────┘
```

---

## Implementation Strategy

### Option 1: **Multiprocessing First** (Recommended)
Use Python's native `multiprocessing` or `joblib.Parallel` for parallel CV before adding Spark complexity.

**Pros**:
- No new dependencies (joblib already used)
- Simpler to implement and debug
- Works on single machine (most common use case)
- Lower overhead than Spark for medium datasets

**Cons**:
- Doesn't scale to true clusters
- Limited to single machine memory

### Option 2: **PySpark for ETL + Multiprocessing for CV**
Use Spark only for data preprocessing, then fall back to multiprocessing for model training.

**Pros**:
- Best of both worlds
- Spark excels at ETL, not ML training
- scikit-survival models aren't Spark-native

**Cons**:
- Added complexity
- Requires data conversion (Spark → pandas)

### Option 3: **Full PySpark with MLlib Integration**
Implement survival models using Spark MLlib or custom estimators.

**Pros**:
- True distributed computing
- Scales to massive datasets and clusters

**Cons**:
- Major rewrite required
- Spark MLlib has limited survival analysis support
- Complex debugging
- Overkill for single-machine use case

### **Recommendation**: Hybrid Approach

**Phase 1**: Multiprocessing for parallel CV (quick win)  
**Phase 2**: PySpark for data preprocessing (scales to large data)  
**Phase 3**: Evaluate custom Spark estimators if needed

---

## Configuration System

### Execution Modes

```python
class ExecutionMode(Enum):
    PANDAS = "pandas"           # Default: Single-threaded pandas
    MULTIPROCESSING = "mp"      # Parallel CV with multiprocessing
    PYSPARK_LOCAL = "spark_local"  # PySpark in local mode
    PYSPARK_CLUSTER = "spark_cluster"  # Distributed Spark cluster

class ExecutionConfig:
    mode: ExecutionMode = ExecutionMode.PANDAS
    n_jobs: int = -1  # -1 = use all cores
    spark_master: str = "local[*]"
    spark_memory: str = "4g"
    auto_mode: bool = True  # Auto-select based on data size
    size_threshold_mb: int = 100  # Switch to Spark above this size
```

### Auto-Detection Logic

```python
def select_execution_mode(df_size_mb: int, n_cores: int) -> ExecutionMode:
    """Auto-select execution mode based on data characteristics."""
    
    # Small data: use pandas (overhead not worth it)
    if df_size_mb < 10:
        return ExecutionMode.PANDAS
    
    # Medium data: use multiprocessing for CV
    if df_size_mb < 100:
        return ExecutionMode.MULTIPROCESSING
    
    # Large data: check if Spark is available
    if df_size_mb >= 100:
        if spark_available():
            return ExecutionMode.PYSPARK_LOCAL
        else:
            # Fallback to multiprocessing
            return ExecutionMode.MULTIPROCESSING
    
    return ExecutionMode.PANDAS
```

### Usage Examples

```bash
# Auto-detect (default)
python src/main.py --input data.pkl --run-type production

# Force multiprocessing
python src/main.py --input data.pkl --run-type production --execution-mode mp --n-jobs 8

# Force PySpark local mode
python src/main.py --input data.pkl --run-type production --execution-mode spark_local

# PySpark cluster
python src/main.py --input data.pkl --run-type production \
  --execution-mode spark_cluster \
  --spark-master spark://master:7077 \
  --spark-memory 8g
```

---

## Backwards Compatibility

### Requirements

1. ✅ **Zero Breaking Changes**: Existing code must work without modification
2. ✅ **Optional Dependencies**: PySpark should be optional (not required)
3. ✅ **Graceful Degradation**: Fall back to pandas if Spark unavailable
4. ✅ **Same API**: Users shouldn't need to change their code

### Implementation Approach

```python
# Old code (still works)
from main import run_pipeline
run_pipeline("data.csv", run_type="production")

# New code (opt-in to parallelism)
from main import run_pipeline
run_pipeline("data.csv", run_type="production", execution_mode="mp")
```

### Dependency Management

```python
# requirements.txt (existing)
pandas>=1.5
scikit-survival>=0.21
...

# requirements-spark.txt (new, optional)
pyspark>=3.5
pyarrow>=10.0  # For efficient Spark <-> pandas conversion
```

---

## Performance Targets

### Baseline (Current - Pandas)
- 287K records: ~30-60 minutes (5 models × 5 folds)
- CPU utilization: ~25% (1 core out of 4)
- Memory: ~1GB peak

### Target (Multiprocessing)
- 287K records: **~6-12 minutes** (5-10× speedup)
- CPU utilization: ~100% (all cores)
- Memory: ~2-3GB peak (multiple processes)

### Target (PySpark - Future)
- 1M+ records: **~10-20 minutes** (distributed preprocessing)
- 10M+ records: **~30-60 minutes** (cluster mode)
- Scalability: Linear with cluster size

### Metrics to Track
- Wall-clock time per model
- Wall-clock time per fold
- Total pipeline time
- CPU utilization %
- Memory usage (peak)
- Speedup vs. baseline

---

## Implementation Phases

### Phase 1: Parallel Cross-Validation (Week 1)
**Goal**: Parallelize the CV loop using multiprocessing

**Tasks**:
- [ ] Add `ExecutionConfig` class
- [ ] Implement parallel CV with `joblib.Parallel`
- [ ] Add `--execution-mode` and `--n-jobs` CLI flags
- [ ] Test with sample and production data
- [ ] Benchmark speedup vs. baseline
- [ ] Update documentation

**Files to Modify**:
- `src/survival_framework/train.py`
- `src/main.py`
- `src/survival_framework/config.py` (new)

**Expected Outcome**: 5-10× speedup on multi-core machines

### Phase 2: Parallel Model Training (Week 2)
**Goal**: Train different models in parallel

**Tasks**:
- [ ] Parallelize outer model loop
- [ ] Combine with CV parallelism (nested parallelism)
- [ ] Optimize memory usage (avoid pickle overhead)
- [ ] Test memory constraints
- [ ] Benchmark

**Expected Outcome**: Additional 2-3× speedup (total 10-20× vs baseline)

### Phase 3: PySpark Data Preprocessing (Week 3-4)
**Goal**: Use Spark for distributed ETL on large datasets

**Tasks**:
- [ ] Create `spark_data.py` module
- [ ] Implement PySpark DataFrame loader
- [ ] Implement distributed imputation
- [ ] Implement distributed scaling
- [ ] Implement distributed one-hot encoding
- [ ] Add Spark ↔ pandas conversion utilities
- [ ] Auto-detection of execution mode
- [ ] Integration tests

**Files to Add**:
- `src/survival_framework/spark_data.py`
- `src/survival_framework/spark_utils.py`

**Expected Outcome**: Scalability to 10M+ records

### Phase 4: Optimization and Production (Week 5)
**Goal**: Production-ready PySpark integration

**Tasks**:
- [ ] Performance profiling
- [ ] Memory optimization
- [ ] Cluster deployment guide
- [ ] CI/CD for Spark tests
- [ ] Documentation updates
- [ ] Example notebooks

---

## Technical Considerations

### Challenges

1. **Model Serialization**: scikit-survival models must be picklable for multiprocessing
2. **Memory Overhead**: Multiple processes = multiple data copies
3. **Spark Overhead**: Spark has startup cost (~5-10 seconds)
4. **Nested Parallelism**: Parallel models × parallel folds = complexity
5. **Random State**: Ensure reproducibility with parallel execution

### Solutions

1. **Use `loky` backend**: Better than pickle for sklearn models
2. **Shared memory**: Use `mmap_mode` for large arrays
3. **Lazy Spark init**: Only initialize when data size exceeds threshold
4. **Sequential outer loop**: Parallelize CV only, not models (simpler)
5. **Seed management**: Set seeds per process for reproducibility

### Key Libraries

- `joblib`: For parallel CV (already in dependencies)
- `pyspark`: For distributed data processing (optional)
- `pyarrow`: For fast Spark ↔ pandas conversion
- `dask` (alternative): Consider instead of Spark for simpler cases

---

## Success Criteria

### Must Have
- ✅ 5× speedup on 8-core machine for production data
- ✅ No breaking changes to existing API
- ✅ Graceful degradation if PySpark not available
- ✅ Works on sample data (backward compatible)

### Should Have
- ✅ 10× speedup with nested parallelism
- ✅ Auto-detection of execution mode
- ✅ Comprehensive benchmarks
- ✅ Updated documentation

### Nice to Have
- ✅ Cluster deployment example
- ✅ Dask integration as alternative
- ✅ Progress bars for parallel execution
- ✅ Resource usage monitoring

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Start Phase 1**: Implement parallel CV with multiprocessing
3. **Benchmark**: Measure actual speedup on production data
4. **Iterate**: Adjust plan based on results

---

## Open Questions

1. **Dask vs PySpark**: Should we consider Dask as a simpler alternative to Spark?
2. **Nested Parallelism**: Parallel models AND parallel CV, or just one?
3. **Cloud Deployment**: Should we plan for cloud Spark clusters (EMR, Databricks)?
4. **Ray**: Should we consider Ray for distributed training instead of Spark?

---

**Status**: Planning Phase  
**Next**: Review plan → Implement Phase 1 → Benchmark
