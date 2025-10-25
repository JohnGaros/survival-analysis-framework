# Logging Policy Skill

## Purpose

This skill guides the implementation of a comprehensive logging strategy for the survival analysis framework to provide:
- Real-time visibility into execution status
- Performance metrics and timing data
- Warning and error tracking
- Audit trail for production runs
- Debugging information for development

## Logging Architecture

### 1. Multi-Level Logging Strategy

```python
import logging
from datetime import datetime
from pathlib import Path

# Logging levels by use case:
# DEBUG: Detailed diagnostic info (variable values, intermediate results)
# INFO: High-level progress updates (model started, fold completed)
# WARNING: Recoverable issues (convergence warnings, data quality issues)
# ERROR: Failures that don't stop execution (single fold failure)
# CRITICAL: Fatal errors that stop execution
```

### 2. Logger Hierarchy

```
survival_framework (root logger)
├── survival_framework.data (data loading & preprocessing)
├── survival_framework.models (model training)
│   ├── survival_framework.models.cox_ph
│   ├── survival_framework.models.gbsa
│   └── survival_framework.models.rsf
├── survival_framework.validation (cross-validation)
├── survival_framework.metrics (metric computation)
└── survival_framework.tracking (MLflow integration)
```

### 3. Log File Organization

```
data/outputs/{run_type}/logs/
├── main_{timestamp}.log              # Full detailed log
├── performance_{timestamp}.log       # Timing and metrics only
├── warnings_{timestamp}.log          # Warnings and errors only
└── debug_{timestamp}.log             # Debug level (dev only)
```

## Implementation Guidelines

### Step 1: Create Logging Configuration Module

Create `src/survival_framework/logging_config.py`:

```python
"""Centralized logging configuration for survival framework."""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from survival_framework.data import RunType


class PerformanceFilter(logging.Filter):
    """Filter to capture only performance-related messages."""

    def filter(self, record):
        return hasattr(record, 'is_performance') and record.is_performance


class WarningErrorFilter(logging.Filter):
    """Filter to capture only warnings and errors."""

    def filter(self, record):
        return record.levelno >= logging.WARNING


def setup_logging(
    run_type: RunType = "sample",
    log_level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """Setup comprehensive logging for survival framework.

    Args:
        run_type: Type of run (sample/production) - determines log directory
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output logs to console

    Returns:
        Configured root logger

    Example:
        >>> logger = setup_logging(run_type="production", log_level=logging.INFO)
        >>> logger.info("Starting training pipeline")
    """
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"data/outputs/{run_type}/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get root logger
    logger = logging.getLogger("survival_framework")
    logger.setLevel(logging.DEBUG)  # Capture everything, filter at handlers

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    performance_formatter = logging.Formatter(
        fmt='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='%(levelname)-8s | %(message)s'
    )

    # 1. Console handler (INFO and above)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # 2. Main detailed log file (all messages)
    main_handler = logging.FileHandler(
        log_dir / f"main_{timestamp}.log",
        mode='w',
        encoding='utf-8'
    )
    main_handler.setLevel(logging.DEBUG)
    main_handler.setFormatter(detailed_formatter)
    logger.addHandler(main_handler)

    # 3. Performance log file (timing and metrics)
    perf_handler = logging.FileHandler(
        log_dir / f"performance_{timestamp}.log",
        mode='w',
        encoding='utf-8'
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(performance_formatter)
    perf_handler.addFilter(PerformanceFilter())
    logger.addHandler(perf_handler)

    # 4. Warning/Error log file
    warning_handler = logging.FileHandler(
        log_dir / f"warnings_{timestamp}.log",
        mode='w',
        encoding='utf-8'
    )
    warning_handler.setLevel(logging.WARNING)
    warning_handler.setFormatter(detailed_formatter)
    warning_handler.addFilter(WarningErrorFilter())
    logger.addHandler(warning_handler)

    # 5. Debug log file (development only)
    if log_level == logging.DEBUG:
        debug_handler = logging.FileHandler(
            log_dir / f"debug_{timestamp}.log",
            mode='w',
            encoding='utf-8'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(detailed_formatter)
        logger.addHandler(debug_handler)

    logger.info(f"Logging initialized for {run_type} run")
    logger.info(f"Log directory: {log_dir.absolute()}")

    return logger


def log_performance(logger: logging.Logger, message: str, **kwargs):
    """Log a performance-related message with timing data.

    Args:
        logger: Logger instance
        message: Performance message
        **kwargs: Additional context (duration, records_processed, etc.)

    Example:
        >>> log_performance(logger, "GBSA fold 0 completed",
        ...                 duration_min=12.5, cindex=0.87)
    """
    # Add performance flag for filtering
    extra = {'is_performance': True}
    extra.update(kwargs)

    # Format message with metrics
    if kwargs:
        metrics_str = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        full_message = f"{message} | {metrics_str}"
    else:
        full_message = message

    logger.info(full_message, extra=extra)
```

### Step 2: Add Timing Decorators

```python
"""Timing utilities for performance logging."""
import time
import functools
import logging
from typing import Callable


def log_execution_time(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution time.

    Args:
        logger: Logger instance (uses module logger if None)

    Example:
        >>> @log_execution_time()
        ... def train_model(X, y):
        ...     # training code
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(f"survival_framework.{func.__module__}")

            start_time = time.time()
            logger.info(f"Starting {func.__name__}")

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Log to performance log
                from survival_framework.logging_config import log_performance
                log_performance(
                    logger,
                    f"{func.__name__} completed",
                    duration_sec=round(duration, 2),
                    duration_min=round(duration / 60, 2)
                )

                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{func.__name__} failed after {duration:.2f}s: {str(e)}",
                    exc_info=True
                )
                raise

        return wrapper
    return decorator


class Timer:
    """Context manager for timing code blocks.

    Example:
        >>> with Timer(logger, "Model training"):
        ...     model.fit(X, y)
    """

    def __init__(self, logger: logging.Logger, description: str):
        self.logger = logger
        self.description = description
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.description}")
        return self

    def __exit__(self, *args):
        duration = time.time() - self.start_time
        from survival_framework.logging_config import log_performance
        log_performance(
            self.logger,
            f"Completed: {self.description}",
            duration_sec=round(duration, 2),
            duration_min=round(duration / 60, 2)
        )
```

### Step 3: Add Progress Tracking

```python
"""Progress tracking for long-running operations."""
import logging
from typing import Iterable, Optional
from tqdm import tqdm


class ProgressLogger:
    """Logs progress updates for iterations.

    Example:
        >>> progress = ProgressLogger(logger, total=5, desc="Cross-validation")
        >>> for fold in range(5):
        ...     # training code
        ...     progress.update(1, metrics={'cindex': 0.85})
    """

    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        desc: str,
        log_interval: int = 1
    ):
        self.logger = logger
        self.total = total
        self.desc = desc
        self.log_interval = log_interval
        self.current = 0

    def update(self, n: int = 1, metrics: Optional[dict] = None):
        """Update progress by n steps."""
        self.current += n

        if self.current % self.log_interval == 0 or self.current == self.total:
            pct = (self.current / self.total) * 100
            msg = f"{self.desc}: {self.current}/{self.total} ({pct:.1f}%)"

            if metrics:
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                msg += f" | {metrics_str}"

            self.logger.info(msg)
```

### Step 4: Warning Capture and Categorization

```python
"""Capture and categorize warnings from libraries."""
import warnings
import logging
from contextlib import contextmanager


class WarningLogger:
    """Captures warnings and logs them appropriately."""

    WARNING_CATEGORIES = {
        'convergence': ['ConvergenceWarning', 'did not converge'],
        'numerical': ['overflow', 'underflow', 'invalid value'],
        'data': ['unknown categories', 'missing values'],
        'statistical': ['Hessian', 'variance_matrix'],
    }

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.warning_counts = {cat: 0 for cat in self.WARNING_CATEGORIES}

    def categorize_warning(self, message: str) -> str:
        """Categorize a warning message."""
        message_lower = message.lower()
        for category, keywords in self.WARNING_CATEGORIES.items():
            if any(kw.lower() in message_lower for kw in keywords):
                return category
        return 'other'

    def log_warning(self, message: str, category: str = None):
        """Log a warning with category."""
        if category is None:
            category = self.categorize_warning(message)

        self.warning_counts[category] += 1
        self.logger.warning(f"[{category.upper()}] {message}")

    def summary(self) -> dict:
        """Return summary of warnings by category."""
        return {k: v for k, v in self.warning_counts.items() if v > 0}


@contextmanager
def capture_warnings(logger: logging.Logger):
    """Context manager to capture warnings.

    Example:
        >>> with capture_warnings(logger) as warning_logger:
        ...     model.fit(X, y)  # warnings will be logged
        >>> print(warning_logger.summary())
    """
    warning_logger = WarningLogger(logger)

    def warning_handler(message, category, filename, lineno, file=None, line=None):
        warning_logger.log_warning(str(message))

    old_showwarning = warnings.showwarning
    warnings.showwarning = warning_handler

    try:
        yield warning_logger
    finally:
        warnings.showwarning = old_showwarning

        # Log summary
        summary = warning_logger.summary()
        if summary:
            logger.info(f"Warning summary: {summary}")
```

## Usage Examples

### Example 1: Main Training Pipeline

```python
# In src/main.py
from survival_framework.logging_config import setup_logging, log_performance
from survival_framework.timing import Timer

def run_pipeline(input_file, run_type="sample"):
    # Setup logging
    logger = setup_logging(run_type=run_type, log_level=logging.INFO)

    logger.info("="*70)
    logger.info(f"SURVIVAL ANALYSIS FRAMEWORK - {run_type.upper()} RUN")
    logger.info("="*70)
    logger.info(f"Input file: {input_file}")

    # Training phase
    with Timer(logger, "Complete training pipeline"):
        train_all_models(input_file, run_type=run_type, logger=logger)

    logger.info("Pipeline completed successfully")
```

### Example 2: Model Training with Progress

```python
# In src/survival_framework/train.py
from survival_framework.logging_config import log_performance
from survival_framework.timing import log_execution_time, Timer
from survival_framework.progress import ProgressLogger

@log_execution_time()
def train_all_models(file_path, run_type="sample", logger=None):
    logger = logger or logging.getLogger("survival_framework.train")

    # Load data
    with Timer(logger, "Data loading"):
        df = load_data(file_path)
        logger.info(f"Loaded {len(df):,} records")

    # Train models
    models = build_models()
    progress = ProgressLogger(logger, total=len(models), desc="Model training")

    for model_name, model in models.items():
        model_logger = logging.getLogger(f"survival_framework.models.{model_name}")

        with Timer(model_logger, f"{model_name} training"):
            with capture_warnings(model_logger) as warning_logger:
                # Cross-validation
                for fold in range(n_folds):
                    fold_logger = logging.getLogger(
                        f"survival_framework.models.{model_name}.fold{fold}"
                    )

                    # Train and evaluate
                    metrics = evaluate_model(model, X_train, y_train, X_val, y_val)

                    log_performance(
                        fold_logger,
                        f"Fold {fold} completed",
                        cindex=metrics['cindex'],
                        ibs=metrics['ibs']
                    )

        progress.update(1, metrics=avg_metrics)
```

### Example 3: Error Handling and Recovery

```python
# In src/survival_framework/validation.py
def evaluate_model(model, X_train, y_train, X_val, y_val, logger=None):
    logger = logger or logging.getLogger("survival_framework.validation")

    try:
        # Fit model
        with Timer(logger, "Model fitting"):
            model.fit(X_train, y_train)

        # Compute metrics
        with Timer(logger, "Metric computation"):
            metrics = compute_all_metrics(model, X_val, y_val)

        logger.info(f"Evaluation complete: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        # Return None or default metrics to continue with other folds
        return None
```

## Log Analysis Tools

### Log Parser Script

Create `scripts/analyze_logs.py`:

```python
"""Analyze training logs for performance and issues."""
import re
from pathlib import Path
from collections import defaultdict

def parse_performance_log(log_file: Path) -> dict:
    """Extract timing metrics from performance log."""
    metrics = defaultdict(list)

    with open(log_file) as f:
        for line in f:
            if 'duration_min=' in line:
                # Extract operation and duration
                match = re.search(r'(.*) \| duration_min=([\d.]+)', line)
                if match:
                    operation = match.group(1).strip()
                    duration = float(match.group(2))
                    metrics[operation].append(duration)

    return metrics

def summarize_warnings(log_file: Path) -> dict:
    """Summarize warnings by category."""
    categories = defaultdict(int)

    with open(log_file) as f:
        for line in f:
            if 'WARNING' in line:
                # Extract category
                match = re.search(r'\[(\w+)\]', line)
                if match:
                    category = match.group(1)
                    categories[category] += 1

    return dict(categories)
```

## Logging Checklist

When implementing logging in a module:

- [ ] Import logging and get module-specific logger
- [ ] Add `@log_execution_time()` to long-running functions
- [ ] Use `Timer` for major code blocks
- [ ] Log INFO for major milestones
- [ ] Log DEBUG for detailed state
- [ ] Log WARNING for recoverable issues
- [ ] Log ERROR with `exc_info=True` for exceptions
- [ ] Use `log_performance()` for metrics
- [ ] Capture library warnings with `capture_warnings()`
- [ ] Add progress logging for iterations
- [ ] Include relevant context (fold number, model name, etc.)

## Performance Logging Standards

For consistency, use these standard metric names:

**Timing Metrics**:
- `duration_sec`: Duration in seconds
- `duration_min`: Duration in minutes
- `records_processed`: Number of records
- `records_per_sec`: Processing rate

**Model Metrics**:
- `cindex`: Concordance index
- `ibs`: Integrated Brier Score
- `mean_auc`: Mean time-dependent AUC
- `fold`: Fold number

**Data Metrics**:
- `n_records`: Total records
- `n_features`: Number of features
- `n_events`: Number of events
- `censoring_rate`: Proportion censored

## Integration with Existing Code

### Priority 1 (High Impact)
1. `src/main.py` - Setup logging at entry point
2. `src/survival_framework/train.py` - Track model training progress
3. `src/survival_framework/validation.py` - Log CV progress and warnings

### Priority 2 (Medium Impact)
4. `src/survival_framework/data.py` - Log data loading and preprocessing
5. `src/survival_framework/models.py` - Capture model-specific warnings

### Priority 3 (Low Impact)
6. `src/survival_framework/metrics.py` - Log metric computation
7. `src/survival_framework/tracking.py` - Log MLflow operations

## Testing Logging

```python
# In tests/test_logging.py
def test_logging_setup(tmp_path):
    """Test logging configuration."""
    logger = setup_logging(run_type="sample", log_level=logging.INFO)

    logger.info("Test info message")
    logger.warning("Test warning")

    # Check log files created
    log_dir = Path("data/outputs/sample/logs")
    assert log_dir.exists()
    assert any(log_dir.glob("main_*.log"))
    assert any(log_dir.glob("performance_*.log"))
    assert any(log_dir.glob("warnings_*.log"))
```

## Expected Output Structure

After implementation, logs should look like:

**Console Output**:
```
INFO     | Starting training pipeline
INFO     | Loaded 287,013 records
INFO     | Starting: Data preprocessing
INFO     | Completed: Data preprocessing | duration_min=0.5
INFO     | Model training: 1/5 (20.0%)
INFO     | Starting: cox_ph training
INFO     | Completed: cox_ph training | duration_min=2.3 | avg_cindex=0.85
WARNING  | [CONVERGENCE] Cox model did not converge on fold 2
INFO     | Model training: 2/5 (40.0%)
...
```

**performance_TIMESTAMP.log**:
```
2025-10-25 08:00:00 | Data preprocessing completed | duration_min=0.5 | records_processed=287013
2025-10-25 08:02:30 | cox_ph training completed | duration_min=2.3 | avg_cindex=0.8521 | avg_ibs=0.1034
2025-10-25 08:05:00 | coxnet training completed | duration_min=2.5 | avg_cindex=0.8678 | avg_ibs=0.0942
2025-10-25 14:23:15 | gbsa training completed | duration_min=378.4 | avg_cindex=0.8947 | avg_ibs=0.0780
```

**warnings_TIMESTAMP.log**:
```
2025-10-25 08:02:15 | WARNING | survival_framework.models.cox_ph | [CONVERGENCE] Optimization did not converge: Maximum iterations exceeded
2025-10-25 08:15:30 | WARNING | survival_framework.models.weibull_aft | [STATISTICAL] Hessian was not invertible, using pseudo-inverse
```

## Notes

- Logs are gitignored (in `data/outputs/`)
- Keep performance log lightweight for easy parsing
- Use structured logging for metrics (key=value format)
- Always include timestamps and module names
- Capture full exception traces with `exc_info=True`
- Consider log rotation for long-running production jobs
- Archive logs after each run for historical analysis
