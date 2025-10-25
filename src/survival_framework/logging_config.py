"""Centralized logging configuration for survival framework.

This module provides comprehensive logging capabilities including:
- Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Multiple log files (main, performance, warnings, debug)
- Performance metric logging with timing data
- Warning categorization and tracking
- Progress tracking for long-running operations

Example:
    >>> from survival_framework.logging_config import setup_logging, log_performance
    >>> logger = setup_logging(run_type="production", log_level=logging.INFO)
    >>> logger.info("Starting training")
    >>> log_performance(logger, "Model trained", duration_min=12.5, cindex=0.87)
"""
import logging
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
from contextlib import contextmanager
from collections import defaultdict


RunType = Literal["sample", "production"]


class PerformanceFilter(logging.Filter):
    """Filter to capture only performance-related messages.

    Messages tagged with 'is_performance' attribute will pass through.
    """

    def filter(self, record):
        """Check if record is a performance metric."""
        return hasattr(record, 'is_performance') and record.is_performance


class WarningErrorFilter(logging.Filter):
    """Filter to capture only warnings and errors."""

    def filter(self, record):
        """Check if record is WARNING level or above."""
        return record.levelno >= logging.WARNING


def setup_logging(
    run_type: RunType = "sample",
    log_level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """Setup comprehensive logging for survival framework.

    Creates multiple log files in data/outputs/{run_type}/logs/:
    - main_{timestamp}.log: All log messages
    - performance_{timestamp}.log: Performance metrics only
    - warnings_{timestamp}.log: Warnings and errors only
    - debug_{timestamp}.log: Debug messages (if log_level=DEBUG)

    Args:
        run_type: Type of run (sample/production) - determines log directory
        log_level: Minimum log level (DEBUG=10, INFO=20, WARNING=30, ERROR=40)
        console_output: Whether to output logs to console (default: True)

    Returns:
        Configured root logger for survival_framework

    Example:
        >>> logger = setup_logging(run_type="production", log_level=logging.INFO)
        >>> logger.info("Pipeline started")
        >>> logger.warning("Convergence issue detected")
    """
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"data/outputs/{run_type}/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get root logger
    logger = logging.getLogger("survival_framework")
    logger.setLevel(logging.DEBUG)  # Capture everything, filter at handlers

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Prevent propagation to root logger
    logger.propagate = False

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

    This creates entries in both the main log and the dedicated performance log.
    Use this for tracking execution times, throughput, and model metrics.

    Args:
        logger: Logger instance
        message: Performance message description
        **kwargs: Additional context (duration, records_processed, metrics, etc.)

    Example:
        >>> log_performance(logger, "GBSA fold 0 completed",
        ...                 duration_min=12.5, cindex=0.87, ibs=0.08)
        # Output: "GBSA fold 0 completed | duration_min=12.5 | cindex=0.87 | ibs=0.08"
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


class WarningLogger:
    """Captures warnings and categorizes them for analysis.

    Categories:
    - convergence: Model convergence issues
    - numerical: Overflow, underflow, invalid values
    - data: Data quality issues (missing values, unknown categories)
    - statistical: Statistical warnings (Hessian issues, variance problems)
    - other: Uncategorized warnings
    """

    WARNING_CATEGORIES = {
        'convergence': ['ConvergenceWarning', 'did not converge', 'maximum iterations'],
        'numerical': ['overflow', 'underflow', 'invalid value', 'divide by zero'],
        'data': ['unknown categories', 'missing values', 'found unknown'],
        'statistical': ['Hessian', 'variance_matrix', 'covariance'],
    }

    def __init__(self, logger: logging.Logger):
        """Initialize warning logger.

        Args:
            logger: Logger instance to send warnings to
        """
        self.logger = logger
        self.warning_counts = {cat: 0 for cat in self.WARNING_CATEGORIES}
        self.warning_counts['other'] = 0

    def categorize_warning(self, message: str) -> str:
        """Categorize a warning message based on keywords.

        Args:
            message: Warning message text

        Returns:
            Category name (convergence, numerical, data, statistical, other)
        """
        message_lower = message.lower()
        for category, keywords in self.WARNING_CATEGORIES.items():
            if any(kw.lower() in message_lower for kw in keywords):
                return category
        return 'other'

    def log_warning(self, message: str, category: str = None):
        """Log a warning with category tag.

        Args:
            message: Warning message
            category: Category name (auto-detected if None)
        """
        if category is None:
            category = self.categorize_warning(message)

        self.warning_counts[category] += 1
        self.logger.warning(f"[{category.upper()}] {message}")

    def summary(self) -> dict:
        """Return summary of warnings by category.

        Returns:
            Dictionary of {category: count} for categories with warnings
        """
        return {k: v for k, v in self.warning_counts.items() if v > 0}


@contextmanager
def capture_warnings(logger: logging.Logger):
    """Context manager to capture and log warnings from libraries.

    Redirects Python warnings to the logging system, categorizes them,
    and provides a summary at the end.

    Args:
        logger: Logger instance

    Yields:
        WarningLogger instance for accessing warning counts

    Example:
        >>> with capture_warnings(logger) as warning_logger:
        ...     model.fit(X, y)  # warnings will be logged and categorized
        >>> print(warning_logger.summary())
        {'convergence': 3, 'numerical': 1}
    """
    warning_logger = WarningLogger(logger)

    def warning_handler(message, category, filename, lineno, file=None, line=None):
        """Custom warning handler that logs to our logger."""
        warning_text = f"{message}"
        warning_logger.log_warning(warning_text)

    # Save old handler
    old_showwarning = warnings.showwarning
    warnings.showwarning = warning_handler

    try:
        yield warning_logger
    finally:
        # Restore old handler
        warnings.showwarning = old_showwarning

        # Log summary
        summary = warning_logger.summary()
        if summary:
            summary_str = ", ".join(f"{k}={v}" for k, v in summary.items())
            logger.info(f"Warning summary: {summary_str}")


class ProgressLogger:
    """Logs progress updates for iterations.

    Example:
        >>> progress = ProgressLogger(logger, total=5, desc="Cross-validation")
        >>> for fold in range(5):
        ...     # training code
        ...     progress.update(1, metrics={'cindex': 0.85})
        # Output: "Cross-validation: 1/5 (20.0%) | cindex=0.8500"
    """

    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        desc: str,
        log_interval: int = 1
    ):
        """Initialize progress logger.

        Args:
            logger: Logger instance
            total: Total number of iterations
            desc: Description of the operation
            log_interval: Log every N updates (default: 1 = every update)
        """
        self.logger = logger
        self.total = total
        self.desc = desc
        self.log_interval = log_interval
        self.current = 0

    def update(self, n: int = 1, metrics: Optional[dict] = None):
        """Update progress by n steps.

        Args:
            n: Number of steps to advance (default: 1)
            metrics: Optional dict of metrics to include in log message
        """
        self.current += n

        if self.current % self.log_interval == 0 or self.current == self.total:
            pct = (self.current / self.total) * 100
            msg = f"{self.desc}: {self.current}/{self.total} ({pct:.1f}%)"

            if metrics:
                metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in metrics.items())
                msg += f" | {metrics_str}"

            self.logger.info(msg)
