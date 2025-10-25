"""Timing utilities for performance logging.

Provides decorators and context managers to measure and log execution times
for functions and code blocks.

Example:
    >>> from survival_framework.timing import log_execution_time, Timer
    >>>
    >>> @log_execution_time()
    ... def train_model(X, y):
    ...     model.fit(X, y)
    ...
    >>> with Timer(logger, "Data preprocessing"):
    ...     X = preprocess(X)
"""
import time
import functools
import logging
from typing import Callable, Optional


def log_execution_time(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution time.

    Measures and logs the duration of function execution. If the function
    succeeds, logs an INFO message with timing. If it fails, logs an ERROR
    with the exception trace.

    Args:
        logger: Logger instance (uses function's module logger if None)

    Returns:
        Decorated function that logs its execution time

    Example:
        >>> @log_execution_time()
        ... def train_model(X, y):
        ...     model.fit(X, y)
        ...     return model
        >>> model = train_model(X, y)
        INFO     | Completed: train_model | duration_min=2.3
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(f"survival_framework.{func.__module__}")

            start_time = time.time()
            logger.info(f"Starting: {func.__name__}")

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Log to performance log
                from survival_framework.logging_config import log_performance
                log_performance(
                    logger,
                    f"Completed: {func.__name__}",
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

    Measures the duration of a code block and logs it as a performance metric.
    Useful for timing sections of code that aren't separate functions.

    Args:
        logger: Logger instance
        description: Description of the operation being timed

    Example:
        >>> with Timer(logger, "Model training"):
        ...     model.fit(X, y)
        INFO     | Starting: Model training
        INFO     | Completed: Model training | duration_min=5.2
    """

    def __init__(self, logger: logging.Logger, description: str):
        """Initialize timer.

        Args:
            logger: Logger instance to send timing messages to
            description: Description of what is being timed
        """
        self.logger = logger
        self.description = description
        self.start_time = None
        self.duration = None

    def __enter__(self):
        """Start timing when entering context."""
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.description}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing when exiting context and log duration."""
        self.duration = time.time() - self.start_time

        if exc_type is None:
            # Normal completion
            from survival_framework.logging_config import log_performance
            log_performance(
                self.logger,
                f"Completed: {self.description}",
                duration_sec=round(self.duration, 2),
                duration_min=round(self.duration / 60, 2)
            )
        else:
            # Exception occurred
            self.logger.error(
                f"{self.description} failed after {self.duration:.2f}s: {exc_val}",
                exc_info=True
            )

        # Don't suppress exception
        return False

    def elapsed(self) -> float:
        """Get elapsed time in seconds (during execution).

        Returns:
            Elapsed time in seconds since entering the context

        Example:
            >>> with Timer(logger, "Long operation") as timer:
            ...     do_something()
            ...     print(f"So far: {timer.elapsed():.1f}s")
            ...     do_more()
        """
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
