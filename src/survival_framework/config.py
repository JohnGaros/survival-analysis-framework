"""Configuration module for execution modes and parallelization settings.

This module provides configuration for different execution modes:
- pandas: Single-threaded pandas (default, backward compatible)
- multiprocessing: Parallel cross-validation using joblib
- pyspark_local: PySpark in local mode (future)
- pyspark_cluster: Distributed PySpark cluster (future)
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import os
import multiprocessing


class ExecutionMode(str, Enum):
    """Execution mode for survival analysis pipeline.

    Attributes:
        PANDAS: Single-threaded pandas execution (default, backward compatible)
        MULTIPROCESSING: Parallel cross-validation using multiprocessing
        PYSPARK_LOCAL: PySpark in local mode (future implementation)
        PYSPARK_CLUSTER: Distributed PySpark cluster (future implementation)
    """
    PANDAS = "pandas"
    MULTIPROCESSING = "mp"
    PYSPARK_LOCAL = "spark_local"
    PYSPARK_CLUSTER = "spark_cluster"


@dataclass
class ExecutionConfig:
    """Configuration for execution mode and parallelization.

    Attributes:
        mode: Execution mode (pandas, mp, spark_local, spark_cluster)
        n_jobs: Number of parallel jobs. -1 means use all cores, 1 means sequential
        verbose: Verbosity level for joblib (0=silent, 10=progress bar, 50=detailed)
        backend: Joblib backend ('loky', 'threading', 'multiprocessing')
        auto_mode: If True, automatically select mode based on data size
        size_threshold_mb: Data size threshold for auto-selecting execution mode
        spark_master: Spark master URL (for spark modes)
        spark_memory: Memory allocation for Spark executor

    Example:
        >>> # Default configuration (pandas, single-threaded)
        >>> config = ExecutionConfig()

        >>> # Multiprocessing with all cores
        >>> config = ExecutionConfig(mode=ExecutionMode.MULTIPROCESSING, n_jobs=-1)

        >>> # Auto-detect mode based on data size
        >>> config = ExecutionConfig(auto_mode=True)
    """
    mode: ExecutionMode = ExecutionMode.PANDAS
    n_jobs: int = 1
    verbose: int = 10
    backend: str = "loky"
    auto_mode: bool = False
    size_threshold_mb: float = 10.0
    spark_master: str = "local[*]"
    spark_memory: str = "4g"

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Convert string mode to enum if needed
        if isinstance(self.mode, str):
            self.mode = ExecutionMode(self.mode)

        # Normalize n_jobs
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        elif self.n_jobs < 1:
            raise ValueError(f"n_jobs must be -1 or positive, got {self.n_jobs}")

        # For pandas mode, force n_jobs=1
        if self.mode == ExecutionMode.PANDAS:
            self.n_jobs = 1

    def get_effective_n_jobs(self) -> int:
        """Get the effective number of jobs for parallel execution.

        Returns:
            Number of parallel jobs to use (1 for sequential, >1 for parallel)

        Example:
            >>> config = ExecutionConfig(mode=ExecutionMode.MULTIPROCESSING, n_jobs=-1)
            >>> config.get_effective_n_jobs()
            8  # On an 8-core machine
        """
        if self.mode == ExecutionMode.PANDAS:
            return 1
        return self.n_jobs

    def is_parallel(self) -> bool:
        """Check if parallel execution is enabled.

        Returns:
            True if execution mode supports parallelism and n_jobs > 1

        Example:
            >>> config = ExecutionConfig(mode=ExecutionMode.PANDAS)
            >>> config.is_parallel()
            False

            >>> config = ExecutionConfig(mode=ExecutionMode.MULTIPROCESSING, n_jobs=4)
            >>> config.is_parallel()
            True
        """
        return self.mode != ExecutionMode.PANDAS and self.n_jobs > 1

    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"ExecutionConfig(mode={self.mode.value}, "
            f"n_jobs={self.n_jobs}, "
            f"parallel={self.is_parallel()})"
        )


def select_execution_mode(
    data_size_mb: float,
    n_cores: Optional[int] = None,
    force_mode: Optional[ExecutionMode] = None
) -> ExecutionMode:
    """Auto-select execution mode based on data characteristics.

    Selection logic:
    - Small data (<10MB): pandas (overhead not worth it)
    - Medium data (10-100MB): multiprocessing (parallel CV)
    - Large data (>100MB): multiprocessing (or spark if available)

    Args:
        data_size_mb: Dataset size in megabytes
        n_cores: Number of available CPU cores (auto-detected if None)
        force_mode: Force a specific mode (overrides auto-detection)

    Returns:
        Recommended execution mode

    Example:
        >>> # Small data: use pandas
        >>> select_execution_mode(5.0)
        <ExecutionMode.PANDAS: 'pandas'>

        >>> # Medium data: use multiprocessing
        >>> select_execution_mode(50.0)
        <ExecutionMode.MULTIPROCESSING: 'mp'>

        >>> # Force a specific mode
        >>> select_execution_mode(50.0, force_mode=ExecutionMode.PANDAS)
        <ExecutionMode.PANDAS: 'pandas'>
    """
    if force_mode is not None:
        return force_mode

    if n_cores is None:
        n_cores = multiprocessing.cpu_count()

    # Small data: pandas (overhead not worth it)
    if data_size_mb < 10.0:
        return ExecutionMode.PANDAS

    # Medium to large data: use multiprocessing if we have multiple cores
    if n_cores > 1:
        return ExecutionMode.MULTIPROCESSING

    # Fallback to pandas for single-core machines
    return ExecutionMode.PANDAS


def create_execution_config(
    mode: Optional[str] = None,
    n_jobs: int = -1,
    auto_mode: bool = True,
    verbose: int = 10
) -> ExecutionConfig:
    """Factory function to create ExecutionConfig with sensible defaults.

    Args:
        mode: Execution mode string ('pandas', 'mp', 'spark_local', 'spark_cluster')
        n_jobs: Number of parallel jobs (-1 = all cores)
        auto_mode: Enable auto-detection of execution mode
        verbose: Verbosity level (0=silent, 10=progress, 50=detailed)

    Returns:
        ExecutionConfig instance

    Example:
        >>> # Default: auto-detect mode, use all cores
        >>> config = create_execution_config()

        >>> # Force multiprocessing with 4 jobs
        >>> config = create_execution_config(mode='mp', n_jobs=4)

        >>> # Silent execution
        >>> config = create_execution_config(verbose=0)
    """
    if mode is None:
        execution_mode = ExecutionMode.PANDAS if not auto_mode else ExecutionMode.MULTIPROCESSING
    else:
        execution_mode = ExecutionMode(mode)

    return ExecutionConfig(
        mode=execution_mode,
        n_jobs=n_jobs,
        auto_mode=auto_mode,
        verbose=verbose
    )


def get_data_size_mb(file_path: str) -> float:
    """Get file size in megabytes.

    Args:
        file_path: Path to data file

    Returns:
        File size in MB

    Example:
        >>> get_data_size_mb('data/inputs/production/large_data.pkl')
        30.5
    """
    if not os.path.exists(file_path):
        return 0.0

    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb
