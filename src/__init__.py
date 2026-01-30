"""Anomaly Detection package.

This package provides implementations of Isolation Forest algorithms for anomaly detection:
- isolation: Standard Isolation Forest using random partitioning
- kmeans_isolation: K-Means-based Isolation Forest using cluster-based partitioning
"""

from . import isolation, kmeans_isolation

__all__ = [
    "isolation",
    "kmeans_isolation",
]
