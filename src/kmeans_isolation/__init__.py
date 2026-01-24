"""K-Means-based Isolation Forest implementation for anomaly detection.

This package provides a variant of Isolation Forest that uses K-Means clustering
for partitioning instead of random splits.
"""

from .forest import KMeansIsolationForest
from .tree import KMeansIsolationTree, KMeansIsolationTreeNode, find_optimal_kmeans

__all__ = [
    "KMeansIsolationTree",
    "KMeansIsolationTreeNode",
    "KMeansIsolationForest",
    "find_optimal_kmeans",
]
