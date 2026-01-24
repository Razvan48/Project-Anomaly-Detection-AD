"""Isolation Forest implementation for anomaly detection.

This package provides the standard Isolation Forest algorithm using random
partitioning of the feature space.
"""

from .forest import IsolationForest
from .tree import IsolationTree, IsolationTreeNode

__all__ = [
    "IsolationTree",
    "IsolationTreeNode", 
    "IsolationForest",
]
