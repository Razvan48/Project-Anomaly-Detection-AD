"""Isolation Tree implementation for anomaly detection.

This module contains the IsolationTreeNode and IsolationTree classes that implement
the standard Isolation Forest algorithm using random partitioning.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class IsolationTreeNode:
    """Node in an Isolation Tree that partitions the feature space.
    
    Each node either represents a split in the feature space or a leaf node
    containing data points. The tree is built by recursively partitioning
    the space using random features and random thresholds.
    
    Attributes:
        depth: Current depth of the node in the tree (root is 0).
        feature_limits: List of [min, max] boundaries for each feature dimension.
        idx_feature: Index of the feature used for splitting (None for leaf nodes).
        split_threshold: Threshold value for the split (None for leaf nodes).
        children: List of child nodes (empty for leaf nodes).
        data: Data points contained in this leaf node (None for internal nodes).
    """
    
    def __init__(self, depth: int, feature_limits: list[list[float]]) -> None:
        """Initialize an IsolationTreeNode.
        
        Args:
            depth: Depth of this node in the tree.
            feature_limits: Boundaries for each feature dimension [[min1, max1], [min2, max2], ...].
        """
        self.depth = depth
        self.feature_limits = feature_limits

        self.idx_feature: int | None = None
        self.split_threshold: float | None = None

        self.children: list[IsolationTreeNode] = []
        self.data: npt.NDArray[np.floating[Any]] | None = None

    def partition_space(
        self, 
        Xs: npt.NDArray[np.floating[Any]], 
        MAX_DEPTH: int
    ) -> None:
        """Recursively partition the feature space using random splits.
        
        Creates a binary tree by randomly selecting a feature and threshold,
        then recursively partitioning child nodes until max depth is reached
        or no further splits are possible.
        
        Args:
            Xs: Training data samples of shape (n_samples, n_features).
            MAX_DEPTH: Maximum depth to build the tree.
        """
        if self.depth == MAX_DEPTH or Xs.shape[0] <= 1 or np.all(Xs == Xs[0]):
            self.data = Xs
            return

        self.idx_feature = np.random.randint(Xs.shape[1])
        self.split_threshold = np.random.uniform(
            self.feature_limits[self.idx_feature][0],
            self.feature_limits[self.idx_feature][1]
        )

        Xs_lower = Xs[Xs[:, self.idx_feature] < self.split_threshold]
        Xs_upper = Xs[Xs[:, self.idx_feature] >= self.split_threshold]

        feature_limits_lower = deepcopy(self.feature_limits)
        feature_limits_lower[self.idx_feature][1] = self.split_threshold
        feature_limits_upper = deepcopy(self.feature_limits)
        feature_limits_upper[self.idx_feature][0] = self.split_threshold

        child_lower = IsolationTreeNode(
            depth=self.depth + 1,
            feature_limits=feature_limits_lower
        )
        child_upper = IsolationTreeNode(
            depth=self.depth + 1,
            feature_limits=feature_limits_upper
        )
        
        self.children = [child_lower, child_upper]
        self.children[0].partition_space(Xs_lower, MAX_DEPTH)
        self.children[1].partition_space(Xs_upper, MAX_DEPTH)

    def get_path_length(self, X: npt.NDArray[np.floating[Any]]) -> float:
        """Calculate the path length from root to the node containing X.
        
        For leaf nodes, estimates the expected path length adjustment based
        on the number of points in the leaf using the harmonic number.
        
        Args:
            X: Single data sample of shape (n_features,).
            
        Returns:
            Path length to reach this sample.
        """
        if self.data is not None:
            if self.data.shape[0] <= 1:
                return 0.0
            else:
                HARMONIC_NUMBER = np.log(self.data.shape[0] - 1) + 0.5772156649
                return 2.0 * (HARMONIC_NUMBER - (self.data.shape[0] - 1) / self.data.shape[0])
        
        assert self.idx_feature is not None
        assert self.split_threshold is not None
        
        if X[self.idx_feature] < self.split_threshold:
            return 1 + self.children[0].get_path_length(X)
        else:
            return 1 + self.children[1].get_path_length(X)
    
    def plot_partition_space_2D(self) -> None:
        """Visualize the 2D space partitioning created by this node and its children.
        
        Plots vertical/horizontal lines for each split and scatters points in leaf nodes.
        Only works for 2-dimensional data.
        """
        if self.data is not None:
            plt.scatter(self.data[:, 0], self.data[:, 1], c='lightgray', s=5)
            return

        if self.idx_feature is not None and self.split_threshold is not None:
            if self.idx_feature == 0:
                plt.plot([self.split_threshold, self.split_threshold],
                         [self.feature_limits[1][0], self.feature_limits[1][1]], c='gray')
            else:
                plt.plot([self.feature_limits[0][0], self.feature_limits[0][1]],
                         [self.split_threshold, self.split_threshold], c='gray')
                
        for child in self.children:
            child.plot_partition_space_2D()


class IsolationTree:
    """Single Isolation Tree for anomaly detection.
    
    An Isolation Tree isolates anomalies by randomly partitioning the feature space.
    Anomalies are easier to isolate (require fewer splits) than normal points.
    
    Attributes:
        feature_limits: Boundaries for each feature dimension.
        root: Root node of the tree.
        expected_path_length: Expected path length for the training subsample size.
        contamination: Expected proportion of anomalies in the dataset.
        anomaly_threshold: Score threshold above which points are classified as anomalies.
        PADDING: Padding added to feature limits to handle edge cases.
    """
    
    def __init__(self) -> None:
        """Initialize an IsolationTree."""
        self.feature_limits: list[list[float]] | None = None
        self.root: IsolationTreeNode | None = None

        self.expected_path_length: float | None = None
        self.contamination: float | None = None
        self.anomaly_threshold: float | None = None

        self.PADDING = 1.0
    
    def fit(
        self, 
        Xs: npt.NDArray[np.floating[Any]], 
        subsample_size: int | None = 256, 
        contamination: float | None = 0.1
    ) -> None:
        """Train the isolation tree on data.
        
        Builds the tree structure by partitioning a subsample of the data,
        then calculates the anomaly threshold based on the contamination parameter.
        
        Args:
            Xs: Training data of shape (n_samples, n_features).
            subsample_size: Number of samples to use for building the tree. 
                If None or >= n_samples, uses all samples.
            contamination: Expected proportion of anomalies (between 0 and 1).
                If None, no threshold is computed (used in ensemble training).
        """
        self.feature_limits = []
        for i in range(Xs.shape[1]):
            self.feature_limits.append([np.min(Xs[:, i]) - self.PADDING, np.max(Xs[:, i]) + self.PADDING])

        self.root = IsolationTreeNode(
            depth=0,
            feature_limits=self.feature_limits
        )

        if subsample_size is not None and subsample_size < Xs.shape[0]:
            subsample_indices = np.random.choice(Xs.shape[0], subsample_size, replace=False)
            Xs_train = Xs[subsample_indices]
        else:
            Xs_train = Xs

        if Xs_train.shape[0] <= 1:
            self.expected_path_length = 0.0
        else:
            HARMONIC_NUMBER = np.log(Xs_train.shape[0] - 1) + 0.5772156649
            self.expected_path_length = 2.0 * (HARMONIC_NUMBER - (Xs_train.shape[0] - 1) / Xs_train.shape[0])

        # MAX_DEPTH = int(np.ceil(np.log2(Xs_train.shape[0])))
        MAX_DEPTH = 9
        self.root.partition_space(Xs_train, MAX_DEPTH)

        self.contamination = contamination
        if self.contamination is None:
            self.anomaly_threshold = 0.5
        else:
            Xs_train_depths = []
            for i in range(Xs_train.shape[0]):
                depth = self.root.get_path_length(Xs_train[i])
                Xs_train_depths.append(depth)
            Xs_train_depths_arr = np.array(Xs_train_depths)

            Xs_train_anomaly_scores = 2.0 ** (-Xs_train_depths_arr / self.expected_path_length)
            self.anomaly_threshold = float(np.quantile(Xs_train_anomaly_scores, 1.0 - self.contamination))

    def scores(self, Xs: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        """Compute anomaly scores for samples.
        
        Anomaly scores are in [0, 1] where higher scores indicate anomalies.
        Based on the formula: 2^(-path_length / expected_path_length).
        
        Args:
            Xs: Data samples of shape (n_samples, n_features).
            
        Returns:
            Anomaly scores for each sample of shape (n_samples,).
        """
        assert self.root is not None
        assert self.expected_path_length is not None
        
        scores_list = []
        for i in range(Xs.shape[0]):
            depth = self.root.get_path_length(Xs[i])
            score = 2.0 ** (-depth / self.expected_path_length)
            scores_list.append(score)
            
        scores_arr = np.array(scores_list)
        return scores_arr
    
    def predict(self, Xs: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.int_]:
        """Predict anomaly labels for samples.
        
        Args:
            Xs: Data samples of shape (n_samples, n_features).
            
        Returns:
            Binary labels (0=normal, 1=anomaly) of shape (n_samples,).
        """
        scores_arr = self.scores(Xs)
        predictions = (scores_arr >= self.anomaly_threshold).astype(int)
        return predictions
    
    def get_path_lengths(self, Xs: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        """Get path lengths for all samples.
        
        Args:
            Xs: Data samples of shape (n_samples, n_features).
            
        Returns:
            Path lengths for each sample of shape (n_samples,).
        """
        assert self.root is not None
        
        path_lengths = []
        for i in range(Xs.shape[0]):
            path_length = self.root.get_path_length(Xs[i])
            path_lengths.append(path_length)

        path_lengths_arr = np.array(path_lengths)
        return path_lengths_arr

    def plot_partition_space_2D(self) -> None:
        """Visualize the 2D space partitioning created by this tree.
        
        Shows the boundary box and all internal splits. Only works for 2D data.
        """
        assert self.feature_limits is not None
        assert self.root is not None
        
        plt.title('Space Partition Isolation Tree')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.plot([self.feature_limits[0][0], self.feature_limits[0][1]],
                 [self.feature_limits[1][0], self.feature_limits[1][0]], c='gray')
        plt.plot([self.feature_limits[0][0], self.feature_limits[0][1]],
                 [self.feature_limits[1][1], self.feature_limits[1][1]], c='gray')
        plt.plot([self.feature_limits[0][0], self.feature_limits[0][0]],
                 [self.feature_limits[1][0], self.feature_limits[1][1]], c='gray')
        plt.plot([self.feature_limits[0][1], self.feature_limits[0][1]],
                 [self.feature_limits[1][0], self.feature_limits[1][1]], c='gray')

        self.root.plot_partition_space_2D()
        plt.show()
