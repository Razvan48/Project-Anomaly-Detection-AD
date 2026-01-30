"""
This module contains the IsolationTreeNode and IsolationTree classes that
implement the standard Isolation Forest algorithm using random partitioning.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class IsolationTreeNode:
    """
    Node in an Isolation Tree.
    Each node either represents a split in the feature space or a leaf node containing data points. 
    The tree is built by recursively partitioning the space using random features and random thresholds.
    Attributes:
        depth: current depth of the node in the tree (root is 0).
        feature_limits: List of [min, max] boundaries for each feature dimension.
        idx_feature: Index of the feature used for splitting (None for leaf nodes).
        split_threshold: Threshold value for the split (None for leaf nodes).
        children: List of child nodes (empty for leaf nodes).
        data: Data points contained in this leaf node (None for internal nodes).
    """
    def __init__(
        self,
        depth: int,
        feature_limits: list[list[float]],
    ) -> None:
        """
        Initialize an IsolationTreeNode.
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
        MAX_DEPTH: int,
    ) -> None:
        """
        Recursively partition the feature space using random splits.
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
            self.feature_limits[self.idx_feature][1],
        )

        Xs_lower = Xs[Xs[:, self.idx_feature] < self.split_threshold]
        Xs_upper = Xs[Xs[:, self.idx_feature] >= self.split_threshold]

        feature_limits_lower = deepcopy(self.feature_limits)
        feature_limits_lower[self.idx_feature][1] = self.split_threshold

        feature_limits_upper = deepcopy(self.feature_limits)
        feature_limits_upper[self.idx_feature][0] = self.split_threshold

        child_lower = IsolationTreeNode(
            depth=self.depth + 1,
            feature_limits=feature_limits_lower,
        )

        child_upper = IsolationTreeNode(
            depth=self.depth + 1,
            feature_limits=feature_limits_upper,
        )

        self.children = [child_lower, child_upper]
        self.children[0].partition_space(Xs_lower, MAX_DEPTH)
        self.children[1].partition_space(Xs_upper, MAX_DEPTH)

    def get_path_lengths_batch(self, Xs: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        """
        Args:
            Xs: Data samples of shape (n_samples, n_features).
        Returns:
            Path lengths for each sample of shape (n_samples,).
        """
        n_samples = Xs.shape[0]
        path_lengths = np.zeros(n_samples, dtype=np.float64)
        
        if self.data is not None:
            # Leaf node - calculate harmonic correction for all samples
            if self.data.shape[0] <= 1:
                return path_lengths  # All zeros
            HARMONIC_NUMBER = np.log(self.data.shape[0] - 1) + 0.5772156649
            adjustment = 2.0 * (HARMONIC_NUMBER - (self.data.shape[0] - 1) / self.data.shape[0])
            return np.full(n_samples, adjustment, dtype=np.float64)
        
        assert self.idx_feature is not None
        assert self.split_threshold is not None
        
        mask_lower = Xs[:, self.idx_feature] < self.split_threshold
        
        if np.any(mask_lower):
            path_lengths[mask_lower] = 1 + self.children[0].get_path_lengths_batch(Xs[mask_lower])

        if np.any(~mask_lower):
            path_lengths[~mask_lower] = 1 + self.children[1].get_path_lengths_batch(Xs[~mask_lower])
        
        return path_lengths

    def plot_partition_space_2D(self) -> None:
        """
        Plots vertical/horizontal lines for each split and scatters points in leaf nodes.
        Only works for 2-dimensional data.
        """
        if self.data is not None:
            plt.scatter(self.data[:, 0], self.data[:, 1], c="lightgray", s=5)
            return

        if self.idx_feature is not None and self.split_threshold is not None:
            if self.idx_feature == 0:
                plt.plot([self.split_threshold, self.split_threshold],
                         [self.feature_limits[1][0], self.feature_limits[1][1]], c="gray")
            else:
                plt.plot([self.feature_limits[0][0], self.feature_limits[0][1]],
                         [self.split_threshold, self.split_threshold], c="gray")

        for child in self.children:
            child.plot_partition_space_2D()


class IsolationTree:
    """
    Single Isolation Tree for anomaly detection.
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
        contamination: float | None = 0.1,
    ) -> None:
        """
        Builds the tree structure by partitioning a subsample of the data,
        then calculates the anomaly threshold based on the contamination parameter.
        Args:
            Xs: Training data of shape (n_samples, n_features).
            subsample_size: Number of samples to use for building the tree.
                If None or >= n_samples, uses all samples.
            contamination: Expected proportion of anomalies (between 0 and 1).
                If None, no threshold is computed (used in ensemble training).
        """
        mins = np.min(Xs, axis=0) - self.PADDING
        maxs = np.max(Xs, axis=0) + self.PADDING

        self.feature_limits = [[float(mins[i]), float(maxs[i])] for i in range(len(mins))]
        self.root = IsolationTreeNode(
            depth=0,
            feature_limits=self.feature_limits,
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

        # Set maximum tree depth
        # Paper suggests: MAX_DEPTH = ceil(log2(n)) where n = subsample_size
        # For n=256: ceil(log2(256)) = 8
        # We use 9 to allow one extra level for edge cases and better partitioning
        MAX_DEPTH = 9
        self.root.partition_space(Xs_train, MAX_DEPTH)

        self.contamination = contamination
        if self.contamination is None:
            self.anomaly_threshold = 0.5
        else:
            Xs_train_depths_arr = self.root.get_path_lengths_batch(Xs_train)
            Xs_train_anomaly_scores = 2.0 ** (-Xs_train_depths_arr / self.expected_path_length)
            self.anomaly_threshold = float(np.quantile(Xs_train_anomaly_scores, 1.0 - self.contamination))

    def scores(self, Xs: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        """
        Anomaly scores are in [0, 1] where higher scores indicate anomalies.
        Based on the formula: 2^(-path_length / expected_path_length).
        Args:
            Xs: Data samples of shape (n_samples, n_features).
        Returns:
            Anomaly scores for each sample of shape (n_samples,).
        """
        assert self.root is not None
        assert self.expected_path_length is not None

        depths = self.root.get_path_lengths_batch(Xs)
        scores_arr = 2.0 ** (-depths / self.expected_path_length)
        return scores_arr

    def predict(self, Xs: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.int_]:
        """
        Args:
            Xs: Data samples of shape (n_samples, n_features).
        Returns:
            Binary labels (0=normal, 1=anomaly) of shape (n_samples,).
        """
        scores_arr = self.scores(Xs)
        return (scores_arr >= self.anomaly_threshold).astype(int)

    def get_path_lengths(self, Xs: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        """
        Args:
            Xs: Data samples of shape (n_samples, n_features).
        Returns:
            Path lengths for each sample of shape (n_samples,).
        """
        assert self.root is not None

        return self.root.get_path_lengths_batch(Xs)

    def plot_partition_space_2D(self) -> None:
        """
        Visualize the 2D space partitioning created by this tree.
        Only works for 2D data.
        """
        assert self.feature_limits is not None
        assert self.root is not None

        plt.title("Space Partition Isolation Tree")
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.plot([self.feature_limits[0][0], self.feature_limits[0][1]],
                 [self.feature_limits[1][0], self.feature_limits[1][0]], c="gray")
        plt.plot([self.feature_limits[0][0], self.feature_limits[0][1]],
                 [self.feature_limits[1][1], self.feature_limits[1][1]], c="gray")
        plt.plot([self.feature_limits[0][0], self.feature_limits[0][0]],
                 [self.feature_limits[1][0], self.feature_limits[1][1]], c="gray")
        plt.plot([self.feature_limits[0][1], self.feature_limits[0][1]],
                 [self.feature_limits[1][0], self.feature_limits[1][1]], c="gray")

        self.root.plot_partition_space_2D()
        plt.show()
    