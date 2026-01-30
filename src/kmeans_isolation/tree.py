"""K-Means-based Isolation Tree implementation for anomaly detection.

This module contains the KMeansIsolationTreeNode and KMeansIsolationTree classes
that implement a variant of Isolation Forest using K-Means clustering for partitioning.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.cluster import MiniBatchKMeans


def find_optimal_kmeans(
    Xs: npt.NDArray[np.floating[Any]]
) -> tuple[int, MiniBatchKMeans]:
    """
    Find the optimal number of clusters using the elbow method.
    Uses MiniBatchKMeans for better performance on larger datasets.
    Args:
        Xs: Data samples of shape (n_samples, n_features).
    Returns:
        Tuple of (optimal_k, fitted_MiniBatchKMeans_model).
    """
    initial_Ks = [2, 3, 5, 7]
    Ks = [k for k in initial_Ks if k <= Xs.shape[0]]

    kmeans_list = []
    inertias = []

    for k in Ks:
        batch_size = min(100, Xs.shape[0])
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=23, batch_size=batch_size)
        kmeans.fit(Xs)

        kmeans_list.append(kmeans)
        inertias.append(kmeans.inertia_)

    if len(Ks) <= 2:
        return Ks[0], kmeans_list[0]

    start_point = np.array([Ks[0], inertias[0]])
    end_point = np.array([Ks[-1], inertias[-1]])
    distances = []

    for i in range(len(Ks)):
        current_point = np.array([Ks[i], inertias[i]])

        area = np.abs(np.cross(end_point - start_point, current_point - start_point))
        base_length = np.linalg.norm(end_point - start_point)

        distance = area / base_length
        distances.append(distance)

    optimal_k = Ks[np.argmax(distances)]
    optimal_kmeans = kmeans_list[np.argmax(distances)]
    return optimal_k, optimal_kmeans


class KMeansIsolationTreeNode:
    """
    Node in a K-Means-based Isolation Tree that partitions using clustering.
    Attributes:
        depth: Current depth of the node in the tree (root is 0).
        feature_limits: List of [min, max] boundaries for each feature dimension.
        idx_feature: Index of the feature used for splitting (None for leaf nodes).
        split_thresholds: List of threshold values between clusters (empty for leaf nodes).
        cluster_centers: List of cluster center positions (empty for leaf nodes).
        cluster_radii: List of cluster radii for normality scoring (empty for leaf nodes).
        children: List of child nodes (empty for leaf nodes).
        data: Data points contained in this leaf node (None for internal nodes).
    """
    
    def __init__(self, depth: int, feature_limits: list[list[float]]) -> None:
        """
        Initialize a KMeansIsolationTreeNode.
        Args:
            depth: Depth of this node in the tree.
            feature_limits: Boundaries for each feature dimension [[min1, max1], [min2, max2], ...].
        """
        self.depth = depth
        self.feature_limits = feature_limits

        self.idx_feature: int | None = None
        self.split_thresholds: list[float] = []
        self.cluster_centers: list[float] = []
        self.cluster_radii: list[float] = []

        self.children: list[KMeansIsolationTreeNode] = []
        self.data: npt.NDArray[np.floating[Any]] | None = None

    def partition_space(
        self,
        Xs: npt.NDArray[np.floating[Any]],
        MAX_DEPTH: int
    ) -> None:
        """
        Creates a multi-way tree by randomly selecting a feature, clustering
        the projected data, and creating splits between cluster centers.
        
        Args:
            Xs: Training data samples of shape (n_samples, n_features).
            MAX_DEPTH: Maximum depth to build the tree.
        """
        if self.depth == MAX_DEPTH or Xs.shape[0] <= 1 or np.all(Xs == Xs[0]):
            self.data = Xs
            return

        self.idx_feature = np.random.randint(Xs.shape[1])
        Xs_projected = Xs[:, self.idx_feature]

        _, kmeans = find_optimal_kmeans(Xs_projected.reshape(-1, 1))

        # Recalculate cluster centers based on actual assigned labels
        # MiniBatchKMeans can have centers that don't match the mean of assigned points
        # due to incremental mini-batch updates, which causes issues with radius calculation

        labels = kmeans.labels_
        recalculated_centers = []
        for cluster_idx in range(kmeans.n_clusters):
            cluster_mask = labels == cluster_idx
            if np.any(cluster_mask):
                cluster_mean = float(np.mean(Xs_projected[cluster_mask]))
                recalculated_centers.append(cluster_mean)
            else:
                recalculated_centers.append(float(kmeans.cluster_centers_[cluster_idx, 0]))

        self.cluster_centers = sorted(recalculated_centers)
        self.split_thresholds = []
        for idx in range(1, len(self.cluster_centers)):
            split_threshold = (self.cluster_centers[idx - 1] + self.cluster_centers[idx]) / 2.0
            self.split_thresholds.append(split_threshold)
        
        cluster_centers_array = np.array(self.cluster_centers).reshape(1, -1)  # (1, n_clusters)
        Xs_projected_reshaped = Xs_projected.reshape(-1, 1)  # (n_samples, 1)
        distances = np.abs(Xs_projected_reshaped - cluster_centers_array)  # (n_samples, n_clusters)
        closest_clusters = np.argmin(distances, axis=1)  # (n_samples,)

        # For each cluster, find max distance from any point assigned to it
        self.cluster_radii = []
        for cluster_idx in range(len(self.cluster_centers)):
            mask = closest_clusters == cluster_idx
            if np.any(mask):
                max_dist = np.max(distances[mask, cluster_idx])
                self.cluster_radii.append(float(max_dist))
            else:
                self.cluster_radii.append(0.0)

        self.children = []
        for idx in range(len(self.split_thresholds)):
            if idx == 0:
                threshold_lower = self.feature_limits[self.idx_feature][0]
            else:
                threshold_lower = self.split_thresholds[idx - 1]
            threshold_upper = self.split_thresholds[idx]

            Xs_between = Xs[
                (threshold_lower <= Xs[:, self.idx_feature]) &
                (Xs[:, self.idx_feature] < threshold_upper)
            ]

            feature_limits_between = deepcopy(self.feature_limits)
            feature_limits_between[self.idx_feature][0] = threshold_lower
            feature_limits_between[self.idx_feature][1] = threshold_upper

            child = KMeansIsolationTreeNode(
                depth=self.depth + 1,
                feature_limits=feature_limits_between
            )
            self.children.append(child)
            self.children[-1].partition_space(Xs_between, MAX_DEPTH)

        threshold_lower = self.split_thresholds[-1]
        threshold_upper = self.feature_limits[self.idx_feature][1]

        Xs_between = Xs[
            (threshold_lower <= Xs[:, self.idx_feature]) &
            (Xs[:, self.idx_feature] <= threshold_upper)
        ]

        feature_limits_between = deepcopy(self.feature_limits)
        feature_limits_between[self.idx_feature][0] = threshold_lower
        feature_limits_between[self.idx_feature][1] = threshold_upper

        child = KMeansIsolationTreeNode(
            depth=self.depth + 1,
            feature_limits=feature_limits_between
        )
        self.children.append(child)
        self.children[-1].partition_space(Xs_between, MAX_DEPTH)

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
            return path_lengths  # All zeros for leaf nodes
        
        assert self.idx_feature is not None
        
        X_projected = Xs[:, self.idx_feature]  # (n_samples,)
        
        for idx in range(len(self.split_thresholds)):
            if idx == 0:
                mask = X_projected < self.split_thresholds[idx]
            else:
                mask = (X_projected >= self.split_thresholds[idx-1]) & (X_projected < self.split_thresholds[idx])
            
            if np.any(mask):
                path_lengths[mask] = 1 + self.children[idx].get_path_lengths_batch(Xs[mask])
        
        # Last child
        mask = X_projected >= self.split_thresholds[-1]
        if np.any(mask):
            path_lengths[mask] = 1 + self.children[-1].get_path_lengths_batch(Xs[mask])
        
        return path_lengths

    def get_normality_scores_batch(self, Xs: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        """
        Args:
            Xs: Data samples of shape (n_samples, n_features).
        Returns:
            Normality scores for each sample of shape (n_samples,).
        """
        n_samples = Xs.shape[0]
        
        if self.data is not None:
            return np.zeros(n_samples, dtype=np.float64)
        
        assert self.idx_feature is not None
        
        X_projected = Xs[:, self.idx_feature]  # (n_samples,)
        
        cluster_centers_array = np.array(self.cluster_centers).reshape(1, -1)
        X_proj_reshaped = X_projected.reshape(-1, 1)
        distances = np.abs(X_proj_reshaped - cluster_centers_array)  # (n_samples, n_clusters)
        
        # Find closest cluster for each sample
        closest_cluster_indices = np.argmin(distances, axis=1)
        
        # Calculate normality scores vectorized
        cluster_radii_array = np.array(self.cluster_radii)
        closest_distances = distances[np.arange(n_samples), closest_cluster_indices]
        closest_radii = cluster_radii_array[closest_cluster_indices]
        
        # Calculate normality scores vectorized
        # Handle zero-radius case properly:
        # - If radius=0 AND distance=0: normality = 1.0 (exact match to single-point cluster)
        # - If radius=0 AND distance>0: normality = 0.0 (not at cluster center, outlier)
        # - If radius>0: normality = 1.0 - (distance / radius), clamped to [0, 1]
        normality_scores = np.zeros(n_samples, dtype=np.float64)
        
        # Handle zero-radius clusters
        zero_radius_mask = closest_radii == 0.0
        if np.any(zero_radius_mask):
            # For zero-radius: normality = 1.0 only if distance is also zero
            exact_match_mask = zero_radius_mask & (closest_distances == 0.0)
            normality_scores[exact_match_mask] = 1.0
            # Points not at center of zero-radius cluster get normality = 0.0 (already set)
        
        # Handle non-zero radius clusters
        non_zero_mask = closest_radii != 0.0
        if np.any(non_zero_mask):
            normality_scores[non_zero_mask] = 1.0 - (closest_distances[non_zero_mask] / closest_radii[non_zero_mask])
            normality_scores[non_zero_mask] = np.maximum(0.0, normality_scores[non_zero_mask])  # Clamp to [0, 1]
        
        # Recursively process children - vectorized routing
        child_scores = np.zeros(n_samples, dtype=np.float64)
        
        for idx in range(len(self.split_thresholds)):
            if idx == 0:
                mask = X_projected < self.split_thresholds[idx]
            else:
                mask = (X_projected >= self.split_thresholds[idx-1]) & (X_projected < self.split_thresholds[idx])
            
            if np.any(mask):
                child_scores[mask] = self.children[idx].get_normality_scores_batch(Xs[mask])
        
        # Last child
        mask = X_projected >= self.split_thresholds[-1]
        if np.any(mask):
            child_scores[mask] = self.children[-1].get_normality_scores_batch(Xs[mask])
        
        return normality_scores + child_scores

    def get_path_length(self, X: npt.NDArray[np.floating[Any]]) -> float:
        """
        Args:
            X: Single data sample of shape (n_features,).
        Returns:
            Path length to reach this sample.
        """
        if self.data is not None:
            return 0.0
        
        assert self.idx_feature is not None
        
        for idx in range(len(self.split_thresholds)):
            if X[self.idx_feature] < self.split_thresholds[idx]:
                return 1 + self.children[idx].get_path_length(X)
        return 1 + self.children[-1].get_path_length(X)
    
    def get_normality_score(self, X: npt.NDArray[np.floating[Any]]) -> float:
        """
        Computes cumulative normality score based on how close the sample is
        to cluster centers at each level. Score is inversely proportional to
        distance from cluster center normalized by cluster radius.
        Args:
            X: Single data sample of shape (n_features,).
        Returns:
            Cumulative normality score (higher = more normal).
        """

        if self.data is not None:
            return 0.0

        assert self.idx_feature is not None
        
        X_projected = X[self.idx_feature]

        cluster_distances = [np.abs(X_projected - cluster_center) for cluster_center in self.cluster_centers]
        closest_cluster_idx = int(np.argmin(cluster_distances))
        cluster_radius = self.cluster_radii[closest_cluster_idx]

        if cluster_radius == 0.0:
            normality_score = 1.0
        else:
            normality_score = 1.0 - (cluster_distances[closest_cluster_idx] / cluster_radius)
        normality_score = float(np.maximum(0.0, normality_score))  # to keep scores in [0.0, 1.0] for unseen data (which might be outside cluster radius)
        
        for idx in range(len(self.split_thresholds)):
            if X_projected < self.split_thresholds[idx]:
                return normality_score + self.children[idx].get_normality_score(X)
        return normality_score + self.children[-1].get_normality_score(X)
    
    def plot_partition_space_2D(self) -> None:
        """
        Plots vertical/horizontal lines for each split and scatters points in leaf nodes.
        Only works for 2-dimensional data.
        """
        if self.data is not None:
            plt.scatter(self.data[:, 0], self.data[:, 1], c='lightgray', s=5)
            return

        if self.idx_feature is not None:
            if self.idx_feature == 0:
                for split_threshold in self.split_thresholds:
                    plt.plot([split_threshold, split_threshold],
                            [self.feature_limits[1][0], self.feature_limits[1][1]], c='gray')
            else:
                for split_threshold in self.split_thresholds:
                    plt.plot([self.feature_limits[0][0], self.feature_limits[0][1]],
                            [split_threshold, split_threshold], c='gray')
                
        for child in self.children:
            child.plot_partition_space_2D()


class KMeansIsolationTree:
    """
    Single K-Means-based Isolation Tree for anomaly detection.
    Attributes:
        feature_limits: Boundaries for each feature dimension.
        root: Root node of the tree.
        contamination: Expected proportion of anomalies in the dataset.
        anomaly_threshold: Score threshold above which points are classified as anomalies.
        PADDING: Padding added to feature limits to handle edge cases.
    """
    
    def __init__(self) -> None:
        """Initialize a KMeansIsolationTree."""
        self.feature_limits: list[list[float]] | None = None
        self.root: KMeansIsolationTreeNode | None = None

        self.contamination: float | None = None
        self.anomaly_threshold: float | None = None

        self.PADDING = 1.0
    
    def fit(
        self,
        Xs: npt.NDArray[np.floating[Any]],
        subsample_size: int | None = 256,
        contamination: float | None = 0.1
    ) -> None:
        """
        Builds the tree structure by partitioning a subsample of the data using
        K-Means clustering, then calculates the anomaly threshold based on
        normality scores.
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

        self.root = KMeansIsolationTreeNode(
            depth=0,
            feature_limits=self.feature_limits
        )

        if subsample_size is not None and subsample_size < Xs.shape[0]:
            subsample_indices = np.random.choice(Xs.shape[0], subsample_size, replace=False)
            Xs_train = Xs[subsample_indices]
        else:
            Xs_train = Xs

        MAX_DEPTH = 9
        self.root.partition_space(Xs_train, MAX_DEPTH)

        self.contamination = contamination
        if self.contamination is None:
            self.anomaly_threshold = 0.5
        else:
            # Vectorized threshold calculation
            Xs_train_normality_scores_arr = self.root.get_normality_scores_batch(Xs_train)
            Xs_train_anomaly_scores = 1.0 - Xs_train_normality_scores_arr
            self.anomaly_threshold = float(np.quantile(Xs_train_anomaly_scores, 1.0 - self.contamination))

    def normality_scores(self, Xs: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        """
        Normality scores are based on proximity to cluster centers at each level.
        Higher scores indicate more normal samples.
        Args:
            Xs: Data samples of shape (n_samples, n_features).
        Returns:
            Normality scores for each sample of shape (n_samples,).
        """
        assert self.root is not None
        return self.root.get_normality_scores_batch(Xs)
    
    def scores(self, Xs: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        """
        Anomaly scores are derived from normality scores: anomaly = 1 - normality.
        Args:
            Xs: Data samples of shape (n_samples, n_features).
        Returns:
            Anomaly scores for each sample of shape (n_samples,).
        """
        normality_scores_arr = self.normality_scores(Xs)
        anomaly_scores = 1.0 - normality_scores_arr
        return anomaly_scores
    
    def predict(self, Xs: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.int_]:
        """
        Args:
            Xs: Data samples of shape (n_samples, n_features).
        Returns:
            Binary labels (0=normal, 1=anomaly) of shape (n_samples,).
        """
        anomaly_scores = self.scores(Xs)
        predictions = (anomaly_scores >= self.anomaly_threshold).astype(int)
        return predictions
    
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
        Shows the boundary box and all internal splits. Only works for 2D data.
        """
        assert self.feature_limits is not None
        assert self.root is not None
        
        plt.title('Space Partition KMeans Isolation Tree')
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
