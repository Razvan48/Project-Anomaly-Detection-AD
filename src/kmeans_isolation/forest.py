"""
This module contains the KMeansIsolationForest class that implements 
an ensemble of K-Means-based isolation trees for robust anomaly detection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed

from .tree import KMeansIsolationTree


def _fit_single_kmeans_tree(
    seed: int,
    Xs: npt.NDArray[np.floating[Any]],
    subsample_size: int | None,
) -> KMeansIsolationTree:
    """
    This function is designed to be called in parallel using joblib. 
    Args:
        seed: Random seed for this tree (integer).
        Xs: Training data of shape (n_samples, n_features).
        subsample_size: Number of samples to use for building the tree.
    Returns:
        Fitted KMeansIsolationTree instance.
    """
    np.random.seed(seed)
    
    tree = KMeansIsolationTree()
    tree.fit(Xs, subsample_size=subsample_size, contamination=None)
    return tree


def _score_single_tree_kmeans(
    tree: KMeansIsolationTree,
    Xs: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """
    This function is designed to be called in parallel using joblib. 
    Args:
    prediction speedup.
    Args:
        tree: Fitted KMeansIsolationTree instance.
        Xs: Data samples of shape (n_samples, n_features).
    Returns:
        Normality scores for each sample of shape (n_samples,).
    """
    return tree.root.get_normality_scores_batch(Xs)  # type: ignore[union-attr]


class KMeansIsolationForest:
    """
    Similar to IsolationForest but uses K-Means-based trees that score samples
    based on proximity to cluster centers rather than just path length.
    Attributes:
        ensemble_size: Number of trees in the ensemble.
        n_jobs: Number of parallel jobs to run. -1 means using all processors.
        random_state: Random seed for reproducibility.
        contamination: Expected proportion of anomalies in the dataset.
        anomaly_threshold: Score threshold above which points are classified as anomalies.
        trees: List of fitted KMeansIsolationTree instances.
    """
    def __init__(
        self,
        ensemble_size: int = 100,
        n_jobs: int = 1,
        random_state: int | None = None
    ) -> None:
        """
        Initialize a KMeansIsolationForest.
        Args:
            ensemble_size: Number of K-Means isolation trees to create in the ensemble.
            n_jobs: Number of parallel jobs to run for tree building.
                - If 1 (default): sequential execution (no parallelization)
                - If -1: use all available processors
                - If > 1: use specified number of processors
            random_state: Random seed for reproducibility. If None, results will
                vary between runs. If an integer, same seed produces identical results
                in both sequential (n_jobs=1) and parallel (n_jobs=-1) modes.
        """
        self.ensemble_size = ensemble_size
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.contamination: float | None = None
        self.anomaly_threshold: float | None = None

        self.trees: list[KMeansIsolationTree] = []
    
    def fit(
        self, 
        Xs: npt.NDArray[np.floating[Any]], 
        subsample_size: int | None = 256, 
        contamination: float = 0.1
    ) -> None:
        """
        Creates multiple K-Means isolation trees, each trained on a random subsample
        of the data, then calculates the anomaly threshold based on the average
        normality scores across all trees.

        Args:
            Xs: Training data of shape (n_samples, n_features).
            subsample_size: Number of samples to use for building each tree.
                If None or >= n_samples, uses all samples.
            contamination: Expected proportion of anomalies (between 0 and 1).
        """

        rng = np.random.RandomState(self.random_state)
        MAX_INT = np.iinfo(np.int32).max
        seeds = rng.randint(MAX_INT, size=self.ensemble_size)
        
        if self.n_jobs == 1:
            # Sequential execution
            self.trees = []
            for seed in seeds:
                tree = _fit_single_kmeans_tree(seed, Xs, subsample_size)
                self.trees.append(tree)
        else:
            # Parallel execution using joblib
            trees_list = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(_fit_single_kmeans_tree)(seed, Xs, subsample_size)
                for seed in seeds
            )
            self.trees = list(trees_list)  # type: ignore[arg-type]

        self.contamination = contamination
        if self.contamination is None:
            self.anomaly_threshold = 0.5
        else:
            normality_matrix = np.zeros((Xs.shape[0], len(self.trees)))
            for tree_idx, tree in enumerate(self.trees):
                normality_matrix[:, tree_idx] = tree.root.get_normality_scores_batch(Xs)  # type: ignore
            Xs_train_mean_normality_scores_arr = np.mean(normality_matrix, axis=1)

            Xs_train_anomaly_scores = 1.0 - Xs_train_mean_normality_scores_arr
            self.anomaly_threshold = float(np.quantile(Xs_train_anomaly_scores, 1.0 - self.contamination))

    def scores(self, Xs: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        """
        Anomaly scores are derived from normality scores averaged across all trees.
        Higher scores indicate anomalies.
        Args:
            Xs: Data samples of shape (n_samples, n_features).
        Returns:
            Anomaly scores for each sample of shape (n_samples,).
        """

        if self.n_jobs == 1:
            # Sequential execution
            normality_matrix = np.zeros((Xs.shape[0], len(self.trees)))
            for tree_idx, tree in enumerate(self.trees):
                normality_matrix[:, tree_idx] = tree.root.get_normality_scores_batch(Xs)  # type: ignore[union-attr]
        else:
            # Parallel execution using joblib
            normality_results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(_score_single_tree_kmeans)(tree, Xs) for tree in self.trees
            )
            normality_matrix = np.column_stack(list(normality_results))  # type: ignore[arg-type]

        mean_normality_scores = np.mean(normality_matrix, axis=1)
        anomaly_scores_arr = 1.0 - mean_normality_scores
        return anomaly_scores_arr
    
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
