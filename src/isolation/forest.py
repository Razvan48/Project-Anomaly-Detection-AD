"""Isolation Forest implementation for anomaly detection.

This module contains the IsolationForest class that implements an ensemble
of isolation trees for robust anomaly detection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed

from .tree import IsolationTree


def _fit_single_tree(
    seed: int,
    Xs: npt.NDArray[np.floating[Any]],
    subsample_size: int | None,
) -> IsolationTree:
    """Worker function to fit a single isolation tree with a given seed.
    
    This function is designed to be called in parallel using joblib. Each worker
    receives an integer seed to ensure reproducibility.
    
    Args:
        seed: Random seed for this tree (integer).
        Xs: Training data of shape (n_samples, n_features).
        subsample_size: Number of samples to use for building the tree.
    
    Returns:
        Fitted IsolationTree instance.
    """
    # Set the numpy random seed for this worker
    np.random.seed(seed)
    
    tree = IsolationTree()
    tree.fit(Xs, subsample_size=subsample_size, contamination=None)
    return tree


class IsolationForest:
    """Ensemble of Isolation Trees for anomaly detection.
    
    An Isolation Forest combines multiple isolation trees to provide more robust
    anomaly detection. Each tree is trained on a random subsample of the data,
    and predictions are made by averaging scores across all trees.
    
    Attributes:
        ensemble_size: Number of trees in the ensemble.
        n_jobs: Number of parallel jobs to run. -1 means using all processors.
        random_state: Random seed for reproducibility.
        expected_path_length: Expected path length for the training subsample size.
        contamination: Expected proportion of anomalies in the dataset.
        anomaly_threshold: Score threshold above which points are classified as anomalies.
        trees: List of fitted IsolationTree instances.
    """
    
    def __init__(
        self, 
        ensemble_size: int = 100, 
        n_jobs: int = 1,
        random_state: int | None = None
    ) -> None:
        """Initialize an IsolationForest.
        
        Args:
            ensemble_size: Number of isolation trees to create in the ensemble.
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

        self.expected_path_length: float | None = None
        self.contamination: float | None = None
        self.anomaly_threshold: float | None = None

        self.trees: list[IsolationTree] = []
    
    def fit(
        self, 
        Xs: npt.NDArray[np.floating[Any]], 
        subsample_size: int | None = 256, 
        contamination: float = 0.1
    ) -> None:
        """Train the ensemble of isolation trees on data.
        
        Creates multiple isolation trees, each trained on a random subsample
        of the data, then calculates the anomaly threshold based on the
        average scores across all trees.
        
        Trees are built in parallel using joblib if n_jobs != 1. Reproducibility
        is ensured by generating seeds sequentially in the main process, then
        passing them to workers.
        
        Args:
            Xs: Training data of shape (n_samples, n_features).
            subsample_size: Number of samples to use for building each tree.
                If None or >= n_samples, uses all samples.
            contamination: Expected proportion of anomalies (between 0 and 1).
        """
        # Generate seeds sequentially for reproducibility
        rng = np.random.RandomState(self.random_state)
        MAX_INT = np.iinfo(np.int32).max
        seeds = rng.randint(MAX_INT, size=self.ensemble_size)
        
        # Build trees in parallel or sequentially
        if self.n_jobs == 1:
            # Sequential execution
            self.trees = []
            for seed in seeds:
                tree = _fit_single_tree(seed, Xs, subsample_size)
                self.trees.append(tree)
        else:
            # Parallel execution using joblib
            trees_list = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(_fit_single_tree)(seed, Xs, subsample_size)
                for seed in seeds
            )
            self.trees = list(trees_list)  # type: ignore[arg-type]

        if subsample_size is not None and subsample_size < Xs.shape[0]:
            train_size = subsample_size
        else:
            train_size = Xs.shape[0]

        if train_size <= 1:
            self.expected_path_length = 0.0
        else:
            HARMONIC_NUMBER = np.log(train_size - 1) + 0.5772156649
            self.expected_path_length = 2.0 * (HARMONIC_NUMBER - (train_size - 1) / train_size)

        self.contamination = contamination
        if self.contamination is None:
            self.anomaly_threshold = 0.5
        else:
            Xs_train_mean_depths = []
            for i in range(Xs.shape[0]):
                mean_depth = np.mean([tree.root.get_path_length(Xs[i]) for tree in self.trees])  # type: ignore
                Xs_train_mean_depths.append(mean_depth)
            Xs_train_mean_depths_arr = np.array(Xs_train_mean_depths)

            Xs_train_anomaly_scores = 2.0 ** (-Xs_train_mean_depths_arr / self.expected_path_length)
            self.anomaly_threshold = float(np.quantile(Xs_train_anomaly_scores, 1.0 - self.contamination))

    def scores(self, Xs: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        """Compute anomaly scores for samples.
        
        Anomaly scores are in [0, 1] where higher scores indicate anomalies.
        Scores are averaged across all trees in the ensemble.
        
        Args:
            Xs: Data samples of shape (n_samples, n_features).
            
        Returns:
            Anomaly scores for each sample of shape (n_samples,).
        """
        assert self.expected_path_length is not None
        
        scores_list = []
        for i in range(Xs.shape[0]):
            mean_depth = np.mean([tree.root.get_path_length(Xs[i]) for tree in self.trees])  # type: ignore
            score = 2.0 ** (-mean_depth / self.expected_path_length)
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
