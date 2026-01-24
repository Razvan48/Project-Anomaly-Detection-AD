"""Test script to verify parallelization produces identical results.

This script verifies that both IsolationForest and KMeansIsolationForest
produce identical results when trained with n_jobs=1 (sequential) vs 
n_jobs=-1 (parallel) using the same random_state.
"""

import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.isolation.forest import IsolationForest
from src.kmeans_isolation.forest import KMeansIsolationForest


def generate_test_data(n_samples=1000, n_features=5, random_state=42):
    """Generate synthetic test data with anomalies."""
    np.random.seed(random_state)
    
    # Normal samples
    normal = np.random.randn(int(n_samples * 0.9), n_features)
    
    # Anomalies (outliers)
    anomalies = np.random.randn(int(n_samples * 0.1), n_features) * 3 + 5
    
    # Combine
    X = np.vstack([normal, anomalies])
    y = np.array([0] * len(normal) + [1] * len(anomalies))
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    return X.astype(np.float64), y


def test_isolation_forest_reproducibility():
    """Test IsolationForest reproducibility across n_jobs values."""
    print("=" * 80)
    print("Testing IsolationForest Reproducibility")
    print("=" * 80)
    
    # Generate test data
    X_train, y_train = generate_test_data(n_samples=1000, random_state=42)
    X_test, y_test = generate_test_data(n_samples=200, random_state=43)
    
    # Train with n_jobs=1 (sequential)
    print("\n[1/3] Training with n_jobs=1 (sequential)...")
    if_seq = IsolationForest(
        ensemble_size=50,
        n_jobs=1,
        random_state=12345
    )
    if_seq.fit(X_train, subsample_size=256, contamination=0.1)
    scores_seq = if_seq.scores(X_test)
    predictions_seq = if_seq.predict(X_test)
    
    # Train with n_jobs=-1 (parallel)
    print("[2/3] Training with n_jobs=-1 (parallel)...")
    if_par = IsolationForest(
        ensemble_size=50,
        n_jobs=-1,
        random_state=12345
    )
    if_par.fit(X_train, subsample_size=256, contamination=0.1)
    scores_par = if_par.scores(X_test)
    predictions_par = if_par.predict(X_test)
    
    # Compare results
    print("[3/3] Comparing results...")
    scores_match = np.allclose(scores_seq, scores_par, rtol=1e-9, atol=1e-12)
    predictions_match = np.array_equal(predictions_seq, predictions_par)
    
    print(f"\n{'Results':.<40} {'Status'}")
    print("-" * 80)
    print(f"{'Scores identical (within tolerance)':<40} {'✓ PASS' if scores_match else '✗ FAIL'}")
    print(f"{'Predictions identical':<40} {'✓ PASS' if predictions_match else '✗ FAIL'}")
    
    if scores_match and predictions_match:
        print(f"\n{'Overall':<40} ✓ PASS")
    else:
        print(f"\n{'Overall':<40} ✗ FAIL")
        print(f"\nScore difference stats:")
        print(f"  Max absolute difference: {np.max(np.abs(scores_seq - scores_par)):.2e}")
        print(f"  Mean absolute difference: {np.mean(np.abs(scores_seq - scores_par)):.2e}")
        
    return scores_match and predictions_match


def test_kmeans_isolation_forest_reproducibility():
    """Test KMeansIsolationForest reproducibility across n_jobs values."""
    print("\n" + "=" * 80)
    print("Testing KMeansIsolationForest Reproducibility")
    print("=" * 80)
    
    # Generate test data
    X_train, y_train = generate_test_data(n_samples=1000, random_state=42)
    X_test, y_test = generate_test_data(n_samples=200, random_state=43)
    
    # Train with n_jobs=1 (sequential)
    print("\n[1/3] Training with n_jobs=1 (sequential)...")
    kmif_seq = KMeansIsolationForest(
        ensemble_size=50,
        n_jobs=1,
        random_state=12345
    )
    kmif_seq.fit(X_train, subsample_size=256, contamination=0.1)
    scores_seq = kmif_seq.scores(X_test)
    predictions_seq = kmif_seq.predict(X_test)
    
    # Train with n_jobs=-1 (parallel)
    print("[2/3] Training with n_jobs=-1 (parallel)...")
    kmif_par = KMeansIsolationForest(
        ensemble_size=50,
        n_jobs=-1,
        random_state=12345
    )
    kmif_par.fit(X_train, subsample_size=256, contamination=0.1)
    scores_par = kmif_par.scores(X_test)
    predictions_par = kmif_par.predict(X_test)
    
    # Compare results
    print("[3/3] Comparing results...")
    scores_match = np.allclose(scores_seq, scores_par, rtol=1e-9, atol=1e-12)
    predictions_match = np.array_equal(predictions_seq, predictions_par)
    
    print(f"\n{'Results':.<40} {'Status'}")
    print("-" * 80)
    print(f"{'Scores identical (within tolerance)':<40} {'✓ PASS' if scores_match else '✗ FAIL'}")
    print(f"{'Predictions identical':<40} {'✓ PASS' if predictions_match else '✗ FAIL'}")
    
    if scores_match and predictions_match:
        print(f"\n{'Overall':<40} ✓ PASS")
    else:
        print(f"\n{'Overall':<40} ✗ FAIL")
        print(f"\nScore difference stats:")
        print(f"  Max absolute difference: {np.max(np.abs(scores_seq - scores_par)):.2e}")
        print(f"  Mean absolute difference: {np.mean(np.abs(scores_seq - scores_par)):.2e}")
        
    return scores_match and predictions_match


def main():
    """Run all reproducibility tests."""
    print("\n" + "█" * 80)
    print("PARALLELIZATION REPRODUCIBILITY TEST SUITE")
    print("█" * 80)
    print("\nVerifying that n_jobs=1 (sequential) and n_jobs=-1 (parallel)")
    print("produce identical results with the same random_state.\n")
    
    # Run tests
    test1_passed = test_isolation_forest_reproducibility()
    test2_passed = test_kmeans_isolation_forest_reproducibility()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"IsolationForest:        {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"KMeansIsolationForest:  {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print("=" * 80)
    
    if test1_passed and test2_passed:
        print("\n✓ All tests PASSED! Parallelization is reproducible.")
        return 0
    else:
        print("\n✗ Some tests FAILED! Parallelization may not be reproducible.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
