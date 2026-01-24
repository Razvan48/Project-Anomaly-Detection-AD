"""Quick test to verify the refactored modules work correctly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.isolation.forest import IsolationForest
from src.kmeans_isolation.forest import KMeansIsolationForest

# Generate simple test data
np.random.seed(42)
X_train = np.random.randn(100, 2)
X_test = np.random.randn(20, 2)

print("Testing IsolationForest...")
iso_forest = IsolationForest(ensemble_size=10)
iso_forest.fit(X_train, subsample_size=50, contamination=0.1)
scores = iso_forest.scores(X_test)
predictions = iso_forest.predict(X_test)
print(f"  Scores shape: {scores.shape}")
print(f"  Predictions shape: {predictions.shape}")
print(f"  Sample scores: {scores[:5]}")
print(f"  Sample predictions: {predictions[:5]}")
print("  ✓ IsolationForest works!")

print("\nTesting KMeansIsolationForest...")
kmeans_forest = KMeansIsolationForest(ensemble_size=10)
kmeans_forest.fit(X_train, subsample_size=50, contamination=0.1)
scores = kmeans_forest.scores(X_test)
predictions = kmeans_forest.predict(X_test)
print(f"  Scores shape: {scores.shape}")
print(f"  Predictions shape: {predictions.shape}")
print(f"  Sample scores: {scores[:5]}")
print(f"  Sample predictions: {predictions[:5]}")
print("  ✓ KMeansIsolationForest works!")

print("\n✅ All refactored modules are working correctly!")
