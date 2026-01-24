# Project-Anomaly-Detection-AD
&emsp; Project Anomaly Detection (AD), MSc Optional, Year 1, Semester 1, Faculty of Mathematics and Computer Science, University of Bucharest

<br/>

&emsp;&emsp;&emsp; **Students:** <br/>
<br/>
&emsp; Capatina Razvan-Nicolae ($407$) <br/>
&emsp; Buca Mihnea-Vicentiu ($408$) <br/>

<br/>
<br/>
<br/>

## Project Structure

```
src/
├── isolation/           # Standard Isolation Forest implementation
│   ├── tree.py         # IsolationTree, IsolationTreeNode
│   └── forest.py       # IsolationForest ensemble
└── kmeans_isolation/   # K-Means-based Isolation Forest
    ├── tree.py         # KMeansIsolationTree, KMeansIsolationTreeNode
    └── forest.py       # KMeansIsolationForest ensemble

notebooks/
└── original-presentation.ipynb  # Original analysis and results
```

## Quick Start

```python
import numpy as np
from src.isolation.forest import IsolationForest
from src.kmeans_isolation.forest import KMeansIsolationForest

# Generate data
X_train = np.random.randn(100, 2)
X_test = np.random.randn(20, 2)

# Standard Isolation Forest
iso_forest = IsolationForest(ensemble_size=100)
iso_forest.fit(X_train, subsample_size=256, contamination=0.1)
predictions = iso_forest.predict(X_test)

# K-Means Isolation Forest
kmeans_forest = KMeansIsolationForest(ensemble_size=100)
kmeans_forest.fit(X_train, subsample_size=256, contamination=0.1)
predictions = kmeans_forest.predict(X_test)
```

# TODO:

- [ ] abstract

- [x] **possible refactoring (split implementations in two .py files) and use notebook only for results and conclusions** ✅
  - Completed: Extracted to modular `src/` structure with full type hints
  - See `REFACTORING_SUMMARY.md` for details

- [ ] for introduction, general problem, existing methods, applications

- [ ] comparison execution time for different runs (isolation tree, kmeans tree, isolation ensemble, kmeans ensemble)
- [ ] **parallelize the ensembles (both isolation and kmeans)**
  - Next: Add `n_jobs` parameter using joblib

- [ ] compare times execution with already existing implementations (only isolation tree (can use forest with t=1) and isolation ensemble) (these two are from sklearn)
try other github implementations (also for kmeans tree and ensemble)

- [ ] complexity (brief explanation)

- [ ] brief mention of other approaches from same familty SOTA + brief technical details

- [ ] pseudocode in latex for implementations
- [ ] comparison with original authors (if they have any implementations) (both papers don't seem to have)

- [ ] **sensibility to parameters (higher contaminations for real datasets, ensemble size, subsampling size?)**
  - Next: Create `notebooks/sensitivity.ipynb`

- [ ] conclusions + future improvements

