from __future__ import annotations
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from src.ml.detectors.base import BaseAnomalyDetector

class IsolationForestDetector(BaseAnomalyDetector):
    # Isolation Forest based anomaly detector operating on windowed feature vectors.

    def __init__(
        self,
        n_estimators: int = 100, # no of trees in IF
        contamination: float = 0.05, # expected fraction of anomalies in the data
        random_state: int = 42, # random seed for determination
        threshold_quantile: float = 95.0, #percentile of anomaly scores used to define decision threshold tau
        **kwargs,
    ):
        super().__init__(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        threshold_quantile=threshold_quantile,
        **kwargs,
        )

        self.model = SklearnIsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.threshold_quantile = threshold_quantile
        self._tau: Optional[float] = None
    
    def fit(self, X: np.ndarray) -> None:
        # Fit IF on windowed feature matrix

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_windows, n_features)")
        
        self.model.fit(X)

        # Pre-compute threshold using training scores
        train_scores = self.score(X)
        self._tau = np.percentile(train_scores, self.threshold_quantile) #self.threshold(train_scores)
        self._is_fitted = True

    def score(self, X: np.ndarray) -> np.ndarray:
        # compute continuous anomaly scores, higher values - more anomalous
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_windows, n_features)")

        return -self.model.score_samples(X)
    
    def threshold(self, scores: np.ndarray) -> float:
        # Compute anomaly decision threshold t using a fixed quantile
        if scores.ndim != 1:
            raise ValueError("scores must be a 1D array")
        
        if self._tau is None:
            raise RuntimeError("Detector must be fitted before thresholding")
        
        return self._tau
        