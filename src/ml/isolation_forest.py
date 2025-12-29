from sklearn.ensemble import IsolationForest
import numpy as np
from typing import Tuple, Dict 

def train_isolation_forest(
    X_train: np.ndarray,
    random_state: int = 42,
) -> Tuple[IsolationForest, Dict]:
    # Train an IF baseline model on normal-operation training data.
    # X_train: feature matrix of shape (num_windows, features)

    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array of shape (num_windows, features)")
    
    if X_train.shape[0] == 0:
        raise ValueError("X_train is empty. Cannot train ")
    
    model = IsolationForest(
        n_estimators = 100,
        max_samples = 'auto',
        contamination = 'auto',
        random_state = random_state,
    )

    model.fit(X_train)
    model_metadata = {
        "model_type": "IsolationForest",
        "n_estimators": model.n_estimators,
        "max_samples": model.max_samples,
        "contamination": model.contamination,
        "random_state": random_state,
        "num_training_samples": X_train.shape[0],
        "num_features": X_train.shape[1],
    }
    return model, model_metadata