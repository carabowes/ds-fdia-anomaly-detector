from sklearn.ensemble import IsolationForest
import numpy as np
from typing import Tuple, Dict 

# NOTE
# early implementation - replaced by ml/detectors/if.py

def train_isolation_forest(
    X_train: np.ndarray,
    random_state: int = 42,
) -> Tuple[IsolationForest, Dict]:
    # Train an IF baseline model on normal-operation training data.
    # X_train: feature matrix of shape (num_windows, features)

    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array of shape (num_windows, features)")
    
    if X_train.shape[0] == 0:
        raise ValueError("X_train is empty. Cannot train Isolation Forest ")
    
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

def compute_anomaly_scores(
    model: IsolationForest,
    X: np.ndarray,
    window_metadata: Dict,
    attack_mask: np.ndarray,
) -> Dict:
    
    """
    Compute anomaly scores and binary predictions for a full windowed dataset
    using a trained Isolation Forest model.

    Returns a dictionary with:
    - anomaly_score (float per window)
    - binary_prediction
    - window_start_indices
    - window_attack_labels
    """

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (num_windows, features)")
    
    start_indices = window_metadata["start_indices"]
    window_size = window_metadata["window_size"]
    
    if np.max(start_indices) + window_size > attack_mask.shape[0]:
        raise ValueError("attack_mask is too short for window alignment")

    if len(start_indices) != X.shape[0]:
        raise ValueError("Mismatch between X and window start indices")
    
    # Compute anomaly scores
    anomaly_scores = model.decision_function(X)
    binary_predictions = model.predict(X) # 1 for normal, -1 for anomaly (sklearn)
    binary_anomaly_label = (binary_predictions == -1).astype(int)  # Convert to 1 for anomaly, 0 for normal

    # Derive window-level attack labels
    window_attack_labels = []
    for start_t in start_indices:
        end_t = start_t + window_size
        is_attacked = np.any(attack_mask[start_t:end_t])
        window_attack_labels.append(int(is_attacked))
    
    results = {
        "anomaly_score": anomaly_scores,
        "binary_prediction": binary_predictions,
        "window_start_indices": start_indices,
        "window_attack_label": np.array(window_attack_labels),
        "binary_anomaly_label": binary_anomaly_label,
    }
    return results