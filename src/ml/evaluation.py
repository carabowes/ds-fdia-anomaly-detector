import numpy as np
from typing import Dict, Optional

def evaluate_timestep_detection(
    attack_mask: np.ndarray,
    timestep_alarms: np.ndarray,
) -> Dict[str, Optional[float]]:
    
    if attack_mask.shape != timestep_alarms.shape:
        raise ValueError("attack_mask and timestep_alarms must have same shape")
    
    y_true = attack_mask.astype(int)
    y_pred = timestep_alarms.astype(int)

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    # TPR = recall
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = (
        2 * precision * TPR / (precision + TPR)
        if (precision + TPR) > 0
        else 0.0
    )

    # Detection delay
    attack_indices = np.where(y_true ==1)[0]
    if len(attack_indices) == 0:
        detection_delay = None
    else:
        attack_start = attack_indices[0]
        alarm_indices = np.where((y_pred == 1) & (np.arange(len(y_pred)) >= attack_start))[0]
        detection_delay = (int(alarm_indices[0] - attack_start) if len(alarm_indices) > 0 else None)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "TPR": TPR,
        "FPR": FPR,
        "detection_delay": detection_delay,
        "precision": precision,
        "accuracy": accuracy,
        "recall": recall,
        "f1": f1,
    }