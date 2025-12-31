import numpy as np

def build_normal_training_set(X_all, window_metadata, window_size, attack_mask):
    """
    Select normal windows from a pre-built window dataset.
    Normal = no attacked timesteps inside the window.
    """

    T = attack_mask.shape[0]
    start_indices = window_metadata["start_indices"]

    train_indices = []
    for i, start_t in enumerate(start_indices):
        end_t = start_t + window_size
        if end_t > T:
            continue
        if not np.any(attack_mask[start_t:end_t]):
            train_indices.append(i)

    if len(train_indices) == 0:
        raise ValueError("No normal windows available for training.")
    X_train = X_all[train_indices]
    train_metadata = {
        "num_total_windows": len(start_indices),
        "num_normal_windows": len(train_indices),
        "window_size": window_size,
        "train_indices": train_indices,
    }
    return X_train, train_metadata
