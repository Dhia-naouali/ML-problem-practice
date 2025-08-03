import numpy as np

def instance_normalization(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Instance Normalization over a 4D tensor X of shape (B, C, H, W).
    gamma: scale parameter of shape (C,)
    beta: shift parameter of shape (C,)
    epsilon: small value for numerical stability
    Returns: normalized array of same shape as X
    """

    mean = np.mean(x, axis=(2, 3), keepdims=True)
    std = np.std(x, axis=(2, 3), keepdims=True) + epsilon


    return beta + gamma * (x - mean) / std
