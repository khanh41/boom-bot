import numpy as np

def flatten_obs(obs: np.ndarray) -> np.ndarray:
    # Flatten channels-first tensor to 1D
    return obs.reshape(-1).astype(np.float32)
