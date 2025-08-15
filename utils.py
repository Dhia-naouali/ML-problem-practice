import os
import numpy as np

def load_data():
    features_path = os.path.join("dummy_data", "features.npy")
    targets_path = os.path.join("dummy_data", "targets.npy")
    
    x = np.load(features_path)
    y = np.load(targets_path)
    return x, y