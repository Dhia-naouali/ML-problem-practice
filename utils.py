import os
import numpy as np

def _load_data():
    features_path = os.path.join("dummy_data", "features.npy")
    targets_path = os.path.join("dummy_data", "targets.npy")
    
    x = np.load(features_path)
    y = np.load(targets_path)
    return x, y

def split(*args, test_size=.2):
    splitted_args = []
    for array in args:
        n = len(array)
        n_test = int(test_size * n)
        splitted_args.extend([
            array[n_test:],
            array[:n_test]
        ])
        
    return splitted_args

def load_data():
    return split(*_load_data())