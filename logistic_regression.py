import numpy as np

class LogisticRegression:
    def __init__(self, in_dim, out_dim):
        self.w = np.random.randn(in_dim, out_dim)
        self.b = np.random.randn(1, out_dim)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def forward(self, x):
        self.x = x
        return self.sigmoid(x @ self.w + self.b)
        
    def backward(self, loss):
        ...
        del self.x