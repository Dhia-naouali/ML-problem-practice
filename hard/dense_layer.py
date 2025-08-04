
import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

	def set_input_shape(self, shape):
    
		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
	def __init__(self, n_units, input_shape=None):
		self.layer_input = None
		self.input_shape = input_shape
		self.n_units = n_units
		self.trainable = True
		self.w = None
		self.w0 = None

    def initialize(self, optimizer):
        a = 1 / math.sqrt(self.input_shape[0])
        self.w = np.random.uniform(-a, a, size=(self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        self.optimizer = optimizer

	def forward_pass(self, x):
        self.x = x if self.trainable else None
        return x @ self.w + self.w0


	def backward_pass(self, accum_grad):
        if self.trainable:
            w_grad = self.x.T @ accum_grad
            self.x = None
            w0_grad = accum_grad.mean(axis=0)
            self.optimizer.update(self.w, w_grad)
            self.optimizer.update(self.w0, w0_grad)
        return accum_grad @ self.w.T


	def number_of_parameters(self):
        return self.w.size + self.w0.size if self.trainable else 0


    