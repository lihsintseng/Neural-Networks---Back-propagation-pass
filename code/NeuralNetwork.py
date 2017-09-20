# Credit: Li-Hsin Tseng
import torch
import numpy as np
import math

'''
Part A:
Create a Python script and create an object model of class NeuralNetwork.
Initializing class using __init__(), with the list (in, h1, h2, …, out) as argument, 
will populate the network dictionary with the Θ(layer) matrices 
(which are mapping layer layer to layer + 1), 
initialised to random values (0 mean, 1/sqrt(layer_size) standard deviation). 
The size of the input layer is in, the size of the hidden layers are h1, h2, …, 
and the size of the output layer is out.
getLayer(layer) will return Θ(layer).
By running forward(input) the script will perform the forward propagation 
pass on the network previously built using sigmoid nonlinearities.
'''

class NeuralNetwork(object):
	# -- create the dictionary of matrices Θ
	# [nil] __init__(([int] in, [int] h1, [int] h2, …, [int] out))
	'''
	You may want to create two additional attributes (dictionaries) a and z in NeuralNetwork class.
	'''
	def __init__(self, arr):
		self.ThetaT = []
		self.dE_dTheta = []
		self.a = []
		self.delta = []
		self.row_num = 0
		for i in range(len(arr) - 1):
			tmp = np.random.normal(0, len(arr) ** 0.5, (arr[i]+1) * arr[i+1])
			self.ThetaT.append(tmp.reshape((arr[i+1], arr[i]+1)))

	# -- returns Θ(layer)
	# [2D DoubleTensor] getLayer([int] layer)
	def getLayer(self, layer_num):
		return torch.from_numpy(np.transpose(self.ThetaT[layer_num]))

	# -- feedforward pass single vector
	# [1D DoubleTensor] forward([1D DoubleTensor] input)
	# -- feedforward pass transposed design matrix
	# [2D DoubleTensor] forward([2D DoubleTensor] input)
	def forward(self, input):
		self.dE_dTheta = []
		self.a = []
		self.delta = []
		self.error = 0
		tmp = input.numpy()
		if tmp.ndim == 1:
			tmp = np.transpose([tmp])
			self.row_num = 1
		else:
			self.row_num = len(tmp[0])
		for i in range(len(self.ThetaT)):
			tmp = np.concatenate(([[1.0] * self.row_num], tmp), axis=0)
			self.a.append(tmp)
			tmp = np.matmul(self.ThetaT[i], tmp)
			for i in range(len(tmp)):
				for j in range(len(tmp[0])):
					sig = 1 / (1 + math.exp(-tmp[i][j]))
					tmp[i][j] = sig
		self.a.append(tmp)
		if self.row_num == 1:
			return torch.squeeze(torch.from_numpy(tmp.T))
		return torch.from_numpy(tmp)

	# -- back-propagation pass single target (computes ∂E/∂Θ)
	# [nil] backward([1D FloatTensor] target)
	# -- back-propagation pass target matrix
	# -- (computes the average of ∂E/∂Θ across seen samples)
	'''
	Implement back-propagation with a Mean Square Error loss function.
	Add one more parameter to backward([1D DoubleTensor] target) which 
	becomes now backward(target, [string] loss). 
	By default loss is MSE. Nevertheless, loss can be CE, which stands for cross-entropy.
	The cross-entropy loss function is defined as following:
	loss(h, y) = -log(exp(h[y]) / (\sum_j exp(h[j])))
	           = -h[y] + log(\sum_j exp(h[j]))
	'''
	# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
	# [nil] backward([2D FloatTensor] target)
	def backward(self, target, loss = 'MSE'):
		tmp = target.numpy()
		if tmp.ndim == 1:
			tmp = np.transpose([tmp])
		for i in reversed(range(len(self.ThetaT))):
			errors = []
			if i == len(self.ThetaT) - 1:
				for j in range(len(self.ThetaT[i])):
					errors.append([self.a[i+1][j][k] - tmp[j][k] for k in range(self.row_num)])		
			elif i == len(self.ThetaT) - 2:
				errors = np.matmul(np.transpose(self.ThetaT[i+1]), self.delta)
			else:
				errors = np.matmul(np.transpose(self.ThetaT[i+1]), self.delta[1:])

			self.error = np.mean(errors)

			self.delta = [[0] * self.row_num for _ in range(len(self.a[i+1]))]
			for j in range(len(self.a[i+1])):
				for k in range(self.row_num):
					self.delta[j][k] = errors[j][k] * self.a[i+1][j][k] * (1 - self.a[i+1][j][k])
			if i == len(self.ThetaT) - 1:
				self.dE_dTheta.append(np.matmul(self.a[i], np.transpose(self.delta)))
			else:
				self.dE_dTheta.append(np.matmul(self.a[i], np.transpose(self.delta[1:])))
		'''
		if loss == 'MSE':
		else if loss == 'CE':
		'''

	# -- update parameters
	# [nil] updateParams([float] eta)
	'''
	Update the matrices Theta with updateParams(eta) based on the 
	learning rate eta and the gradient of the error with respect of 
	the parameters dE_dTheta.
	'''
	def updateParams(self, eta):
		for i in range(len(self.ThetaT)):
			for j in range(len(self.ThetaT[i])):
				for k in range(len(self.ThetaT[i][j])):
					self.ThetaT[i][j][k] -= eta * np.transpose(self.dE_dTheta[-1-i])[j][k]

	