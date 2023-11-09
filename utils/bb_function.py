import sys
sys.path.append('..')

import numpy as np
import torch
from torch import nn
import GPy
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from utils.config import Config

class BBFunction():
	def __init__(self, **kwargs):
		pass

	def __call__(self, *args, **kwargs):
		pass


class GPRandomFunction(BBFunction):
	def __init__(self, parameter_set, kernel:GPy.kern.Kern, mean=0):
		super().__init__()
		self.mu = np.zeros([len(parameter_set)]) + mean
		self.kernel = kernel.copy()
		self.parameter_set = np.atleast_2d(parameter_set)
		C = self.kernel.K(self.parameter_set, self.parameter_set)
		self.f = np.random.multivariate_normal(self.mu, C, 1).squeeze()

	def __call__(self, idx):
		return self.f[idx]


class NNRandomFunction(BBFunction):
	""" Generate Neural network random function by random sampling points and overfitting a neural network """
	def __init__(self, dim, xrange, yrange, nn_dim, act=nn.LeakyReLU, ckpt=None):
		"""

		:param dim:
		:param xrange: [#samples, dim]
		:param yrange: [2]
		:param nn_dim: List[int], dimension of neural network
		:param act:
		:param ckpt:
		"""
		super().__init__()
		self.dim = dim
		if self.dim == 1 and len(xrange.shape) == 1:
			self.xrange = xrange[:, None]

		layers = []
		for i in range(len(nn_dim) - 2):
			layers.append(nn.Linear(nn_dim[i], nn_dim[i + 1]))
			layers.append(act())
		layers.append(nn.Linear(nn_dim[-2], nn_dim[-1]))
		self.nn = nn.Sequential(*layers)

		if ckpt is not None:
			self.nn.load_state_dict(torch.load(ckpt))
		else:
			xs = np.random.uniform(low=xrange[0], high=xrange[-1], size=[Config.nn_random_function_samples, dim])
			ys = np.random.uniform(low=yrange[0], high=yrange[1], size=[Config.nn_random_function_samples])
			xs = torch.from_numpy(xs).float()
			ys = torch.from_numpy(ys).float()

			loss = nn.MSELoss()
			optimizer = torch.optim.Adam(self.nn.parameters(), lr=1e-3)
			for i in range(10000):
				optimizer.zero_grad()
				y_pred = self.nn(xs)
				output = loss(y_pred[:, 0], ys)
				output.backward()
				optimizer.step()

		ys = self.nn(torch.from_numpy(self.xrange).float()).detach().numpy()[:,0]
		self.f = ys

	def __call__(self, idx):
		if isinstance(idx, int):
			return self.f[idx]
		return np.array([
			self.f[i] for i in idx
		])


if __name__ == '__main__':
	# parameter_set = np.linspace(-5, 5, 100)[:,None]
	# kernel = GPy.kern.Matern32(1)
	# f = GPRandomFunction(parameter_set, kernel)
	# plt.plot(parameter_set, f.f)
	# plt.show()

	# print(f(parameter_set))
	# print(f(parameter_set))

	fun = NNRandomFunction(
		dim=1,
		xrange=np.linspace(-5, 5, 500),
		yrange=np.array([-5, 5]),
		nn_dim=[1, 4, 8, 16, 8, 4, 1],
		act=nn.LeakyReLU,
		ckpt=None
	)
	plt.plot(fun.xrange[:, 0], fun.f)
	plt.show()