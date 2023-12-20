import sys
sys.path.append('..')

import numpy as np
import torch
from torch import nn
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import GPy
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
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
			optimizer = torch.optim.Adam(self.nn.parameters(), lr=1e-3, weight_decay=1e-7)
			for i in range(5000):
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

class Zinc(BBFunction):
	def __init__(self):
		super().__init__()
		self.zinc = pd.read_csv('../zinc/zinc_subsample.csv')
		cols = [
		        'penalized_logP',
				'qed',
		        # 'exact_mol_wt',
		        'fp_density_morgan_1',
		        'fp_density_morgan_2',
		        'fp_density_morgan_3',
		        'heavy_atom_mol_wt',
		        'max_abs_partial_charge',
		        'max_partial_charge',
		        'min_partial_charge',
		        'mol_weight',
		        'num_valence_electons']
		self.zinc = self.zinc[cols].to_numpy()
		self.Y = self.zinc[:, :2]
		self.X = self.zinc[:, 2:]

		# compute lipschitz constant for SafeOpt and StageOpt
		d = pairwise_distances(self.X, self.X)
		idx = np.tril_indices(len(self.X), k=-1)
		dy1 = self.Y[:, 0][:, None] - self.Y[:, 0][None, :]
		dy2 = self.Y[:, 1][:, None] - self.Y[:, 1][None, :]
		L1 = np.max(np.abs(dy1[idx] / d[idx]))
		L2 = np.max(np.abs(dy2[idx] / d[idx]))
		self.L = np.array([L1, L2])