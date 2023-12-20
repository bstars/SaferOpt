import sys
sys.path.append('..')

from collections.abc import Sequence
import numpy as np
import GPy
from scipy.spatial.distance import cdist
from scipy.special import expit

from utils.config import Config


class GPOpt(object):
	def __init__(self,
	             xs: np.array,
	             ys:np.array,
	             parameter_set:np.array,
	             fmin,
	             beta,
	             kernel=None,
	             noise_var=Config.gp_noise_var):
		"""

		A class for GP optimization.
		Most of the code is taken from https://github.com/befelix/SafeOpt

		:param xs:
			np.array, [num_seed, dim]
			Seed set points
		:param ys:
			np.array, [num_seed, num_func]
			Function values at seed set points
		:param parameter_set:
			np.array, [num_para, dim]
			Explore region
		:param fmin:
			np.array or list, [num_func]
			Threshold for each function.
			If no threshold considered for the i-th function, fmin[i] = -np.inf
		:param beta:
			float or callable
			Scheduling for confidence interval
		:param kernel: kernel for GP
		"""

		if kernel is None:
			kernel = GPy.kern.RBF(xs.shape[1])

		self.kernel = kernel

		self.gps = []
		for i in range(ys.shape[1]):
			self.gps.append(
				GPy.models.GPRegression(xs, ys[:, i][:, None], kernel=kernel.copy(), noise_var=noise_var,
				                        normalizer=False)
			)

		self.fmin = fmin
		self.beta = beta if hasattr(beta, '__call__') else lambda t: beta
		self.parameter_set = parameter_set
		self.xs = xs
		self.ys = ys

		# Q[:, ::2] is the lower bound
		# Q[:, 1::2] is the upper bound
		self.Q = self.confidence_interval()

	@property
	def t(self):
		return self.xs.shape[0]

	def confidence_interval(self):
		"""
		Compute the confidence interval for each function at each point in the parameter set
		and update the Q matrix
		Q[:, ::2] is the lower bound
		Q[:, 1::2] is the upper bound
		"""

		Q = np.zeros([
			len(self.parameter_set), 2 * len(self.gps)
		])
		for i, gp in enumerate(self.gps):
			mu, var = gp.predict_noiseless(self.parameter_set)
			mu, var = mu.squeeze(), var.squeeze()
			std = np.sqrt(var)
			Q[:, 2 * i] = mu - self.beta(self.t) * std
			Q[:, 2 * i + 1] = mu + self.beta(self.t) * std
		return Q

	def _add_data_to_gp(self, gp, x, y):
		gp.set_XY(
			np.concatenate([gp.X, x], axis=0),
			np.concatenate([gp.Y, y], axis=0)
		)

	def _remove_last_data_from_gp(self, gp):
		gp.set_XY(gp.X[:-1], gp.Y[:-1])

	def add_new_data_point(self, x, y):
		"""
		:param x: np.array, [num_pts, dim]
		:param y: np.array, [num_pts, num_func]
		:return: None
		"""
		x = np.atleast_2d(x)
		y = np.atleast_2d(y)
		self.xs = np.concatenate([self.xs, x], axis=0)
		self.ys = np.concatenate([self.ys, y], axis=0)
		for i, gp in enumerate(self.gps):
			self._add_data_to_gp(gp, x, y[:, i][:, None])
		self.Q = self.confidence_interval()

	def remove_last_data_point(self):
		self.xs = self.xs[:-1]
		self.ys = self.ys[:-1]
		for gp in self.gps:
			self._remove_last_data_from_gp(gp)
		self.Q = self.confidence_interval()