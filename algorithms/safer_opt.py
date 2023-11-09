import sys
sys.path.append('..')

import numpy as np

from algorithms.robust_nn_regression import RobustNNRegression
from algorithms.bayesian_opt import UCB
from utils.utilities import array_to_tensor, tensor_to_array
from utils.config import Config

class SaferOpt():
	def __init__(self,
	             xs:np.array,
	             ys:np.array,
	             parameter_set:np.array,
	             fmin,
	             beta,
	             aq_func=UCB(),
	             kde_bandwidth=0.1):
		self.dim = xs.shape[1]
		self.xs = xs
		self.ys = ys
		self.parameter_set = parameter_set
		self.fmin = fmin
		self.beta = beta if hasattr(beta, '__call__') else lambda t: beta
		self.aq_func = aq_func
		self.name = 'RR+' + "Constrained" + aq_func.name

		self.rrs = [
			RobustNNRegression(
				nn_dim=[self.dim] + Config.nn_dims,
				kde_bandwidth=kde_bandwidth
			)
			for _ in range(self.ys.shape[1])
		]

		[
			rr.fit(
				Xsrc=self.xs,
				ysrc=self.ys[:, i],
				Xtrg=self.parameter_set,
				max_iteration=1000,
				safe_threshold=self.fmin[i] if self.fmin[i] != -np.inf else None
			)
			for i, rr in enumerate(self.rrs)
		]

		self.Q = np.zeros([
			len(self.parameter_set), 2 * len(self.rrs)
		])
		self.S = np.zeros([
			parameter_set.shape[0]
		], dtype=bool)

		self.compute_set()

	@property
	def t(self): return self.xs.shape[0]

	def compute_set(self):
		Q = np.zeros(
			[self.parameter_set.shape[0], 2 * len(self.rrs)]
		)
		for i, rr in enumerate(self.rrs):
			mus, vars = rr.predict(self.parameter_set)
			stds = np.sqrt(vars)
			Q[:, 2 * i] = mus - self.beta(self.t) * stds
			Q[:, 2 * i + 1] = mus + self.beta(self.t) * stds
		self.Q = Q

		self.S = np.all(
			self.Q[:, ::2] >= self.fmin, axis=1
		)

	def propose_evaluation(self):
		l = self.Q[self.S, 0]
		u = self.Q[self.S, 1]
		mu = (l + u) / 2
		std = (u - l) / 2
		idx = self.aq_func(mu, std, r_best=np.max(self.ys[:, 0]), t=self.t)
		idx = np.where(self.S)[0][idx]
		return idx, self.parameter_set[idx]

	def add_new_data_point(self, x, y):
		"""
		:param x: [num_pts, dim]
		:param y: [num_pts, num_func]
		:return:
		"""
		self.xs = np.concatenate([self.xs, x], axis=0)
		self.ys = np.concatenate([self.ys, y], axis=0)
		[
			rr.fit(
				Xsrc=self.xs,
				ysrc=self.ys[:, i],
				Xtrg=self.parameter_set,
				max_iteration=1500,
				safe_threshold=self.fmin[i] if self.fmin[i] != -np.inf else None,
			)
			for i, rr in enumerate(self.rrs)
		]
		self.compute_set()



