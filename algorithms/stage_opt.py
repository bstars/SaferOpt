import sys
sys.path.append('..')

import numpy as np
import GPy
from scipy.spatial.distance import cdist
from algorithms.gp_opt import GPOpt

class StageOpt(GPOpt):
	"""
		A simple and clear re-implementation of StageOpt algorithm from https://arxiv.org/pdf/1806.07555.pdf
		The original code is from https://github.com/vzhuang/StageOpt

		(StageOpt is very similar to SafeOpt in computing sets)

		Difference:
			1. In original algorithm, an error is raised if the safe set is empty, we'll just use ucb criterion to select the next point
			2. we don't consider contexts here, only basic functionalities are implemented
			3. We return not only the next evaluation point, but also the index of the next evaluation point in the parameter set
		"""

	def __init__(self,
	             lipschitz,
	             xs: np.array,
	             ys: np.array,
	             parameter_set: np.array,
	             fmin,
	             beta,
	             kernel=None,
	             explore_threshold=0.1,
	             switch_time=10):
		"""

		:param lipschitz:
			List[float] or np.array, [num_func] or None
			If lipschitz is None:
				use Lower Confidence Bound to compute the expanders
				https://github.com/vzhuang/StageOpt/blob/main/gp_opt.py#L567
			If lipschitz is not None:
				use Lipschitz constant to compute the expanders
				https://github.com/vzhuang/StageOpt/blob/main/gp_opt.py#L548 (also see Line 11 in Algorithm 1 in the paper)

		:param explore_threshold:
			float
			All point with confidence interval width less than this threshold will not be explored anymore
			https://github.com/vzhuang/StageOpt/blob/main/gp_opt.py#L526

		Other parameters: see gp_opt.py
		"""
		super().__init__(xs, ys, parameter_set, fmin, beta, kernel=kernel)
		self.lipschitz = lipschitz
		# self.S = np.zeros([parameter_set.shape[0]], dtype=bool)
		self.S = np.all(
			self.Q[:, ::2] >= self.fmin, axis=1
		)
		self.G = self.S.copy()  # possible expanders
		self.M = self.S.copy()  # possible maximizers
		self.d = cdist(
			self.parameter_set,
			self.parameter_set
		)

		self.use_lipschitz = False if self.lipschitz is None else True
		self.explore_threshold = explore_threshold
		self.switch_time = switch_time
		self.name = 'StageOpt'
		self.compute_set()

	def compute_safe_set(self):
		"""
		Remark:
			In line 7 of algorithm 1 in the paper, the safe set is computed using Lipschitz constant,
			but in the code, it is computed using lower confidence bound even if Lipschitz constant is provided.
			https://github.com/vzhuang/StageOpt/blob/main/gp_opt.py#L469

			But we also implement the Lipschitz version here.
		"""
		if self.use_lipschitz:
			d = self.d
			# S = np.zeros_like(self.S, dtype=bool)
			S = np.ones_like(self.S, dtype=bool)

			for i in range(len(self.gps)):
				# S = np.logical_or(
				# 	S,
				# 	np.any(self.Q[:, 2 * i] - self.lipschitz[i] * d > self.fmin[i], axis=1)
				# )

				S = np.logical_and(
					S,
					np.any(self.Q[self.S, 2 * i][None,:] - self.lipschitz[i] * d[:,self.S] > self.fmin[i], axis=1)
				)
			self.S = S
		else:
			self.S = np.all(
				self.Q[:, ::2] >= self.fmin, axis=1
			)


	def compute_set(self):
		self.compute_safe_set()
		self.M[:] = False
		self.G[:] = False

		# if no point is considered safe
		if not np.any(self.S):
			return

		l = self.Q[:, 0]
		u = self.Q[:, 1]
		self.M[self.S] = u[self.S] >= np.max(l[self.S])

		# if the confidence interval at a point is too small,
		# then we don't consider this point in the set of possible expanders
		# https://github.com/vzhuang/StageOpt/blob/main/gp_opt.py#L526
		s = np.logical_and(
			self.S,
			np.any(
				self.Q[:, 1::2] - self.Q[:, ::2] > self.explore_threshold * self.beta(self.t),
				axis=1
			)
		)
		s[s] = np.logical_and(s[s], ~self.M[s])

		# possible expanders
		G_safe = np.zeros_like(self.S, dtype=bool)
		for idx in np.where(s)[0]:

			if np.all(np.abs(self.Q[idx, 1::2] - self.Q[idx, ::2]) / 2 < self.explore_threshold):
				continue

			# https://github.com/vzhuang/StageOpt/blob/main/gp_opt.py#L548
			if self.use_lipschitz:
				d = cdist(
					self.parameter_set[idx][None, :],
					self.parameter_set[~self.S, :],
				)[0]  # distance from this safe point to all unsafe points

				for i in range(len(self.gps)):
					if self.fmin[i] == -np.inf:
						continue

					G_safe[idx] = np.any(
						self.Q[idx, 2 * i + 1] - self.lipschitz[i] * d > self.fmin[i]
					)

					if not G_safe[idx]:
						break

			else:
				for i in range(len(self.gps)):
					if self.fmin[i] == -np.inf:
						continue

					self._add_data_to_gp(
						self.gps[i],
						self.parameter_set[idx][None, :],
						np.atleast_2d(self.Q[idx, 2 * i + 1])
					)
					mean_after, var_after = self.gps[i].predict_noiseless(self.parameter_set[~self.S, :])
					self._remove_last_data_from_gp(self.gps[i])

					mean_after = mean_after.squeeze()
					std_after = np.sqrt(var_after).squeeze()

					G_safe[idx] = np.any(mean_after - std_after > self.fmin[i])

					if not G_safe[idx]:
						break

		self.G = np.logical_and(self.S, G_safe)

	def propose_evaluation(self, ucb=False):

		if not np.any(self.S):
			idx = np.argmax(self.Q[:, 1])
			return idx, self.parameter_set[idx]

		if self.should_exploit() or ucb or np.all(~self.G):
			idx = np.argmax(self.Q[self.S, 1])
			x = self.parameter_set[self.S][idx]
			idx = np.where(self.S)[0][idx]
			return idx, x

		uml = self.Q[self.G, 1] - self.Q[self.G, 0]
		idx = np.argmax(uml)
		idx = np.where(self.G)[0][idx]
		x = self.parameter_set[idx]
		return idx, x

	def should_exploit(self):
		if self.switch_time is None: return False
		return self.t >= self.switch_time

	def add_new_data_point(self, x, y):
		super().add_new_data_point(x, y)
		self.compute_set()

	def remove_last_data_point(self):
		super().remove_last_data_point()
		self.compute_set()