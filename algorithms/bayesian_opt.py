import sys
sys.path.append('..')

import numpy as np
from scipy.stats import norm
import GPy

from algorithms.gp_opt import GPOpt
from utils.config import Config

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

class AcquisitionFunction(object):
	def __init__(self):
		self.name = ''

	def __call__(self, means, stds, **kwargs):
		"""
		return the index maximizes the acquisition function
		"""
		raise NotImplementedError

class UCB(AcquisitionFunction):
	def __init__(self):
		super().__init__()
		self.name = 'UCB'

	def __call__(self, means, stds, **kwargs):
		return np.argmax(means + stds)


class DynamicUCB(AcquisitionFunction):
	"""
	DynamicUCB chooses argmax_x s(t) * u(x) + (1 - s(t)) * std(x)
	"""
	def __init__(self):
		super().__init__()
		self.name = 'DUCB'

	def __call__(self, means, stds, **kwargs):
		t = kwargs.pop('t')
		s = sigmoid(t * Config.sig_step + Config.sig_start)
		# print(s)
		acq = s * means + (1 - s) * stds
		return int(np.argmax(acq, axis=0))

class ExpectedImprovement(AcquisitionFunction):

	def __init__(self):
		super().__init__()
		self.name = 'EoI'

	def __call__(self, means, stds, **kwargs):
		r_best = kwargs.pop('r_best')
		EIs = (means - r_best) * (1 - norm.cdf((r_best - means) / stds)) \
			+ stds * norm.pdf((r_best - means) / stds)
		return int(np.argmax(EIs, axis=0))

class ProbOfImprovement(AcquisitionFunction):
	def __init__(self):
		super().__init__()
		self.name = 'ProbOfImprovement'

	def __call__(self, means, stds, **kwargs):
		r_best = kwargs.pop('r_best')
		PoI = norm.cdf((means - r_best) / stds)
		return int(np.argmax(PoI, axis=0))


class EpsGreedy(AcquisitionFunction):
	"""
	Epsilon-Greedy choose
		the point with maximum mean with probability 1-eps
		other points with probability eps, the probability is proportional to their stds
	"""
	def __init__(self):
		super().__init__()
		self.name = 'EpsGreedy'

	def __call__(self, mean, stds, **kwargs):
		t = kwargs.pop('t')
		eps = Config.eps(t)
		# print(eps)

		ps = np.zeros_like(mean)
		idx = np.argmax(mean + stds)
		ps[idx] = 1 - eps
		ps += eps * stds / np.sum(stds)
		return int(np.random.choice(np.arange(len(mean)), p=ps))


class BayesianOpt(GPOpt):
	"""
	Bayesian Optimization with Gaussian Process and custom acquisition function.

	By default this considers safety constraint,
	if no safety constraint for i-th function, just set fmin[i] = -np.inf
	"""

	def __init__(self,
	             xs: np.array,
	             ys: np.array,
	             parameter_set: np.array,
	             fmin,
	             beta,
	             kernel=None,
	             aq_func: AcquisitionFunction = UCB(),
	             noise_var = Config.gp_noise_var):
		super().__init__(xs, ys, parameter_set, fmin, beta, kernel, noise_var)
		self.S = None
		self.aq_func = aq_func

		self.name = 'GP_' + self.kernel.name + "+" + (
			'' if np.all(np.array(fmin) == -np.inf) else 'Constrained') + aq_func.name
		self.compute_safe_set()

	def propose_evaluation(self):
		Q = self.Q

		# S = np.where(
		# 	np.all(Q[:, ::2] > self.fmin, axis=1)
		# )[0]
		S = self.S

		if len(S) == 0:
			idx = np.argmax(Q[:, 1])
			return idx, self.parameter_set[idx]

		mus = 0.5 * (Q[S, 0] + Q[S, 1])
		stds = 0.5 * (Q[S, 1] - Q[S, 0])
		idx = self.aq_func(
			mus, stds, r_best=np.max(self.ys[:, 0]), t=self.t
		)
		idx = np.where(S)[0][idx]
		return idx, self.parameter_set[idx]

	def compute_safe_set(self):
		self.Q = self.confidence_interval()
		self.S = np.all(self.Q[:, ::2] > self.fmin, axis=1)

	def add_new_data_point(self, x, y):
		# recompute the safe set after adding a new data point
		super().add_new_data_point(x, y)
		self.compute_safe_set()
