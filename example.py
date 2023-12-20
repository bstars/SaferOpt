import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch

from algorithms.bayesian_opt import BayesianOpt, UCB, DynamicUCB, ExpectedImprovement, ProbOfImprovement, EpsGreedy
from algorithms.safe_opt import SafeOpt
from algorithms.stage_opt import StageOpt
from algorithms.robust_nn_regression import RobustNNRegression
from algorithms.safer_opt import SaferOpt

from utils.bb_function import NNRandomFunction
from utils.config import Config


def eg_bayesian_opt():
	def truef(xs):
		return 2 * (np.sin(xs) * np.cos(xs + 0.7) + np.log(np.abs(xs) + 3) - 1.)


	num = 500
	fmin = 0.
	xs = np.linspace(-5, 5, num)
	ys = truef(xs)

	print(xs.shape, ys.shape)

	xs = np.linspace(Config.xrange[0], Config.xrange[1], Config.xrange[2])
	truef = NNRandomFunction(
		dim=1,
		xrange=np.linspace(Config.xrange[0], Config.xrange[1], Config.xrange[2]),
		yrange=np.array([Config.yrange[0], Config.yrange[1]]),
		nn_dim=[1, 4, 8, 8, 4, 1],
		act=nn.LeakyReLU,
		ckpt=None
	)
	ys = truef.f

	idx = np.random.choice(
		np.where(np.logical_and(fmin <= ys, ys <= fmin + 0.4))[0], replace=False, size=3
	)



	opt = BayesianOpt(
		xs[idx][:, None],
		ys[idx][:, None],
		xs[:, None],
		[0.],
		1.0,
		kernel=None,
		aq_func=UCB()
		# aq_func=DynamicUCB()
		# aq_func=ExpectedImprovement()
		# aq_func=ProbOfImprovement()
		# aq_func=EpsGreedy()
	)

	for i in range(20):
		idx, x = opt.propose_evaluation()

		plt.scatter(opt.xs[:, 0], opt.ys[:, 0], c='r', label='Observations')
		plt.plot(xs, ys, 'k-', label='True Function')
		plt.fill_between(
			opt.parameter_set[:, 0],
			opt.Q[:, 0],
			opt.Q[:, 1],
			alpha=0.15,
			label='Confidence Interval',
		)
		plt.scatter(
			opt.parameter_set[opt.S, 0],
			np.ones([np.sum(opt.S)]) * opt.fmin,
			c='g', label='Safe Set', s=15)
		plt.scatter(xs[idx], ys[idx], c='b', label='Proposed Evaluation')
		plt.show()

		opt.add_new_data_point(
			xs[idx], ys[idx]
		)


def eg_safe_opt():
	def truef(xs):
		return 2 * (np.sin(xs) * np.cos(xs + 0.7) + np.log(np.abs(xs) + 3) - 1.)

	num = 500
	fmin = 0.
	xs = np.linspace(-5, 5, num)
	ys = truef(xs)
	L = np.max(np.abs(ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1]))
	idx = np.random.choice(
		np.where(np.logical_and(fmin <= ys, ys <= fmin + 1))[0], replace=False, size=3
	)

	opt = SafeOpt(
		# None,
		[L],
		xs[idx][:, None],
		ys[idx][:, None],
		xs[:, None],
		[0.],
		1.0,
		kernel=None,
		explore_threshold=0.02,
	)


	for _ in range(100):
		opt.compute_set()

		plt.figure(figsize=(10, 6))
		plt.scatter(opt.xs[:, 0], opt.ys[:, 0], c='r', label='Observations')
		plt.plot(xs, ys, 'k-', label='True Function')
		plt.fill_between(
			opt.parameter_set[:, 0],
			opt.Q[:, 0],
			opt.Q[:, 1],
			alpha=0.15,
			label='Confidence Interval',
		)
		plt.plot(xs, np.ones([len(xs)]) * opt.fmin, 'b--', label='Threshold')
		plt.scatter(
			opt.parameter_set[opt.S, 0],
			np.ones([np.sum(opt.S)]) * opt.fmin,
			c='g', label='Safe Set', s=15)
		plt.scatter(
			opt.parameter_set[opt.M, 0],
			np.ones([np.sum(opt.M)]) * opt.fmin,
			c='r', label='Possible Maximizers', s=15)
		plt.scatter(
			opt.parameter_set[opt.G, 0],
			np.ones([np.sum(opt.G)]) * opt.fmin,
			c='b', label='Expander', s=5)

		idx, x = opt.propose_evaluation()
		plt.scatter(x[0], truef(x[0]), c='y', label='Proposed Evaluation')
		plt.legend()
		plt.show()

		opt.add_new_data_point(x, truef(x))

def eg_stage_opt():
	def truef(xs):
		return 2 * (np.sin(xs) * np.cos(xs + 0.7) + np.log(np.abs(xs) + 3) - 1.)

	num = 500
	fmin = 0.
	xs = np.linspace(-5, 5, num)
	ys = truef(xs)
	L = np.max(np.abs(ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1]))
	idx = np.random.choice(
		np.where(np.logical_and(fmin <= ys, ys <= fmin + 1))[0], replace=False, size=3
	)



	opt = StageOpt(
		# None,
		[L],
		xs[idx][:, None],
		ys[idx][:, None],
		xs[:, None],
		[0.],
		1.0,
		kernel=None,
		switch_time=20
	)

	for _ in range(100):
		opt.compute_set()

		plt.scatter(opt.xs[:, 0], opt.ys[:, 0], c='r', label='Observations')
		plt.plot(xs, ys, 'k-', label='True Function')
		plt.fill_between(
			opt.parameter_set[:, 0],
			opt.Q[:, 0],
			opt.Q[:, 1],
			alpha=0.15,
			label='Confidence Interval',
		)
		plt.plot(xs, np.ones([len(xs)]) * opt.fmin, 'b--', label='Threshold')
		plt.scatter(
			opt.parameter_set[opt.S, 0],
			np.ones([np.sum(opt.S)]) * opt.fmin,
			c='g', label='Safe Set', s=15)
		plt.scatter(
			opt.parameter_set[opt.M, 0],
			np.ones([np.sum(opt.M)]) * opt.fmin,
			c='r', label='Possible Maximizers', s=15)
		plt.scatter(
			opt.parameter_set[opt.G, 0],
			np.ones([np.sum(opt.G)]) * opt.fmin,
			c='b', label='Expander', s=5)

		idx, x = opt.propose_evaluation()
		plt.scatter(x[0], truef(x[0]), c='y', label='Proposed Evaluation')
		plt.legend()
		plt.show()

		opt.add_new_data_point(x, truef(x))

def eg_robust_nn_regression():
	""" Example of Robust NN Regression """
	def truef(xs):
		return 2 * (np.sin(xs) * np.cos(xs + 0.7) + np.log(np.abs(xs) + 3) - 1.2)

	def truef_(xs):
		ys = np.zeros_like(xs)
		ys[xs<0] = - xs[xs<0]
		ys[xs>0] = xs[xs>0]
		return ys

	num = 500
	xs = np.linspace(-5, 5, num)
	ys = truef(xs)
	n = len(ys)

	src_idx = np.random.randint(0, n, size=5) # uniformly sample source data
	trg_idx = np.setdiff1d(np.arange(n), src_idx)

	src_idx = np.sort(src_idx)
	trg_idx = np.sort(trg_idx)

	Xsrc = xs[src_idx]
	ysrc = ys[src_idx]
	Xtrg = xs[trg_idx]
	ytrg = ys[trg_idx]

	rr = RobustNNRegression(nn_dim=[1, 4, 4, 4], kde_bandwidth=0.2)
	rr.fit(Xsrc, ysrc, Xtrg, max_iteration=20000, verbose=False, primal_lr=2e-4, dual_lr=2e-4, safe_threshold=None)

	mus, sigma = rr.predict(Xtrg)

	plt.scatter(Xsrc, ysrc, facecolor='none', edgecolor='red', s=30, label='Source data')
	plt.scatter(Xtrg, ytrg, c='black', s=5, label='Target data')
	plt.plot(Xtrg, mus, label='$\mu$')

	plt.fill_between(Xtrg, mus - 1.96 * np.sqrt(sigma), mus + 1.96 * np.sqrt(sigma),
	                 alpha=0.2,
	                 color='red', label='$95\%$ confidence interval')
	plt.legend()
	plt.title("Robust NN Regression ")
	plt.show()


def eg_safer_opt():
	num = 500
	threshold = -0.1
	num_seed = 5
	steps = 50

	def truef(xs):
		xs = xs * 1
		return 2 * (np.sin(xs) * np.cos(xs + 0.7) + np.log(np.abs(xs) + 3) - 1.2)

	xs = np.linspace(-5, 5, num)
	ys = truef(xs)
	seed_set_idx = np.random.choice(
		np.where(np.logical_and(threshold < ys, ys < threshold + 4))[0],
		size=num_seed, replace=False
	)
	# print(
	# 	xs[seed_set_idx][:, None].shape,
	# 	np.stack([ys[seed_set_idx], ys[seed_set_idx]], axis=1).shape
	# )
	opt = SaferOpt(
		xs[seed_set_idx][:, None],
		np.stack([ys[seed_set_idx], ys[seed_set_idx]], axis=1),
		xs[:, None],
		[-np.inf, threshold],
		1.0,
		aq_func=DynamicUCB()
	)

	fbest = np.max(ys[seed_set_idx])
	true_best = np.max(ys)
	history = []

	for i in range(steps):


		plt.figure(figsize=(10, 6))
		plt.scatter(opt.xs[:, 0], opt.ys[:, 0], c='r', label='Observations')
		plt.plot(xs, ys, 'k-', label='True Function')
		plt.fill_between(
			opt.parameter_set[:, 0],
			opt.Q[:, 0],
			opt.Q[:, 1],
			alpha=0.15,
			label='Reward Confidence Interval',
		)

		plt.fill_between(
			opt.parameter_set[:, 0],
			opt.Q[:, 2],
			opt.Q[:, 3],
			alpha=0.15,
			label='Constraint Confidence Interval',
		)

		plt.plot(xs, np.ones([len(xs)]) * threshold, 'b--', label='Threshold')
		plt.scatter(
			opt.parameter_set[opt.S, 0],
			np.ones([np.sum(opt.S)]) * threshold,
			c='g', label='Safe Set', s=15)

		idx, x = opt.propose_evaluation()
		x = x[0]
		plt.scatter(x, truef(x), c='y', label='Proposed Evaluation')
		plt.legend()
		plt.show()

		opt.add_new_data_point(
			np.atleast_2d(x),
			np.array([truef(x), truef(x)])[None, :]
		)

if __name__ == '__main__':
	eg_bayesian_opt()
	# eg_safe_opt()
	# eg_stage_opt()
	# eg_robust_nn_regression()
	# eg_safer_opt()