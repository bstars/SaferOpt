import sys
sys.path.append('..')

import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Callable
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch import nn
import torch
import os
from scipy.io import savemat, loadmat
import GPy

from algorithms.safe_opt import SafeOpt
from algorithms.safer_opt import SaferOpt
from algorithms.stage_opt import StageOpt
from algorithms.bayesian_opt import BayesianOpt, DynamicUCB, ExpectedImprovement, ProbOfImprovement, UCB

from utils.bb_function import Zinc
from utils.config import Config
from experiments.run_opt_1d import run_experiment_1d

opts = [
	'GP_rbf+UCB',
	'RR+ConstrainedDUCB',
	'SafeOpt',
	'StageOpt'
]

def run_experiment(xs,
                   ys,
                   opt:SafeOpt or SaferOpt or StageOpt,
                   steps=100,
                   save_path='',
                   threshold=0., save=True):
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	true_best = ys[np.where(ys[:,1] > threshold)[0], 0].max()
	fbest = opt.ys[:, 0].max()
	proposed = []
	best_achieved = []
	num_violation = 0
	true_S = threshold < ys
	first_violation = False
	ratio_true_safe = None
	ratio_false_safe = None

	for i in range(steps):
		idx, x = opt.propose_evaluation()
		# print(opt.name, i, true_best, fbest, idx, ys.shape)

		print('%s, iteration %d, true best: %.3f, best achieved: %.3f, proposed: %.3f, violate: %d' % (
		opt.name, i, true_best, fbest, ys[idx, 0], ys[idx, 1] < threshold))

		if ys[idx, 0] > fbest and ys[idx, 1] > threshold:
			fbest = ys[idx, 0]

		opt.add_new_data_point(np.atleast_2d(xs[idx]), ys[idx])

		proposed.append(ys[idx,0])
		best_achieved.append(fbest)
	# 	if ys[idx, 1] < threshold:
	# 		num_violation += 1
	# 		if not first_violation:
	# 			first_violation = True
	# 			ratio_true_safe = np.sum(np.logical_and(true_S, opt.S)) / np.sum(true_S)
	# 			ratio_false_safe = np.sum(np.logical_and(~true_S, opt.S)) / np.sum(~true_S)
	#
	# if not first_violation:
	# 	first_violation = True
	# 	ratio_true_safe = np.sum(np.logical_and(true_S, opt.S)) / np.sum(true_S)
	# 	ratio_false_safe = np.sum(np.logical_and(~true_S, opt.S)) / np.sum(opt.S)

	savemat(
		os.path.join(save_path, 'result.mat'),
		{
			'threshold': threshold,
			'true_best': true_best,
			'proposed': np.array(proposed),
			'best_achieved': np.array(best_achieved),
			'num_violation': num_violation
		}
	)

def experiment_one_trial():

	zinc = Zinc()

	num_seed = 15
	steps = 150
	beta = lambda t: 0.5 / np.log(t / 4 + 2)

	xs = zinc.X
	ys = zinc.Y
	threshold = np.mean(ys[:,1])

	for trial in range(5):

		seed_set_idx = np.random.choice(
			np.where(
				np.logical_and(
					ys[:,1] > threshold + 0.01,
					ys[:,0] < 0.5 * (np.max(ys[:,0]) + np.mean(ys[:,0]))
				)
			)[0],
			size=num_seed, replace=True
		)

		opts = [
			BayesianOpt(
				xs[seed_set_idx],
				ys[seed_set_idx],
				xs,
				[-np.inf, -np.inf],
				beta,
				kernel=GPy.kern.RBF(xs.shape[1], variance=15),
				aq_func=DynamicUCB()
			),
			# SafeOpt(
			# 	lipschitz=zinc.L,
			# 	xs=xs[seed_set_idx],
			# 	ys=ys[seed_set_idx],
			# 	parameter_set=xs,
			# 	fmin=[-np.inf, threshold],
			# 	beta=beta,
			# 	kernel=None
			# ),
			# StageOpt(
			# 	lipschitz=zinc.L,
			# 	xs=xs[seed_set_idx],
			# 	ys=ys[seed_set_idx],
			# 	parameter_set=xs,
			# 	fmin=[-np.inf, threshold],
			# 	beta=beta,
			# 	kernel=None,
			# 	switch_time=30
			# ),
			# SaferOpt(
			# 	xs=xs[seed_set_idx],
			# 	ys=ys[seed_set_idx],
			# 	parameter_set=xs,
			# 	fmin=[-np.inf, threshold],
			# 	beta=beta,
			# 	aq_func=DynamicUCB(),
			# 	kde_bandwidth=0.3
			# )
		]

		for opt in opts:
			run_experiment(xs, ys, opt, steps, save_path='results2/zinc/%s/trial_%d' % (opt.name, trial), threshold=threshold)


def experiment_1d():

	zinc = Zinc()

	num_seed = 5
	steps = 150
	beta = lambda t: 0.5 / np.log(t / 4 + 2)

	xs = zinc.X[:,0]
	ys = zinc.Y[:,0]
	idx = np.argsort(xs)[:200]
	xs = xs[idx]
	ys = ys[idx]
	threshold = np.mean(ys)

	plt.scatter(xs, ys)
	plt.plot(xs, ys)
	plt.show()

	for trial in range(5):

		seed_set_idx = np.random.choice(
			# np.where(
			# 	np.logical_and(
			# 		threshold < ys,
			# 		ys < (np.max(ys) + threshold) / 2
			# 	)
			# )[0],
			np.arange(len(xs)),
			size=num_seed, replace=False
		)

		# if num_seed == 1:
		# 	seed_set_idx = np.array(seed_set_idx)

		opts = [
			# simplest bayesian optimization without any constraints
			# BayesianOpt(
			# 	xs[seed_set_idx][:, None],
			# 	ys[seed_set_idx][:, None],
			# 	xs[:, None],
			# 	[0.],
			# 	1.0,
			# 	kernel=None,
			# 	aq_func=DynamicUCB(),
			# ),

			SaferOpt(
				xs[seed_set_idx][:, None],
				np.stack([ys[seed_set_idx], ys[seed_set_idx]], axis=1),
				xs[:, None],
				[-np.inf, threshold],
				1.0,
				aq_func=DynamicUCB()
			)

		]

		for opt in opts:
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
				# plt.scatter(
				# 	opt.parameter_set[opt.S, 0],
				# 	np.ones([np.sum(opt.S)]) * opt.fmin,
				# 	c='g', label='Safe Set', s=15)
				plt.scatter(xs[idx], ys[idx], c='b', label='Proposed Evaluation')
				plt.show()

				# opt.add_new_data_point(
				# 	xs[idx], ys[idx]
				# )

				opt.add_new_data_point(
					np.atleast_2d(xs[idx]),
					np.array([ys[idx], ys[idx]])[None, :]
				)


if __name__ == '__main__':
	# experiment_one_trial()
	experiment_1d()