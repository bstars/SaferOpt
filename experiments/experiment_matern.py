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

from utils.bb_function import GPRandomFunction
from utils.config import Config
from experiments.run_opt_1d import run_experiment_1d


opts = [
	'GP_Mat32+ConstrainedUCB',
	'GP_rbf+UCB',
	'RR+ConstrainedDUCB',
	'SafeOpt',
	'StageOpt'
]

def experiment_one_gp_function(gp_function = 0):
	num_trials = 5
	threshold = 0.
	num_seed = 2
	steps = 150

	# nn_function = 0
	# beta = lambda t: 1 / np.log(t / 10 + 2)
	beta = 2.

	xs = np.linspace(Config.xrange[0], Config.xrange[1], Config.xrange[2])
	truef = GPRandomFunction(xs[:,None], GPy.kern.Matern32(1))
	ys = truef.f

	L = np.max(np.abs(
		(ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
	))


	for i in range(num_trials):
		seed_set_idx = np.random.choice(
			np.where(
				np.logical_and(
					threshold < ys,
					ys < (np.max(ys) + threshold) / 2
				)
			)[0],
			size=num_seed, replace=False
		)

		opts = [
			# simplest bayesian optimization without any constraints
			BayesianOpt(
				xs[seed_set_idx][:, None],
				ys[seed_set_idx][:, None],
				xs[:, None],
				[-np.inf],
				beta,
				kernel=None,
				aq_func=UCB()
			),

			# bayesian optimization with matern kernel
			BayesianOpt(
				xs[seed_set_idx][:, None],
				ys[seed_set_idx][:, None],
				xs[:, None],
				[threshold],
				beta,
				kernel=GPy.kern.Matern32(1),
				aq_func=UCB()
			),

			# SafeOpt
			SafeOpt(
				lipschitz=[L],
				xs=xs[seed_set_idx][:, None],
				ys=ys[seed_set_idx][:, None],
				parameter_set=xs[:, None],
				fmin=[threshold],
				beta=beta,
				kernel=None
			),

			# StageOpt
			StageOpt(
				lipschitz=[L],
				xs=xs[seed_set_idx][:, None],
				ys=ys[seed_set_idx][:, None],
				parameter_set=xs[:, None],
				fmin=[threshold],
				beta=beta,
				kernel=None,
				switch_time=20
			),

			# SaferOpt
			SaferOpt(
				xs=xs[seed_set_idx][:, None],
				ys=np.stack([ ys[seed_set_idx], ys[seed_set_idx] ], axis=1),
				parameter_set=xs[:, None],
				fmin=[-np.inf, threshold],
				beta=beta,
				aq_func=DynamicUCB(),
				kde_bandwidth=0.2
			)
		]

		for opt in opts:
			run_experiment_1d(
				xs,
				ys,
				opt,
				steps,
				save_path='./results_single/gp_matern/%d/%s/trial_%d' % (gp_function, opt.name, i),
			)

def statistics():
	dict = {opt:[] for opt in opts}

	for func_id in range(5):
		path = './results_single/gp_matern/%d' % func_id
		for opt in opts:
			for trial in range(5):
				result = loadmat(os.path.join(path, opt, 'trial_%d' % trial, 'result.mat'))
				regret = result['true_best'].squeeze() - result['proposed'].squeeze()
				num_vio = result['num_violation'].squeeze()
				dict[opt].append(
					(regret, num_vio)
				)

	for opt in opts:
		result = dict[opt]
		regret = np.array([r for r, _ in result])
		num_vio = np.array([n for _, n in result])
		print(regret.shape)
		plt.plot(
			np.cumsum(
				np.mean(regret, axis=0)
			),
			label=opt + ':%.2f' % np.mean(num_vio)
		)

	plt.legend()
	plt.savefig('./results_single/gp_matern/avgall.png')
	plt.close()

	for opt in opts:
		result = dict[opt]
		regret = np.array([r for r, _ in result])
		num_vio = np.array([n for _, n in result])

		r = np.mean(regret[:,-1])
		v = np.mean(num_vio)

		plt.scatter(r, v, label=opt)
	plt.legend()
	plt.savefig('./results_single/gp_matern/avgall_2d.png')
	plt.close()

if __name__ == '__main__':
	# for i in range(5):
	# 	experiment_one_gp_function(i)
	statistics()
