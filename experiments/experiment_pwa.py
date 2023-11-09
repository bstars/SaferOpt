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

from utils.bb_function import NNRandomFunction
from utils.config import Config
from experiments.run_opt import run_experiment


opts = [
	'GP_rbf+UCB',
	'SafeOpt',
	'StageOpt',
	'RR+ConstrainedDUCB'
]

def experiment_one_pwa_function(pwa_function = 0):
	num_trials = 5
	threshold = 0.
	num_seed = 15
	steps = 150
	# nn_function = 0
	# beta = lambda t: 1 / np.log(t / 10 + 2)
	beta = 2.

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
			run_experiment(
				xs,
				ys,
				opt,
				steps,
				save_path='./results/nn_pwa/%d/%s/trial_%d' % (pwa_function, opt.name, i),
			)

def statistics():
	dict = {opt:[] for opt in opts}

	for func_id in range(1):
		path = './results/nn_pwa/%d' % func_id
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
	plt.savefig('./results/nn_pwa/avgall.png')
	plt.close()

	for opt in opts:
		result = dict[opt]
		regret = np.array([r for r, _ in result])
		num_vio = np.array([n for _, n in result])

		r = np.mean(regret[:,-1])
		v = np.mean(num_vio)

		plt.scatter(r, v, label=opt)
	plt.legend()
	plt.savefig('./results/nn_pwa/avgall_2d.png')
	plt.close()


if __name__ == '__main__':
	for i in range(1):
		experiment_one_pwa_function(i)
	# statistics()
