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
from scipy.io import savemat


from algorithms.safe_opt import SafeOpt
from algorithms.stage_opt import StageOpt
from algorithms.bayesian_opt import DynamicUCB, ExpectedImprovement, ProbOfImprovement, UCB
from algorithms.safer_opt import SaferOpt

def run_experiment(xs,
                   ys,
                   opt:SafeOpt or SaferOpt or StageOpt,
                   steps=100,
                   plot=True,
                   save_path='',
                   threshold=0.):
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	true_best = ys.max()
	fbest = opt.ys[:,0].max()
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

		print('%s, iteration %d, true best: %.3f, best achieved: %.3f, proposed: %.3f' % (opt.name, i, true_best, fbest, ys[idx]))

		if i % 10 == 0 and plot:
			plt.scatter(opt.xs[:, 0], opt.ys[:, 0], c='r', label='Observations')
			plt.plot(xs, ys, 'k-', label='True Function')
			plt.plot(xs, np.ones([len(xs)]) * threshold, label='Threshold')

			if isinstance(opt, SaferOpt):
				plt.fill_between(
					opt.parameter_set[:, 0],
					opt.Q[:, 0],
					opt.Q[:, 1],
					alpha=0.1,
					label='Reward Confidence Interval',
				)
				plt.fill_between(
					opt.parameter_set[:, 0],
					opt.Q[:, 2],
					opt.Q[:, 3],
					alpha=0.1,
					label='Constraint Confidence Interval',
				)
			else:
				plt.fill_between(
					opt.parameter_set[:, 0],
					opt.Q[:, 0],
					opt.Q[:, 1],
					alpha=0.15,
					label='Confidence Interval',
				)

			plt.scatter(
				opt.parameter_set[opt.S, 0],
				np.ones([np.sum(opt.S)]) * threshold,
				c='g', label='Safe Set', s=15)

			if isinstance(opt, SafeOpt) or isinstance(opt, StageOpt):
				plt.scatter(opt.parameter_set[opt.M, 0], np.ones([np.sum(opt.M)]) * threshold, c='b', label='Maximizers', s=5)
				plt.scatter(opt.parameter_set[opt.G, 0], np.ones([np.sum(opt.G)]) * threshold, c='r', label='Expanders', s=5)


			plt.scatter(xs[idx], ys[idx], c='y', label='Proposed Evaluation')
			plt.legend()
			plt.title('iteration %d' % i)
			plt.savefig(os.path.join(save_path, 'iteration_%d.png' % i))
			plt.close()
			# plt.show()

		if ys[idx] > fbest:
			fbest = ys[idx]
		if isinstance(opt, SaferOpt):
			opt.add_new_data_point(
				np.atleast_2d(xs[idx]),
				np.array([ys[idx], ys[idx]])[None, :]
			)
		else:
			opt.add_new_data_point(
				np.atleast_2d(xs[idx]),
				np.atleast_2d(ys[idx])
			)

		# keep the record
		proposed.append(ys[idx])
		best_achieved.append(fbest)
		if ys[idx] < threshold:
			num_violation += 1
			if not first_violation:
				first_violation = True
				ratio_true_safe = np.sum(np.logical_and(true_S, opt.S)) / np.sum(true_S)
				ratio_false_safe = np.sum(np.logical_and(~true_S, opt.S)) / np.sum(~true_S)

	if not first_violation:
		first_violation = True
		ratio_true_safe = np.sum(np.logical_and(true_S, opt.S)) / np.sum(true_S)
		ratio_false_safe = np.sum(np.logical_and(~true_S, opt.S)) / np.sum(opt.S)

	savemat(
		os.path.join(save_path, 'result.mat'),
		{
			'threshold' : threshold,
			'true_best': true_best,
			'proposed': np.array(proposed),
			'best_achieved': np.array(best_achieved),
			'num_violation': num_violation,
			'ratio_true_safe': ratio_true_safe,
			'ratio_false_safe': ratio_false_safe
		}
	)



if __name__ == '__main__':
	def truef(xs):
		return 2 * (np.sin(xs) * np.cos(xs + 0.7) + np.log(np.abs(xs) + 3) - 1.2)


	num = 500
	threshold = -0.1
	num_seed = 5
	steps = 50
	xs = np.linspace(-5, 5, num)
	ys = truef(xs)
	seed_set_idx = np.random.choice(
		np.where(np.logical_and(threshold < ys, ys < threshold + 4))[0],
		size=num_seed, replace=False
	)

	opt = SaferOpt(
		xs[seed_set_idx][:, None],
		np.stack([ys[seed_set_idx], ys[seed_set_idx]], axis=1),
		xs[:, None],
		[-np.inf, threshold],
		1.0,
		aq_func=UCB()
	)

	run_experiment(xs, ys, opt, steps, save_path='safer_opt', threshold=threshold)

