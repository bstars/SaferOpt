import sys
sys.path.append('..')

import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np


from scipy.stats import norm
from torch import nn
import torch
import os
from scipy.io import savemat


from algorithms.safe_opt import SafeOpt
from algorithms.stage_opt import StageOpt
from algorithms.bayesian_opt import DynamicUCB, ExpectedImprovement, ProbOfImprovement, UCB
from algorithms.safer_opt import SaferOpt

def run_experiment_1d(xs,
                   ys,
                   opt:SafeOpt or SaferOpt or StageOpt,
                   steps=100,
                   plot=True,
                   save_path='',
                   threshold=0., save=True):
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
		mat = {}
		idx, x = opt.propose_evaluation()
		# print(opt.name, i, true_best, fbest, idx, ys.shape)

		print('%s, iteration %d, true best: %.3f, best achieved: %.3f, proposed: %.3f' % (opt.name, i, true_best, fbest, ys[idx]))

		if save:
			# save for re-plotting
			mat['xs'] = xs
			mat['ys'] = ys
			mat['true_best'] = true_best
			mat['Q'] = opt.Q
			mat['S'] = opt.S
			if isinstance(opt, SafeOpt) or isinstance(opt, StageOpt):
				mat['M'] = opt.M
				mat['G'] = opt.G

			mat['propose_idx'] = idx
			mat['proposed_x'] = x
			savemat(os.path.join(save_path, 'iteration_%d.mat' % i), mat)


		if i % 1 == 0 and plot:
			plt.scatter(opt.xs[:, 0], opt.ys[:, 0], c='r', label='Observations')
			plt.plot(xs, ys, 'k-', label='True Function')
			plt.plot(xs, np.ones([len(xs)]) * threshold, label='Threshold')

			if isinstance(opt, SaferOpt):
				plt.fill_between(
					opt.parameter_set[:, 0],
					opt.Q[:, 0],
					opt.Q[:, 1],
					alpha=0.5,
					label='Reward Confidence Interval',
				)
				plt.fill_between(
					opt.parameter_set[:, 0],
					opt.Q[:, 2],
					opt.Q[:, 3],
					alpha=0.5,
					label='Constraint Confidence Interval',
				)
			else:
				plt.fill_between(
					opt.parameter_set[:, 0],
					opt.Q[:, 0],
					opt.Q[:, 1],
					alpha=0.5,
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
			plt.legend(loc='best', prop={'size': 6})

			plt.title('iteration %d' % i)
			plt.ylim([np.min(ys)-1, np.max(ys)+1])
			plt.savefig(os.path.join(save_path, 'iteration_%d.png' % i), bbox_inches='tight')
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
