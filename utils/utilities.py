import sys
sys.path.append('..')

from collections.abc import Sequence            # isinstance(...,Sequence)
import numpy as np
import torch
from utils.config import Config




def linearly_spaced_combinations(bounds, num_samples):
	"""
	Return 2-D array with all linearly spaced combinations with the bounds.

	Parameters
	----------
	bounds: sequence of tuples
		The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
	num_samples: integer or array_likem
		Number of samples to use for every dimension. Can be a constant if
		the same number should be used for all, or an array to fine-tune
		precision. Total number of data points is num_samples ** len(bounds).

	Returns
	-------
	combinations: 2-d array
		A 2-d arrray. If d = len(bounds) and l = prod(num_samples) then it
		is of size l x d, that is, every row contains one combination of
		inputs.
	"""
	num_vars = len(bounds)

	if not isinstance(num_samples, Sequence):
		num_samples = [num_samples] * num_vars

	if len(bounds) == 1:
		return np.linspace(bounds[0][0], bounds[0][1], num_samples[0])[:, None]

	# Create linearly spaced test inputs
	inputs = [np.linspace(b[0], b[1], n) for b, n in zip(bounds,
														 num_samples)]

	# Convert to 2-D array
	return np.array([x.ravel() for x in np.meshgrid(*inputs)]).T




def array_to_tensor(arr):
	return torch.from_numpy(arr).to(Config.device)

def tensor_to_array(t):
	return t.cpu().numpy()

if __name__ == '__main__':
	xranges = [(-5, 5)] * 1
	xs = linearly_spaced_combinations(xranges, 10)
	print(xs.shape)