import numpy as np
import torch

def get_device():
	if torch.cuda.is_available():
		return torch.device('cuda')
	if torch.backends.mps.is_available() and torch.backends.mps.is_built():
		return torch.device('mps')
	return torch.device('cpu')

class Config:
	# device = get_device()
	device = 'cpu'
	nn_dims = [16, 32, 64, 32, 8]


	# Dynamic UCB scheduling
	sig_start = -2.5 # -2.6
	sig_step = 0.05 # 1 / 20

	# eps scheduling
	eps = lambda t : 1 / np.sqrt(t + 1)

	# gp
	gp_noise_var = 0.001

	# nn random function range
	xrange = [-5, 5, 500]
	yrange = [-2, 2]
	nn_random_function_samples = 30

