import numpy as np
from sklearn.neighbors import KernelDensity
import torch
from torch import nn

from utils.config import Config

class RobustNNRegression():
	def __init__(self,
	             nn_dim,
	             kde_bandwidth=0.1,
	             act=nn.ELU,
	             normalize=True):
		"""
		:param nn_dim:
		:param kde_bandwidth:
		:param act:
		:param normalize:
		"""
		self.nn_dim = nn_dim
		self.kde_bandwidth = kde_bandwidth
		self.normalize = normalize

		self.M11 = torch.ones([1,], requires_grad=False).to(Config.device)
		self.M12 = torch.ones([nn_dim[-1],], requires_grad=False).to(Config.device) * 0.1

		self.kde_src = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth)
		self.kde_trg = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth)

		self.x_mean = None
		self.x_std = None
		self.y_mean = None
		self.y_std = None

		self.mu0 = None
		self.sigma0 = None

		layers = []
		for i in range(len(nn_dim) - 2):
			layers.append(
				nn.Linear(nn_dim[i], nn_dim[i + 1])
			)
			layers.append(act())
		layers.append(
			nn.Linear(nn_dim[-2], nn_dim[-1])
		)
		self.nn = nn.Sequential(*layers).to(Config.device)

	def gaussian_params(self, nn_x, M11, M12, mu0, sigma0, Pratio):
		"""
		:param nn_x: torch.Tensor, shape [batch, dim]
		:param M11: torch.Tensor, shape [1,]
		:param M12:  torch.Tensor, shape [dim,]
		:param mu0: scalar or torch.Tensor, shape [1,]
		:param sigma0: scalar or torch.Tensor, shape [1,]
		:param Pratio: torch.Tensor, shape [batch,]
		"""
		sigmas = 1 / (1 / sigma0 + 2 * M11 * Pratio)
		mus = sigmas * (-2 * Pratio * (nn_x @ M12) + mu0 / sigma0)
		return mus, sigmas

	def fit(self,
	        Xsrc,
	        ysrc,
	        Xtrg,
	        verbose=False,
	        max_iteration=10000,
	        primal_lr=5e-4,
	        dual_lr=5e-4,
	        safe_threshold=None):
		"""
		Args:
			Xsrc (np.array):
				Samples from source domain
				Shape [n_src_samples, dim] if dim != 1
				Shape [n_src_samples,] if dim == 1
			ysrc (np.array):
				Labels of samples from source domain
				Shape [n_src_samples,]
			Xtrg (np.array):
				Samples from target domain
				Shape [n_trg_samples, dim] if dim != 1
				Shape [n_trg_samples,] if dim == 1

		Returns:
		"""

		if len(Xsrc.shape) == 1:
			Xsrc = Xsrc[:, None]
			Xtrg = Xtrg[:, None]

		# Normalization
		self.X_mean = np.mean(Xsrc, axis=0, keepdims=True)
		self.X_std = np.std(Xsrc, axis=0, keepdims=True)
		self.y_mean = np.mean(ysrc, axis=0, keepdims=False)
		self.y_std = np.std(ysrc, axis=0, keepdims=False)
		self.y_std = 1 if np.abs(self.y_std) < 1e-4 else self.y_std

		if self.normalize:
			Xsrc = (Xsrc - self.X_mean) / self.X_std
			Xtrg = (Xtrg - self.X_mean) / self.X_std
			ysrc = (ysrc - self.y_mean) / self.y_std

		self.mu0 = (np.max(ysrc) + np.min(ysrc)) / 2
		# self.sigma0 = (0.5 * (np.max(ysrc) - self.mu0)) ** 2 + 1e-6
		self.sigma0 = (4. * (np.max(ysrc) - np.min(ysrc))) + 1e-6
		# self.sigma0 = 1e-2

		if safe_threshold is not None:
			if self.normalize:
				self.mu0 = (safe_threshold - self.y_mean) / self.y_std
			else:
				self.mu0 = safe_threshold

		self.kde_src.fit(Xsrc)
		self.kde_trg.fit(Xtrg)

		# P_src(x) / P_trg(x)
		Pratio_src = np.exp(self.kde_src.score_samples(Xsrc) - self.kde_trg.score_samples(Xsrc))
		Pratio_src = torch.from_numpy(Pratio_src).float().to(Config.device)
		Xsrc = torch.from_numpy(Xsrc).float().to(Config.device)
		ysrc = torch.from_numpy(ysrc).float().to(Config.device)

		optimizer = torch.optim.Adam(self.nn.parameters(), lr=primal_lr, weight_decay=1e-3)
		for t in range(1, max_iteration):
			# Dual update
			self.nn.eval()
			nn_x = self.nn(Xsrc)
			mus, sigmas = self.gaussian_params(nn_x, self.M11, self.M12, self.mu0, self.sigma0, Pratio_src)
			dM11 = torch.mean(mus ** 2 + sigmas - ysrc ** 2)
			dM12 = 2 * torch.mean(torch.einsum('ij,i->ij', nn_x, mus - ysrc), dim=0)

			if verbose:
				print(t, dM11, dM12)

			self.M11.data += dual_lr * dM11
			self.M12.data += dual_lr * dM12

			self.M11.data = torch.clamp(self.M11.data, 0, 1e6)

			# Primal update

			self.nn.train()
			nn_x = self.nn(Xsrc)
			mus, sigmas = self.gaussian_params(nn_x, self.M11, self.M12, self.mu0, self.sigma0, Pratio_src)
			mus = mus.cpu().detach()

			loss = 2 * torch.mean(
				torch.einsum('ij,i->ij', nn_x, ysrc - mus.to(Config.device)) @ self.M12
			)  # This is where I got confused

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	def predict(self, Xtrg):
		"""
		Args:
			Xtrg (np.array):
				Shape [n_trg_samples, dim] if dim != 1
				Shape [n_trg_samples,] if dim == 1
		Returns:
		"""

		if len(Xtrg.shape) == 1:
			Xtrg = Xtrg[:, None]

		if self.normalize:
			Xtrg = (Xtrg - self.X_mean) / self.X_std

		Pratio = np.exp(self.kde_src.score_samples(Xtrg) - self.kde_trg.score_samples(Xtrg))
		Pratio = torch.from_numpy(Pratio).float().to(Config.device)
		Xtrg_th = torch.from_numpy(Xtrg).float().to(Config.device)
		nn_x = self.nn(Xtrg_th).to(Config.device)

		mus, sigmas = self.gaussian_params(nn_x, self.M11, self.M12, self.mu0, self.sigma0, Pratio)
		mus = mus.cpu().detach().numpy()
		sigmas = sigmas.cpu().detach().numpy()

		if self.normalize:
			return mus * self.y_std + self.y_mean, sigmas * self.y_std ** 2
		else:
			return mus, sigmas