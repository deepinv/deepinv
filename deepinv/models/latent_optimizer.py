from __future__ import annotations
import abc
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.optim as optim
from scipy.interpolate import griddata
from collections import defaultdict
import warnings
from tqdm import tqdm

from deepinv.models import Reconstructor, VAE, DCGANGenerator, MbtCAE
from deepinv.models.cae import LowerBound
from deepinv.physics import Physics, LinearPhysics, Inpainting
from deepinv.optim import DataFidelity
from deepinv.loss import cal_psnr


class LatentOptimizer(Reconstructor):
    r"""
    Base class for latent optimization algorithms. Those algorithms solve a given inverse problem
    in the latent space of a learned unconditional generative model.

    .. note::

        Two types of algorithms are implemented:

        - Deterministic optimization algorithms (CSGM and MAP-z, see `"Compressed Sensing using Generative Models" <https://arxiv.org/abs/1703.03208>`_),
          which consist in optimizing :math:`\min_z F(D_\theta(z),y) + \lambda R_\theta(z)`
          where :math:`F` is the data-fidelity term, :math:`R_\theta` is a regularization term in the latent space,
          and $\theta$ are the parameters of the generative model.

        - Variational inference algorithms using VBLE-z and VBLE-xz formulation (see
          `"Variational Bayes Image Restoration with Compressive Autoencoders" <https://arxiv.org/abs/2311.17744>`_ and
          `"Deep priors for satellite image restoration with accurate uncertainties" <https://arxiv.org/abs/2412.04130>`_).
          These algorithms consist in optimizing the parameters of an approximate posterior distribution :math:`q(z) or :math:`q(x|z)q(z)` by maximizing the
          Evidence Lower Bound (ELBO): :math:`\max_{q(x,z)} \mathbb{E}_{q(x,z)}[\log p(y|x)] - \lambda D_{KL}(q(x,z) || p_\theta(x,z))`.

    |sep|

    :param nn.Module prior: Pretrained generative model to use as prior. Currently supported models are
        `deepinv.models.VAE`, `deepinv.models.DCGANGenerator` and `deepinv.models.MbtCAE`.
    :param float lamb: Weight of the latent regularization term.
    :param int max_iters: Maximum number of optimization iterations.
    :param float lr: Learning rate for the optimizer.
    :param str optimizer_name: Name of the optimizer to use ('adam' or 'sgd').
    :param int n_samples_sgvb: Number of samples to use for SGVB estimation of the ELBO (only for variational inference algorithms).
    :param int rate_scale: Scaling factor to modulate the regularization strength when using compressive autoencoder priors. Between (0, 1],
        a value < 1 increases the regularization strength.
    :param float clip_grad_norm: Maximum norm for gradient clipping. If None, no clipping is applied.
    """

    def __init__(
        self,
        prior: VAE | DCGANGenerator | MbtCAE,
        lamb: float = 1.0,
        max_iters: int = 200,
        lr: float = 0.1,
        optimizer_name: str = "adam",
        n_samples_sgvb: int = 1,
        rate_scale: int = 1,
        clip_grad_norm: float | None = None,
    ):
        super().__init__()

        self.prior = prior
        self.prior.eval()
        self.lamb = lamb
        self.max_iters = max_iters
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.n_samples_sgvb = n_samples_sgvb
        self.optimize_h = isinstance(self.prior, MbtCAE)
        self.gaussian_sampling = isinstance(self.prior, VAE)
        if rate_scale <= 0 or rate_scale > 1:
            rate_scale = 1
            warnings.warn(
                "rate_scale should be in (0, 1]. Setting rate_scale=1 (no scaling)."
            )
        if rate_scale < 1 and not isinstance(self.prior, MbtCAE):
            rate_scale = 1
            warnings.warn(
                "rate_scale < 1 is only supported for compressive autoencoder priors. Setting rate_scale=1 (no scaling)."
            )
        self.rate_scale = torch.Tensor([rate_scale]).view(1, 1, 1, 1)
        self.clip_grad_norm = clip_grad_norm

        if isinstance(self.prior, VAE) or isinstance(self.prior, MbtCAE):
            self.decode_fn = self.prior.decode
        elif isinstance(self.prior, DCGANGenerator):

            def decode_fn(latent_dict, decode_mean_only=True):
                z = latent_dict["z"]
                x_rec = self.prior(z)
                return {"x_rec": x_rec}

            self.decode_fn = decode_fn
        else:
            raise NotImplementedError("Prior model not supported in LatentOptimizer.")

    def forward(
        self,
        y: Tensor,
        physics: Physics,
        datafidelity: DataFidelity,
        x_target: torch.Tensor | None = None,
        compute_metrics: bool = False,
        verbose: bool = True,
        freq_metrics: int = 10,
    ) -> torch.Tensor:
        r"""
        Solves the inverse problem using gradient descent on the optimized variables.

        :param torch.Tensor y: Measurement (batch restoration not yet supported).
        :param deepinv.physics.Physics physics: Forward model.
        :param deepinv.optim.DataFidelity datafidelity: Data-fidelity term.
        :param torch.Tensor x_target: Ground truth image for metric computation.
            If None, the metrics are not computed.
        :param bool compute_metrics: Whether to compute metrics during optimization.
        :param bool verbose: Whether to print optimization progress.
        :param int freq_metrics: Frequency (in iterations) to compute metrics.

        :return: Reconstructed image (the deterministic estimate for MAP-z and CSGM, one
            sample from the approximate posterior for VBLE-z and VBLE-xz).
        """

        assert y.shape[0] == 1, "Batch processing not yet supported in LatentOptimizer."

        device = y.device

        if isinstance(physics, Inpainting):
            # interpolation
            x_init = interpolate_init_inpainting(y.detach().cpu().numpy())
            x_init = torch.tensor(x_init).to(device)
        elif isinstance(physics, LinearPhysics):
            x_init = physics.A_adjoint(y)
        else:
            x_init = physics.A_dagger(y)
        self.init_optimized_parameters(x_init)
        n_pixels = np.prod(x_init.shape[-2:])

        # Initialize loss and metrics dictionaries
        self.init_dicos()

        iter = 0

        optimizer = self.get_optimizer(self.optimized_params)

        self.lb_x = LowerBound(0.0).to(self.device)
        self.up_x = LowerBound(-1.0).to(self.device)

        for iter in tqdm(range(1, self.max_iters + 1), disable=not verbose):
            optimizer.zero_grad()
            generated_dict = self.generate_solution(self.n_samples_sgvb)

            datafid_loss = datafidelity(generated_dict["x_rec"], y, physics) / (
                n_pixels * self.n_samples_sgvb
            )
            image_loss = self.image_loss()
            latent_loss = self.latent_loss(generated_dict)
            total_loss = datafid_loss + self.lamb * (image_loss + latent_loss)
            total_loss.backward()

            if self.clip_grad_norm:
                _ = torch.nn.utils.clip_grad_norm_(
                    self.optimized_params.parameters(), self.clip_grad_norm
                )
            optimizer.step()

            # save running losses
            self.update_loss_dict(
                total_loss.item(),
                datafid_loss.item(),
                latent_loss.item(),
                image_loss.item(),
            )

            if iter % freq_metrics == 0 and compute_metrics and x_target is not None:
                self.update_metrics_dict(x_target, generated_dict["x_rec"])

        return self.generate_solution()["x_rec"]

    def generate_solution(self, n_samples: int = 1):
        r"""
        Generate a solution from the optimized parameters.
        """
        return self.optimized_params.generate_solution(
            self.decode_fn, n_samples=n_samples, rate_scale=self.rate_scale
        )

    def init_optimized_parameters(self, x_init: torch.Tensor):
        r"""
        Initialize the optimized parameters (``optimized_params`` attribute) during latent optimization.

        :param torch.Tensor x_init: Initial estimate of the reconstructed image.
        """
        dict_inf_params_init = {}

        if isinstance(self.prior, DCGANGenerator):
            z_size = [(1, 100, 1, 1)]
            dict_inf_params_init["zbar"] = (
                torch.rand(*z_size[0]).to(x_init.device) * 2 - 1
            )
        else:
            out_x_init = self.prior(x_init * self.rate_scale)

            dict_inf_params_init["zbar"] = out_x_init["z"]
            if self.optimize_h:
                dict_inf_params_init["hbar"] = out_x_init["h"]
            z_size = (
                [dict_inf_params_init["zbar"].shape, dict_inf_params_init["hbar"].shape]
                if self.optimize_h
                else [dict_inf_params_init["zbar"].shape]
            )
        self.optimized_params = self.optimized_param_class(
            z_size, xdim=x_init.shape, gaussian_sampling=self.gaussian_sampling
        ).to(self.device)
        self.optimized_params.init_from_paramvalues(dict_inf_params_init)

    def to(self, device):
        """Move the model to the specified device."""
        super(LatentOptimizer, self).to(device)
        self.device = device
        if hasattr(self, "prior"):
            self.prior = self.prior.to(device)
        if hasattr(self, "optimized_params"):
            self.optimized_params = self.optimized_params.to(device)
        if hasattr(self, "rate_scale"):
            self.rate_scale = self.rate_scale.to(device)
        return self

    def init_dicos(self):
        r"""
        Initialize the loss and metrics dictionaries.
        """
        self.dico_loss = defaultdict(list)
        self.dico_metrics = defaultdict(list)

    def update_metrics_dict(self, x_target: torch.Tensor, x_rec: torch.Tensor):
        r"""
        Update the metrics dictionary.

        :param torch.Tensor x_target: Ground truth image.
        :param torch.Tensor x_rec: Reconstructed image.
        """
        with torch.no_grad():
            psnr = cal_psnr(x_rec, x_target).item()
        self.dico_metrics["psnr"].append(psnr)

    def get_loss(self) -> dict:
        r"""
        Returns the loss dictionary.
        """
        return {k: np.array(v).reshape((1, -1)) for k, v in self.dico_loss.items()}

    def get_metrics(self) -> dict:
        r"""
        Returns the metrics dictionary.
        """
        return {k: np.array(v).reshape((1, -1)) for k, v in self.dico_metrics.items()}

    def get_optimizer(self, inference_params: OptimizedParameters) -> optim.Optimizer:
        r"""
        Returns the optimizer for the latent optimization.

        :param OptimizedParameters inference_params: Parameters to optimize.
        """
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(
                [param for param in inference_params.parameters()], lr=self.lr
            )

        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(
                [param for param in inference_params.parameters()],
                lr=self.lr,
                momentum=0.9,
                nesterov=True,
            )

        else:
            raise NotImplementedError

        return optimizer


class CSGMOptimizer(LatentOptimizer):
    r"""
    Optimizer class for Compressed Sensing using Generative Models (CSGM) algorithm (see `"seminal work" <https://arxiv.org/abs/1703.03208>`_).
    This class implements the CSGM algorithm which consists in optimizing :math:`\min_z - F(D_\theta(z),y)` with :math:`F` a data-fidelity term.

    :param nn.Module prior: Pretrained generative model to use as prior. Currently supported models are
        `deepinv.models.VAE`, `deepinv.models.DCGANGenerator` and `deepinv.models.MbtCAE`.
    :param int max_iters: Maximum number of optimization iterations.
    :param float lr: Learning rate for the optimizer.
    :param str optimizer_name: Name of the optimizer to use ('adam' or 'sgd').
    :param int rate_scale: Scaling factor to modulate the regularization strength when using compressive autoencoder priors. Between (0, 1],
        a value < 1 increases the regularization strength.
    :param float | None clip_grad_norm: Maximum norm for gradient clipping. If None, no clipping is applied.
    """

    def __init__(
        self,
        prior: VAE | DCGANGenerator | MbtCAE,
        max_iters: int = 200,
        lr: float = 0.1,
        optimizer_name: str = "adam",
        rate_scale: int = 1,
        clip_grad_norm: float | None = None,
    ):
        super().__init__(
            prior=prior,
            lamb=0.0,
            max_iters=max_iters,
            lr=lr,
            optimizer_name=optimizer_name,
            n_samples_sgvb=1,
            rate_scale=rate_scale,
            clip_grad_norm=clip_grad_norm,
        )

        self.optimized_param_class = MAPzParameters

    def latent_loss(self, gen_dict):
        return torch.tensor(0.0).to(gen_dict["x_rec"].device)

    def image_loss(self):
        return torch.tensor(0.0).to(next(self.optimized_params.parameters()).device)

    def update_loss_dict(
        self,
        total_loss: float,
        datafid_loss: float = 0.0,
        latent_loss: float = 0.0,
        image_loss: float = 0.0,
    ):
        self.dico_loss["total_loss"].append(total_loss)


class MAPzOptimizer(LatentOptimizer):
    r"""
    Optimizer class for Maximum A Posteriori estimation in the latent space (MAP-z)
    algorithm (see `"seminal work" <https://arxiv.org/abs/1703.03208>`_).

    This class implements the MAP-z algorithm which consists in optimizing :math:`\min_z - \log p(z|y) - \lambda \log p_\theta(z)`
    where :math:`p_\theta(z)` is the prior distribution in the latent space of the generative model.

    :param nn.Module prior: Pretrained generative model to use as prior. Currently supported models are
        `deepinv.models.VAE`, `deepinv.models.DCGANGenerator` and `deepinv.models.MbtCAE`.
    :param float lamb: Weight of the latent regularization term. `lamb=1` corresponds to the Bayesian MAP estimate.
    :param int max_iters: Maximum number of optimization iterations.
    :param float lr: Learning rate for the optimizer.
    :param str optimizer_name: Name of the optimizer to use ('adam' or 'sgd').
    :param int rate_scale: Scaling factor to modulate the regularization strength when using compressive autoencoder priors. Between (0, 1],
        a value < 1 increases the regularization strength.
    :param float | None clip_grad_norm: Maximum norm for gradient clipping. If None, no clipping is applied.
    """

    def __init__(
        self,
        prior: VAE | DCGANGenerator | MbtCAE,
        lamb: float = 1.0,
        max_iters: int = 200,
        lr: float = 0.1,
        optimizer_name: str = "adam",
        rate_scale: int = 1,
        clip_grad_norm: float | None = None,
    ):
        super().__init__(
            prior=prior,
            lamb=lamb,
            max_iters=max_iters,
            lr=lr,
            optimizer_name=optimizer_name,
            n_samples_sgvb=1,
            rate_scale=rate_scale,
            clip_grad_norm=clip_grad_norm,
        )

        self.optimized_param_class = MAPzParameters

    def latent_loss(self, gen_dict: dict):
        _, _, H, W = gen_dict["x_rec"].size()
        num_pixels = H * W
        if isinstance(self.prior, MbtCAE):
            latent_loss = self.prior.compute_bpp_loss(gen_dict)
        elif isinstance(self.prior, VAE):
            latent_loss = 0.5 * (torch.sum(gen_dict["z"].pow(2)) / num_pixels)
        elif isinstance(self.prior, DCGANGenerator):
            latent_loss = torch.tensor(0.0).to(gen_dict["x_rec"].device)
        else:
            raise NotImplementedError("Prior model not supported in MAPzOptimizer.")
        return latent_loss

    def image_loss(self):
        return torch.tensor(0.0).to(next(self.optimized_params.parameters()).device)

    def update_loss_dict(
        self,
        total_loss: float,
        datafid_loss: float = 0.0,
        latent_loss: float = 0.0,
        image_loss: float = 0.0,
    ):
        self.dico_loss["total_loss"].append(total_loss)
        self.dico_loss["datafid_loss"].append(datafid_loss)
        self.dico_loss["latent_reg_loss"].append(latent_loss)


class VBLEzOptimizer(LatentOptimizer):
    r"""
    Optimizer class for VBLE-z algorithm (see `"Variational Bayes Image Restoration with Compressive Autoencoders" <https://arxiv.org/abs/2311.17744>`_).
    This class implements the VBLE-z algorithm which consists in optimizing the parameters of an approximate posterior distribution :math:`q(z)`
    by maximizing the Evidence Lower Bound (ELBO): :math:`\max_{q(z)} \mathbb{E}_{q(z)}[\log p(y|x)] - \lambda D_{KL}(q(z) || p_\theta(z))`.
    :math: `q(z)` is either Gaussian (for VAE priors) or uniform (for CAE and GAN priors).

    :param nn.Module prior: Pretrained generative model to use as prior. Currently supported models are
        `deepinv.models.VAE`, `deepinv.models.DCGANGenerator` and `deepinv.models.MbtCAE`.
    :param float lamb: Weight of the latent regularization term.
    :param int max_iters: Maximum number of optimization iterations.
    :param float lr: Learning rate for the optimizer.
    :param str optimizer_name: Name of the optimizer to use ('adam' or 'sgd').
    :param int n_samples_sgvb: Number of samples to use for SGVB estimation of the ELBO.
    :param int rate_scale: Scaling factor to modulate the regularization strength when using compressive autoencoder priors. Between (0, 1],
        a value < 1 increases the regularization strength.
    :param float | None clip_grad_norm: Maximum norm for gradient clipping. If None, no clipping is applied.
    """

    def __init__(
        self,
        prior: VAE | DCGANGenerator | MbtCAE,
        lamb: float = 1.0,
        max_iters: int = 200,
        lr: float = 0.1,
        optimizer_name: str = "adam",
        n_samples_sgvb: int = 1,
        rate_scale: int = 1,
        clip_grad_norm: float | None = None,
    ):
        super().__init__(
            prior=prior,
            lamb=lamb,
            max_iters=max_iters,
            lr=lr,
            optimizer_name=optimizer_name,
            n_samples_sgvb=n_samples_sgvb,
            rate_scale=rate_scale,
            clip_grad_norm=clip_grad_norm,
        )

        self.optimized_param_class = VBLEzParameters

    def latent_loss(self, gen_dict: dict) -> torch.Tensor:
        r"""
        Computes the KL divergence loss in the latent space :math:`D_{KL}(q(z) || p_\theta(z))`.

        :param dict gen_dict: Dictionary containing the generated samples and relevant information.

        :returns: (torch.Tensor) Computed KL divergence loss.
        """
        return self.prior.compute_kl_loss(gen_dict)

    def image_loss(self):
        r"""Unused image loss for VBLE-z algorithm."""
        return torch.tensor(0.0).to(next(self.optimized_params.parameters()).device)

    def update_loss_dict(
        self,
        total_loss: float,
        datafid_loss: float,
        latent_loss: float,
        image_loss: float = 0.0,
    ):
        self.dico_loss["total_loss"].append(total_loss)
        self.dico_loss["datafid_loss"].append(datafid_loss)
        self.dico_loss["latent_reg_loss"].append(latent_loss)


class VBLExzOptimizer(LatentOptimizer):
    r"""
    Optimizer class for VBLE-xz algorithm (see `"Deep priors for satellite image restoration with accurate uncertainties" <https://arxiv.org/abs/2412.04130>`_).
    This class implements the VBLE-xz algorithm which consists in optimizing the parameters of an approximate posterior distribution :math:`q(x|z)q(z)`
    by maximizing the Evidence Lower Bound (ELBO): :math:`\max_{q(x,z)} \mathbb{E}_{q(x,z)}[\log p(y|x)] - \lambda D_{KL}(q(x,z) || p_\theta(x,z))`.
    :math: `q(z)` is either Gaussian (for VAE priors) or uniform (for CAE and GAN priors), and :math:`q(x|z)` is a
    Gaussian distribution with mean :math:`D_\theta(z)` and diagonal covariance :math:`\mathrm{diag}(b * \sigma^2_\theta(z))` with
    :math:`\sigma_\theta(z)` the decoder deviation predicted by a second decoder.

    :param nn.Module prior: Pretrained generative model to use as prior. Currently supported models are
        `deepinv.models.VAE` and `deepinv.models.MbtCAE`.
    :param float lamb: Weight of the latent regularization term.
    :param int max_iters: Maximum number of optimization iterations.
    :param float lr: Learning rate for the optimizer.
    :param str optimizer_name: Name of the optimizer to use ('adam' or 'sgd').
    :param int n_samples_sgvb: Number of samples to use for SGVB estimation of the ELBO.
    :param int rate_scale: Scaling factor to modulate the regularization strength when using compressive autoencoder priors. Between (0, 1],
        a value < 1 increases the regularization strength.
    :param float | None clip_grad_norm: Maximum norm for gradient clipping. If None, no clipping is applied.
    """

    def __init__(
        self,
        prior: VAE | MbtCAE,
        lamb: float = 1.0,
        max_iters: int = 200,
        lr: float = 0.1,
        optimizer_name: str = "adam",
        n_samples_sgvb: int = 1,
        rate_scale: int = 1,
        clip_grad_norm: float | None = None,
    ):
        super().__init__(
            prior=prior,
            lamb=lamb,
            max_iters=max_iters,
            lr=lr,
            optimizer_name=optimizer_name,
            n_samples_sgvb=n_samples_sgvb,
            rate_scale=rate_scale,
            clip_grad_norm=clip_grad_norm,
        )

        assert isinstance(self.prior, VAE) or isinstance(
            self.prior, MbtCAE
        ), "VBLE-xz algorithm only supported for VAE and MbtCAE priors."

        self.optimized_param_class = VBLExzParameters

    def latent_loss(self, gen_dict: dict) -> torch.Tensor:
        r"""
        Computes the KL divergence loss in the latent space :math:`D_{KL}(q(x,z) || p_\theta(x,z))`.

        :param dict gen_dict: Dictionary containing the generated samples and relevant information.

        :returns: (torch.Tensor) Computed KL divergence loss.
        """
        return self.prior.compute_kl_loss(gen_dict)

    def image_loss(self) -> torch.Tensor:
        r"""
        Computes the KL divergence loss in the image space :math:`D_{KL}(q(x|z) || p_\theta(x|z))`.

        :returns: (torch.Tensor) Computed KL divergence loss.
        """
        b = self.optimized_params.get_b()
        _, _, H, W = b.size()
        num_pixels = H * W

        kl_x = 0.5 * (torch.sum(b.pow(2) - 2 * torch.log(b)) - num_pixels) / num_pixels
        return kl_x

    def update_loss_dict(
        self,
        total_loss: float,
        datafid_loss: float,
        latent_loss: float,
        image_loss: float,
    ):
        self.dico_loss["total_loss"].append(total_loss)
        self.dico_loss["datafid_loss"].append(datafid_loss)
        self.dico_loss["latent_reg_loss"].append(latent_loss)
        self.dico_loss["image_reg_loss"].append(image_loss)


## PARAMETERS CLASSES ##


class OptimizedParameters(nn.Module):
    r"""
    Base class for optimized parameters in latent optimization algorithms.
    Provides a template for defining optimized parameters.
    """

    def __init__(self):
        super(OptimizedParameters, self).__init__()

    @abc.abstractmethod
    def generate_solution(
        self,
        decode_fn: callable,
        n_samples: int = 0,
        rate_scale: torch.Tensor | float = 1.0,
    ) -> dict:
        r"""
        Generate a solution given the optimized parameters.

        :param callable decode_fn: Decoding function of the generative model.
        :param int n_samples: Number of samples to generate (used only for stochastic latent optimization algorithms).
        :param torch.Tensor | float rate_scale: Scaling factor to modulate the regularization strength when using compressive autoencoder priors.
        """
        pass

    @abc.abstractmethod
    def init_from_paramvalues(self, init_dict: dict):
        r"""
        Initialize the class from a dictionary of parameters values

        :param dict init_dict: Dictionary containing the parameter values.
        """
        pass


class MAPzParameters(OptimizedParameters):
    r"""
    Class to handle CSGM and MAP-z optimized parameters.

    The optimized parameters are:
    - zbar: latent variable z
    - hbar: second latent variable h (if compressive autoencoder network)

    :param list[tuple[int]] zdim: Dimensions of the latent variable z (and h if compressive autoencoder network)
    """

    def __init__(
        self,
        zdim: list[tuple[int]],
        **kwargs,
    ):
        super(MAPzParameters, self).__init__()

        self.zdim = zdim

        ## init of z_bar ##
        self.zbar = nn.Parameter(torch.zeros(self.zdim[0]), requires_grad=True)

        ## init of h_bar (second latent variable) if compressive autoencoder network ##
        if len(zdim) == 2:
            self.hbar = nn.Parameter(torch.zeros(self.zdim[1]), requires_grad=True)
            self.optimize_h = True
        else:
            self.optimize_h = False

    def get_zbar(self):
        return self.zbar

    def get_hbar(self):
        return self.hbar

    def generate_solution(
        self,
        decode_fn: callable,
        n_samples: int = 0,
        rate_scale: torch.Tensor | float = 1.0,
    ):
        r"""
        Generate a solution from the optimized parameters using :math:`x_{rec} = D_\theta(zbar)`.

        :param callable decode_fn: Decoding function of the generative model.
        :param int n_samples: Unused parameter for MAP-z.
        :param torch.Tensor | float rate_scale: Scaling factor to modulate the regularization strength when using compressive autoencoder priors.
        """
        out_dict = {"z": self.get_zbar()}
        if self.optimize_h:
            out_dict["h"] = self.get_hbar()

        x_rec = decode_fn(out_dict, decode_mean_only=True)["x_rec"]
        out_dict.update({"x_rec": x_rec / rate_scale})

        return out_dict

    def init_from_paramvalues(self, init_dict):
        """
        Initialize the class from a dictionary of parameters values
        """
        for k, v in init_dict.items():
            if k in ["hbar", "zbar"]:
                getattr(self, k).data = v


class VBLEzParameters(MAPzParameters):
    """
    Class to handle the variational inference parameters for VBLE-z algorithm.

    The parameters include:
    - zbar: mean of the latent variable z
    - hbar: mean of the second latent variable h (if compressive autoencoder network
    - az: deviation of the latent variable z
    - ah: deviation of the second latent variable h (if compressive autoencoder network)

    :param list[tuple[int]] zdim: Dimensions of the latent variable z (and h if compressive autoencoder network)
    :param bool gaussian_sampling: Whether to use Gaussian sampling (True for VAE priors, False for CAE and GAN priors).
    """

    def __init__(
        self,
        zdim: list[tuple[int]],
        gaussian_sampling: bool,
        **kwargs,
    ):
        super(VBLEzParameters, self).__init__(zdim)

        self.bound_a = init_bound(vmin=1.5e-5, vmax=20)

        ## init of a_z ##
        self.az = nn.Parameter(torch.zeros(zdim[0]), requires_grad=True)

        ## init of a_h (deviation for second latent variable) if compressive autoencoder network ##
        if self.optimize_h:
            self.ah = nn.Parameter(torch.zeros(zdim[1]), requires_grad=True)

        self.latent_sampling_fn = (
            torch.distributions.normal.Normal(0.0, 1.0)
            if gaussian_sampling
            else torch.distributions.uniform.Uniform(-0.5, 0.5)
        )

    def sample_latent(self, size: tuple[int], std: torch.Tensor):
        r"""Helper function to sample latent variables."""
        return self.latent_sampling_fn.sample(size).to(std.device) * std

    def get_az(self):
        return torch.exp(self.bound_a(self.az))

    def get_ah(self):
        return torch.exp(self.bound_a(self.ah))

    def init_from_paramvalues(self, init_dict):
        """
        Initialize the class from a dictionary of parameters values
        """
        super().init_from_paramvalues(init_dict)
        for k, v in init_dict.items():
            if k in ["az", "ah"]:
                getattr(self, k).data = torch.log(v)

    def generate_solution(
        self,
        decode_fn: callable,
        n_samples: int = 1,
        rate_scale: torch.Tensor | int = 1.0,
    ):
        r"""
        Sample from the approximate posterior distribution using :math:`z \sim q(z), x = D_\theta(z)`.

        :param callable decode_fn: Decoding function :math:`D_\theta`.
        :param int n_samples: Number of samples to draw from the approximate posterior.
        :param torch.Tensor | float rate_scale: Scaling factor to modulate the regularization strength when using compressive autoencoder priors.
        """

        # Sampling of z
        z_bar = self.get_zbar()
        z = z_bar.expand((n_samples, *z_bar.shape[1:]))
        az = self.get_az()
        z = z + self.sample_latent(z.size(), az)
        out_dict = {"sd_q_z": az, "mu_q_z": z_bar, "z": z}

        # Sampling of the second latent variable h for CAEs
        if self.optimize_h:
            h_bar = self.get_hbar()
            h = h_bar.expand((n_samples, *h_bar.shape[1:]))
            ah = self.get_ah()
            h = h + self.sample_latent(h.size(), ah)
            out_dict.update({"sd_q_h": ah, "mu_q_h": h_bar, "h": h})

        # Decoding
        out_decoder_dict = decode_fn(out_dict, decode_mean_only=True)
        out_dict["x_rec"] = out_decoder_dict["x_rec"] / rate_scale
        return out_dict


class VBLExzParameters(VBLEzParameters):
    r"""
    Class to handle the variational inference parameters for VBLE-xz algorithm.

    The parameters include:
    - zbar: mean of the latent variable z
    - hbar: mean of the second latent variable h (if compressive autoencoder network
    - az: deviation of the latent variable z
    - ah: deviation of the second latent variable h (if compressive autoencoder network)
    - b: deviation of the image variable x

    :param list[tuple[int]] zdim: Dimensions of the latent variable z (and h if compressive autoencoder network)
    :param tuple[int] xdim: Dimension of the observed variable x
    :param bool gaussian_sampling: Whether to use Gaussian sampling (True for VAE priors, False for CAE and GAN priors).
    """

    def __init__(
        self,
        zdim: list[tuple[int]],
        xdim: tuple[int],
        gaussian_sampling: bool,
    ):
        super(VBLExzParameters, self).__init__(zdim, gaussian_sampling)

        self.xdim = xdim

        self.bound_b = init_bound(vmin=1e-4, vmax=3)

        ## init of b ##
        self.b = nn.Parameter(torch.zeros(xdim), requires_grad=True)

        self.image_sampling_fn = torch.distributions.normal.Normal(0.0, 1.0)

    def sample_image(self, size: tuple[int], std: torch.Tensor):
        r"""Helper function to sample image variables."""
        return self.image_sampling_fn.sample(size).to(std.device) * std

    def get_b(self):
        return torch.exp(self.bound_b(self.b))

    def get_param_names(self):
        return self.optimized_param_names

    def init_from_paramvalues(self, init_dict):
        """
        Initialize the class from a dictionary of parameters values
        """
        super().init_from_paramvalues(init_dict)
        for k, v in init_dict.items():
            if k in ["az", "ah", "b"]:
                getattr(self, k).data = torch.log(v)

    def generate_solution(
        self,
        decode_fn: callable,
        n_samples: int = 1,
        rate_scale: torch.Tensor | int = 1.0,
    ):
        r"""
        Sample from the approximate posterior distribution using :math:`z \sim q(z), x \sim q(x|z)`.

        :param callable decode_fn: Decoding function :math:`D_\theta`.
        :param int n_samples: Number of samples to draw from the approximate posterior.
        :param torch.Tensor | float rate_scale: Scaling factor to modulate the regularization strength when using compressive autoencoder priors.
        """

        # Sampling of z
        z_bar = self.get_zbar()
        z = z_bar.expand((n_samples, *z_bar.shape[1:]))
        az = self.get_az()
        z = z + self.sample_latent(z.size(), az)
        out_dict = {"sd_q_z": az, "mu_q_z": z_bar, "z": z}

        # Sampling of the second latent variable h for CAEs
        if self.optimize_h:
            h_bar = self.get_hbar()
            h = h_bar.expand((n_samples, *h_bar.shape[1:]))
            ah = self.get_ah()
            h = h + self.sample_latent(h.size(), ah)
            out_dict.update({"sd_q_h": ah, "mu_q_h": h_bar, "h": h})

        # Decoding
        out_decoder_dict = decode_fn(out_dict, decode_mean_only=False)

        # Sampling in image space
        x_bar = out_decoder_dict["x_rec"].expand((n_samples, -1, -1, -1))
        x_rec_std = out_decoder_dict["x_rec_std"].expand(
            (n_samples, *out_decoder_dict["x_rec_std"].shape[1:])
        )
        x_std = self.get_b() * x_rec_std.detach()
        x_sample = x_bar + self.sample_image(x_bar.size(), x_std)

        out_dict.update({"x_rec": x_sample / rate_scale})
        return out_dict


def interpolate_init_inpainting(x: np.ndarray) -> np.ndarray:
    r"""
    Image interpolation for inpainting initialization.

    :param np.ndarray x: Input image with shape (1, C, H, W) containing holes (zero values).

    :returns: Interpolated image with shape (1, C, H, W) with holes filled using linear interpolation.
    """
    xshape = x.shape
    for i in range(xshape[1]):
        mask_values = x[0, i] != 0
        X, Y = np.mgrid[0 : xshape[2], 0 : xshape[3]]
        x_values = X[mask_values]
        y_values = Y[mask_values]
        img_values = x[0, i][mask_values]
        interpolation = griddata(
            (x_values, y_values), img_values, (X, Y), method="linear", fill_value=0.5
        )
        x[0, i] = interpolation
    return x


def init_bound(vmin: float = 1.5e-5, vmax: float = 20, device: str = "cuda"):
    """
    Initialize bounding function for standard deviations used in VBLE algorithms.
    """

    lb_min = LowerBound(np.log(vmin)).to(device)
    lb_max = LowerBound(-np.log(vmax)).to(device)

    def bound_fn(z):
        return lb_min(-lb_max(-z))

    return bound_fn
