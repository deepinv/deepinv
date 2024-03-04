import torch.nn as nn
import torch
from deepinv.utils import patch_extractor
from deepinv.optim.utils import conjugate_gradient
from deepinv.utils.demo import load_torch_url
from deepinv.physics import Denoising, GaussianNoise
from tqdm import tqdm


class EPLL(nn.Module):
    r"""
    Patch prior via Gaussian mixture models.

    The forward method evaluates the negative log likelihood of the GMM.
    The reconstruction function implements the approximated half-quadratic splitting method as in Zoran, D., and Weiss, Y.  "From learning models of natural image patches to whole image restoration." (ICCV 2011).

    :param deepinv.models.GaussianMixtureModel GMM: Gaussian mixture defining the distribution on the patch space.
        None creates a GMM with n_components components of dimension accordingly to the arguments patch_size and channels.
    :param int n_components: number of components of the generated GMM if GMM is None.
    :param str pretrained: Path to pretrained weights of the GMM with file ending .pt. None for no pretrained weights, "download" for pretrained weights on the BSDS500 dataset, "GMM_lodopab_small" for the weights from the limited-angle CT example.
    :param int patch_size: patch size.
    :param int channels: number of color channels (e.g. 1 for gray-valued images and 3 for RGB images)
    :param str device: defines device (cpu or cuda)
    """

    def __init__(
        self,
        GMM=None,
        n_components=200,
        pretrained="download",
        patch_size=6,
        channels=1,
        device="cpu",
    ):
        super(EPLL, self).__init__()
        if GMM is None:
            self.GMM = GaussianMixtureModel(
                n_components, patch_size**2 * channels, device=device
            )
        else:
            self.GMM = GMM
        self.patch_size = patch_size
        self.denoising_operator = Denoising(GaussianNoise())
        if pretrained:
            if pretrained == "download":
                if patch_size == 6 and (channels == 1 or channels == 3):
                    pretrained = "GMM_BSDS_gray" if channels == 1 else "GMM_BSDS_color"
                else:
                    raise ValueError("Pretrained weights not found!")
            if pretrained[-3:] == ".pt":
                weights = torch.load(pretrained)
            else:
                if pretrained == "GMM_lodopab_small":
                    assert patch_size == 3
                    assert channels == 1
                    url = "https://drive.google.com/uc?export=download&id=1SBe1tVqGscDa-JqaaKxenbO6WmGBkctH"
                elif pretrained == "GMM_BSDS_gray":
                    assert patch_size == 6
                    assert channels == 1
                    url = "https://drive.google.com/uc?export=download&id=17d40IPycCf8Cb5RmOcrlPTq_AniBlYcK"
                elif pretrained == "GMM_BSDS_color":
                    assert patch_size == 6
                    assert channels == 3
                    url = "https://www.googleapis.com/drive/v3/files/1SndTEXBDyPAOFepWSPTC1fxh-d812F75?alt=media&key=AIzaSyDVCNpmfKmJ0gPeyZ8YWMca9ZOKz0CWdgs"
                else:
                    raise ValueError("Pretrained weights not found!")
                file_name = pretrained + ".pt"
                weights = torch.hub.load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=file_name
                )
            self.load_state_dict(weights)

    def forward(self, x, sigma, betas=None, batch_size=-1):
        r"""
        Calls the reconstruction for denoising

        :param torch.Tensor x: tensor of noisy images. Shape: batch size x ...
        :param float sigma: noise level
        :param list of floats betas: parameters from the half-quadratic splitting. None uses the standard choice 1/sigma_sq [1,4,8,16,32]
        :param int batch_size: batching the patch estimations for large images. No effect on the output, but a small value reduces the memory consumption
            but might increase the computation time. -1 for considering all patches at once.
        """
        return self.reconstruction(
            x, x.clone(), sigma, self.denoising_operator, batch_size=batch_size
        )

    def negative_log_likelihood(self, x):
        r"""
        Takes patches and returns the negative log likelihood of the GMM for each patch.

        :param torch.Tensor x: tensor of patches of shape batch_size x number of patches per batch x patch_dimensions
        """
        B, n_patches = x.shape[0:2]
        logpz = self.GMM(x.view(B * n_patches, -1))
        return logpz.view(B, n_patches)

    def reconstruction(self, y, x_init, sigma, physics, betas=None, batch_size=-1):
        r"""
        Approximated half-quadratic splitting method for image reconstruction as proposed by Zoran and Weiss.

        :param torch.Tensor y: tensor of observations. Shape: batch size x ...
        :param torch.Tensor x_init: tensor of initializations. Shape: batch size x channels x height x width
        :param float sigma: noise level (acts as regularization parameter)
        :param deepinv.physics.LinearPhysics physics: Forward operator. Has to be linear. Requires physics.A and physics.A_adjoint.
        :param list of floats betas: parameters from the half-quadratic splitting. None uses the standard choice 1/sigma_sq [1,4,8,16,32]
        :param int batch_size: batching the patch estimations for large images. No effect on the output, but a small value reduces the memory consumption
            but might increase the computation time. -1 for considering all patches at once.
        """
        if betas is None:
            # default choice as suggested in Parameswaran et al. "Accelerating GMM-Based Patch Priors for Image Restoration: Three Ingredients for a 100× Speed-Up"
            betas = [beta / sigma**2 for beta in [1.0, 4.0, 8.0, 16.0, 32.0]]
        if y.shape[0] > 1:
            # vectorization over a batch of images not implemented....
            return torch.cat(
                [
                    self.reconstruction(
                        y[i : i + 1],
                        x_init[i : i + 1],
                        betas=betas,
                        batch_size=batch_size,
                    )
                    for i in range(y.shape[0])
                ],
                0,
            )
        x = x_init
        Aty = physics.A_adjoint(y)
        for beta in betas:
            x = self._reconstruction_step(Aty, x, sigma**2, beta, physics, batch_size)
        return x

    def _reconstruction_step(self, Aty, x, sigma_sq, beta, physics, batch_size):
        # precomputations for GMM with covariance regularization
        self.GMM.set_cov_reg(1.0 / beta)
        N, M = x.shape[2:4]
        total_patch_number = (N - self.patch_size + 1) * (M - self.patch_size + 1)
        if batch_size == -1 or batch_size > total_patch_number:
            batch_size = total_patch_number

        # compute sum P_i^T z and sum P_i^T P_i on the fly with batching
        x_tilde_flattened = torch.zeros_like(x).reshape(-1)
        patch_multiplicities = torch.zeros_like(x).reshape(-1)

        # batching loop over all patches in the image
        ind = 0
        while ind < total_patch_number:
            # extract patches
            n_patches = min(batch_size, total_patch_number - ind)
            patch_inds = torch.LongTensor(range(ind, ind + n_patches)).to(x.device)
            patches, linear_inds = patch_extractor(
                x, n_patches, self.patch_size, position_inds_linear=patch_inds
            )
            patches = patches.reshape(patches.shape[0] * patches.shape[1], -1)
            linear_inds = linear_inds.reshape(patches.shape[0], -1)

            # Gaussian selection
            k_star = self.GMM.classify(patches, cov_regularization=True)

            # Patch estimation
            estimation_matrices = torch.bmm(
                self.GMM.get_cov_inv_reg(), self.GMM.get_cov()
            )
            estimation_matrices_k_star = estimation_matrices[k_star]
            patch_estimates = torch.bmm(
                estimation_matrices_k_star, patches[:, :, None]
            ).reshape(patches.shape[0], patches.shape[1])

            # update on-the-fly parameters
            # the following two lines are the same like
            # patch_multiplicities[linear_inds] += 1.0
            # x_tilde_flattened[linear_inds] += patch_estimates
            # where values of multiple indices are accumulated.
            patch_multiplicities.index_put_(
                (linear_inds,), torch.ones_like(patch_estimates), accumulate=True
            )
            x_tilde_flattened.index_put_(
                (linear_inds,), patch_estimates, accumulate=True
            )
            ind = ind + n_patches
        # compute x_tilde
        x_tilde_flattened /= patch_multiplicities

        # Image estimation by CG method
        rhs = Aty + beta * sigma_sq * x_tilde_flattened.view(x.shape)
        op = lambda im: physics.A_adjoint(physics.A(im)) + beta * sigma_sq * im
        hat_x = conjugate_gradient(op, rhs, max_iter=1e2, tol=1e-5)
        return hat_x


class GaussianMixtureModel(nn.Module):
    r"""
    Gaussian mixture model including parameter estimation.

    Implements a Gaussian Mixture Model, its negative log likelihood function and an EM algorithm
    for parameter estimation.

    :param int n_components: number of components of the GMM
    :param int dimension: data dimension
    :param str device: gpu or cpu.
    """

    def __init__(self, n_components, dimension, device="cpu"):
        super(GaussianMixtureModel, self).__init__()
        self._covariance_regularization = None
        self.n_components = n_components
        self.dimension = dimension
        self._weights = nn.Parameter(
            torch.ones((n_components,), device=device), requires_grad=False
        )
        self.set_weights(self._weights)
        self.mu = nn.Parameter(
            torch.zeros((n_components, dimension), device=device), requires_grad=False
        )
        self._cov = nn.Parameter(
            0.1
            * torch.eye(dimension, device=device)[None, :, :].tile(n_components, 1, 1),
            requires_grad=False,
        )
        self._cov_inv = nn.Parameter(
            0.1
            * torch.eye(dimension, device=device)[None, :, :].tile(n_components, 1, 1),
            requires_grad=False,
        )
        self._cov_inv_reg = nn.Parameter(
            0.1
            * torch.eye(dimension, device=device)[None, :, :].tile(n_components, 1, 1),
            requires_grad=False,
        )
        self._cov_reg = nn.Parameter(
            0.1
            * torch.eye(dimension, device=device)[None, :, :].tile(n_components, 1, 1),
            requires_grad=False,
        )
        self._logdet_cov = nn.Parameter(self._weights.clone(), requires_grad=False)
        self._logdet_cov_reg = nn.Parameter(self._weights.clone(), requires_grad=False)
        self.set_cov(self._cov)

    def set_cov(self, cov):
        r"""
        Sets the covariance parameters to cov and maintains their log-determinants and inverses

        :param torch.Tensor cov: new covariance matrices in a n_components x dimension x dimension tensor
        """
        self._cov.data = cov.detach().to(self._cov)
        self._logdet_cov.data = torch.logdet(self._cov).detach().clone()
        self._cov_inv.data = torch.linalg.inv(self._cov).detach().clone()
        if self._covariance_regularization:
            self._cov_reg.data = (
                self._cov.detach().clone()
                + self._covariance_regularization
                * torch.eye(self.dimension, device=self._cov.device)[None, :, :].tile(
                    self.n_components, 1, 1
                )
            )
            self._logdet_cov_reg.data = torch.logdet(self._cov_reg).detach().clone()
            self._cov_inv_reg.data = torch.linalg.inv(self._cov_reg).detach().clone()

    def set_cov_reg(self, reg):
        r"""
        Sets covariance regularization parameter for evaluating
        Needed for EPLL.

        :param float reg: covariance regularization parameter
        """
        self._covariance_regularization = reg
        self._cov_reg.data = (
            self._cov.detach().clone()
            + self._covariance_regularization
            * torch.eye(self.dimension, device=self._cov.device)[None, :, :].tile(
                self.n_components, 1, 1
            )
        )
        self._logdet_cov_reg.data = torch.logdet(self._cov_reg).detach().clone()
        self._cov_inv_reg.data = torch.linalg.inv(self._cov_reg).detach().clone()

    def get_cov(self):
        r"""
        get method for covariances
        """
        return self._cov.clone()

    def get_cov_inv_reg(self):
        r"""
        get method for covariances
        """
        return self._cov_inv_reg.clone()

    def set_weights(self, weights):
        r"""
        sets weight parameter while ensuring non-negativity and summation to one

        :param torch.Tensor weights: non-zero weight tensor of size n_components with non-negative entries
        """
        assert torch.min(weights) >= 0.0
        assert torch.sum(weights) > 0.0
        self._weights.data = (weights / torch.sum(weights)).detach().to(self._weights)

    def get_weights(self):
        r"""
        get method for weights
        """
        return self._weights.clone()

    def load_state_dict(self, *args, **kwargs):
        r"""
        Override load_state_dict to maintain internal parameters.
        """
        super().load_state_dict(*args, **kwargs)
        self.set_cov(self._cov)
        self.set_weights(self._weights)

    def component_log_likelihoods(self, x, cov_regularization=False):
        r"""
        returns a tensor containing the log likelihood values of x for each component

        :param torch.Tensor x: input data of shape batch_dimension x dimension
        :param bool cov_regularization: whether using regularized covariance matrices
        """
        if cov_regularization:
            cov_inv = self._cov_inv_reg
            logdet_cov = self._logdet_cov_reg
        else:
            cov_inv = self._cov_inv
            logdet_cov = self._logdet_cov
        centered_x = x[None, :, :] - self.mu[:, None, :]
        exponent = torch.sum(torch.bmm(centered_x, cov_inv) * centered_x, 2)
        component_log_likelihoods = (
            -0.5 * logdet_cov[:, None]
            - 0.5 * exponent
            - 0.5 * self.dimension * torch.log(torch.tensor(2 * torch.pi).to(x))
        )
        return component_log_likelihoods.T

    def forward(self, x):
        r"""
        evaluate negative log likelihood function

        :param torch.Tensor x: input data of shape batch_dimension x dimension
        """
        component_log_likelihoods = self.component_log_likelihoods(x)
        component_log_likelihoods = component_log_likelihoods + torch.log(
            self._weights[None, :]
        )
        log_likelihoods = torch.logsumexp(component_log_likelihoods, -1)
        return -log_likelihoods

    def classify(self, x, cov_regularization=False):
        """
        returns the index of the most likely component

        :param torch.Tensor x: input data of shape batch_dimension x dimension
        :param bool cov_regularization: whether using regularized covariance matrices
        """
        component_log_likelihoods = self.component_log_likelihoods(
            x, cov_regularization=cov_regularization
        )
        component_log_likelihoods = component_log_likelihoods + torch.log(
            self._weights[None, :]
        )
        val, ind = torch.max(component_log_likelihoods, 1)
        return ind

    def fit(
        self,
        dataloader,
        max_iters=100,
        stopping_criterion=None,
        data_init=True,
        cov_regularization=1e-5,
        verbose=False,
    ):
        """
        Batched Expectation Maximization algorithm for parameter estimation.


        :param torch.utils.data.DataLoader dataloader: containing the data
        :param int max_iters: maximum number of iterations
        :param float stopping_criterion: stop when objective decrease is smaller than this number.
            None for performing exactly max_iters iterations
        :param bool data_init: True for initialize mu by the first data points, False for using current values as initialization
        :param bool verbose: Output progress information in the console
        """
        if data_init:
            first_data = next(iter(dataloader))[0][: self.n_components].to(self.mu)
            if first_data.shape[0] == self.n_components:
                self.mu = first_data
            else:
                # if the first batch does not contain enough data points, fill up the others randomly...
                self.mu[: first_data.shape[0]] = first_data
                self.mu[first_data.shape[0] :] = torch.randn_like(
                    self.mu[first_data.shape[0] :]
                ) * torch.std(first_data, 0, keepdim=True) + torch.mean(
                    first_data, 0, keepdim=True
                )

        objective = 1e100
        for step in (progress_bar := tqdm(range(max_iters), disable=not verbose)):
            weights_new, mu_new, cov_new, objective_new = self._EM_step(
                dataloader, verbose
            )
            # stopping criterion
            self.set_weights = weights_new
            self.mu.data = mu_new
            cov_new_reg = cov_new + cov_regularization * torch.eye(self.dimension)[
                None, :, :
            ].tile(self.n_components, 1, 1).to(cov_new)
            self.set_cov(cov_new_reg)
            if stopping_criterion:
                if objective - objective_new < stopping_criterion:
                    return
            objective = objective_new
            progress_bar.set_description(
                "Step {}, Objective {:.4f}".format(step + 1, objective.item())
            )

    def _EM_step(self, dataloader, verbose):
        """
        one step of the EM algorithm

        :param torch.data.Dataloader dataloader: containing the data
        :param bool verbose: Output progress information in the console
        """
        objective = 0
        weights_new = torch.zeros_like(self._weights)
        mu_new = torch.zeros_like(self.mu)
        C_new = torch.zeros_like(self._cov)
        n = 0
        objective = 0
        for x, _ in tqdm(dataloader, disable=not verbose):
            x = x.to(self.mu)
            n += x.shape[0]
            component_log_likelihoods = self.component_log_likelihoods(x)
            log_betas = component_log_likelihoods + torch.log(self._weights[None, :])
            log_beta_sum = torch.logsumexp(log_betas, -1)
            log_betas = log_betas - log_beta_sum[:, None]
            objective -= torch.sum(log_beta_sum)
            betas = torch.exp(log_betas)
            weights_new += torch.sum(betas, 0)
            beta_times_x = x[None, :, :] * betas.T[:, :, None]
            mu_new += torch.sum(beta_times_x, 1)
            C_new += torch.bmm(
                beta_times_x.transpose(1, 2),
                x[None, :, :].tile(self.n_components, 1, 1),
            )

        # prevents division by zero if weights_new is zero
        weights_new = torch.maximum(weights_new, torch.tensor(1e-5).to(weights_new))

        mu_new = mu_new / weights_new[:, None]
        cov_new = C_new / weights_new[:, None, None] - torch.matmul(
            mu_new[:, :, None], mu_new[:, None, :]
        )
        weights_new = weights_new / n
        objective = objective / n
        return weights_new, mu_new, cov_new, objective
