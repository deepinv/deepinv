import torch
from tqdm import tqdm


class GaussianMixtureModel:
    r"""
    Implements a Gaussian Mixture Model, its negative log likelihood function and an EM algorithm
    for parameter estimation.

    :param int n_components: number of components of the GMM
    :param int dimension: data dimension
    :param str device: gpu or cpu.
    """

    def __init__(self, n_components, dimension, device="cpu"):
        self._covariance_regularization = None
        self.n_components = n_components
        self.dimension = dimension
        self._weights = torch.ones((n_components,), device=device)
        self.set_weights(self._weights)
        self.mu = torch.zeros((n_components, dimension), device=device)
        self._cov = 0.1 * torch.eye(dimension, device=device)[None, :, :].tile(
            n_components, 1, 1
        )
        self.set_cov(self._cov)

    def set_cov(self, cov):
        r"""
        Sets the covariance parameters to cov and maintains their log-determinants and inverses

        :param torch.Tensor cov: new covariance matrices in a n_components x dimension x dimension tensor
        """
        self._cov = cov.to(self._cov)
        self._logdet_cov = torch.logdet(self._cov)
        self._cov_inv = torch.linalg.inv(self._cov)
        if self._covariance_regularization:
            self._cov_reg = self._cov + self._covariance_regularization * torch.eye(
                self.dimension, device=self._cov.device
            )[None, :, :].tile(self.n_components, 1, 1)
            self._logdet_cov_reg = torch.logdet(self._cov_reg)
            self._cov_inv_reg = torch.linalg.inv(self._cov_reg)

    def set_cov_reg(self, reg):
        r"""
        Sets covariance regularization parameter for evaluating
        Needed for EPLL.

        :param float reg: covariance regularization parameter
        """
        self._covariance_regularization = reg
        self._cov_reg = self._cov + self._covariance_regularization * torch.eye(
            self.dimension, device=self._cov.device
        )[None, :, :].tile(self.n_components, 1, 1)
        self._logdet_cov_reg = torch.logdet(self._cov_reg)
        self._cov_inv_reg = torch.linalg.inv(self._cov_reg)

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
        self._weights = (weights / torch.sum(weights)).to(self._weights)

    def get_weights(self):
        r"""
        get method for weights
        """
        return self._weights.clone()

    def get_parameter_dict(self):
        r"""
        Creates dict with parameters
        """
        parameters = {
            "weights": self._weights.clone().cpu(),
            "mu": self.mu.cpu(),
            "cov": self._cov.clone().cpu(),
        }
        return parameters

    def load_parameter_dict(self, parameter_dict):
        r"""
        Loads dict with parameters

        :param dict parameter_dict: dictionary containing parameters
        """
        self.mu = parameter_dict["mu"].to(self.mu)
        self.set_weights(parameter_dict["weights"])
        self.set_cov(parameter_dict["cov"])

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

    def negative_log_likelihood(self, x):
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


        :param torch.data.Dataloader dataloader: containing the data
        :param int max_iters: maximum number of iterations
        :param float stopping_criterion, None: stop when objective decrease is smaller than this number.
            None for performing exactly max_iters iterations
        :param bool data_init: True for initialize mu by the first data points, False for using current values as initialization
        :param bool verbose: Output progress information in the console
        """
        if data_init:
            self.mu = next(iter(dataloader))[0][: self.n_components].to(self.mu)

        objective = 1e100
        for step in (progress_bar := tqdm(range(max_iters), disable=not verbose)):
            weights_new, mu_new, cov_new, objective_new = self._EM_step(
                dataloader, verbose
            )
            # stopping criterion
            self.set_weights = weights_new
            self.mu = mu_new
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
        mu_new = mu_new / weights_new[:, None]
        cov_new = C_new / weights_new[:, None, None] - torch.matmul(
            mu_new[:, :, None], mu_new[:, None, :]
        )
        weights_new = weights_new / n
        objective = objective / n
        return weights_new, mu_new, cov_new, objective
