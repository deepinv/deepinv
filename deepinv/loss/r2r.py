import torch
from deepinv.loss.loss import Loss


class R2RLoss(Loss):
    r"""
    Recorrupted-to-Recorrupted (R2R) Loss

    This loss can be used for unsupervised image denoising with unorganized noisy images.

    The loss is designed for the noise model:

    .. math::

        y \sim\mathcal{N}(u,\sigma^2 I) \quad \text{with}\quad u= A(x).

    The loss is computed as:

    .. math::

        \| y^- - AR(y^+) \|_2^2 \quad \text{s.t.} \quad y^+ = y + \alpha z, \quad y^- = y -  z / \alpha

    where :math:`R` is the trainable network, :math:`A` is the forward operator,
    :math:`y` is the noisy measurement,
    :math:`z` is the additional Gaussian noise of standard deviation :math:`\eta`,
    and :math:`\alpha` is a scaling factor.


    This loss is statistically equivalent to the supervised loss function defined on noisy/clean image pairs
    according to authors in https://ieeexplore.ieee.org/document/9577798

    .. note::

        :math:`\eta` should be chosen equal or close to :math:`\sigma` to obtain the best performance.

    .. note::

        To obtain the best test performance, the trained model should be averaged at test time
        over multiple realizations of the added noise, i.e. :math:`\hat{x} = \frac{1}{N}\sum_{i=1}^N R(y+\alpha z_i)`
        where :math:`N>1`. This can be achieved using :meth:`deepinv.loss.r2r_eval`.

    :param float eta: standard deviation of the Gaussian noise used for the perturbation.
    :param float alpha: scaling factor of the perturbation.
    """

    def __init__(self, metric=torch.nn.MSELoss(), eta=0.1, alpha=0.5):
        super(R2RLoss, self).__init__()
        self.name = "r2r"
        self.metric = metric
        self.eta = eta
        self.alpha = alpha

    def forward(self, y, physics, model, **kwargs):
        r"""
        Computes the R2R Loss.

        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction model.
        :return: (torch.Tensor) R2R loss.
        """

        pert = torch.randn_like(y) * self.eta

        y_plus = y + pert * self.alpha
        y_minus = y - pert / self.alpha

        output = model(y_plus, physics)

        return self.metric(physics.A(output), y_minus)


def r2r_eval(model, eta=0.1, alpha=0.5, MC_samples=5):
    r"""
    Average over multiple added noise realizations at evaluation time.

    To obtain the best test performance, the trained model using :meth:`deepinv.loss.R2RLoss`
    should be averaged at test time over multiple realizations of the added noise:

    .. math::

        \hat{x} = \frac{1}{N}\sum_{i=1}^N R(y+\alpha z_i)


    where :math:`N\geq 1`.

    :param torch.nn.Module model: Reconstruction model.
    :param float eta: standard deviation of the Gaussian noise used for the perturbation.
    :param float alpha: scaling factor of the perturbation.
    :param int MC_samples: number of samples used for the Monte Carlo approximation.
    :return: (torch.nn.Module) Model modified for evaluation.
    """

    class R2REvalModel(torch.nn.Module):
        def __init__(self, model):
            super(R2REvalModel, self).__init__()
            self.model = model

        def forward(self, y, physics):
            if self.training:
                return self.model(y, physics)
            else:
                with torch.no_grad():
                    out = 0
                    for i in range(MC_samples):
                        pert = torch.randn_like(y) * eta
                        y_plus = y + pert * alpha
                        out += self.model(y_plus, physics)
                    out = out / MC_samples
                return out

    return R2REvalModel(model)
