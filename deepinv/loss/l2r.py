from __future__ import annotations
import torch
import math
import warnings
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
import torch.nn.functional as F 
import torch.nn as nn



class Recorruptor(torch.nn.Module):
    r"""
    Trainable re-corruption network used by :class:`L2RLoss`.

    Given a random input tensor :math:`\omega` and measurement :math:`y`, this module
    outputs :math:`h(\omega, y)`, i.e. an additive perturbation used to build
    re-corrupted measurements :math:`y_1 = y + \alpha h(\omega, y)`.

    :param int depth: Depth of the internal MLP definition.
    :param int feats: Number of hidden features in the MLP.
    :param int kernel_size: Spatial kernel size used to filter the output
        perturbation. If ``kernel_size=1``, a scalar scale is used instead.
    :param bool multiplicative: If ``True``, modulates perturbations by
        :math:`\sqrt{y}` to mimic signal-dependent noise.
    :param float sigma: Initialization value for the scalar scale when
        ``kernel_size=1``.
    """

    def __init__(self, depth=5, feats=4, kernel_size=3, multiplicative=False, sigma=0.1, net=None):
        super(Recorruptor, self).__init__()

        self.multiplicative = multiplicative
        self.kernel_size = kernel_size


        feats_list = [1] + [feats] * depth + [1]
        t_in = [1]


        # self.net = MonotonicFullyConnectedNet(feats_list, t_in=t_in, base_act=F.softplus)

        if net == "identity":
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Linear(1, feats),
                nn.Softplus(),
                nn.Linear(feats, feats),
                nn.Softplus(),
                nn.Linear(feats, 1)
            )


        self.norm_layer  = nn.BatchNorm1d(1, affine=False, momentum=0.9)

        if self.kernel_size > 1:
            self.sigma = nn.Parameter(torch.randn(1, 1, self.kernel_size, self.kernel_size) * 0.1 , requires_grad=True) 
        else:
            self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=True)

    
    def forward(self, w, y):

        c = y.shape[1]

        hw = self.net(w.reshape(-1, 1))
        hw = self.norm_layer(hw)
        hw = hw.reshape_as(w)

        if self.kernel_size > 1:
            kernel = self.sigma
            kernel = kernel.repeat(c, 1, 1, 1)
            hw = F.conv2d(hw, kernel, padding=self.kernel_size//2, groups=c)
        else:
            hw = self.sigma * hw

        if self.multiplicative:
            hw = hw * y.clamp(min=1e-6).sqrt()
            
        return hw



class L2RLoss(Loss):
    r"""
    Learning to Recorrupt (L2R) Loss

    This self-supervised loss can be used when the measurement noise model is
    unknown. L2R introduces a trainable re-corruption mechanism:

    .. math::

        y_1 = y + \alpha h(\omega),

    where :math:`h` is a trainable network, :math:`\omega` is an i.i.d. Gaussian
    random tensor, and :math:`\alpha` is a scaling factor.

    Let :math:`R` be the trainable reconstruction network and :math:`A` the
    forward operator. The L2R objective is:

    .. math::

        \|AR(y_1) - y\|_2^2 + \frac{2}{\alpha}\langle AR(y_1), h(\omega)\rangle.

    During training, this objective is minimized with respect to the
    reconstruction model parameters, while the re-corruption network parameters
    are updated in the opposite direction (maximization step).

    In practice, :math:`h` is parameterized as a lightweight neural network that
    learns perturbations adapted to the observed data.

    .. warning::

        The reconstruction model should be adapted before training using
        :meth:`adapt_model` so that re-corruption is applied at model input.

    .. note::

        To obtain better test performance, predictions can be averaged over
        multiple re-corruptions:

        .. math::

            \hat{x} = \frac{1}{N}\sum_{i=1}^N R(y_1^{(i)}), \quad N > 1.

        This is handled automatically by :meth:`adapt_model` in evaluation mode.

    :param Metric, torch.nn.Module metric: Metric used to compute the main data
        term. Defaults to MSE when set to ``None``.
    :param float alpha: Scaling factor controlling the re-corruption strength.
    :param int eval_n_samples: Number of Monte Carlo samples used at test time.
    :param torch.nn.Module recorruptor: Trainable re-corruption network
        :math:`h`. If ``None``, a default :class:`Recorruptor` is used.

    |sep|

    :Example:

    >>> import torch
    >>> import deepinv as dinv
    >>> physics = dinv.physics.Denoising()
    >>> model = dinv.models.MedianFilter()
    >>> loss = dinv.loss.L2RLoss(metric=torch.nn.MSELoss(), alpha=0.5, eval_n_samples=2)
    >>> model = loss.adapt_model(model)  # important step!
    >>> x = torch.ones((1, 1, 8, 8))
    >>> y = physics(x)
    >>> x_net = model(y, physics, update_parameters=True)  # save corruption in forward pass
    >>> l = loss(x_net, y, physics, model)
    >>> print(l.item() >= 0)
    True
    """

    def __init__(
        self,
        metric: Metric | torch.nn.Module | None = None,
        alpha: float = 0.5,
        eval_n_samples: int = 5,
        recorruptor: torch.nn.Module | None = None,
        device: torch.device | None = None,
        **kwargs,
    ) -> None:
        r"""
        Initializes the L2R loss.

        :param Metric, torch.nn.Module metric: Metric used to compute the main
            data term.
        :param float alpha: Scaling factor controlling re-corruption strength.
        :param int eval_n_samples: Number of Monte Carlo samples used at test
            time.
        :param torch.nn.Module recorruptor: Trainable re-corruption module.
        """

        if metric is None:
            metric = torch.nn.MSELoss()

        super(L2RLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.eval_n_samples = eval_n_samples

        if recorruptor is None:
            self.recorruptor = Recorruptor(multiplicative=True)
        else:
            self.recorruptor = recorruptor
        
        self.recorruptor.to(device)

        self.recorruptor_optimizer = torch.optim.Adam(self.recorruptor.parameters(), 
                                                      lr=1e-6, 
                                                      weight_decay=1e-6)



    def forward(self, x_net, y, physics, model, **kwargs):
        r"""
        Computes the L2R objective.

        :param torch.Tensor x_net: Reconstructed image estimate.
        :param torch.Tensor y: Noisy measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with
            the measurements.
        :param torch.nn.Module model: Adapted reconstruction model exposing
            :meth:`get_corruption`.
        :return: (:class:`torch.Tensor`) L2R loss value.
        """
        
        hw = model.get_corruption()
        y_pred = physics.A(x_net)

        if model.training:
            loss = - (2/self.alpha) * ( y_pred.detach() * hw ).mean()
            self.update_recorruptor(loss)

        return self.metric(y_pred, y) + (2/self.alpha) * ( y_pred * hw.detach() ).mean()

    
    def adapt_model(self, model, **kwargs):
        r"""
        Adapts a reconstruction model to include L2R re-corruption at input.

        During training, one re-corruption sample is used for efficiency. During
        evaluation, multiple samples can be averaged for improved robustness.

        :param torch.nn.Module model: Reconstruction model.
        :return: (:class:`torch.nn.Module`) Adapted L2R model.
        """

        if isinstance(model, L2RModel):
            model = model
        else:
            model = L2RModel(
                model,
                self.recorruptor,
                self.alpha,
                self.eval_n_samples,
                **kwargs,
            )

        return model


    def update_recorruptor(self, loss, **kwargs):
        r"""
        Applies one optimization step to the re-corruption network.

        :param torch.Tensor loss: Objective used to update re-corruption
            parameters.
        """
        self.recorruptor_optimizer.zero_grad()
        loss.backward()
        self.recorruptor_optimizer.step()


class L2RModel(torch.nn.Module):
    r"""
    Learning to Recorrupt (L2R) wrapper model.

    This wrapper injects trainable re-corruption before calling the underlying
    reconstruction model, and optionally stores the sampled corruption during
    training for use in :class:`L2RLoss`.
    """

    def __init__(self, model, recorruptor, alpha, eval_n_samples, **kwargs):
        super(L2RModel, self).__init__()

        self.model = model
        self.recorruptor = recorruptor
        self.alpha = alpha
        self.eval_n_samples = eval_n_samples
        self.name = "l2r"
    
    def forward(self, y, physics, update_parameters=False, more_evals=0, x=None):
        r"""
        Runs the adapted model with L2R re-corruption.

        :param torch.Tensor y: Input measurements.
        :param deepinv.physics.Physics physics: Forward operator.
        :param bool update_parameters: If ``True`` in training mode, stores the
            sampled corruption for subsequent loss computation.
        :param int more_evals: Extra Monte Carlo samples to add during
            evaluation.
        :param torch.Tensor x: Unused argument kept for API compatibility.
        :return: (:class:`torch.Tensor`) Averaged model output.
        """

        eval_n_samples = 1 if self.training else self.eval_n_samples
        out = 0
        eval_n_samples = eval_n_samples + more_evals

        with torch.set_grad_enabled(self.training):

            for _ in range(eval_n_samples):

                hw = self.recorruptor( torch.randn_like(y), y )
                y1 = y + self.alpha * hw

                out += self.model(y1.detach(), physics)

            if self.training and update_parameters:
                self.corruption = hw

            out = out / eval_n_samples

        return out
    
    def get_corruption(self):
        r"""Returns the most recently stored re-corruption sample."""
        return self.corruption