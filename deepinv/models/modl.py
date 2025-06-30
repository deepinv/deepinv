from typing import Union
from torch.nn import Module

from deepinv.optim.optimizers import create_iterator
from deepinv.optim.prior import PnP
from deepinv.optim.data_fidelity import L2
from deepinv.models import DnCNN, Denoiser
from deepinv.unfolded import BaseUnfold


class MoDL(BaseUnfold):
    def __init__(
        self,
        denoiser: Union[Denoiser, Module] = None,
        num_iter: int = 3,
    ):
        """Unrolled network proposed in MoDL.

        The model is a simple unrolled network using half-quadratic splitting
        where the prox is replaced by a trainable denoising prior.

        This was proposed for MRI reconstruction in :footcite:t:`aggarwal2018modl`.

        :param Denoiser, torch.nn.Module denoiser: backbone denoiser model. If ``None``, uses :class:`deepinv.models.DnCNN`
        :param int num_iter: number of unfolded layers ("cascades"), defaults to 3.

        """
        # Select the data fidelity term
        data_fidelity = L2()

        # If the prior dict value is initialized with a table of length max_iter, then a distinct model is trained for each
        # iteration. For fixed trained model prior across iterations, initialize with a single model.
        denoiser = (
            denoiser
            if denoiser is not None
            else DnCNN(
                in_channels=2,  # real + imaginary parts
                out_channels=2,
                pretrained=None,
                depth=7,
            )
        )
        prior = PnP(denoiser=denoiser)

        # Unrolled optimization algorithm parameters
        lamb = [1.0] * num_iter  # initialization of the regularization parameter
        stepsize = [1.0] * num_iter  # initialization of the step sizes.
        sigma_denoiser = [0.01] * num_iter  # initialization of the denoiser parameters
        params_algo = (
            {  # wrap all the restoration parameters in a 'params_algo' dictionary
                "stepsize": stepsize,
                "g_param": sigma_denoiser,
                "lambda": lamb,
            }
        )

        trainable_params = [
            "lambda",
            "stepsize",
            "g_param",
        ]  # define which parameters from 'params_algo' are trainable

        # Define the unfolded trainable model.
        iterator = create_iterator("HQS", prior=prior)
        super().__init__(
            iterator,
            max_iter=num_iter,
            trainable_params=trainable_params,
            has_cost=iterator.has_cost,
            data_fidelity=data_fidelity,
            prior=prior,
            params_algo=params_algo,
        )
