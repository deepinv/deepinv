import torch.nn as nn


class SamplingIterator(nn.Module):
    r"""
    Base class for sampling iterators.
    """

    def __init__(self, **kwargs):
        super(SamplingIterator, self).__init__()

    def forward(
        self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs
    ):
        r"""
        map from X_t -> X_{t+1}

        :param dict X: Dictionary containing the current iterate(s?).
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: Instance of the physics modeling the observation.
        :return: Dictionary `{"est": (x, z), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        pass
