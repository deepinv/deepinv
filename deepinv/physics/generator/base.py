# %%
import torch
import torch.nn as nn


class Generator:
    r"""
    Base class for parameter generation of physics.

    :param torch.Tensor params: the parameter of a physic, e.g., the filter of the Blur physic.
    :param dict kwargs: default keyword arguments to be passed to :meth:`Generator` for generating new parameters.
    """

    def __init__(self, params: torch.Tensor, **kwargs) -> None:
        self.params = params
        self.kwargs = kwargs
        self.factory_kwargs = {"device": params.device, "dtype": params.dtype}
        # Set attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    def step(self, *args, **kwargs):
        r"""
        Updates the parameter
        """

        if not kwargs:
            self.kwargs = kwargs

        new_params = self.__call__(*args, **self.kwargs)

        self.params.zero_()
        self.params.add_(new_params)

    def __call__(self, *args, **kwargs):
        r"""
        Return new parameter
        """
        return torch.zeros_like(self.params)


if __name__ == "__main__":

    class Physic(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            # self.params = nn.Parameter(torch.tensor([1., 2., 3.]), requires_grad=False)
            self.params = torch.tensor([1.0, 2.0, 3.0])
            self.kwargs = kwargs

        def forward(self, *args, **kwargs):
            pass

    # %%
    P = Physic()
    print(P.params)
    G = Generator(P.params, l=1, n=2)
    G.step()
    print(P.params)
    # %%
