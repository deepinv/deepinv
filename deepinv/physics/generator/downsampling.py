import torch
from typing import List
from deepinv.physics.generator import PhysicsGenerator
from deepinv.physics.blur import gaussian_blur, bilinear_filter, bicubic_filter


class DownsamplingGenerator(PhysicsGenerator):
    r"""
    Random downsampling generator.

    Generates random downsampling factors and filters.
    This can be used for generating parameters to be passed to the
    :class:`Downsampling <deepinv.physics.Downsampling>` class.

    >>> from deepinv.physics.generator import DownsamplingGenerator
    >>> list_filters = ["bilinear", "bicubic", "gaussian"]
    >>> list_factors = [2, 4]
    >>> generator = DownsamplingGenerator(filters=list_filters, factors=list_factors)
    >>> ds = generator.step(batch_size=1)  # dict_keys(['filter', 'factor'])
    >>> filter = ds['filter']
    >>> factor = ds['factor']

    .. note::
        Each batch element has the same downsampling factor and filter, but these can vary from batch to batch.

    :param List[str] filters: list of filters to use for downsampling. Default is ["gaussian", "bilinear", "bicubic"].
    :param List[int] factors: list of factors to use for downsampling. Default is [2, 4].
    :param rng: random number generator. Default is None.
    :param device: device to use. Default is "cpu".
    :param dtype: data type to use. Default is torch.float32.
    """

    def __init__(
        self,
        filters: [str, List[str]] = ["gaussian", "bilinear", "bicubic"],
        factors: [int, List[int]] = [2, 4],
        rng: torch.Generator = None,
        device: str = "cpu",
        dtype: type = torch.float32,
    ) -> None:
        if isinstance(filters, str):
            filters = [filters]
        if isinstance(factors, int):
            factors = [factors]
        kwargs = {
            "list_filters": filters,
            "list_factors": factors,
        }
        super().__init__(device=device, dtype=dtype, rng=rng, **kwargs)

    def str2filter(self, filter_name: str, factor: int):
        r"""
        Returns the filter associated to a given filter name and factor.
        """
        if filter_name == "gaussian":
            filter = torch.nn.Parameter(
                gaussian_blur(sigma=(factor, factor)), requires_grad=False
            ).to(self.device)
        elif filter_name == "bilinear":
            filter = torch.nn.Parameter(
                bilinear_filter(factor), requires_grad=False
            ).to(self.device)
        elif filter_name == "bicubic":
            filter = torch.nn.Parameter(bicubic_filter(factor), requires_grad=False).to(
                self.device
            )
        return filter

    def get_kernel(self, filter_str: str = None, factor=None):
        r"""
        Returns a batched tensor of filters associated to a given filter name and factor.

        :param str filter_str: filter name. Default is None.
        :param int factor: downsampling factor. Default is None.
        """
        batched_kernels = self.str2filter(filter_str, factor)
        return batched_kernels

    def step(
        self,
        batch_size: int = 1,
        seed: int = None,
    ):
        r"""
        Generates a random downsampling factor and filter.

        :param int batch_size: batch size. Default is 1.
        :param int seed: seed for random number generator. Default is None.
        """
        self.rng_manual_seed(seed)

        random_indices = torch.randint(
            low=0,
            high=len(self.list_factors),
            size=(2,),
            generator=self.rng,
            **self.factory_kwargs,
        )
        factor = self.list_factors[int(random_indices[0].item())]
        filter_str = self.list_filters[int(random_indices[1].item())]
        filters = self.get_kernel(filter_str, factor)
        return {
            "filter": filters,
            "factor": factor,
        }
