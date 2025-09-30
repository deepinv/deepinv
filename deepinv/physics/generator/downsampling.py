from __future__ import annotations
import torch
from deepinv.physics.generator import PhysicsGenerator
from deepinv.physics.blur import gaussian_blur, bilinear_filter, bicubic_filter
from deepinv.utils.compat import zip_strict


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
        If batch size = 1, a random filter and factor is sampled in (filters, factors) at each step.
        If batch size > 1, a unique factor needs to be sampled for the whole batch, but filters can vary. In this case,
        it is recommended to set the `psf_size` argument to ensure that all filters in the batch have the same shape.

    :param list[str] filters: list of filters to use for downsampling. Default is ["gaussian", "bilinear", "bicubic"].
    :param list[int] factors: list of factors to use for downsampling. Default is [2, 4].
    :param tuple[int, int] psf_size: size of the point spread function (PSF) to use for the filters, necessary to stack different filters. If None, the default size of the filter from the filter functions will be used. Default is None.
    :param rng: random number generator. Default is None.
    :param device: device to use. Default is "cpu".
    :param dtype: data type to use. Default is torch.float32.
    """

    def __init__(
        self,
        filters: str | list[str] = ("gaussian", "bilinear", "bicubic"),
        factors: int | list[int] = (2, 4),
        psf_size: tuple[int, int] = None,
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
            "psf_size": psf_size,
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

        if self.psf_size is not None:
            dH = self.psf_size[0] - filter.shape[-2]
            dW = self.psf_size[1] - filter.shape[-1]

            pad_top, pad_bottom = dH // 2, dH - dH // 2
            pad_left, pad_right = dW // 2, dW - dW // 2

            filter = torch.nn.functional.pad(
                filter, (pad_left, pad_right, pad_top, pad_bottom)
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

        factor_indices = torch.randint(
            low=0,
            high=len(self.list_factors),
            size=(batch_size,),
            generator=self.rng,
            **self.factory_kwargs,
        )
        filter_indices = torch.randint(
            low=0,
            high=len(self.list_filters),
            size=(batch_size,),
            generator=self.rng,
            **self.factory_kwargs,
        )
        factors = [self.list_factors[int(i)] for i in factor_indices.tolist()]
        filters = [self.list_filters[int(i)] for i in filter_indices.tolist()]

        filters = [
            self.get_kernel(f_str, f) for f_str, f in zip_strict(filters, factors)
        ]

        if not all([f.shape == filters[0].shape for f in filters]):
            raise ValueError(
                "Generated filters have different shapes in batch. Consider limiting factors/filters to one type per batch or limiting batch size = 1. If using a single factor, set the psf_size argument to ensure that all filters in the batch have the same shape."
            )

        return {
            "filter": torch.cat(filters),
            "factor": torch.tensor(factors),
        }
