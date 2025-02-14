from typing import Union, Iterable
import torch
from deepinv.transform.base import Transform, TransformParam


class CPABDiffeomorphism(Transform):
    r"""
    Continuous Piecewise-Affine-based Diffeomorphism.

    This requires the libcpab package which you can install from our `maintained fork <https://github.com/Andrewwango/libcpab>`_
    using ``pip install libcpab``.

    Wraps CPAB from a modified version of the `original implementation <https://github.com/SkafteNicki/libcpab>`_.
    From the paper Freifeld et al. `Transformations Based on Continuous Piecewise-Affine Velocity Fields <https://ieeexplore.ieee.org/abstract/document/7814343>`_.

    These diffeomorphisms benefit from fast GPU-accelerated transform + fast inverse.

    Generates ``n_trans`` randomly transformed versions.

    See :class:`deepinv.transform.Transform` for further details and examples.

    .. warning::

        This implementation does not allow using a ``torch.Generator`` to generate reproducible transformations.
        You may be able to achieve reproducibility by using a global seed instead.

    :param int n_trans: number of transformed versions generated per input image.
    :param int constant_batch: if ``True``, all images in batch transformed with same params.
    :param int n_tesselation: number of cells in tesselation in all dimensions.
        See ``libcpab.Cpab`` `docs <https://github.com/SkafteNicki/libcpab?tab=readme-ov-file#how-to-use>`_ for more info.
    :param bool zero_boundary: see ``libcpab.Cpab`` docs.
    :param bool volume_perservation: see ``libcpab.Cpab`` docs.
    :param bool override: see ``libcpab.Cpab`` docs.
    :param str, torch.device device: torch device.
    """

    def __init__(
        self,
        *args,
        constant_batch: bool = True,
        n_tesselation: int = 3,
        zero_boundary: bool = True,
        volume_perservation: bool = True,
        override: bool = True,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.constant_batch = constant_batch

        try:
            from libcpab import Cpab
        except ImportError:
            raise ImportError("Install libcpab using pip install libcpab")

        self.cpab = Cpab(
            [n_tesselation, n_tesselation],
            device=device,
            zero_boundary=zero_boundary,
            volume_perservation=volume_perservation,
            override=override,
        )

    def _get_params(self, x: torch.Tensor) -> dict:
        """Generate random diffeomorphism parameters.

        If ``constant_batch`` is set, then all images in batch will use same parameters.

        :param torch.Tensor x: input image
        """
        return {
            "diffeo": self.cpab.sample_transformation(
                self.n_trans * (1 if self.constant_batch else len(x))
            )
        }

    def _transform(
        self,
        x: torch.Tensor,
        diffeo: Union[torch.Tensor, Iterable, TransformParam] = [],
        **kwargs,
    ) -> torch.Tensor:
        """Transform image deterministically.

        :param torch.Tensor x: input image
        :param Union[torch.Tensor, Iterable, TransformParam] diffeo: CPAB diffeomorphism parameters.
        """
        if self.constant_batch:
            diffeo = diffeo.repeat_interleave(len(x), dim=0)

        x = torch.cat([x] * self.n_trans)

        return self.cpab.transform_data(x, diffeo, outsize=x.shape[-2:])
