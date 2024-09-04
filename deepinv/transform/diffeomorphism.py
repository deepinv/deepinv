from typing import Union, Iterable
import torch
from deepinv.transform.base import Transform, TransformParam


class CPADiffeomorphism(Transform):
    r"""
    Continuous Piecewise-Affine-based Diffeomorphism.

    Wraps CPAB from the `original implementation <https://github.com/SkafteNicki/libcpab>`_.
    From the paper Freifeld et al. `Transformations Based on Continuous Piecewise-Affine Velocity Fields <https://ieeexplore.ieee.org/abstract/document/7814343>`_.

    These diffeomorphisms benefit from fast GPU-accelerated transform + fast inverse.

    Requires installing ``libcpab`` using ``pip install git+https://github.com/Andrewwango/libcpab.git``.

    Generates n_trans randomly transformed versions.

    See :class:`deepinv.transform.Transform` for further details and examples.

    ..warning ::

        This implementation does not allow using a ``torch.Generator`` to generate reproducible transformations.
        You may be able to achieve reproducibility by using a global seed instead.

    :param int n_trans: number of transformed versions generated per input image.
    :param int n_tesselation: see ``libcpab.Cpab`` docs
    :param bool zero_boundary: see ``libcpab.Cpab`` docs
    :param bool volume_perservation: see ``libcpab.Cpab`` docs
    :param bool override: see ``libcpab.Cpab`` docs
    :param str, torch.device device: torch device.
    """

    def __init__(
        self,
        *args,
        n_tesselation=3,
        zero_boundary=True,
        volume_perservation=True,
        override=True,
        device="cpu",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        try:
            from libcpab import Cpab
        except ImportError:
            raise ImportError(
                "Install libcpab using pip install git+https://github.com/Andrewwango/libcpab.git"
            )

        self.cpab = Cpab(
            [n_tesselation, n_tesselation],
            backend="pytorch",
            device=device,
            zero_boundary=zero_boundary,
            volume_perservation=volume_perservation,
            override=override,
        )

    def _get_params(self, x: torch.Tensor) -> dict:
        return {"diffeo": self.cpab.sample_transformation(self.n_trans)}

    def _transform(
        self,
        x: torch.Tensor,
        diffeo: Union[torch.Tensor, Iterable, TransformParam] = [],
        **kwargs,
    ) -> torch.Tensor:
        B = len(x)
        x = torch.cat([x] * len(diffeo))
        diffeo = diffeo.repeat_interleave(B, dim=0)

        return torch.cat(
            [
                self.cpab.transform_data(x[[i]], diffeo[[i]], outsize=x.shape[-2:])
                for i in range(len(x))
            ]
        )
