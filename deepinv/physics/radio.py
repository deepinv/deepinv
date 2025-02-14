import torch
from deepinv.physics import LinearPhysics

try:
    import torchkbnufft as tkbn
except:
    tkbn = ImportError("The torchkbnufft package is not installed.")


class RadioInterferometry(LinearPhysics):
    r"""
    Radio Interferometry measurement operator.

    The operator handles ungridded measurements using the non-uniform FFT (NUFFT), which is based in Kaiser-Bessel
    kernel interpolation. This particular implementation relies on the `torchkbnufft <https://github.com/mmuckley/torchkbnufft>`_ package.

    The forward operator is defined as :math:`A:x \mapsto y`,
    where :math:`A` can be decomposed as :math:`A = GFZ \in \mathbb{C}^{m \times n}`.
    There, :math:`G \in \mathbb{C}^{m \times d}` is a sparse interpolation matrix,
    encoding the non-uniform Fourier transform,
    :math:`F \in \mathbb{C}^{d\times d}` is the 2D Discrete orthonormal Fourier Transform,
    :math:`Z \in \mathbb{R}^{d\times n}` is a zero-padding operator,
    incorporating the correction for the convolution performed through the operator :math:`G`.

    :param tuple img_size: Size of the target image, e.g., (H, W).
    :param torch.Tensor samples_loc: Normalized sampling locations in the Fourier domain (Size: N x 2).
    :param torch.Tensor dataWeight: Data weighting for the measurements (Size: N). Default is ``torch.tensor([1.0])`` (i.e. no weighting).
    :param Union[int, Sequence[int]] interp_points: Number of neighbors to use for interpolation in each dimension. Default is ``7``.
    :param float k_oversampling: Oversampling of the k space grid, should be between ``1.25`` and ``2``. Default is ``2``.
    :param bool real_projection: Apply real projection after the adjoint NUFFT. Default is ``True``.
    :param torch.device device: Device where the operator is computed. Default is ``cpu``.

    .. warning::
        If the ``real_projection`` parameter is set to ``False``, the output of the adjoint will have a complex type rather than a real typed.

    """

    def __init__(
        self,
        img_size,
        samples_loc,
        dataWeight=torch.tensor(
            [
                1.0,
            ]
        ),
        k_oversampling=2,
        interp_points=7,
        real_projection=True,
        device="cpu",
        **kwargs,
    ):
        super(RadioInterferometry, self).__init__(**kwargs)

        self.device = device
        self.k_oversampling = k_oversampling
        self.interp_points = interp_points
        self.img_size = img_size
        self.real_projection = real_projection

        # Check image size format
        assert len(self.img_size) == 2

        # Define oversampled grid
        self.grid_size = (
            int(img_size[0] * self.k_oversampling),
            int(img_size[1] * self.k_oversampling),
        )

        self.samples_loc = samples_loc.to(self.device)
        self.dataWeight = dataWeight.to(self.device)

        self.nufftObj = tkbn.KbNufft(
            im_size=self.img_size,
            grid_size=self.grid_size,
            numpoints=self.interp_points,
            device=self.device,
        )
        self.adjnufftObj = tkbn.KbNufftAdjoint(
            im_size=self.img_size,
            grid_size=self.grid_size,
            numpoints=self.interp_points,
            device=self.device,
        )

        # Define adjoint operator projection
        if self.real_projection:
            self.adj_projection = lambda x: torch.real(x).to(torch.float)
        else:
            self.adj_projection = lambda x: x

    def setWeight(self, w):
        self.dataWeight = w.to(self.device)

    def A(self, x):
        r"""
        Applies the weighted NUFFT operator to the input image.

        :param torch.Tensor x: input image
        :return: (:class:`torch.Tensor`) containing the measurements
        """
        return (
            self.nufftObj(x.to(torch.cfloat), self.samples_loc, norm="ortho")
            * self.dataWeight
        )

    def A_adjoint(self, y):
        r"""
        Applies the adjoint of the weighted NUFFT operator.

        :param torch.Tensor y: input measurements
        :return: (:class:`torch.Tensor`) containing the reconstructed image
        """
        return self.adj_projection(
            self.adjnufftObj(y * self.dataWeight, self.samples_loc, norm="ortho")
        )
