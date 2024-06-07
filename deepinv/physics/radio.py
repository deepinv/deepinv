import torch
from deepinv.physics import LinearPhysics

try:
    import torchkbnufft as tkbn
except:
    tkbn = ImportError("The torchkbnufft package is not installed.")


class RadioInterferometry(LinearPhysics):
    r"""
    Radio Interferometry measurement operator.

    The operator handles ungridded measurements using the NUFFT, which is based in Kaiser-Bessel kernel interpolation.


    :param tuple img_size: Size of the target image, e.g., (H, W).
    :param torch.Tensor samples_loc: Normalized sampling locations in the Fourier domain (Size: N x 2).
    :param torch.Tensor dataWeight: Data weighting for the measurements (Size: N). Default is `torch.tensor([1.0])`.
    :param Union[int, Sequence[int]] interp_points: Number of neighbors to use for interpolation in each dimension. Default is `7`.
    :param float k_oversampling: Oversampling of the k space grid, should be between `1.25` and `2`. Default is `2`.
    :param bool real_projection: Apply real projection after the adjoint NUFFT. Warning: If the `real_projection` is `False`, the output of the adjoint will have a complex-typed rather than a float-typed.
    :param torch.device device: Device where the operator is computed.
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
        return (
            self.nufftObj(x.to(torch.complex64), self.samples_loc, norm="ortho")
            * self.dataWeight
        )

    def A_adjoint(self, y):
        return self.adj_projection(
            self.adjnufftObj(y * self.dataWeight, self.samples_loc, norm="ortho")
        )
