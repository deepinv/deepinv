from __future__ import annotations
from .forward import LinearPhysics
from .noise import PoissonNoise
import torch
from typing import TYPE_CHECKING
import contextlib
import io

if TYPE_CHECKING:
    import parallelproj


class PET(LinearPhysics):
    r"""
    Non time-of-flight Positron emission tomography (PET) physics model.

    This operator relies on the `parallelproj` library by :footcite:t:`schramm2024parallelproj`.

    The PET forward model is defined as

    .. math::

        y \sim \gamma \mathcal{P}(\frac{c \circ H(g*x) + b}{\gamma})

    where :math:`H \in \mathbb{R}_{+}^{m \times n}` is the projection operator,
    :math:`g \in \mathbb{R}_{+}^{n}` is a Gaussian blur kernel, :math:`x\in\mathbb{R}_{+}^{n}`
    is the emission image, :math:`b \in \mathbb{R}_{+}^{m}` is the (expected) background,
    :math:`\mathcal{P}` denotes Poisson noise with gain :math:`\gamma > 0`,
    :math:`c=\exp(-H\mu)\in \mathbb{R}_{+}^{m}` is an (optional) attenuation term
    with :math:`\mu \in \mathbb{R}_{+}^{n}` an attenuation map (typically obtained through an auxiliary CT scan).

    The operator **can be used on 2D images or 3D volumes**.

    The operator relies on parameters `background` and `attenuation` that can be updated through the
    :meth:`physics.update <deepinv.physics.Physics.update>` method or when evaluating
    :meth:`physics.A <deepinv.physics.Physics.A>` or :meth:`physics.A_adjoint <deepinv.physics.LinearPhysics.A_adjoint>`.

    .. note::

        This operator requires the `parallelproj` package to be installed.
        This in turn requires :ref:`installing deepinv via pixi or conda <install>`,
        but not pypi/uv (as `parallelproj` is not currently available on pypi).

        If you are working on a conda environment, you can install `parallelproj` as

        ::

            conda install -c conda-forge parallelproj


        If you are working on a pixi installation, simply do

        ::

            pixi install -e full

        which installs all optional dependencies.

        Check the `parallelproj` documentation for more details: https://parallelproj.readthedocs.io/en/stable/.


    .. tip::

        Check out the :ref:`2D <sphx_glr_auto_examples_physics_demo_pet2d.py>` and
        :ref:`3D <sphx_glr_auto_examples_physics_demo_pet3d.py>` examples to get started with this operator.

    .. note::

        This operator is defined to work on sinogram (binned) data.
        List-mode data is not supported yet, but it can be pre-binned into a sinogram using `parallelproj`.

    :param tuple img_size: shape of the input 2D `(H, W)` or 3D volumes `(D, H, W)`.
    :param tuple voxel_size: voxel size in mm. Default is 2 x 2 x 2 mm.
    :param float fwhm_data_mm: full width at half maximum (FWHM) of the Gaussian blur :math:`g`. It has a crucial impact on the maximum achievable resolution,
        which is typically a fraction of the FWHM.
    :param None, parallelproj.pet_scanners.ModularizedPETScannerGeometry scanner: Scanner configuration. If None, the default scanner from parallelproj is used.
    :param int radial_trim: radial trim of rays on the sides of the volume to improve efficiency.
    :param float gain: gain factor :math:`\gamma` for the Poisson noise model.
    :param bool normalize: If `True` the forward operator is normalized such that :math:`\|A\|=1`.
    :param bool normalize_counts: If `False` the :math:`\gamma` term in front of the Poisson noise is removed,
        that is the measurements are true counts.
    :param str | torch.device device: device to run the computations on, e.g. `"cpu"` or `"cuda"`
    :param torch.Tensor background: background sinogram :math:`b`, i.e. the expected number of background events in each LOR, with shape `(num_lors,)`
    :param torch.Tensor attenuation: attenuation map. Can be provided either in **image space** as :math:`\mu`
        (linear attenuation coefficients, shape `(H,W)` for 2D or `(D,H,W)` for 3D — typically from an auxiliary CT scan),
        or in **sinogram/projection space** as :math:`c=\exp(-H\mu)`. The space is inferred automatically
        by comparing the spatial dimensions of the tensor against `img_size`: if they match, image space is assumed
        and the attenuation is projected; otherwise, sinogram space is assumed and the tensor is used directly.
        Providing the attenuation in image space allows computing gradients with respect to it efficiently.

    |sep|

    :Example:

    Simulate 2D PET measurements

    >>> from deepinv.physics import PET
    >>> import torch
    >>> img_size = (64, 64)
    >>> physics = PET(img_size=img_size)
    >>> x = torch.rand((1, 1) + img_size)
    >>> background = torch.ones_like(physics.A(x))
    >>> attenuation = torch.rand((1, 1,) + img_size)
    >>> y = physics(x, attenuation=attenuation, background=background)
    >>> y.shape
    torch.Size([1, 1, 539, 272])

    """

    def __init__(
        self,
        img_size: tuple,
        voxel_size: tuple = (2, 2, 2),
        fwhm_data_mm: float | tuple = 4.0,
        scanner: None | parallelproj.pet_scanners.ModularizedPETScannerGeometry = None,
        radial_trim: int = 3,
        gain: float = 1.0,
        normalize: bool = False,
        normalize_counts: bool = False,
        device: str | torch.device = "cpu",
        background: torch.Tensor | None = None,
        attenuation: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(img_size, tuple):
            # remove first entries = 1
            while len(img_size) > 0 and img_size[0] == 1:
                img_size = img_size[1:]
        if not isinstance(img_size, tuple) or not len(img_size) in (2, 3):
            raise ValueError(
                "img_size must be a tuple of length 2 or 3, e.g. (H, W) or (D, H, W)"
            )

        try:  # avoids doctest failing when parallelproj prints banner
            with contextlib.redirect_stdout(io.StringIO()):
                import parallelproj
                from array_api_compat import torch as torch_compat
        except ImportError:
            raise ImportError(
                "parallelproj package is required for PET physics model. "
                "You can install it via conda: conda install -c conda-forge parallelproj. "
                "Check the parallelproj documentation for more details: https://parallelproj.readthedocs.io/en/stable/."
            )

        if isinstance(voxel_size, (int, float)):
            voxel_size = (voxel_size, voxel_size, voxel_size)
        elif len(voxel_size) == 2:
            voxel_size = voxel_size + (voxel_size[-1],)

        self.img_size = img_size

        if len(img_size) == 2:
            img_size = img_size + (1,)
            self.is_2d = True
        else:
            self.is_2d = False

        if scanner is None:
            scanner = parallelproj.pet_scanners.DemoPETScannerGeometry(
                torch_compat, dev=device, num_rings=1 if self.is_2d else 16
            )

        # setup the LOR descriptor that defines the sinogram
        lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
            scanner,
            radial_trim=radial_trim,
            sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
        )

        self.proj = parallelproj.RegularPolygonPETProjector(
            lor_desc, img_shape=img_size, voxel_size=voxel_size
        )

        if background is not None:
            background = background.to(device)
        else:
            background = (
                self.proj(torch.zeros(img_size, device=device))
                .unsqueeze(0)
                .unsqueeze(0)
            )
            if self.is_2d:
                background = background.squeeze(-1)

        # Default attenuation to zero in image space (no attenuation)
        if attenuation is None:
            attenuation = torch.zeros((1, 1) + self.img_size, device=device)

        self.res_model = parallelproj.GaussianFilterOperator(
            img_size, sigma=fwhm_data_mm / (2.35 * self.proj.voxel_size)
        )
        self.pet_lin_op = parallelproj.CompositeLinearOperator(
            (self.proj, self.res_model)
        )

        self.normalize = normalize

        self.register_buffer("operator_norm", torch.ones(1, device=device))
        self.register_buffer("background", background)
        self.register_buffer("attenuation", attenuation)
        self.update_parameters(background=background, attenuation=attenuation)
        self.noise_model = PoissonNoise(gain=gain, normalize=normalize_counts)
        self.to(device)

    def A(
        self,
        x: torch.Tensor,
        add_background: bool = False,
        background: torch.Tensor | None = None,
        attenuation: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Apply the linear operator :math:`Ax=c \circ H(g*x)` to a signal :math:`x`

        :param torch.Tensor x: input image or volume
        :param torch.Tensor add_background: whether to add background :math:`s`. By default, no background is added.
        :param torch.Tensor background: If not `None`, update the background :math:`s` of the operator.
        :param torch.Tensor attenuation: If not `None`, update the attenuation of the operator.
            The space (image or sinogram) is inferred automatically from the tensor shape.

        """
        if x.shape[1] != 1:
            raise ValueError(
                f"Input volume must have 1 channel, got {x.shape[1]} channels"
            )
        self.update_parameters(attenuation=attenuation, background=background)
        attenuation = self.attenuation
        if self.is_2d:
            x = x.unsqueeze(-1)
            attenuation = attenuation.unsqueeze(-1)

        out = LinearSingleChannelOperator.apply(x, self.pet_lin_op) * attenuation
        if self.is_2d:
            out = out.squeeze(-1)

        out /= self.operator_norm

        if add_background:
            out = out + self.background
        return out

    def A_adjoint(
        self, y: torch.Tensor, attenuation=None, background=None, **kwargs
    ) -> torch.Tensor:
        r"""
        Apply the adjoint of the linear operator :math:`A^{\top}y` where :math:`A=c \circ H(g*\cdot)` to a sinogram :math:`y`

        :param torch.Tensor y: input sinogram
        :param torch.Tensor attenuation: If not `None`, update the attenuation
        :param torch.Tensor background: If not `None`, update the background
        """
        if y.shape[1] != 1:
            raise ValueError(
                f"Input measurements must have 1 channel, got {y.shape[1]} channels"
            )
        self.update_parameters(attenuation=attenuation, background=background)
        attenuation = self.attenuation
        if self.is_2d:
            y = y.unsqueeze(-1)
            attenuation = attenuation.unsqueeze(-1)
        out = (
            AdjointLinearSingleChannelOperator.apply(y * attenuation, self.pet_lin_op)
            / self.operator_norm
        )
        if self.is_2d:
            out = out.squeeze(-1)
        return out

    def plot_geometry(self):
        r"""
        Plot the scanner geometry.

        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        self.proj.show_geometry(ax)
        # set labels of axes
        ax.set_xlabel("mm")
        ax.set_ylabel("mm")
        ax.set_zlabel("mm")
        fig.tight_layout()
        fig.show()

    def forward(
        self, x: torch.Tensor, attenuation=None, background=None, **kwargs
    ) -> torch.Tensor:
        r"""
        Generate PET measurements.

        :param torch.Tensor x: input image or volume
        :param torch.Tensor attenuation: If not `None`, update the attenuation
        :param torch.Tensor background: If not `None`, update the background
        """
        self.update_parameters(attenuation=attenuation, background=background)
        return self.noise_model(self.A(x, **kwargs, add_background=True))

    def generate_background(self, expected_background: torch.Tensor) -> torch.Tensor:
        r"""
        Generate a random PET background based on the expected background.

        :param torch.Tensor expected_background: Expected background.
        """
        return self.noise_model(expected_background)

    def update_parameters(
        self,
        attenuation: torch.Tensor | None = None,
        background: torch.Tensor | None = None,
        **kwargs,
    ):
        r"""
        Update the background and/or attenuation parameters.

        The space of the attenuation tensor is inferred automatically: if the last
        ``len(img_size)`` dimensions match ``img_size``, the tensor is treated as an
        image-space attenuation map :math:`\mu` and projected; otherwise it is treated as
        a sinogram-space attenuation :math:`c=\exp(-H\mu)` and used directly.

        :param torch.Tensor attenuation: If not `None`, update the attenuation. Can be in
            image space (shape matching `img_size`) or sinogram space.
        :param torch.Tensor background: If not `None`, update the background :math:`b`.
        """
        if attenuation is not None:
            n = len(self.img_size)
            is_image_space = tuple(attenuation.shape[-n:]) == tuple(self.img_size)
            if is_image_space:
                if self.is_2d:
                    attenuation = attenuation.unsqueeze(-1)

                proj_att = LinearSingleChannelOperator.apply(attenuation, self.proj)
                if self.is_2d:
                    proj_att = proj_att.squeeze(-1)
                self.attenuation = torch.exp(-proj_att)
            else:
                self.attenuation = attenuation

            if self.normalize:
                self.operator_norm = torch.ones(1, device=self.attenuation.device)
                self.operator_norm = self.compute_norm(
                    torch.ones((1, 1) + self.img_size, device=self.attenuation.device),
                    squared=False,
                    verbose=False,
                )
        if background is not None:
            self.background = background


class LinearSingleChannelOperator(torch.autograd.Function):
    """
    Function representing a linear operator acting on a mini batch of single channel images
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, operator: parallelproj.LinearOperator
    ) -> torch.Tensor:
        r"""
        Forward pass for a mini-batch of 3D images using a linear operator.

        :param ctx: context object
            Context object that can be used to store information for the backward pass.
        :param x: torch.Tensor
            Mini-batch of 3D images with dimension (batch_size, 1, num_voxels_x, num_voxels_y, num_voxels_z).
        :param operator: parallelproj.LinearOperator
            Linear operator that can act on a single 3D image.

        :return: torch.Tensor
            Mini-batch of 3D images with dimension (batch_size, 1, operator.out_shape).
        """
        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        ctx.set_materialize_grads(False)
        ctx.operator = operator

        batch_size = x.shape[0]
        y = torch.zeros(
            (batch_size, 1) + operator.out_shape, dtype=x.dtype, device=x.device
        )

        for i in range(batch_size):
            y[i, 0, ...] = operator(x[i, 0, ...].detach())

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        r"""
        Backward pass of the forward operation.

        :param ctx: context object
            Context object that can be used to obtain information from the forward pass.
        :param grad_output: torch.Tensor
            Mini-batch of gradients with dimension (batch_size, 1, operator.out_shape).

        :return: tuple[torch.Tensor, None]
            Mini-batch of 3D images with dimension (batch_size, 1, operator.img_size), and None.
        """
        # For details on how to implement the backward pass, see
        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use

        # since forward takes two input arguments (x, operator)
        # we have to return two arguments (the latter is None)
        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator

            batch_size = grad_output.shape[0]
            x = torch.zeros(
                (batch_size, 1) + operator.in_shape,
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

            for i in range(batch_size):
                x[i, 0, ...] = operator.adjoint(grad_output[i, 0, ...].detach())

            return x, None


# %%
# Setup the back projection layer
# -------------------------------
#
# We subclass :class:`torch.autograd.Function` to define a custom pytorch layer
# that is compatible with pytorch's autograd engine.
# see also: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html


class AdjointLinearSingleChannelOperator(torch.autograd.Function):
    """
    Function representing the adjoint of a linear operator acting on a mini batch of single channel images
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, operator: parallelproj.LinearOperator
    ) -> torch.Tensor:
        """forward pass of the adjoint of the linear operator

        Parameters
        ----------
        ctx : context object
            that can be used to store information for the backward pass
        x : torch.Tensor
            mini batch of 3D images with dimension (batch_size, 1, operator.out_shape)
        operator : parallelproj.LinearOperator
            linear operator that can act on a single 3D image

        Returns
        -------
        torch.Tensor
            mini batch of 3D images with dimension (batch_size, 1, opertor.img_size)
        """

        ctx.set_materialize_grads(False)
        ctx.operator = operator

        batch_size = x.shape[0]
        y = torch.zeros(
            (batch_size, 1) + operator.in_shape, dtype=x.dtype, device=x.device
        )

        # loop over all samples in the batch and apply linear operator
        # to the first channel
        for i in range(batch_size):
            y[i, 0, ...] = operator.adjoint(x[i, ...].detach())

        return y

    @staticmethod
    def backward(ctx, grad_output):
        """backward pass of the forward pass

        Parameters
        ----------
        ctx : context object
            that can be used to obtain information from the forward pass
        grad_output : torch.Tensor
            mini batch of dimension (batch_size, 1, operator.img_size)

        Returns
        -------
        torch.Tensor, None
            mini batch of 3D images with dimension (batch_size, 1, opertor.out_shape)
        """
        # For details on how to implement the backward pass, see
        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use

        # since forward takes two input arguments (x, operator)
        # we have to return two arguments (the latter is None)
        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator

            batch_size = grad_output.shape[0]
            x = torch.zeros(
                (batch_size,) + operator.out_shape,
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

            # loop over all samples in the batch and apply linear operator
            # to the first channel
            for i in range(batch_size):
                x[i, ...] = operator(grad_output[i, 0, ...].detach())

            return x, None
