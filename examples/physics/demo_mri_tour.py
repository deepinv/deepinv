import torch

from deepinv.physics.forward import LinearPhysics
from deepinv.utils.mixins import MRIMixin

class NonCartesianMRI(LinearPhysics, MRIMixin):
    """
    Non-Cartesian MRI via `mri-nufft`.

    Provides a simple for non-Cartesian linear operators via `mri-nufft`, allowing quick initialisation
    of spiral and radial sampling trajectories.

    For more advanced trajectories and multi-coil operator, override this function and provide your own
    `nufft` operator and `density` compensation vector. See `MRI-NUFFT docs <https://mind-inria.github.io/mri-nufft>`_ for more details.

    .. note:
        This operator handles batching via looping.
    
    .. note:
        This operator uses Voronoi density compensation by default.
        Optionally set `density_adjointness=True` so that the forward matches the adjoint,
        by simulating density compensation in the forward too
        (important for self-adjoint normal operators).
    
    :param int num_shots: number of sampling shots.
    :param int num_samples_per_shot: number of samples per shot.
    :param tuple[int, int] img_size: reconstructed image size `(H, W)`.
    :param str backend: mri-nufft backend, defaults to `finufft` (for CPU). Set to `cufinufft` for CUDA.
        `MRI-NUFFT docs <https://mind-inria.github.io/mri-nufft>`_ for more details.
    :param str trajectory: mri-nufft trajectory, defaults to "radial".
        See `MRI-NUFFT docs <https://mind-inria.github.io/mri-nufft>`_ for more details.
    :param bool normalize: if `True`, normalize operator by its empirical spectral norm.
    :param str density_mode: if `compensate` (default), apply Voronoi density compensation before adjoint.
        If `adjointness`, simulate sqrt density in forward and perform sqrt density in adjoint, such that
        adjointness holds exactly. If `None`, no density compensation performed.
    :param torch.device device: device on which to calculate norm and initialise density, defaults to "cpu"

    :Example:

    >>> from deepinv.physics import NonCartesianMRI
    >>> physics = NonCartesianMRI()

    """
    def __init__(
        self,
        num_shots: int,
        num_samples_per_shot: int,
        img_size: tuple[int, int] = (320, 320),
        backend: str = "finufft",
        trajectory: str = "radial",
        normalize: bool = True,
        density_mode: str = "compensate",
        device: torch.device = "cpu",
        **kwargs
    ):
        super().__init__(**kwargs)

        try:
            import mrinufft
        except ImportError:
            raise ImportError(
                "mri-nufft is required for NonCartesianMRI. Install it using `pip install mrinufft[finufft]`" \
                " for CPU (finufft) backend, or `pip install mrinufft[cufinufft]` for GPU backend."
            )
        
        if trajectory == "radial":
            self.samples_loc = mrinufft.initialize_2D_radial(Nc=num_shots, Ns=num_samples_per_shot)
        elif trajectory == "spiral":
            self.samples_loc = mrinufft.initialize_2D_spiral(Nc=num_shots, Ns=num_samples_per_shot, nb_revolutions=1)
        
        self.backend = backend
        self.nufft = mrinufft.get_operator(self.backend)(self.samples_loc, shape=img_size, n_coils=1, n_batchs=1) # we do batching ourselves

        self.density_mode = density_mode
        if self.density_mode is None:
            self.density = None
        elif self.density_mode in ["compensate", "adjointness"]:
            self.density = torch.from_numpy(mrinufft.density.voronoi(self.samples_loc)).to(device).view(1, 1, -1) # 111S
        else:
            raise ValueError("'density_mode' must be None, 'compensate' or 'adjointness'")

        self.operator_norm = 1
        if normalize:
            self.operator_norm = self.compute_norm(torch.randn(1, 2, *img_size[-2:], device=device), squared=False)

    def A(self, x: torch.Tensor) -> torch.Tensor: # B2HW -> B2S
        """
        Forward NUFFT operator.

        .. note:
            Handles the finufft and cufinufft cases separately to ensure shapes are correct.
        
        :param torch.Tensor x: input gridded image of shape B,2,H,W
        :return: :class:`torch.Tensor` NUFFT measurements of shape B,2,S where S is number of samples
        """
        outs = []
        
        for _x in x.double():
            _x = _x.unsqueeze(0) # 12HW

            if self.backend == "finufft":
                out = self.from_torch_complex(
                    self.nufft.op(
                        self.to_torch_complex(_x) # 1HW
                    ).unsqueeze(0) # 1S
                ) # 12S
            
            elif self.backend == "cufinufft":
                out = self.from_torch_complex(
                    self.nufft.op(
                        self.to_torch_complex(_x)#.unsqueeze(1) # 11HW # TODO is unsqueeze here needed?
                    ) # 11S
                ) # 121S # should be 12S
            
            outs += [out]

        y = torch.cat(outs).float() / self.operator_norm

        if self.density_mode and self.density_mode == "adjointness":
            y *= self.density.sqrt()
        else:
            pass

        return y

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor: # B2S -> B2HW
        """
        Adjoint NUFFT operator.

        .. note:
            Handles the finufft and cufinufft cases separately to ensure shapes are correct.
        
        :param torch.Tensor y: NUFFT measurements of shape B,2,S where S is number of samples input
        :return: :class:`torch.Tensor` gridded image of shape B,2,H,W
        """
        if self.density_mode and self.density_mode == "adjointness":
            y_in = y * self.density.sqrt()
        elif self.density_mode and self.density_mode == "compensate":
            y_in = y * self.density
        else:
            y_in = y

        outs = []
        for _y in y_in:
            _y = _y.unsqueeze(0) # 12S

            if self.backend == "finufft":
                out = self.from_torch_complex(
                    self.nufft.adj_op(
                        self.to_torch_complex(_y) # 1S
                    ).unsqueeze(0) # 1HW
                ) # 12HW
            elif self.backend == "cufinufft":
                out = self.from_torch_complex(
                    self.nufft.adj_op(
                        self.to_torch_complex(_y) # 1S
                    ) # 1HW
                ) # 12HW
            
            outs += [out]

        return torch.cat(outs).float() / self.operator_norm


r"""
Tour of MRI functionality in DeepInverse
========================================

This example presents the various datasets, forward physics and models
available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:

-  Physics: :class:`deepinv.physics.MRI`,
   :class:`deepinv.physics.MultiCoilMRI`,
   :class:`deepinv.physics.DynamicMRI`
-  Datasets: raw kspace with the `FastMRI <https://fastmri.med.nyu.edu>`__ dataset
   :class:`deepinv.datasets.FastMRISliceDataset` and an in-memory easy-to-use version
   :class:`deepinv.datasets.SimpleFastMRISliceDataset`, and raw dynamic k-t-space data with the
   `CMRxRecon <https://cmrxrecon.github.io>`__ dataset.
-  Models: :class:`deepinv.models.VarNet`
   (VarNet :footcite:t:`hammernik2018learning`, E2E-VarNet :footcite:t:`sriram2020end`),
   :class:`deepinv.models.MoDL` (a simple MoDL :footcite:t:`aggarwal2018modl` unrolled model)

Contents:

1. Get started with FastMRI (singlecoil + multicoil)
2. Train an accelerated MRI with neural networks
3. Load raw FastMRI data (singlecoil + multicoil)
4. Train using raw data
5. Explore 3D MRI
6. Explore dynamic MRI

"""

# %%
import deepinv as dinv
import torch, torchvision
from torch.utils.data import DataLoader

device = dinv.utils.get_device()
rng = torch.Generator(device=device).manual_seed(0)


# %%
# 1. Get started with FastMRI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# You can get started with our simple
# `FastMRI <https://fastmri.med.nyu.edu>`__ mini slice subsets which provide
# quick, easy-to-use, in-memory datasets which can be used for simulation
# experiments.
#
# .. important::
#
#    By using this dataset, you confirm that you have agreed to and signed the `FastMRI data use agreement <https://fastmri.med.nyu.edu/>`_.
#
# .. seealso::
#
#   Datasets :class:`deepinv.datasets.FastMRISliceDataset` :class:`deepinv.datasets.SimpleFastMRISliceDataset`
#       We provide convenient datasets to easily load both raw and reconstructed FastMRI images.
#       You can download more data on the `FastMRI site <https://fastmri.med.nyu.edu/>`_.
#
# Load mini demo knee and brain datasets (original data is 320x320 but we resize to
# 128 for speed):
#

transform = torchvision.transforms.Resize(128)
knee_dataset = dinv.datasets.SimpleFastMRISliceDataset(
    dinv.utils.get_data_home(),
    anatomy="knee",
    transform=transform,
    train=True,
    download=True,
)
brain_dataset = dinv.datasets.SimpleFastMRISliceDataset(
    dinv.utils.get_data_home(),
    anatomy="brain",
    transform=transform,
    train=True,
    download=True,
)

img_size = knee_dataset[0].shape[-2:]  # (128, 128)
dinv.utils.plot({"knee": knee_dataset[0], "brain": brain_dataset[0]})


# # %%
# # Let's start with single-coil MRI. We can define a constant Cartesian 4x
# # undersampling mask by sampling once from a physics generator. The mask,
# # data and measurements will all be of shape ``(B, C, H, W)`` where
# # ``C=2`` is the real and imaginary parts.
# #

# physics_generator = dinv.physics.generator.GaussianMaskGenerator(
#     img_size=img_size, acceleration=4, rng=rng, device=device
# )
# mask = physics_generator.step()["mask"]

# physics = dinv.physics.MRI(mask=mask, img_size=img_size, device=device)

# x = next(iter(DataLoader(knee_dataset))).to(device)

# dinv.utils.plot(
#     {
#         "x": x,
#         "mask": mask,
#         "y": physics(x).clamp(-1, 1),
#     }
# )
# print("Shapes:", x.shape, physics.mask.shape)


# # %%
# # We can next generate an accelerated single-coil MRI measurement dataset. Let's use knees
# # for training and brains for testing.
# #
# # We can also use the physics generator to randomly sample a new mask per
# # sample, and save the masks alongside the measurements.
# #
# # Note that you could alternatively train using `online_measurements`, where you can generate
# # random measurements on the fly.
# #

# dataset_path = dinv.datasets.generate_dataset(
#     train_dataset=knee_dataset,
#     test_dataset=brain_dataset,
#     val_dataset=None,
#     physics=physics,
#     physics_generator=physics_generator,
#     save_physics_generator_params=True,
#     overwrite_existing=False,
#     device=device,
#     save_dir=dinv.utils.get_data_home(),
#     batch_size=1,
# )

# train_dataset = dinv.datasets.HDF5Dataset(
#     dataset_path, split="train", load_physics_generator_params=True
# )
# test_dataset = dinv.datasets.HDF5Dataset(
#     dataset_path, split="test", load_physics_generator_params=True
# )

# train_dataloader = DataLoader(train_dataset)
# iterator = iter(train_dataloader)

# x0, y0, params0 = next(iterator)
# x1, y1, params1 = next(iterator)

# dinv.utils.plot(
#     {
#         "x0": x0,
#         "mask0": params0["mask"],
#         "x1": x1,
#         "mask1": params1["mask"],
#     }
# )


# # %%
# # We can also simulate multicoil MRI data. Either pass in ground-truth
# # coil maps, or pass an integer to simulate simple birdcage coil maps. The
# # measurements ``y`` are now of shape ``(B, C, N, H, W)``, where ``N`` is
# # the coil-dimension.
# #

# mc_physics = dinv.physics.MultiCoilMRI(img_size=img_size, coil_maps=3, device=device)

# dinv.utils.plot(
#     {
#         "x": x,
#         "mask": mask,
#         "coil_map_0": mc_physics.coil_maps.abs()[:, 0, ...],
#         "coil_map_1": mc_physics.coil_maps.abs()[:, 1, ...],
#         "coil_map_2": mc_physics.coil_maps.abs()[:, 2, ...],
#         "RSS": mc_physics.A_adjoint_A(x, mask=mask, rss=True),
#     }
# )


# # %%
# # 2. Train an accelerated MRI problem with neural networks
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #
# # Next, we train a neural network to solve the MRI inverse problem. We provide various
# # models specifically used for MRI reconstruction. These are unrolled
# # networks which require a backbone denoiser, such as UNet or DnCNN:
# #

# denoiser = dinv.models.UNet(
#     in_channels=2,
#     out_channels=2,
#     scales=2,
# )

# denoiser = dinv.models.DnCNN(
#     in_channels=2,
#     out_channels=2,
#     pretrained=None,
#     depth=2,
# )


# # %%
# # These backbones can be used within specific MRI models, such as
# # VarNet :footcite:t:`hammernik2018learning`, E2E-VarNet :footcite:t:`sriram2020end` and MoDL :footcite:t:`aggarwal2018modl`,
# # for which we provide implementations:
# #

# model = dinv.models.VarNet(denoiser, num_cascades=2, mode="varnet").to(device)

# model = dinv.models.MoDL(denoiser, num_iter=2).to(device)


# # %%
# # Now that we have our architecture defined, we can train it with supervised or self-supervised (using Equivariant
# # Imaging) loss. We use the PSNR metric on the complex magnitude.
# #
# # For the sake of speed in this example, we only use a very small 2-layer DnCNN inside an unrolled
# # network with 2 cascades, and train with 2 images for 1 epoch.
# #

# loss = dinv.loss.SupLoss()
# loss = dinv.loss.EILoss(transform=dinv.transform.CPABDiffeomorphism())

# trainer = dinv.Trainer(
#     model=model,
#     physics=physics,
#     optimizer=torch.optim.Adam(model.parameters()),
#     train_dataloader=train_dataloader,
#     metrics=dinv.metric.PSNR(complex_abs=True),
#     epochs=1,
#     show_progress_bar=False,
#     save_path=None,
# )

# # %%
# # To improve results in the case of this very short training, we start training from a pretrained model state (trained on 900 images):

# url = dinv.models.utils.get_weights_url(
#     model_name="demo", file_name="demo_tour_mri.pth"
# )
# ckpt = torch.hub.load_state_dict_from_url(
#     url, map_location=lambda storage, loc: storage, file_name="demo_tour_mri.pth"
# )
# trainer.model.load_state_dict(ckpt["state_dict"])  # load the state dict
# trainer.optimizer.load_state_dict(ckpt["optimizer"])  # load the optimizer state dict

# model = trainer.train()  # train the model
# trainer.plot_images = True


# # %%
# # Now that our model is trained, we can test it. Notice that we improve the PSNR compared to the zero-filled
# # reconstruction, both on the train (knee) set and the test (brain) set:

# # sphinx_gallery_start_ignore
# # sphinx_gallery_multi_image = "single"
# # sphinx_gallery_end_ignore

# _ = trainer.test(train_dataloader)

# _ = trainer.test(DataLoader(test_dataset))


# # %%
# # 3. Load raw FastMRI data
# # ~~~~~~~~~~~~~~~~~~~~~~~~
# #
# # It is also possible to use the raw data directly.
# # The raw multi-coil FastMRI train/validation data is provided as pairs of ``(x, y)`` where
# # ``y`` are the fully-sampled k-space measurements of arbitrary size, and
# # ``x`` are the cropped root-sum-square (RSS) magnitude reconstructions.
# # Let's download a sample volume and check out its middle slice.
# #

# dinv.datasets.download_archive(
#     dinv.utils.get_image_url("demo_fastmri_brain_multicoil.h5"),
#     dinv.utils.get_data_home() / "brain" / "fastmri.h5",
# )

# dataset = dinv.datasets.FastMRISliceDataset(
#     dinv.utils.get_data_home() / "brain", slice_index="middle"
# )

# x, y = next(iter(DataLoader(dataset)))
# x = x.to(device)
# y = y.to(device)

# img_size, kspace_shape = x.shape[-2:], y.shape[-2:]
# n_coils = y.shape[2]

# print("Shapes:", x.shape, y.shape)  # x (B, 1, W, W); y (B, C, N, H, W)

# # %%
# # Note that we can relate ``x`` and fully-sampled ``y`` using our
# # :class:`deepinv.physics.MultiCoilMRI` (note that since we are not
# # provided with the ground-truth coil-maps, we can only perform the
# # adjoint operator).
# #

# physics = dinv.physics.MultiCoilMRI(
#     img_size=img_size,
#     mask=torch.ones(kspace_shape, device=device),
#     coil_maps=torch.ones(
#         (n_coils,) + kspace_shape, dtype=torch.complex64, device=device
#     ),
#     device=device,
# )

# x_rss = physics.A_adjoint(y, rss=True, crop=True)

# assert torch.allclose(x, x_rss)

# # %%
# # We can also pre-estimate coil sensitivity maps using ESPIRiT from the raw data.
# #

# dataset = dinv.datasets.FastMRISliceDataset(
#     dinv.utils.get_data_home() / "brain",
#     slice_index="middle",
#     transform=dinv.datasets.MRISliceTransform(
#         estimate_coil_maps=True,
#         acs=15,  # Num. low frequency, fix to 15
#     ),
# )

# x, y, params = next(iter(DataLoader(dataset)))

# physics.update(**params)

# dinv.utils.plot(
#     {"x": x, "maps0": physics.coil_maps[:, 0], "maps1": physics.coil_maps[:, 1]}
# )

# # %%
# # 4. Train using raw data
# # ~~~~~~~~~~~~~~~~~~~~~~~
# #
# # For training with multicoil raw data, we can simulate random masks **on-the-fly**:
# #

# dataset = dinv.datasets.FastMRISliceDataset(
#     dinv.utils.get_data_home() / "brain",
#     slice_index="middle",
#     transform=dinv.datasets.MRISliceTransform(
#         mask_generator=dinv.physics.generator.GaussianMaskGenerator(
#             img_size=kspace_shape,
#             acceleration=4,
#         ),
#         seed_mask_generator=False,  # More diversity during training
#         estimate_coil_maps=False,  # Set to true if coil maps are not already set in physics.
#         # This will use ACS size from mask generator. If mask generator is None, then try find ACS size from metadata.
#     ),
# )

# # %%
# #
# # Note if the data is already undersampled raw kspace data (e.g. FastMRI test set)
# # you can also easily directly load it and their associated masks for testing or training
# # (optionally specify separate target folder if targets are in a different folder):
# #
# # ::
# #
# #         dataset = dinv.datasets.FastMRISliceDataset(
# #             root=root,
# #             target_root=target_root,
# #             transform=dinv.datasets.MRISliceTransform()
# #         )
# #
# # We use the E2E-VarNet model designed for
# # multicoil MRI. For this example, we do not perform joint coil sensitivity map estimation and
# # simply assume they are flat. If you want to estimate the maps, either pass a model
# # as the ``sensitivity_model`` parameter, or use a different model which uses precomputed maps.
# #

# model = dinv.models.VarNet(denoiser, num_cascades=2, mode="e2e-varnet").to(device)

# # %%
# # We also need to modify the metrics used to crop the model output and take the magnitude when
# # comparing to the cropped magnitude RSS targets:
# #


# def crop(x_net, x):
#     """Crop to GT shape then take magnitude."""
#     return dinv.utils.MRIMixin().rss(
#         dinv.utils.MRIMixin().crop(x_net, shape=x.shape), multicoil=False
#     )


# class CropPSNR(dinv.metric.PSNR):
#     def forward(self, x_net=None, x=None, *args, **kwargs):
#         return super().forward(crop(x_net, x), x, *args, **kwargs)


# class CropMSE(dinv.metric.MSE):
#     def forward(self, x_net=None, x=None, *args, **kwargs):
#         return super().forward(crop(x_net, x), x, *args, **kwargs)


# trainer = dinv.Trainer(
#     model=model,
#     physics=physics,
#     losses=dinv.loss.SupLoss(metric=CropMSE()),
#     metrics=CropPSNR(),
#     optimizer=torch.optim.Adam(model.parameters()),
#     train_dataloader=DataLoader(dataset),
#     epochs=1,
#     save_path=None,
#     show_progress_bar=False,
#     device=device,
# )
# _ = trainer.train()


# # %%
# # 5. Explore 3D MRI
# # ~~~~~~~~~~~~~~~~~
# #
# # We can also simulate 3D MRI data.
# # Here, we use a demo 3D brain volume of shape ``(181, 217, 181)`` from the
# # `BrainWeb <https://brainweb.bic.mni.mcgill.ca/brainweb/>`_ dataset
# # and simulate 3D single-coil or multi-coil Fourier measurements using
# # :class:`deepinv.physics.MRI` or
# # :class:`deepinv.physics.MultiCoilMRI`.
# #

# x = (
#     dinv.utils.load_np_url(
#         "https://huggingface.co/datasets/deepinv/images/resolve/main/brainweb_t1_ICBM_1mm_subject_0.npy?download=true"
#     )
#     .unsqueeze(0)
#     .unsqueeze(0)
#     .to(device)
# )
# x = torch.cat([x, torch.zeros_like(x)], dim=1)  # add imaginary dimension

# print(x.shape)  # (B, C, D, H, W) where D is depth

# physics = dinv.physics.MultiCoilMRI(img_size=x.shape[1:], three_d=True, device=device)
# physics = dinv.physics.MRI(img_size=x.shape[1:], three_d=True, device=device)

# dinv.utils.plot_ortho3D([x, physics(x)], titles=["x", "y"])


# # %%
# # 6. Explore dynamic MRI
# # ~~~~~~~~~~~~~~~~~~~~~~
# #
# # Finally, we show how to use the dynamic MRI for image sequence data of
# # shape ``(B, C, T, H, W)`` where ``T`` is the time dimension. Note that
# # this is also compatible with 3D MRI. We use dynamic MRI data from the
# # `CMRxRecon <https://cmrxrecon.github.io/>`_ challenge of cardiac cine
# # sequences and load them using :class:`deepinv.datasets.CMRxReconSliceDataset`
# # provided in deepinv. We download demo data from the first patient
# # including ground truth images, undersampled kspace, and associated masks:
# #

# dinv.datasets.download_archive(
#     dinv.utils.get_image_url("CMRxRecon.zip"),
#     dinv.utils.get_data_home() / "CMRxRecon.zip",
#     extract=True,
# )

# dataset = dinv.datasets.CMRxReconSliceDataset(
#     dinv.utils.get_data_home() / "CMRxRecon",
# )

# x, y, params = next(iter(DataLoader(dataset)))

# print(f"""
#     Ground truth: {x.shape} (B, C, T, H, W)
#     Measurements: {y.shape}
#     Acc. mask: {params["mask"].shape}
# """)

# # %%
# # Dynamic MRI data is directly compatible with existing functionality.
# # For example, you can train with this data by passing the dataset to
# # :class:`deepinv.Trainer`, which will automatically load in the data
# # ``x, y, params``. Or, you can use the data directly with the physics
# # :class:`deepinv.physics.DynamicMRI`.
# #
# # You can also pass in a custom k-t acceleration mask generator to
# # generate random time-varying masks:
# #

# physics_generator = dinv.physics.generator.EquispacedMaskGenerator(
#     img_size=x.shape[1:],
#     acceleration=16,
# )
# physics = dinv.physics.DynamicMRI(img_size=(512, 256), device=device)

# dataset = dinv.datasets.CMRxReconSliceDataset(
#     dinv.utils.get_data_home() / "CMRxRecon",
#     mask_generator=physics_generator,
#     mask_dir=None,
# )

# x, y, params = next(iter(DataLoader(dataset)))

# x = x.to(device)
# y = y.to(device)
# params = {k: v.to(device) for k, v in params.items()}

# # %%
# # We provide a video plotting function, :class:`deepinv.utils.plot_videos`. Here, we
# # visualize t=5 frames of the ground truth ``x``, the mask, and the zero-filled
# # reconstruction ``x_zf`` (and crop to square for better visibility):
# #

# x_zf = physics.A_adjoint(y, **params)

# dinv.utils.plot(
#     {
#         f"t={i}": torch.cat([x[:, :, i], params["mask"][:, :, i], x_zf[:, :, i]])[
#             ..., 128:384, :
#         ]
#         for i in range(5)
#     }
# )

# %%
# 6. Non-Cartesian MRI
# ~~~~~~~~~~~~~~~~~~~~
# We provide :class:`deepinv.physics.NonCartesianMRI`, a wrapper for `mri-nufft <https://mind-inria.github.io/mri-nufft>`_,
# to perform non-Cartesian sampling. See the class description for how to use it, and `mri-nufft` docs for more details.
#
# Here, we simulate 
#

x = brain_dataset[0].to(device).unsqueeze(0)  # 1,2,H,W

physics = NonCartesianMRI(
    num_shots=64,
    num_samples_per_shot=64,
    trajectory="spiral",
    density_mode=None,
    normalize=False,
    img_size=x.shape[-2:],
    backend="finufft" if str(device) == "cpu" else "cufinufft",
    noise_model=dinv.physics.GaussianNoise(1e-4), # simulate Gaussian noise
    device=device,
)

y = physics(x)  # simulate measurement

x_adj = physics.A_adjoint(y)

x_dag = physics.A_dagger(y)

print(y.shape, x_adj.shape)

dinv.utils.plot(
    {"GT": x, "Zero-filled": x_adj, "Pseudo-inv": x_dag},
    suptitle="No density compensation",
)

# %%
# Add density compensation:
#

physics = NonCartesianMRI(
    num_shots=64,
    num_samples_per_shot=64,
    trajectory="spiral",
    density_mode="compensate",
    normalize=False,
    img_size=x.shape[-2:],
    backend="finufft" if str(device) == "cpu" else "cufinufft",
    noise_model=dinv.physics.GaussianNoise(1e-4),
    device=device,
)

x_adj = physics.A_adjoint(y)
x_dag = physics.A_dagger(y)

dinv.utils.plot(
    {"GT": x, "Zero-filled": x_adj, "Pseudo-inv": x_dag},
    suptitle="Voronoi density compensated",
)

# %%
# Non-Cartesian reconstruction via deep learning
#
# The non-Cartesian MRI operator by default is unnormalized (i.e. spectral norm not 1), and
# its "adjoint" doesn't satisfy the adjoint condition due to the density compensation.
#
# We can empirically normalize the operator using the power method, and we can apply the square-root method
# so that the operators satisfies adjointness.
#
# .. note::
#     The zero-shot performance is subpar. Fine-tune the model on a few images to obtain better results,
#     e.g. using supervised learning, or self-supervised learning (see :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_scan_specific.py
#     for an example).
#

physics = NonCartesianMRI(
    num_shots=64,
    num_samples_per_shot=64,
    trajectory="spiral",
    density_mode="adjointness",
    normalize=True,
    img_size=x.shape[-2:],
    backend="finufft" if str(device) == "cpu" else "cufinufft",
    noise_model=dinv.physics.GaussianNoise(1e-4),
    device=device,
)

print(physics.adjointness_test(x_adj), physics.compute_norm(x_adj)) # should be ~0, ~1

model = dinv.models.RAM(device=device)

y *= physics.density.sqrt()

with torch.no_grad():
    x_hat = model(y, physics)

dinv.utils.plot({"GT": x, "Adjoint": x_adj, "Pseudo-inv": x_dag, "RAM zero-shot": x_hat})

# %%
# :References:
#
# .. footbibliography::
