from __future__ import annotations
import os
import shutil
import copy
from math import sqrt
import pytest
import warnings
import torch

import numpy as np
from deepinv.physics.forward import adjoint_function
import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.physics.mri import MRI, DynamicMRI, MultiCoilMRI
from deepinv.utils.mixins import MRIMixin
from deepinv.utils import TensorList
from deepinv.transform.rotate import Rotate

# Linear forward operators to test (make sure they appear in find_operator as well)
# We do not include operators for which padding is involved, they are tested separately
OPERATORS = [
    "CS",
    "inpainting",
    "inpainting_clone",
    "demosaicing",
    "denoising",
    "colorize",
    "fftdeblur",
    "singlepixel",
    "deblur_valid",
    "deblur_circular",
    "deblur_reflect",
    "pet_2d",
    "pet_3d",
    "deblur_replicate",
    "deblur_constant",
    "composition",
    "composition2",
    "space_deblur_valid",
    "space_deblur_circular",
    "space_deblur_reflect",
    "space_deblur_replicate",
    "space_deblur_constant",
    "tiled_space_deblur_valid",
    "hyperspectral_unmixing",
    "3Ddeblur_valid",
    "3Ddeblur_circular",
    "super_resolution_valid",
    "super_resolution_circular",
    "super_resolution_reflect",
    "super_resolution_replicate",
    "super_resolution_constant",
    "down_resolution_circular",
    "down_resolution_reflect",
    "down_resolution_replicate",
    "down_resolution_constant",
    "aliased_super_resolution",
    "super_resolution_matlab",
    "fast_singlepixel",
    "fast_singlepixel_cake_cutting",
    "fast_singlepixel_zig_zag",
    "fast_singlepixel_xy",
    "MRI",
    "DynamicMRI",
    "MultiCoilMRI",
    "MultiCoilMRIBirdcage",
    "3DMRI",
    "3DMultiCoilMRI",
    "aliased_pansharpen",
    "pansharpen_valid",
    "pansharpen_circular",
    "pansharpen_reflect",
    "pansharpen_replicate",
    "complex_compressed_sensing",
    "radio",
    "radio_weighted",
    "structured_random",
    "cassi-ss",
    "cassi-sd",
    "ptychography_linear",
    "2DParallelBeamCT",
    "2DFanBeamCT",
    "VirtualLinearPhysics",
    "ultrasound_planewave",
]

NONLINEAR_OPERATORS = [
    "haze",
    "lidar",
    "spatial_unwrapping_round",
    "spatial_unwrapping_floor",
    "scattering",
]

PHASE_RETRIEVAL_OPERATORS = [
    "random_phase_retrieval",
    "structured_random_phase_retrieval",
    "ptychography",
]

NOISES = [
    "Gaussian",
    "Poisson",
    "PoissonGaussian",
    "UniformGaussian",
    "Uniform",
    "Neighbor2Neighbor",
    "LogPoisson",
    "Gamma",
    "FisherTippett",
    "SaltPepper",
]


WRAPPERS = [
    None,
    "LinearPhysicsMultiScaler",
    "PhysicsCropper",
]


def find_operator(name, device, imsize=None, get_physics_param=False):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda
    :return: (:class:`deepinv.physics.Physics`) forward operator.
    """
    img_size = (3, 16, 8) if imsize is None else imsize
    norm = 1
    dtype = torch.float
    padding = next(
        (
            p
            for p in ["valid", "circular", "reflect", "replicate", "constant"]
            if p in name
        ),
        None,
    )

    rng = torch.Generator(device).manual_seed(0)
    if name == "CS":
        m = 30
        p = dinv.physics.CompressedSensing(
            m=m, img_size=img_size, device=device, rng=rng
        )
        norm = (
            1 + np.sqrt(np.prod(img_size) / m)
        ) ** 2 - 0.75  # Marcenko-Pastur law, second term is a small n correction
        params = []
    elif name == "colorize":
        p = dinv.physics.Decolorize(device=device)
        norm = 0.4468
        params = ["srf"]
    elif name == "cassi-ss":
        img_size = (7, 37, 31) if imsize is None else imsize
        p = dinv.physics.CompressiveSpectralImaging(
            img_size, device=device, rng=rng, mode="ss"
        )
        norm = 1 / img_size[0]
        params = ["mask"]
    elif name == "cassi-sd":
        img_size = (7, 37, 31) if imsize is None else imsize
        p = dinv.physics.CompressiveSpectralImaging(
            img_size, device=device, rng=rng, mode="sd"
        )
        norm = 1 / img_size[0]
        params = ["mask"]
    elif name == "inpainting":
        p = dinv.physics.Inpainting(img_size=img_size, mask=0.5, device=device, rng=rng)
        params = ["mask"]
    elif name == "inpainting_clone":
        p = dinv.physics.Inpainting(img_size=img_size, mask=0.5, device=device, rng=rng)
        p = p.clone()
        params = ["mask"]
    elif name == "demosaicing":
        p = dinv.physics.Demosaicing(img_size=img_size, device=device)
        norm = 1.0
        params = []
    elif name == "MRI":
        img_size = (2, 17, 11) if imsize is None else imsize  # C,H,W
        p = MRI(img_size=img_size, device=device)
        params = ["mask"]
    elif name == "3DMRI":
        img_size = (
            (2, 5, 17, 11) if imsize is None else imsize
        )  # C,D,H,W where D is depth
        p = MRI(img_size=img_size, three_d=True, device=device)
        params = ["mask"]
    elif name == "DynamicMRI":
        img_size = (
            (2, 5, 17, 11) if imsize is None else imsize
        )  # C,T,H,W where T is time
        p = DynamicMRI(img_size=img_size, device=device)
        params = ["mask"]
    elif name == "MultiCoilMRI":
        img_size = (2, 17, 11) if imsize is None else imsize  # C,H,W
        n_coils = 7
        maps = torch.ones(
            (1, n_coils, img_size[-2], img_size[-1]),
            dtype=torch.complex64,
            device=device,
        ) / sqrt(
            n_coils
        )  # B,N,H,W where N is coil dimension
        p = MultiCoilMRI(coil_maps=maps, img_size=img_size, device=device)
        params = ["mask", "coil_maps"]
    elif name == "MultiCoilMRIBirdcage":
        pytest.importorskip(
            "sigpy",
            reason="This test requires sigpy. It should be "
            "installed with `pip install "
            "sigpy`",
        )
        img_size = (2, 17, 11) if imsize is None else imsize  # C,H,W
        p = MultiCoilMRI(coil_maps=7, img_size=img_size, device=device)
        params = ["mask", "coil_maps"]
    elif name == "3DMultiCoilMRI":
        img_size = (
            (2, 5, 17, 11) if imsize is None else imsize
        )  # C,D,H,W where D is depth
        n_coils = 15
        maps = torch.ones(
            (1, n_coils, img_size[-3], img_size[-2], img_size[-1]),
            dtype=torch.complex64,
            device=device,
        ) / sqrt(
            n_coils
        )  # B,N,D,H,W where N is coils and D is depth
        p = MultiCoilMRI(coil_maps=maps, img_size=img_size, three_d=True, device=device)
        params = ["mask"]
    elif name == "pet_2d":
        pytest.importorskip(
            "parallelproj",
            reason="This test requires parallelproj. It should be "
            "installed with `conda install -c conda-forge parallelproj`",
        )
        img_size = (1, 16, 16) if imsize is None else imsize  # C,H,W
        attenuation = torch.full(img_size, 0.01, device=device)
        p = dinv.physics.PET(
            img_size,
            normalize=True,
            device=device,
            attenuation=attenuation,
        )
        assert not torch.allclose(p.attenuation, torch.ones_like(p.attenuation))
        p.update(attenuation=torch.ones_like(p.attenuation))
        p.noise_model = dinv.physics.ZeroNoise()
        p.normalize = False  # stop auto-normalize to compute gradients wrt to attn
        params = ["background", "attenuation"]
    elif name == "pet_3d":
        pytest.importorskip(
            "parallelproj",
            reason="This test requires parallelproj. It should be "
            "installed with `conda install -c conda-forge parallelproj`",
        )
        img_size = (1, 16, 16, 16) if imsize is None else imsize  # C,H,W
        p = dinv.physics.PET(
            img_size,
            normalize=True,
            device=device,
        )
        p.noise_model = dinv.physics.ZeroNoise()
        p.normalize = False  # stop auto-normalize to compute gradients wrt to attn
        params = ["attenuation", "background"]
    elif name == "2DParallelBeamCT":
        img_size = (1, 16, 16) if imsize is None else imsize  # C,H,W
        p = dinv.physics.Tomography(
            img_width=img_size[-1], angles=img_size[-1], fan_beam=False, device=device
        )

        params = []
    elif name == "2DFanBeamCT":
        img_size = (1, 16, 16) if imsize is None else imsize  # C,H,W
        p = dinv.physics.Tomography(
            img_width=img_size[-1],
            angles=img_size[-1],
            fan_beam=True,
            device=device,
        )
        params = []
    elif name == "VirtualLinearPhysics":
        base_physics = dinv.physics.Inpainting(
            img_size=img_size, mask=0.5, device=device, rng=rng
        )
        transform = Rotate(n_trans=4, multiples=90, positive=True)
        x0 = torch.zeros(1, *img_size, device=device)
        G_params = transform.get_params(x0)
        G_params = transform.iterate_params(G_params)
        g_params = next(iter(G_params))
        p = dinv.physics.VirtualLinearPhysics(
            physics=base_physics,
            transform=transform,
            g_params=g_params,
        )
        params = []
    elif name == "ultrasound_planewave":
        # Small IQ setup. Image is (2, Z, X): 2 = (I, Q).
        img_size = (2, 16, 16) if imsize is None else imsize
        assert (
            img_size[0] == 2
        ), f"ultrasound expects 2-channel IQ, got img_size={img_size}"
        Z, X = img_size[-2:]
        n_elements = 8
        pitch = 3e-4
        ele_x = torch.linspace(
            -pitch * (n_elements - 1) / 2, pitch * (n_elements - 1) / 2, n_elements
        )
        ele_pos = torch.stack([ele_x, torch.zeros(n_elements)], dim=-1)
        lam = 1540.0 / 5e6
        pixel_size = (lam / 2.0, lam / 2.0)
        common = dict(
            img_size=(Z, X),
            element_positions=ele_pos,
            n_samples=128,
            sampling_frequency=20e6,
            demodulation_frequency=5e6,
            pixel_size=pixel_size,
            normalize=True,
            device=device,
        )
        angles = torch.deg2rad(torch.linspace(-16.0, 16.0, 3))
        p = dinv.physics.UltrasoundPlaneWave(angles=angles, **common)
        params = []
    elif name == "composition":
        img_size = (3, 16, 16) if imsize is None else imsize
        p1 = dinv.physics.Downsampling(
            img_size=img_size, factor=2, device=device, padding="same", filter=None
        )
        p2 = dinv.physics.BlurFFT(
            img_size=img_size,
            device=device,
            filter=dinv.physics.functional.gaussian_blur(sigma=(1.0, 1.0)),
        )
        p = p1 * p2
        norm = 1 / 2**2
        params = ["filter"]
    elif name == "composition2":
        img_size = (3, 16, 16) if imsize is None else imsize
        p1 = dinv.physics.Downsampling(
            img_size=img_size, factor=2, device=device, filter=None
        )
        p2 = dinv.physics.BlurFFT(
            img_size=(3, 8, 8),
            device=device,
            filter=dinv.physics.functional.gaussian_blur(sigma=(0.5, 0.5)),
        )
        p = p2 * p1
        params = ["filter"]
    elif name == "denoising":
        p = dinv.physics.Denoising(dinv.physics.GaussianNoise(0.1, rng=rng))
        params = []
    elif name.startswith("pansharpen"):
        img_size = (3, 30, 32)
        p = dinv.physics.Pansharpen(
            img_size=img_size,
            device=device,
            padding=padding,
            filter="bilinear",
            use_brovey=False,
        )
        norm = 0.4
        params = []
    elif name == "aliased_pansharpen":
        img_size = (3, 30, 32) if imsize is None else imsize
        p = dinv.physics.Pansharpen(
            img_size=img_size, device=device, filter=None, use_brovey=False
        )
        norm = 1.4
        params = []  # no filter in aliased case
    elif name == "fast_singlepixel":
        p = dinv.physics.SinglePixelCamera(
            m=20, fast=True, img_size=img_size, device=device, rng=rng
        )
        params = ["mask"]
    elif name == "fast_singlepixel_cake_cutting":
        p = dinv.physics.SinglePixelCamera(
            m=20,
            fast=True,
            img_size=img_size,
            device=device,
            rng=rng,
            ordering="cake_cutting",
        )
        params = ["mask"]
    elif name == "fast_singlepixel_zig_zag":
        p = dinv.physics.SinglePixelCamera(
            m=20,
            fast=True,
            img_size=img_size,
            device=device,
            rng=rng,
            ordering="zig_zag",
        )
        params = ["mask"]
    elif name == "fast_singlepixel_xy":
        p = dinv.physics.SinglePixelCamera(
            m=20, fast=True, img_size=img_size, device=device, rng=rng, ordering="xy"
        )
        params = []
    elif name == "singlepixel":
        m = 20
        p = dinv.physics.SinglePixelCamera(
            m=m, fast=False, img_size=img_size, device=device, rng=rng
        )
        norm = (
            1 + np.sqrt(np.prod(img_size) / m)
        ) ** 2 - 3.7  # Marcenko-Pastur law, second term is a small n correction
        params = ["mask"]
    elif name.startswith("deblur"):
        img_size = (3, 17, 19) if imsize is None else imsize
        p = dinv.physics.Blur(
            filter=dinv.physics.functional.gaussian_blur(sigma=(0.25, 0.1), angle=0.0),
            padding=padding,
            device=device,
        )
        params = ["filter"]
    elif name == "fftdeblur":
        img_size = (3, 17, 19) if imsize is None else imsize
        p = dinv.physics.BlurFFT(
            img_size=img_size,
            filter=dinv.physics.functional.bicubic_filter(),
            device=device,
        )
        params = ["filter"]
    elif name.startswith("space_deblur"):
        img_size = (3, 20, 13) if imsize is None else imsize
        h = dinv.physics.functional.bilinear_filter(factor=2).unsqueeze(0).to(device)
        h /= torch.sum(h)
        h = torch.cat([h, h], dim=2)
        p = dinv.physics.SpaceVaryingBlur(
            filters=h,
            multipliers=torch.ones(
                (
                    1,
                    img_size[0],
                    2,
                )
                + img_size[-2:],
                device=device,
            ).to(device)
            * 0.5,
            padding=padding,
            device=device,
        )
        params = ["filters", "multipliers"]
    elif name == "tiled_space_deblur_valid":
        img_size = (3, 20, 13) if imsize is None else imsize
        h = dinv.physics.functional.bilinear_filter(factor=2).to(device)
        h = h.unsqueeze(2)  # shape (1,1,1,Hf,Wf)
        num_filters = dinv.physics.TiledSpaceVaryingBlur.num_filters(
            img_size=img_size[-2:],
            patch_size=(8, 5),
            stride=(4, 3),
        )
        h = h.repeat(1, 3, num_filters[0] * num_filters[1], 1, 1)  # shape (1,3,K,Hf,Wf)
        p = dinv.physics.TiledSpaceVaryingBlur(
            filters=h,
            patch_size=(8, 5),
            stride=(4, 3),
            device=device,
        )

        params = ["filters", "multipliers"]

    elif name == "hyperspectral_unmixing":
        img_size = (15, 32, 32) if imsize is None else imsize  # x (E, H, W)
        p = dinv.physics.HyperSpectralUnmixing(E=15, C=64, device=device)
        params = ["M"]
    elif name.startswith("3Ddeblur"):
        img_size = (1, 7, 6, 8) if imsize is None else imsize  # C,D,H,W
        h_size = (1, 1, 4, 3, 5)
        h = torch.rand(h_size)
        h /= h.sum()
        p = dinv.physics.Blur(
            filter=h,
            padding=padding,
            device=device,
        )
        params = ["filter"]
    elif name == "aliased_super_resolution":
        img_size = (1, 32, 32) if imsize is None else imsize
        factor = 2
        norm = 1.0
        p = dinv.physics.Downsampling(
            img_size=img_size,
            factor=factor,
            padding=padding,
            device=device,
            filter=None,
        )
        params = []
    elif name == "super_resolution_matlab":
        img_size = (1, 32, 32)
        factor = 2
        norm = 1.0 / factor**2
        p = dinv.physics.DownsamplingMatlab(factor=factor)
        params = []
    elif name.startswith("super_resolution"):
        img_size = (1, 32, 32) if imsize is None else imsize
        factor = 2
        norm = 1.0 / factor**2
        p = dinv.physics.Downsampling(
            img_size=img_size,
            factor=factor,
            padding=padding,
            device=device,
            filter="bilinear",
            dtype=dtype,
        )
        params = ["filter"]
    elif name.startswith("down_resolution"):
        img_size = (1, 32, 32) if imsize is None else imsize
        factor = 2
        norm = 1.0 / factor**2
        p = dinv.physics.Upsampling(
            img_size=(img_size[0], img_size[1] * factor, img_size[2] * factor),
            factor=factor,
            padding=padding,
            device=device,
            filter="bilinear",
            dtype=dtype,
        )
        params = ["filter"]
    elif name == "complex_compressed_sensing":
        img_size = (1, 8, 8) if imsize is None else imsize
        m = 50
        p = dinv.physics.CompressedSensing(
            m=m,
            img_size=img_size,
            dtype=torch.cdouble,
            device=device,
            compute_inverse=True,
            rng=rng,
        )
        dtype = p.dtype
        norm = (1 + np.sqrt(np.prod(img_size) / m)) ** 2
        params = ["mask"]
    elif "radio" in name:
        dtype = torch.cfloat
        img_size = (1, 64, 64) if imsize is None else imsize
        pytest.importorskip(
            "torchkbnufft",
            reason="This test requires torchkbnufft. It should be "
            "installed with `pip install torchkbnufft`",
        )

        # Generate regular grid for sampling
        y = torch.linspace(-1, 1, img_size[-2])
        x = torch.linspace(-1, 1, img_size[-1])
        grid_y, grid_x = torch.meshgrid(y, x)
        uv = torch.stack((grid_y, grid_x), dim=-1) * torch.pi  # normalize [-pi, pi]

        # Reshape to [nb_points x 2]
        uv = uv.view(-1, 2)
        uv = uv.to(device)

        if "weighted" in name:
            dataWeight = torch.linspace(
                0.01, 0.99, uv.shape[0], device=device
            )  # take a non-trivial weight
        else:
            dataWeight = torch.tensor(
                [
                    1.0,
                ]
            )

        p = dinv.physics.RadioInterferometry(
            img_size=img_size[1:],
            samples_loc=uv.permute((1, 0)),
            dataWeight=dataWeight,
            real_projection=False,
            dtype=dtype,
            device=device,
            noise_model=dinv.physics.GaussianNoise(0.0, rng=rng),
        )
        params = []
    elif name == "structured_random":
        img_size = (1, 8, 8) if imsize is None else imsize
        p = dinv.physics.StructuredRandom(
            img_size=img_size, output_size=img_size, device=device
        )
        params = []
    elif name == "ptychography_linear":
        img_size = (1, 32, 32) if imsize is None else imsize
        dtype = torch.complex64
        norm = 1.32
        p = dinv.physics.PtychographyLinearOperator(
            img_size=img_size,
            probe=None,
            shifts=None,
            device=device,
        )
        params = ["probe", "shifts"]
    else:
        raise Exception("The inverse problem chosen doesn't exist")

    if not get_physics_param:
        return p, img_size, norm, dtype
    else:
        return p, img_size, norm, dtype, params


def find_nonlinear_operator(name, device):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda
    :return: (:class:`deepinv.physics.Physics`) forward operator.
    """
    if name == "haze":
        x = dinv.utils.TensorList(
            [
                torch.randn(1, 1, 16, 16, device=device),
                torch.randn(1, 1, 16, 16, device=device),
                torch.randn(1, device=device),
            ]
        )
        p = dinv.physics.Haze()

    elif name == "scattering":
        try:
            import scipy  # noqa: F401
        except ImportError:
            pytest.skip(
                "This test requires scipy. It should be "
                "installed with `pip install scipy`"
            )
        dtype = torch.complex128
        transmitters, receivers = dinv.physics.scattering.circular_sensors(
            8, radius=1.0, device=device
        )
        p = dinv.physics.Scattering(
            img_width=32,
            device=device,
            background_wavenumber=5 * (2 * torch.pi),
            wave_type="plane_wave",
            transmitters=transmitters,
            receivers=receivers,
            verbose=False,
        )
        x = torch.rand(1, 1, 32, 32, dtype=dtype, device=device) * 0.1  # low contrast
    elif name == "lidar":
        x = torch.rand(1, 3, 16, 16, device=device)
        p = dinv.physics.SinglePhotonLidar(device=device)
    elif name == "spatial_unwrapping_round":
        x = torch.randn(1, 3, 16, 16, device=device)
        p = dinv.physics.SpatialUnwrapping(threshold=1.0, mode="round", device=device)
    elif name == "spatial_unwrapping_floor":
        x = torch.randn(1, 3, 16, 16, device=device)
        p = dinv.physics.SpatialUnwrapping(threshold=1.0, mode="floor", device=device)
    else:
        raise Exception("The inverse problem chosen doesn't exist")
    return p, x


def wrap_physics(wrapper_name, physics, img_size, device):
    if wrapper_name == "LinearPhysicsMultiScaler":
        factors = [2, 4, 8]
        p = dinv.physics.LinearPhysicsMultiScaler(
            physics=physics, img_shape=img_size, factors=factors, device=device
        )
        img_size_out = (img_size[0], img_size[-2] // 4, img_size[-1] // 4)
    elif wrapper_name == "PhysicsCropper":
        crop = (2, 4)
        p = dinv.physics.PhysicsCropper(physics=physics, crop=crop, device=device)
        img_size_out = (
            *img_size[:-2],
            img_size[-2] + crop[-2],
            img_size[-1] + crop[-1],
        )
    else:
        raise Exception(
            f"The wrapper {wrapper_name} is not in the `wrap_physics` function"
        )
    return p, img_size_out


def find_phase_retrieval_operator(name, device):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda
    :return: (deepinv.physics.PhaseRetrieval) forward operator.
    """
    if name == "random_phase_retrieval":
        img_size = (1, 10, 10)
        p = dinv.physics.RandomPhaseRetrieval(m=500, img_size=img_size, device=device)
    elif name == "ptychography":
        img_size = (1, 32, 32)
        p = dinv.physics.Ptychography(
            img_size=img_size,
            probe=None,
            shifts=None,
            device=device,
        )
    elif name == "structured_random_phase_retrieval":
        img_size = (1, 10, 10)
        p = dinv.physics.StructuredRandomPhaseRetrieval(
            img_size=img_size, output_size=img_size, n_layers=2, device=device
        )
    else:
        raise Exception("The inverse problem chosen doesn't exist")
    return p, img_size


def test_stacking(device):
    r"""
    Tests if stacking physics operators is consistent with applying them sequentially.

    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is less than 1e-3
    """
    imsize = (2, 5, 5)
    p1 = dinv.physics.Inpainting(mask=0.5, img_size=imsize, device=device)
    p2 = dinv.physics.Physics(A=lambda x: x**2)
    p3 = p1.stack(p2)

    x = torch.randn(imsize, device=device).unsqueeze(0)
    y1 = p1.A(x)
    y2 = p2.A(x)
    y = p3.A(x)

    assert torch.allclose(y[0], y1)
    assert torch.allclose(y[1], y2)

    assert not isinstance(p3, dinv.physics.StackedLinearPhysics)
    assert isinstance(p3, dinv.physics.StackedPhysics)

    p4 = p1.stack(p1)
    y = p4(x)
    assert isinstance(p4, dinv.physics.StackedLinearPhysics)
    assert len(y) == 2
    assert p4.A_adjoint(y).shape == x.shape

    p5 = p4.stack(p4)
    y = p5(x)
    assert len(p5) == 4
    assert len(y) == 4


@pytest.mark.parametrize("name", OPERATORS)
def test_operators_adjointness(name, device, rng):
    r"""
    Tests if a linear forward operator has a well-defined adjoint.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts adjointness
    """
    physics, imsize, _, dtype = find_operator(name, device)

    if name == "radio":
        dtype = torch.cfloat

    x = torch.randn(imsize, device=device, dtype=dtype, generator=rng).unsqueeze(0)
    error = physics.adjointness_test(x).abs()
    assert error < 1e-3

    if (
        "pansharpen" in name or "radio" in name or "pet" in name
    ):  # automatic adjoint does not work for inputs that are not torch.tensors
        pytest.skip()
    f = adjoint_function(physics.A, x.shape, x.device, x.dtype)

    y = physics.A(x)
    error2 = (f(y) - physics.A_adjoint(y)).flatten().mean().abs()

    assert error2 < 1e-3


LIST_DOWN_OP = [
    "down_resolution_circular",
    "down_resolution_reflect",
    "down_resolution_replicate",
    "down_resolution_constant",
]


@pytest.mark.parametrize("name", LIST_DOWN_OP)
@pytest.mark.parametrize("kernel", ["bilinear", "bicubic", "sinc", "gaussian"])
def test_upsampling(device, rng, name, kernel):
    r"""
    This function tests that the Upsampling and Downsampling operators are effectively adjoint to each other.

    Note that the test does not hold when the padding is not 'valid', as the Upsampling operator
    does not support 'valid' padding.
    """
    padding = name.split("_")[-1]  # get padding type from name
    physics, imsize, _, dtype = find_operator(name, device)
    physics_adjoint, _, _, dtype = find_operator(
        "super_resolution_" + padding, device, imsize=imsize
    )

    # physics.register_buffer("filter", None)
    physics.update_parameters(filter=kernel)

    # physics_adjoint.register_buffer("filter", None)
    physics_adjoint.update_parameters(filter=kernel)

    factor = physics.factor

    x = torch.randn(
        (1, imsize[0], imsize[1], imsize[2]),
        device=device,
        dtype=dtype,
        generator=rng,
    )

    out = physics(x)
    assert out.shape == (1, imsize[0], imsize[1] * factor, imsize[2] * factor)

    y = physics(x)
    err1 = (physics.A_adjoint(y) - physics_adjoint(y)).flatten().mean().abs()
    assert err1 < 1e-6

    imsize_new = (*imsize[:1], imsize[1] * factor, imsize[2] * factor)
    physics_adjoint, _, _, dtype = find_operator(
        "super_resolution_" + padding, device, imsize=imsize_new
    )  # we need to redefine the adjoint operator with the new image size

    # physics_adjoint.register_buffer("filter", None)
    physics_adjoint.update_parameters(filter=kernel)

    x = torch.randn(imsize_new, device=device, dtype=dtype, generator=rng).unsqueeze(0)
    y = physics_adjoint(x)
    err2 = (physics.A(y) - physics_adjoint.A_adjoint(y)).flatten().mean().abs()
    assert err2 < 1e-6


@pytest.mark.parametrize("name", OPERATORS)
def test_operator_multiscale_wrapper(name, device, rng):
    r"""
    Tests if a linear physics operator can be wrapped with a multi-scale wrapper.
    """

    # defining a list of exceptions to skip  # TODO: fix for those?
    list_exceptions = [
        "pansharpen",  # shape handling
        "radio",  # data type (complex)
        "3d",  # shape handling
        "ptychography",  # ?
        "composition2",  # shape handling
        "dynamicmri",  # shape handling
        "complex_compressed_sensing",  # data type (complex)
    ]

    if any(exc in name.lower() for exc in list_exceptions):
        pytest.skip(f"Skipping test for operator '{name}' as it matches an exception.")

    base_shape = (32, 32)
    scale = 2

    _, img_size_orig, _, _ = find_operator(
        name,
        device=device,
    )  # get img_size for the operator
    physics, img_size_orig, _, dtype = find_operator(
        name,
        device=device,
        imsize=(*img_size_orig[:-2], base_shape[-2], base_shape[-1]),
    )  # get physics for the operator with base img size

    image_shape = (
        *img_size_orig[:-2],
        base_shape[-2] // (scale**2),
        base_shape[-1] // (scale**2),
    )
    x = torch.rand((1, *image_shape), dtype=dtype, device=device)  # add batch dim

    new_physics = dinv.physics.LinearPhysicsMultiScaler(
        physics,
        (*image_shape[:-2], *base_shape),
        factors=[2, 4, 8],
        dtype=dtype,
        device=device,
    )  # define a multi-scale physics with base img size (1, 32, 32)
    y = new_physics(x, scale=scale)
    Aty = new_physics.A_adjoint(y, scale=scale)

    assert Aty.shape == x.shape


@pytest.mark.parametrize("name", OPERATORS)
def test_operator_cropper(name, device, rng):
    r"""
    Tests if a linear physics operator can be wrapped with a crop wrapper.
    """

    physics, image_shape, _, dtype = find_operator(
        name,
        device,
    )  # get physics for the operator with base img size

    x = torch.rand((1, *image_shape), dtype=dtype, device=device)  # add batch dim
    padding_shape = (2, 5)
    x_new = torch.nn.functional.pad(x, (padding_shape[1], 0, padding_shape[0], 0))

    new_physics = dinv.physics.PhysicsCropper(physics, padding_shape, device=device)
    y = new_physics(x_new)
    Aty = new_physics.A_adjoint(y)

    assert Aty.shape == x_new.shape


@pytest.mark.parametrize("name", OPERATORS)
@pytest.mark.parametrize("verbose", [True, False])
def test_operators_norm(name, verbose, device, rng):
    r"""
    Tests if a linear physics operator has a norm close to 1.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param imsize: (tuple) image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts norm is in (.8,1.2)
    """
    if name == "radio_weighted":  # weighted nufft norm is not tested
        return

    if name == "singlepixel" or name == "CS":
        device = torch.device("cpu")
        rng = torch.Generator("cpu")

    torch.manual_seed(0)
    physics, imsize, norm_ref, dtype = find_operator(name, device)
    x = torch.randn(imsize, device=device, dtype=dtype, generator=rng).unsqueeze(0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        physics.compute_sqnorm(x, max_iter=1, tol=1e-9, verbose=verbose)
        assert len(w) == 1

    norm = physics.compute_sqnorm(x, max_iter=1000, tol=1e-6, verbose=verbose)
    bound = 1e-2
    # if theoretical bound relies on Marcenko-Pastur law, or if pansharpening, relax the bound
    if (
        name in ["singlepixel", "CS", "complex_compressed_sensing", "radio"]
        or "pansharpen" in name
    ):
        bound = 0.2
    # convolution norm is not simple in those cases
    if (
        "reflect" in name
        or "replicate" in name
        or "constant" in name
        or "valid" in name
    ):
        pass
    else:
        assert torch.abs(norm - norm_ref) < bound


@pytest.mark.parametrize("name", NONLINEAR_OPERATORS)
def test_nonlinear_operators(name, device):
    r"""
    Tests if a linear physics operator has a norm close to 1.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts correct shapes
    """
    physics, x = find_nonlinear_operator(name, device)
    y = physics(x)
    xhat = physics.A_dagger(y)
    assert x.shape == xhat.shape


@pytest.mark.parametrize("name", OPERATORS)
@pytest.mark.parametrize("implicit_backward_solver", [True, False])
def test_pseudo_inverse(name, device, rng, implicit_backward_solver):
    r"""
    Tests if a linear physics operator has a well-defined pseudoinverse.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param imsize: (tuple) image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is less than 1e-3
    """
    physics, imsize, _, dtype = find_operator(name, device)
    physics.implicit_backward_solver = implicit_backward_solver

    x = torch.randn(imsize, device=device, dtype=dtype, generator=rng).unsqueeze(0)

    r = physics.A_adjoint(physics.A(x))  # project to range of A^T
    y = physics.A(r)
    error = torch.linalg.vector_norm(
        physics.A_dagger(y, solver="lsqr", tol=0.0001, max_iter=50, verbose=True) - r
    ) / torch.linalg.vector_norm(r)
    assert error < 0.05


@pytest.mark.parametrize("name", OPERATORS)
def test_decomposable(name, device, rng):
    physics, imsize, _, dtype = find_operator(name, device)
    if isinstance(physics, dinv.physics.DecomposablePhysics):
        x = torch.randn(imsize, device=device, dtype=dtype, generator=rng).unsqueeze(0)

        proj = lambda u: physics.V(physics.V_adjoint(u))
        r = proj(x)  # project
        assert (
            torch.linalg.vector_norm(proj(r) - r) / torch.linalg.vector_norm(r) < 1e-3
        )

        y = physics.A(x)
        proj = lambda u: physics.U(physics.U_adjoint(u))
        r = proj(y)
        assert (
            torch.linalg.vector_norm(proj(r) - r) / torch.linalg.vector_norm(r) < 1e-3
        )


@pytest.fixture
def mri_img_size():
    return 1, 2, 3, 16, 16  # B, C, T, H, W


@pytest.mark.parametrize("mri", [MRI, DynamicMRI, MultiCoilMRI])
def test_MRI(mri, mri_img_size, device, rng):
    r"""
    Test MRI and DynamicMRI functions

    Assert mask is applied to physics wherever it is passed.

    :param mri_img_size: (tuple) image size tuple (B, C, T, H, W)
    :param device: (torch.device) cpu or cuda:x
    :param rng: (torch.Generator)
    """
    if mri is MultiCoilMRI:
        pytest.importorskip(
            "sigpy",
            reason="This test requires sigpy. It should be "
            "installed with `pip install "
            "sigpy`",
        )

    B, C, T, H, W = mri_img_size
    if rng.device != device:
        rng = torch.Generator(device=device).manual_seed(0)
    x, y = (
        torch.rand(mri_img_size, generator=rng, device=device) + 1,
        torch.rand(mri_img_size, generator=rng, device=device) + 1,
    )

    coil_maps_kwarg = {}

    if mri is MRI:
        x = x[:, :, 0, :, :]
        y = y[:, :, 0, :, :]
    elif mri is MultiCoilMRI:
        # y treat T as coil dim for tests
        x = x[:, :, 0, :, :]
        coil_maps_kwarg = {"coil_maps": T}

    for mask_size in [(H, W), (T, H, W), (C, T, H, W), (B, C, T, H, W)]:
        # Remove time dim for static MRI
        _mask_size = mask_size if mri is DynamicMRI else mask_size[:-3] + mask_size[-2:]

        mask, mask2 = (
            torch.ones(_mask_size, device=device)
            - torch.eye(*_mask_size[-2:], device=device),
            torch.zeros(_mask_size, device=device)
            + torch.eye(*_mask_size[-2:], device=device),
        )

        # Empty mask
        physics = mri(img_size=x.shape, device=device, **coil_maps_kwarg)
        y1 = physics(x)
        x1 = physics.A_adjoint(y)
        assert torch.sum(y1 == 0) == 0
        assert torch.sum(x1 == 0) == 0

        # Set mask in constructor
        physics = mri(
            mask=mask, img_size=mri_img_size, device=device, **coil_maps_kwarg
        )
        y1 = physics(x)
        if isinstance(physics, MultiCoilMRI):
            y1 = y1[:, :, 0]  # check 0th coil
        assert torch.all((y1 == 0) == (physics.mask == 0))

        # Set mask in forward
        y1 = physics(x, mask=mask2)
        if isinstance(physics, MultiCoilMRI):
            y1 = y1[:, :, 0]  # check 0th coil
        assert torch.all((y1 == 0) == (mask2 == 0))

        # Mask retained in previous forward
        y1 = physics(x)
        if isinstance(physics, MultiCoilMRI):
            y1 = y1[:, :, 0]  # check 0th coil
        assert torch.all((y1 == 0) == (mask2 == 0))

        # Set mask via update
        physics.update(mask=mask)
        y1 = physics(x)
        if isinstance(physics, MultiCoilMRI):
            y1 = y1[:, :, 0]  # check 0th coil
        assert torch.all((y1 == 0) == (mask == 0))

        # Check mag/rss reduces channel dim
        x_hat = physics.A_adjoint(
            y, **{("rss" if isinstance(physics, MultiCoilMRI) else "mag"): True}
        )
        # (B, 2, ...) -> (B, 1, ...)
        assert x_hat.shape[:2] == (x.shape[0], 1) and y.shape[1] == 2

        # Check rss works for multi-coil
        if isinstance(physics, MultiCoilMRI):
            assert y.shape[:3] == (x.shape[0], 2, T)  # B,C,N(=T)
            xrss = physics.A_adjoint(y, rss=True)
            assert xrss.shape == (x.shape[0], 1, *x.shape[2:])  # B,1,H,W


@pytest.mark.parametrize("mri", [MRI, DynamicMRI, MultiCoilMRI])
def test_MRI_noise_domain(mri, mri_img_size, device, rng):
    r"""
    Test that MRI noise addition is 0 where mask is 0

    :param mri_img_size: (tuple) image size tuple (B, C, T, H, W)
    :param device: (torch.device) cpu or cuda:x
    :param rng: (torch.Generator)
    """
    if mri is MultiCoilMRI:
        pytest.importorskip(
            "sigpy",
            reason="This test requires sigpy. It should be "
            "installed with `pip install "
            "sigpy`",
        )

    B, C, T, H, W = mri_img_size
    if rng.device != device:
        rng = torch.Generator(device=device).manual_seed(0)
    x, y = (
        torch.rand(mri_img_size, generator=rng, device=device) + 1,
        torch.rand(mri_img_size, generator=rng, device=device) + 1,
    )

    coil_maps_kwarg = {}

    if mri is MRI:
        x = x[:, :, 0, :, :]
        y = y[:, :, 0, :, :]
    elif mri is MultiCoilMRI:
        # y treat T as coil dim for tests
        x = x[:, :, 0, :, :]
        coil_maps_kwarg = {"coil_maps": T}

    for mask_size in [(H, W), (T, H, W), (C, T, H, W), (B, C, T, H, W)]:
        # Remove time dim for static MRI
        _mask_size = mask_size if mri is DynamicMRI else mask_size[:-3] + mask_size[-2:]

        mask = torch.ones(_mask_size, device=device) - torch.eye(
            *_mask_size[-2:], device=device
        )

        # Set mask in constructor
        physics = mri(
            mask=mask,
            img_size=mri_img_size,
            device=device,
            noise_model=dinv.physics.noise.GaussianNoise(sigma=0.1).to(device),
            **coil_maps_kwarg,
        )
        y1 = physics(x)
        if isinstance(physics, MultiCoilMRI):
            y1 = y1[:, :, 0]  # check 0th coil

        assert torch.all((y1 == 0) == (physics.mask == 0))


@pytest.mark.parametrize("name", OPERATORS)
def test_concatenation(name, device):
    if "pansharpen" in name:  # TODO: fix pansharpening
        return
    physics, imsize, _, dtype = find_operator(name, device)

    x = torch.rand(imsize, device=device, dtype=dtype).unsqueeze(0)
    y = physics(x)
    physics = (
        dinv.physics.Inpainting(
            img_size=y.size()[1:], mask=0.5, pixelwise=False, device=device
        )
        * physics
    )

    r = physics.A_adjoint(physics.A(x))  # project to range of A^T
    y = physics.A(r)
    error = torch.linalg.vector_norm(
        physics.A_dagger(y, solver="lsqr", tol=0.0001) - r
    ) / torch.linalg.vector_norm(r)
    assert error < 0.01


@pytest.mark.parametrize("name", PHASE_RETRIEVAL_OPERATORS)
def test_phase_retrieval(name, device):
    r"""
    Tests to ensure the phase retrieval operator is behaving as expected.

    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is less than 1e-3
    """
    physics, imsize = find_phase_retrieval_operator(name, device)
    x = torch.randn(imsize, dtype=torch.cfloat, device=device).unsqueeze(0)

    y = physics(x)
    # nonnegativity
    assert (y >= 0).all()
    # same outputes for x and -x
    assert torch.equal(y, physics(-x))

    x_hat = physics.A_dagger(physics(x))
    assert x_hat.shape == x.shape


def test_phase_retrieval_Avjp(device):
    r"""
    Tests if the gradient computed with A_vjp method of phase retrieval is consistent with the autograd gradient.

    :param device: (torch.device) cpu or cuda:x
    :return: assertion error if the relative difference between the two gradients is more than 1e-5
    """
    # essential to enable autograd
    torch.set_grad_enabled(True)
    x = torch.randn((1, 1, 3, 3), dtype=torch.cfloat, device=device, requires_grad=True)
    physics = dinv.physics.RandomPhaseRetrieval(m=10, img_size=(1, 3, 3), device=device)
    loss = L2()
    func = lambda x: loss(x, torch.ones_like(physics(x)), physics)[0]
    grad_value = torch.func.grad(func)(x)
    jvp_value = loss.grad(x, torch.ones_like(physics(x)), physics)
    assert torch.isclose(grad_value[0], jvp_value, rtol=1e-5).all()


def test_linear_physics_Avjp(device, rng):
    r"""
    Tests if the gradient computed with A_vjp method of linear physics is consistent with the autograd gradient.

    :param device: (torch.device) cpu or cuda:x
    :return: assertion error if the relative difference between the two gradients is more than 1e-5
    """
    # essential to enable autograd
    torch.set_grad_enabled(True)
    x = torch.randn(
        (1, 1, 3, 3),
        dtype=torch.float,
        device=device,
        generator=rng,
        requires_grad=True,
    )
    physics = dinv.physics.CompressedSensing(m=10, img_size=(1, 3, 3), device=device)
    loss = L2()
    func = lambda x: loss(x, torch.ones_like(physics(x)), physics)[0]
    grad_value = torch.func.grad(func)(x)
    jvp_value = loss.grad(x, torch.ones_like(physics(x)), physics)
    assert torch.isclose(grad_value[0], jvp_value, rtol=1e-5).all()


def test_physics_Avjp(device):
    r"""
    Tests if the vector Jacobian product computed by A_vjp method of physics is correct.

    :param device: (torch.device) cpu or cuda:x
    :return: assertion error if the relative difference between the computed gradients and expected values is more than 1e-5
    """
    A = torch.eye(3, dtype=torch.float64)

    def A_forward(v):
        return A @ v

    physics = dinv.physics.Physics(A=A_forward)
    for _ in range(100):
        x = torch.randn(3, dtype=torch.float64)
        v = torch.randn(3, dtype=torch.float64)
        # Jacobian in this case should be identity
        assert torch.allclose(physics.A_vjp(x, v), v)


def choose_noise(noise_type, device="cpu"):
    gain = 0.1
    sigma = 0.1
    mu = 0.2
    N0 = 1024.0
    l = torch.ones((1), device=device)
    p, s = 0.025, 0.025
    if noise_type == "PoissonGaussian":
        noise_model = dinv.physics.PoissonGaussianNoise(sigma=sigma, gain=gain)
    elif noise_type == "Gaussian":
        noise_model = dinv.physics.GaussianNoise(sigma)
    elif noise_type == "UniformGaussian":
        noise_model = dinv.physics.UniformGaussianNoise()
    elif noise_type == "Uniform":
        noise_model = dinv.physics.UniformNoise(a=gain)
    elif noise_type == "Poisson":
        noise_model = dinv.physics.PoissonNoise(gain)
    elif noise_type == "Neighbor2Neighbor":
        noise_model = dinv.physics.PoissonNoise(gain)
    elif noise_type == "LogPoisson":
        noise_model = dinv.physics.LogPoissonNoise(N0, mu)
    elif noise_type == "Gamma":
        noise_model = dinv.physics.GammaNoise(l)
    elif noise_type == "SaltPepper":
        noise_model = dinv.physics.SaltPepperNoise(p=p, s=s)
    elif noise_type == "FisherTippett":
        noise_model = dinv.physics.FisherTippettNoise(l)
    else:
        raise Exception("Noise model not found")

    return noise_model


@pytest.mark.parametrize("noise_type", NOISES)
def test_noise(device, noise_type):
    r"""
    Tests noise models.
    """
    physics = dinv.physics.DecomposablePhysics()
    physics.noise_model = choose_noise(noise_type, device)
    x = torch.ones((1, 3, 2), device=device).unsqueeze(0)

    y1 = physics(
        x
        # Note: this works but not physics.A(x) because only the noise is reset (A does not encapsulate noise)
    )
    assert y1.shape == x.shape

    # Test that negative values input are handled correctly
    if noise_type in ["Poisson", "PoissonGaussian", "Gamma"]:
        x_neg = -torch.ones((1, 3, 2), device=device).unsqueeze(0)

        with pytest.raises(ValueError):
            y_neg = physics(x_neg)


def test_noise_domain(device):
    r"""
    Tests that there is no noise outside the domain of the measurement operator, i.e. that in y = Ax+n, we have
    n=0 where Ax=0.
    """
    x = torch.ones((1, 3, 12, 7), device=device)
    mask = torch.ones_like(x[0])
    # mask[:, x.shape[-2]//2-3:x.shape[-2]//2+3, x.shape[-1]//2-3:x.shape[-1]//2+3] = 0
    mask[0, 0, 0] = 0
    mask[1, 1, 1] = 0
    mask[2, 2, 2] = 0

    physics = dinv.physics.Inpainting(img_size=x.shape, mask=mask, device=device)
    physics.noise_model = choose_noise("Gaussian")
    y1 = physics(
        x
        # Note: this works but not physics.A(x) because only the noise is reset (A does not encapsulate noise)
    )
    assert y1.shape == x.shape

    assert y1[0, 0, 0, 0] == 0
    assert y1[0, 1, 1, 1] == 0
    assert y1[0, 2, 2, 2] == 0


@pytest.mark.parametrize(
    "img_size", [(1, 64, 64), (3, 65, 65), (1, 64, 65), (3, 65, 64)]
)
@pytest.mark.parametrize("filter_size", [(1, 5, 5), (1, 6, 6), (1, 6, 5)])
@pytest.mark.parametrize("filter_type", ["random", "directional"])
def test_blur(img_size, filter_size, filter_type, device):
    r"""
    Test that :class:`deepinv.physics.Blur` with `padding="circular"` and :class:`deepinv.physics.BlurFFT` compute the same circular blur.
    """
    torch.manual_seed(0)
    x = torch.randn(*img_size, device=device).unsqueeze(0)
    if filter_type == "random":
        h = torch.rand(*filter_size, device=device).unsqueeze(0)
    elif filter_type == "directional":
        # create a directional filter
        h = torch.zeros(*filter_size, device=device).unsqueeze(0)
        diag_len = min(filter_size[-2], filter_size[-1])
        idx = torch.arange(diag_len, device=device)
        h[..., idx, idx] = 1.0

    h = h / h.sum(dim=[-2, -1], keepdim=True)  # normalize filter
    physics_blur = dinv.physics.Blur(
        filter=h,
        device=device,
        padding="circular",
    )

    physics_blurfft = dinv.physics.BlurFFT(
        img_size=img_size,
        filter=h,
        device=device,
    )

    y1 = physics_blur(x)
    y2 = physics_blurfft(x)

    back1 = physics_blur.A_adjoint(y1)
    back2 = physics_blurfft.A_adjoint(y2)

    assert y1.shape == y2.shape
    assert back1.shape == back2.shape

    assert torch.allclose(y1, y2, atol=1e-5)
    assert torch.allclose(back1, back2, atol=1e-5)


def test_reset_noise(device):
    r"""
    Tests that the reset function works.

    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is > 0
    """
    x = torch.ones((1, 3, 3), device=device).unsqueeze(0)
    rng = torch.Generator(device)
    physics = dinv.physics.Denoising()
    physics.noise_model = dinv.physics.GaussianNoise(0.1, rng=rng)

    y1 = physics(x)
    y2 = physics(x, sigma=0.2)

    assert physics.noise_model.sigma == 0.2

    physics.noise_model = dinv.physics.PoissonNoise(0.1, rng=rng)

    y1 = physics(x)
    y2 = physics(x, gain=0.2)

    assert physics.noise_model.gain == 0.2

    physics.noise_model = dinv.physics.PoissonGaussianNoise(0.5, 0.3, rng=rng)
    y1 = physics(x)
    y2 = physics(x, sigma=0.2, gain=0.2)

    assert physics.noise_model.gain == 0.2
    assert physics.noise_model.sigma == 0.2


@pytest.mark.parametrize("normalize", [True, False, None])
@pytest.mark.parametrize("parallel_computation", [True, False])
@pytest.mark.parametrize("fan_beam", [True, False])
@pytest.mark.parametrize("circle", [True, False])
@pytest.mark.parametrize("adjoint_via_backprop", [True, False])
@pytest.mark.parametrize("fbp_interpolate_boundary", [True, False])
@pytest.mark.parametrize("fbp_pseudo_inverse", [True, False])
@pytest.mark.parametrize("channels", [1, 2])
def test_tomography(
    normalize,
    parallel_computation,
    fan_beam,
    circle,
    adjoint_via_backprop,
    fbp_interpolate_boundary,
    fbp_pseudo_inverse,
    channels,
    device,
):
    r"""
    Tests tomography operator which does not have a numerically precise adjoint.

    :param device: (torch.device) cpu or cuda:x
    """
    imsize = (channels, 16, 16)
    physics = dinv.physics.Tomography(
        img_width=imsize[-1],
        angles=imsize[-1],
        device=device,
        circle=circle,
        fan_beam=fan_beam,
        normalize=normalize,
        adjoint_via_backprop=adjoint_via_backprop,
        fbp_interpolate_boundary=fbp_interpolate_boundary,
        parallel_computation=parallel_computation,
    )

    x = torch.randn(
        imsize, device=device, generator=torch.Generator(device).manual_seed(0)
    ).unsqueeze(0)

    if adjoint_via_backprop:
        assert physics.adjointness_test(x).abs() < 1e-3

    if normalize:
        assert abs(physics.compute_sqnorm(x) - 1.0) < 1e-3

    if normalize is None:
        # when normalize is not set by the user, it should default to True
        assert physics.normalize is True
        assert abs(physics.compute_sqnorm(x) - 1.0) < 1e-3

    r_tol = 0.05 if not fbp_pseudo_inverse else 0.65
    r = physics.A_adjoint(physics.A(x))
    y = physics.A(r)

    error = torch.linalg.vector_norm(
        physics.A_dagger(y, fbp=fbp_pseudo_inverse) - r
    ) / torch.linalg.vector_norm(r)
    assert (
        error < r_tol
    ), f"error: {error} > {r_tol}, fanbeam={fan_beam}, circle={circle}, fbp_interpolate_boundary={fbp_interpolate_boundary}, normalize={normalize}, adjoint_via_backprop={adjoint_via_backprop}, parallel_computation={parallel_computation}, fbp_pseudo_inverse={fbp_pseudo_inverse}"


@pytest.mark.parametrize(
    "padding", ("valid", "constant", "circular", "reflect", "replicate")
)
def test_downsampling_adjointness(padding, device):
    r"""
    Tests downsampling+blur operator adjointness for various image and filter sizes

    :param device: (torch.device) cpu or cuda:x
    """
    torch.manual_seed(0)

    nchannels = ((1, 1), (3, 1), (3, 3))

    for nchan_im, nchan_filt in nchannels:
        size_im = (
            [nchan_im, 5, 5],
            [nchan_im, 6, 6],
            [nchan_im, 5, 6],
            [nchan_im, 6, 5],
        )
        size_filt = (
            [nchan_filt, 3, 3],
            [nchan_filt, 4, 4],
            [nchan_filt, 3, 4],
            [nchan_filt, 4, 3],
        )

        for sim in size_im:
            for sfil in size_filt:
                x = torch.rand(1, *sim).to(device)
                h = torch.rand(1, *sfil).to(device)

                physics = dinv.physics.Downsampling(
                    sim, filter=h, padding=padding, device=device
                )

                Ax = physics.A(x)
                y = torch.rand_like(Ax)
                Aty = physics.A_adjoint(y)
                Axy = torch.sum(Ax * y)
                Atyx = torch.sum(Aty * x)

                assert torch.abs(Axy - Atyx) < 1e-3


def test_prox_l2_downsampling(device):
    nchannels = ((1, 1), (3, 1), (3, 3))

    for nchan_im, nchan_filt in nchannels:
        size_im = ([nchan_im, 16, 16],)
        filters = ["bicubic", "bilinear", "sinc"]

        paddings = ("circular",)

        for pad in paddings:
            for sim in size_im:
                for h in filters:
                    x = torch.rand(sim)[None].to(device)

                    physics = dinv.physics.Downsampling(
                        sim, filter=h, padding=pad, device=device
                    )

                    y = physics(x)
                    # next we test the speedup formula of prox with fft
                    x_prox1 = physics.prox_l2(
                        physics.A_adjoint(y) * 0.0, y, gamma=1e5, use_fft=True
                    )
                    x_prox2 = physics.prox_l2(
                        physics.A_adjoint(y) * 0.0, y, gamma=1e5, use_fft=False
                    )

                    assert torch.abs(x_prox1 - x_prox2).max() < 1e-2


@pytest.mark.parametrize("imsize", ((8, 16),))  # must be even here
@pytest.mark.parametrize("channels", (1, 2))
@pytest.mark.parametrize("factor", (2, 4))
@pytest.mark.parametrize(
    "downsampling", (dinv.physics.Downsampling, dinv.physics.DownsamplingMatlab)
)
def test_downsampling_imsize(imsize, channels, device, factor, downsampling):
    # Test downsampling can update imsize on the fly
    x = torch.rand(1, channels, *imsize, device=device)
    physics = downsampling(device=device, factor=factor)
    assert physics(x).shape == (1, channels, imsize[0] // factor, imsize[1] // factor)
    assert physics.A_adjoint(x).shape == (
        1,
        channels,
        imsize[0] * factor,
        imsize[1] * factor,
    )
    assert physics.adjointness_test(x).abs() < 1e-3


def test_mri_fft():
    """
    Test that our torch FFT is the same as FastMRI FFT implementation.
    The following 5 functions are taken from
    from https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
    """

    def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
        if not data.shape[-1] == 2:
            raise ValueError("Tensor does not have separate complex dim.")

        data = ifftshift(data, dim=[-3, -2])
        data = torch.view_as_real(
            torch.fft.fftn(  # type: ignore
                torch.view_as_complex(data), dim=(-2, -1), norm=norm
            )
        )
        data = fftshift(data, dim=[-3, -2])

        return data

    def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
        shift = shift % x.size(dim)
        if shift == 0:
            return x

        left = x.narrow(dim, 0, x.size(dim) - shift)
        right = x.narrow(dim, x.size(dim) - shift, shift)

        return torch.cat((right, left), dim=dim)

    def roll(x: torch.Tensor, shift: list[int], dim: list[int]) -> torch.Tensor:
        for s, d in zip(shift, dim, strict=True):
            x = roll_one_dim(x, s, d)

        return x

    def fftshift(x: torch.Tensor, dim: list[int] | None = None) -> torch.Tensor:
        if dim is None:
            # this weird code is necessary for toch.jit.script typing
            dim = [0] * (x.dim())
            for i in range(1, x.dim()):
                dim[i] = i

        # also necessary for torch.jit.script
        shift = [0] * len(dim)
        for i, dim_num in enumerate(dim):
            shift[i] = x.shape[dim_num] // 2

        return roll(x, shift, dim)

    def ifftshift(x: torch.Tensor, dim: list[int] | None = None) -> torch.Tensor:
        if dim is None:
            # this weird code is necessary for toch.jit.script typing
            dim = [0] * (x.dim())
            for i in range(1, x.dim()):
                dim[i] = i

        # also necessary for torch.jit.script
        shift = [0] * len(dim)
        for i, dim_num in enumerate(dim):
            shift[i] = (x.shape[dim_num] + 1) // 2

        return roll(x, shift, dim)

    x = torch.randn(4, 2, 16, 8)  # B,C,H,W

    # Our FFT
    xf1 = MRIMixin.from_torch_complex(MRIMixin.fft(MRIMixin.to_torch_complex(x)))

    # FastMRI FFT
    xf2 = fft2c_new(x.moveaxis(1, -1).contiguous()).moveaxis(-1, 1)

    assert torch.all(xf1 == xf2)


@pytest.fixture
def multispectral_channels():
    return 7


@pytest.mark.parametrize("srf", ("flat", "random", "rec601", "list"))
def test_decolorize(srf, device, imsize, multispectral_channels):
    channels = multispectral_channels
    if srf == "list":
        srf = list(range(channels))
        srf = [s / sum(srf) for s in srf]

    physics = dinv.physics.Decolorize(channels=channels, srf=srf, device=device)
    x = torch.ones((1, channels, *imsize[-2:]), device=device)
    x2 = physics.A_adjoint_A(x)

    assert x2.shape == x.shape
    assert torch.allclose(
        physics.srf.sum(), torch.tensor(1.0, device=device), rtol=1e-3
    )
    assert physics.srf.shape[1] == channels


@pytest.mark.parametrize("shear_dir", ["h", "w"])
@pytest.mark.parametrize("cassi_mode", ["ss", "sd"])
def test_CASSI(shear_dir, imsize, device, multispectral_channels, rng, cassi_mode):
    channels = multispectral_channels

    x = torch.ones(1, channels, *imsize[-2:]).to(device)
    physics = dinv.physics.CompressiveSpectralImaging(
        (channels, *imsize[-2:]),
        mask=None,
        mode=cassi_mode,
        shear_dir=shear_dir,
        device=device,
        rng=rng,
    )
    y = physics(x)
    if cassi_mode == "ss":
        assert y.shape == (x.shape[0], 1, *x.shape[2:])
    elif cassi_mode == "sd":
        if shear_dir == "h":
            assert y.shape == (x.shape[0], 1, x.shape[-2] + channels - 1, x.shape[-1])
        elif shear_dir == "w":
            assert y.shape == (x.shape[0], 1, x.shape[-2], x.shape[-1] + channels - 1)

    x_hat = physics.A_adjoint(y)
    assert x_hat.shape == x.shape


def test_unmixing(device):
    physics = dinv.physics.HyperSpectralUnmixing(
        M=torch.tensor(
            [
                [0.5, 0.5, 0.0],  # yellow endmember
                [0.0, 0.0, 1.0],  # blue endmember
            ],
            device=device,
        ),
        device=device,
    )
    # Image of shape B,C,H,W
    # Image consists of 2 pixels, one yellow and one blue
    y = (
        torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            device=device,
        )
        .unsqueeze(-1)
        .unsqueeze(0)
    )
    x_hat = physics.A_adjoint(y)

    assert torch.all(x_hat[:, 0].squeeze() == torch.tensor([1.0, 0.0], device=device))
    assert torch.all(x_hat[:, 1].squeeze() == torch.tensor([0.0, 1.0], device=device))


@pytest.mark.parametrize("name", OPERATORS)
def test_operators_differentiability(name, device):
    r"""
    Tests if a forward operator is differentiable (can perform back-propagation)
    with respect to the input and its physics parameters (if they exist and are floating point tensors).

    :param name: operator name (see find_operator)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts differentiability
    """

    physics, imsize, _, dtype, params = find_operator(
        name, device, get_physics_param=True
    )

    if name == "radio":
        dtype = torch.cfloat

    if "composition" in name:
        pytest.skip("Skip composition operators for differentiability test.")

    # Only test for floating point tensor
    valid_dtype = [torch.float16, torch.float32, torch.float64]
    if dtype in valid_dtype:
        # Differentiate w.r.t to input image
        x = torch.randn(imsize, device=device, dtype=dtype).unsqueeze(0)
        y = physics.A(x)
        x_hat = (
            torch.randn(imsize, device=device, dtype=dtype)
            .unsqueeze(0)
            .requires_grad_(True)
        )
        with torch.enable_grad():
            y_hat = physics.A(x_hat)
            if isinstance(y_hat, TensorList):
                for y_hat_item, y_item in zip(y_hat.x, y.x, strict=True):
                    loss = torch.nn.functional.mse_loss(y_hat_item, y_item)
                    loss.backward()
                    assert x_hat.requires_grad == True
                    assert x_hat.grad is not None
                    assert torch.all(~torch.isnan(x_hat.grad))
            else:
                loss = torch.nn.functional.mse_loss(y_hat, y)
                loss.backward()
                assert x_hat.requires_grad == True
                assert x_hat.grad is not None
                assert torch.all(~torch.isnan(x_hat.grad))

        # Differentiate w.r.t to physics parameters
        # if the buffers are not empty (i.e. there is a parameter)
        if len(physics.state_dict()) > 0 and len(params) > 0:
            x = torch.randn(imsize, device=device, dtype=dtype).unsqueeze(0)
            buffers = copy.deepcopy(dict(physics.named_buffers()))
            parameters = {k: v for k, v in buffers.items() if k in params}
            # Set requires grad
            for k, v in parameters.items():
                if v.dtype in valid_dtype:
                    parameters[k] = v.requires_grad_(True)

            with torch.enable_grad():
                y_hat = physics(x, **parameters)
                if isinstance(y_hat, TensorList):
                    for y_hat_item, y_item in zip(y_hat.x, y.x, strict=True):
                        loss = torch.nn.functional.mse_loss(y_hat_item, y_item)
                        loss.backward()

                        for k, v in parameters.items():
                            if v.dtype in valid_dtype:
                                assert v.requires_grad == True
                                assert v.grad is not None
                                assert torch.all(~torch.isnan(v.grad))

                else:
                    loss = torch.nn.functional.mse_loss(y_hat, y)
                    loss.backward()

                    for k, v in parameters.items():
                        if v.dtype in valid_dtype:
                            assert v.requires_grad == True
                            assert v.grad is not None
                            assert torch.all(~torch.isnan(v.grad))


@pytest.mark.parametrize(
    "name", OPERATORS + NONLINEAR_OPERATORS + PHASE_RETRIEVAL_OPERATORS
)
def test_device_consistency(name):
    r"""
    Tests if a physics can be moved properly between devices.

    :param name: operator name (see find_operator)
    :return: asserts
    """

    def try_find_operator(name):
        physics, imsize, _, dtype = find_operator(name, "cpu")
        return physics, imsize, dtype

    def try_find_nonlinear_operator(name):
        physics, x = find_nonlinear_operator(name, "cpu")
        return physics, x, x[0].dtype if isinstance(x, TensorList) else x.dtype

    def try_find_phase_retrieval_operator(name):
        (
            physics,
            imsize,
        ) = find_phase_retrieval_operator(name, "cpu")
        return physics, imsize, torch.complex64

    for finder in (
        try_find_operator,
        try_find_nonlinear_operator,
        try_find_phase_retrieval_operator,
    ):
        try:
            physics, imsize, dtype = finder(name)
            break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not find an operator for {name}")

    # The current radio physics depends on torchkbnufft, which seems to be not compatible.
    if "radio" in name:
        pytest.skip(
            "Skip 'radio' operator for device consistency test, since the current implementation depends on torchkbnufft, which seems to be not compatible."
        )
    elif "ultrasound" in name:
        # Ultrasound uses scatter_add, which is nondeterministic on CUDA at
        # atomicAdd ordering; CPU vs CUDA differ at ~1e-4 in float32, above
        # the 1e-5 tolerance used here. Adjointness is unaffected.
        pytest.skip(
            "Skip 'ultrasound' operator for device consistency test: "
            "CUDA scatter_add is nondeterministic in float32."
        )
    else:
        # Test CPU
        torch.manual_seed(11)
        # For non linear operators
        if not isinstance(imsize, (torch.Tensor, TensorList)):
            x = torch.randn(imsize, device="cpu", dtype=dtype).unsqueeze(0)
        else:
            x = imsize
        y1 = physics.A(x)
        assert y1.device == torch.device("cpu")
        # Move to GPU if cuda is available
        if torch.cuda.is_available():
            torch.manual_seed(11)
            cuda = torch.device("cuda:0")
            physics = physics.to(cuda)
            x = x.to(cuda)
            y2 = physics.A(x)
            assert y2.device == cuda

            # skip denoising that adds random noise in each forward call
            if not isinstance(physics, dinv.physics.Denoising):
                if isinstance(y2, TensorList):
                    for y11, y22 in zip(y1, y2, strict=True):
                        assert torch.linalg.norm((y11.to(cuda) - y22).ravel()) < 1e-5
                else:
                    assert torch.linalg.norm((y1.to(cuda) - y2).ravel()) < 1e-5


def get_all_tensor_attrs(module, prefix=""):
    """
    Get all tensor attributes of a module.
    """
    tensor_attrs = {}

    def full_name(name):
        return f"{prefix}.{name}" if prefix else name

    # Registered parameters
    for name, parameter in module._parameters.items():
        if parameter is not None:
            tensor_attrs[full_name(name)] = parameter

    # Persistent registered buffers
    for name, buffer in module._buffers.items():
        if buffer is not None and name not in module._non_persistent_buffers_set:
            tensor_attrs[full_name(name)] = buffer

    # Unregistered tensor attributes.
    # Including these preserves the test's ability to detect tensors that
    # should potentially have been registered.
    for name, attr in vars(module).items():
        if isinstance(attr, torch.Tensor):
            tensor_attrs[full_name(name)] = attr

    # Recurse through registered submodules
    for name, submodule in module._modules.items():
        if submodule is not None:
            tensor_attrs.update(
                get_all_tensor_attrs(
                    submodule,
                    prefix=full_name(name),
                )
            )

    return tensor_attrs


def test_get_all_tensor_attrs_is_not_vacuous():
    """
    Test that the get_all_tensor_attrs behaves as expected.
    """

    class TestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.ones(1))
            self.parameter = torch.nn.Parameter(torch.ones(1))
            self.unregistered_tensor = torch.ones(1)

        @property
        def tensor_property(self):
            return torch.ones(1)

    attrs = get_all_tensor_attrs(TestModule())

    assert set(attrs) == {
        "buffer",
        "parameter",
        "unregistered_tensor",
    }


@pytest.mark.parametrize("name", OPERATORS)
def test_physics_state_dict(name, device):
    r"""
    Tests if the physics state dict is well-behaved.

    :param name: operator name (see find_operator)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts state dict is saved.
    """

    def get_all_tensor_attrs(module, prefix=""):
        tensor_attrs = {}

        # Check direct attributes
        for name in dir(module):
            try:
                attr = getattr(module, name)
            except Exception:
                continue  # skip attributes that raise exceptions on access

            full_name = f"{prefix}.{name}" if prefix else name
            if (
                isinstance(attr, torch.Tensor)
                and name not in module._non_persistent_buffers_set
            ):
                tensor_attrs[full_name] = attr
            elif isinstance(attr, torch.nn.ModuleList):
                for i, submodule in enumerate(attr):
                    tensor_attrs.update(
                        get_all_tensor_attrs(submodule, prefix=f"{full_name}.{i}")
                    )
            elif isinstance(attr, torch.nn.Module):
                # Recurse into submodules
                tensor_attrs.update(get_all_tensor_attrs(attr, prefix=full_name))

        return tensor_attrs

    if "ultrasound" in name and str(device).startswith("cuda"):
        pytest.skip(
            "CUDA scatter_add is nondeterministic; two identical forward "
            "passes differ at float32 rounding scale."
        )

    physics, imsize, _, dtype = find_operator(name, device)
    if name == "radio":
        dtype = torch.cfloat

    # A cache dir for saving state dict
    cache_dir = "./cache_test_physics"
    os.makedirs(cache_dir, exist_ok=True)

    # If the buffers are not empty (i.e. there is a parameter)
    if len(physics.state_dict()) > 0:
        state_dict = physics.state_dict()
        # Check that all tensor attributes are in the state dict
        params = get_all_tensor_attrs(physics)

        assert set(state_dict.keys()) == set(params.keys())
        for k, v in params.items():
            assert torch.allclose(state_dict[k], v)

        # Save the state_dict
        torch.save(state_dict, os.path.join(cache_dir, f"{name}.pt"))

        # Reinitialize the physics
        new_physics, _, _, _ = find_operator(name, device)
        # Change to random parameters

        loaded_state_dict = torch.load(os.path.join(cache_dir, f"{name}.pt"))
        new_physics.load_state_dict(loaded_state_dict)
        new_state_dict = new_physics.state_dict()
        # Check that the state dict are identical
        assert set(state_dict.keys()) == set(new_state_dict.keys())
        for k, v in state_dict.items():
            assert torch.equal(v, new_state_dict[k])

        # Check two physics have the same output
        x = torch.randn(imsize, device=device, dtype=dtype).unsqueeze(0)
        y1 = physics(x)
        y2 = new_physics(x)
        if isinstance(y1, TensorList):
            for y1, y2 in zip(physics(x), new_physics(x), strict=True):
                assert torch.allclose(y1, y2)
        else:
            assert torch.allclose(y1, y2)

        # Remove the cache dir
        shutil.rmtree(cache_dir)


def test_composed_physics(device):
    img_size = (3, 32, 32)
    # First physics
    mask_1 = torch.ones(img_size, device=device).unsqueeze(0)
    mask_1[..., 10:15, 13:17] = 0.0
    physics_1 = dinv.physics.Inpainting(img_size=img_size, mask=mask_1, device=device)
    # Second physics
    mask_2 = torch.ones(img_size, device=device).unsqueeze(0)
    mask_2[..., 5:7, 9:13] = 0.0
    physics_2 = dinv.physics.Inpainting(img_size=img_size, mask=mask_2, device=device)

    composed_physics = physics_1 * physics_2  # physics_1(physics_2(.))
    x = torch.randn(img_size, device=device).unsqueeze(0)
    assert torch.equal(composed_physics.A(x), physics_1.A(physics_2.A(x)))
    assert torch.equal(composed_physics.A(torch.ones_like(x)), mask_1 * mask_2)

    # A blur physics
    physics_3 = dinv.physics.BlurFFT(
        img_size=img_size,
        filter=dinv.physics.functional.bilinear_filter(2.0),
        device=device,
    )

    composed_physics = physics_1 * physics_3
    assert torch.equal(composed_physics.A(x), physics_1.A(physics_3.A(x)))
    assert torch.equal(
        composed_physics.A_adjoint(x), physics_3.A_adjoint(physics_1.A_adjoint(x))
    )

    # Compose with Transform:
    physics = dinv.physics.Blur(
        filter=dinv.physics.functional.bicubic_filter(3.0, device=device), device=device
    )
    T = dinv.transform.Shift()
    T_kwargs = {
        "x_shift": torch.tensor([1], device=device),
        "y_shift": torch.tensor([1], device=device),
    }

    physics_mul = physics * dinv.physics.LinearPhysics(
        A=lambda x: T.inverse(x, **T_kwargs),
        A_adjoint=lambda y: T(y, **T_kwargs),
    )
    rng = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(1, 3, 64, 64, device=device, generator=rng)
    assert torch.allclose(physics_mul.A(x), physics.A(T.inverse(x, **T_kwargs)))
    y = physics_mul.A(x)
    assert torch.allclose(physics_mul.A_adjoint(y), T(physics.A_adjoint(y), **T_kwargs))
    assert torch.allclose(
        physics_mul.A_dagger(y),
        T(physics.A_dagger(y), **T_kwargs),
        atol=1e-4,
        rtol=1e-4,
    )

    # test non-linear physics - checking for possible bugs in noise model
    non_lin_physics = dinv.physics.Physics(
        A=lambda x: x**2, noise_model=dinv.physics.GaussianNoise(0.0)
    )
    p = physics * non_lin_physics

    y_2 = physics(non_lin_physics.A(x))
    assert torch.allclose(y_2, p(x))

    assert isinstance(p.noise_model, dinv.physics.ZeroNoise)
    assert isinstance(p, dinv.physics.Physics) and not isinstance(
        p, dinv.physics.LinearPhysics
    )

    p = non_lin_physics * physics

    y_2 = non_lin_physics.A(physics(x))
    assert torch.allclose(y_2, p(x))
    assert p.noise_model.sigma == 0.0


@pytest.mark.parametrize("name", OPERATORS)
def test_adjoint_autograd(name, device):
    # NOTE: The current implementation of adjoint_function does not support
    # physics that return tensor lists or complex tensors. It also does not
    # support RadioInterferometry although it is not entirely clear why.
    if name in {
        "aliased_pansharpen",
        "pansharpen_valid",
        "pansharpen_circular",
        "pansharpen_reflect",
        "pansharpen_replicate",
        "complex_compressed_sensing",
        "ptychography_linear",
        "radio",
        "radio_weighted",
        "pet_2d",
        "pet_3d",
    }:
        pytest.skip(f"Operator {name} is not supported by adjoint_function.")

    if "ultrasound" in name and str(device).startswith("cuda"):
        pytest.skip(
            "CUDA scatter_add is nondeterministic; A vs autograd-adjoint "
            "differ at float32 rounding scale."
        )

    physics, imsize, _, dtype = find_operator(name, device)

    x = torch.randn(imsize, device=device, dtype=dtype).unsqueeze(0)
    y = physics.A(x)

    A_adjoint = adjoint_function(physics.A, x.shape, x.device, x.dtype)

    # Compute Df^\top(z) using autograd where f(z) = A^\top z.
    y.requires_grad_()
    z = torch.randn_like(x, device=device, dtype=dtype)
    l = (z * A_adjoint(y)).sum()
    l.backward()
    # \delta y := \delta_y <z, A^\top y> = Az
    delta_y = y.grad
    Az = physics.A(z)
    assert torch.allclose(delta_y, Az, rtol=1e-5)


@pytest.mark.parametrize(
    "name", OPERATORS + NONLINEAR_OPERATORS + PHASE_RETRIEVAL_OPERATORS
)
def test_clone(name, device):
    if name in OPERATORS:
        physics, imsize, _, dtype = find_operator(name, device)
        if "pet" in name:
            pytest.skip("PET operators cannot be cloned due to parallelproj.")
    elif name in NONLINEAR_OPERATORS:
        if name == "haze":
            pytest.skip(
                "Haze physics takes a TensorList as input, which is not supported by the current test."
            )
        physics, x = find_nonlinear_operator(name, device)
        imsize = x.shape[1:]
        dtype = x.dtype
    elif name in PHASE_RETRIEVAL_OPERATORS:
        physics, imsize = find_phase_retrieval_operator(name, device)
        dtype = torch.complex64

    # Add a dummy parameter used for further testing
    dummy_tensor = torch.randn(
        imsize,
        device=device,
        dtype=dtype,
        generator=torch.Generator(device).manual_seed(0),
    )
    dummy_parameter = torch.nn.Parameter(dummy_tensor)
    physics.register_parameter("dummy", dummy_parameter)

    physics_clone = physics.clone()

    # Test clone type (parent class)
    assert type(physics_clone) == type(physics), "Clone is not of the same type."

    # Check parameters
    parameter_names = set(name for name, _ in physics.named_parameters())
    parameter_names_clone = set(name for name, _ in physics_clone.named_parameters())

    assert parameter_names == parameter_names_clone, "Parameter names do not match."

    for name in parameter_names.intersection(parameter_names_clone):
        param = physics.get_parameter(name)
        param_clone = physics_clone.get_parameter(name)

        # Check that params have been reallocated somewhere else in the memory space
        assert (
            param.data_ptr() != param_clone.data_ptr()
        ), f"Parameter {name} has not been cloned properly."

        # Check that changing one parameter does not change the other
        # NOTE: no_grad is necessary because autograd prevents in-place modifications
        # of leaf variables.
        with torch.no_grad():
            param.fill_(0)
            param_clone.fill_(1)
        assert not torch.allclose(param, param_clone), f"Expected different values"

    # Check buffers
    buffer_names = set(name for name, _ in physics.named_buffers())
    buffer_names_clone = set(name for name, _ in physics_clone.named_buffers())

    assert buffer_names == buffer_names_clone, "Buffer names do not match."

    for name in buffer_names.intersection(buffer_names_clone):
        buffer = physics.get_buffer(name)
        buffer_clone = physics_clone.get_buffer(name)

        # Check that buffers have been reallocated somewhere else in the memory space
        assert (
            buffer.data_ptr() != buffer_clone.data_ptr()
        ), f"Buffer {name} has not been cloned properly."

        # Check that changing one buffer does not change the other
        buffer.fill_(0)
        buffer_clone.fill_(1)
        assert not torch.allclose(buffer, buffer_clone), f"Expected different values"

    # Test that RNGs have been cloned successfully
    rng = getattr(physics, "rng", None)
    rng_clone = getattr(physics_clone, "rng", None)

    assert (rng is not None) == (
        rng_clone is not None
    ), "RNGs are not both set or unset."

    if rng is not None:
        assert torch.all(
            rng.get_state() == rng_clone.get_state()
        ), "RNG state does not match."

        arr = torch.randn(16, device=rng.device, generator=rng)
        arr_clone = torch.randn(16, device=rng_clone.device, generator=rng_clone)
        assert torch.allclose(
            arr, arr_clone
        ), "RNGs do not produce the same random numbers after cloning."

    # Additional tests
    if hasattr(physics, "mask") and physics.mask.dtype != torch.bool:
        # Save original values
        saved_mask = physics.mask
        saved_physics_clone = physics_clone

        physics.mask += 7
        physics_clone = physics.clone()
        assert torch.allclose(
            physics_clone.mask, physics.mask
        ), "Mask has not been cloned properly."

        # Restore original values
        physics_clone.mask = saved_mask
        physics_clone = physics_clone

    # Test other attributes than parameters and buffers
    attr_name = "img_size"
    is_attr = hasattr(physics, attr_name)
    is_parameter = attr_name in [name for name, _ in physics.named_parameters()]
    is_buffer = attr_name in [name for name, _ in physics.named_buffers()]
    if is_attr and not is_parameter and not is_buffer:
        # Save original values
        attr_val = getattr(physics, attr_name)
        attr_val_clone = getattr(physics_clone, attr_name)

        setattr(physics, attr_name, 42)
        physics_clone = physics.clone()
        assert getattr(physics_clone, attr_name) == getattr(
            physics, attr_name
        ), "Attribute has not been cloned properly."

        # Restore original values
        setattr(physics, attr_name, attr_val)
        setattr(physics_clone, attr_name, attr_val_clone)

    # Save original values
    saved_A = physics.A
    physics.A = lambda *args, **kwargs: "hi"

    x = torch.randn(
        imsize,
        device=device,
        dtype=dtype,
        generator=torch.Generator(device).manual_seed(0),
    ).unsqueeze(0)
    assert physics.A(x) == "hi"
    assert physics_clone.A(x) != "hi"

    # Restore original values
    physics.A = saved_A

    # Check requires_grad in parameters and buffers

    saved_physics = physics
    saved_physics_clone = physics_clone

    # Use a clone as the base to avoid mutations across different tests as it
    # may happen when modifying parameters and buffers
    physics = physics.clone()

    for param in physics.parameters():
        if not torch.is_floating_point(param) and not torch.is_complex(param):
            continue
        param.requires_grad = True

    physics_clone = physics.clone()

    for param in physics_clone.parameters():
        if not torch.is_floating_point(param) and not torch.is_complex(param):
            continue
        assert param.requires_grad, "Cloned parameter does not require grad."

    for param in physics.parameters():
        if not torch.is_floating_point(param) and not torch.is_complex(param):
            continue
        param.requires_grad = False

    physics_clone = physics.clone()

    for param in physics_clone.parameters():
        if not torch.is_floating_point(param) and not torch.is_complex(param):
            continue
        assert not param.requires_grad, "Cloned parameter should not require grad."

    for buffer in physics.buffers():
        if not torch.is_floating_point(buffer) and not torch.is_complex(buffer):
            continue
        buffer.requires_grad = True

    physics_clone = physics.clone()

    for buffer in physics_clone.buffers():
        if not torch.is_floating_point(buffer) and not torch.is_complex(buffer):
            continue
        assert buffer.requires_grad, "Cloned buffer does not require grad."

    for buffer in physics.buffers():
        buffer.requires_grad = False

    physics_clone = physics.clone()

    for buffer in physics_clone.buffers():
        assert not buffer.requires_grad, "Cloned buffer should not require grad."

    # Restore original values
    physics = saved_physics
    physics_clone = saved_physics_clone

    # Test autograd
    saved_physics = physics
    saved_physics_clone = physics_clone

    # Use a clone as the base to avoid mutations across different tests as it
    # may happen when modifying parameters and buffers
    physics = physics.clone()

    for param in physics.parameters():
        if not torch.is_floating_point(param) and not torch.is_complex(param):
            continue
        param.requires_grad = True

    physics_clone = physics.clone()

    for param in physics.parameters():
        if not torch.is_floating_point(param):
            continue
        l = param.flatten()[0]
        l.backward()
        assert param.grad is not None, "Parameter gradient is None after backward."

    for param in physics_clone.parameters():
        if not torch.is_floating_point(param):
            continue
        assert param.grad is None, "Cloned parameter should not have a gradient."

    for param in physics.parameters():
        if not torch.is_floating_point(param):
            continue
        param.grad = None  # Reset gradients

    for param in physics_clone.parameters():
        if not torch.is_floating_point(param):
            continue
        param.grad = None  # Reset gradients

    for param in physics_clone.parameters():
        if not torch.is_floating_point(param):
            continue
        l = param.flatten()[0]
        l.backward()
        assert param.grad is not None, "Parameter gradient is None after backward."

    for param in physics.parameters():
        if not torch.is_floating_point(param):
            continue
        assert param.grad is None, "Original parameter should not have a gradient."

    # Restore original values
    physics = saved_physics
    physics_clone = saved_physics_clone


def test_physics_warn_extra_kwargs():
    with pytest.warns(
        UserWarning, match="Arguments {'sigma': 0.5} are passed to Denoising"
    ):
        dinv.physics.Denoising(sigma=0.5)


MULTISCALE_EXCLUSION = [
    # three dimensional signals are currently not supported
    "3Ddeblur_valid",
    "3Ddeblur_circular",
    "3DMRI",
    "3DMultiCoilMRI",
    "pet_3d",
    "DynamicMRI",
    "fast_singlepixel",
    "fast_singlepixel_zig_zag",
    "fast_singlepixel_old_sequency",
    "fast_singlepixel_cake_cutting",
    "fast_singlepixel_xy",
    "ultrasound_planewave",
]


@pytest.mark.parametrize(
    "name", [op for op in OPERATORS if op not in MULTISCALE_EXCLUSION]
)
def test_multiscale_coarse_adjointness(name, device):
    if (
        "MRI" in name
        or "cassi" in name
        or "pet_2d" == name
        or "ptychography_linear" == name
        or "hyperspectral_unmixing" == name
        or "composition2" == name
    ):
        physics, imsize, _, dtype = find_operator(name, device)
    else:
        # make sure the imsize is large enough for multi-scale tests
        imsize = (3, 16, 16)
        physics, imsize, _, dtype = find_operator(name, device, imsize=imsize)

    if not isinstance(physics, dinv.physics.LinearPhysics):
        pytest.skip("Skip " + name + " : not LinearPhysics")

    p_coarse = dinv.physics.to_multiscale(
        physics, imsize, factors=(2,), device=device, dtype=dtype
    )
    p_coarse.set_scale(1)

    assert isinstance(
        p_coarse, dinv.physics.LinearPhysics
    ), "Coarse physics is not LinearPhysics despite base physics being LinearPhysics"

    x = torch.rand(imsize, device=device, dtype=dtype).unsqueeze(0)
    x_coarse = p_coarse.downsample(x)

    error = p_coarse.adjointness_test(x_coarse).abs()
    assert error < 1e-3


@pytest.mark.parametrize(
    "name", [op for op in OPERATORS if op not in MULTISCALE_EXCLUSION]
)
def test_multiscale_A_adjoint_A(name, device):
    if (
        "MRI" in name
        or "cassi" in name
        or "pet_2d" == name
        or "ptychography_linear" == name
        or "hyperspectral_unmixing" == name
        or "composition2" == name
    ):
        physics, imsize, _, dtype = find_operator(name, device)
    else:
        # make sure the imsize is large enough for multi-scale tests
        imsize = (3, 16, 16)
        physics, imsize, _, dtype = find_operator(name, device, imsize=imsize)

    if not isinstance(physics, dinv.physics.LinearPhysics):
        pytest.skip("Skip " + name + " : not LinearPhysics")

    p_coarse = dinv.physics.to_multiscale(
        physics, imsize, factors=(2,), device=device, dtype=dtype
    )
    p_coarse.set_scale(1)

    assert isinstance(
        p_coarse, dinv.physics.LinearPhysics
    ), "Coarse physics is not LinearPhysics despite base physics being LinearPhysics"

    x = torch.rand(imsize, device=device, dtype=dtype).unsqueeze(0)
    x_coarse = p_coarse.downsample(x)

    A = p_coarse.A
    A_adj = p_coarse.A_adjoint
    A_adj_A = p_coarse.A_adjoint_A

    def op_cmp(xc):
        return A_adj(A(xc)) - A_adj_A(xc)

    physics_cmp = dinv.physics.LinearPhysics(
        img_size=imsize, A=op_cmp, A_adjoint=op_cmp
    )

    error = physics_cmp.compute_norm(x_coarse).abs()
    assert error < 0.2


def test_automatic_A_adjoint(device):
    x = torch.randn((2, 3, 8, 8), device=device)
    physics = dinv.physics.LinearPhysics(
        A=lambda x: x.mean(dim=1, keepdim=True), img_size=(3, 8, 8)
    )

    y = physics(x)
    x_adj = physics.A_adjoint(y)
    assert x_adj.shape == x.shape, "A_adjoint shape mismatch."
    assert (
        physics.adjointness_test(x) < 1e-4
    ), "Adjointness test failed for LinearPhysics with automatic A_adjoint."

    # test decomposable physics
    physics = dinv.physics.DecomposablePhysics(
        V_adjoint=lambda s: s.mean(dim=1, keepdim=True), img_size=(3, 8, 8)
    )

    y = physics(x)
    x_adj = physics.A_adjoint(y)

    assert torch.allclose(
        physics.U(x), physics.U_adjoint(x)
    ), "U and U_adjoint should be identity if not provided."
    assert torch.allclose(physics.U(x), x), "U should be identity if not provided."
    assert x_adj.shape == x.shape, "A_adjoint shape mismatch for DecomposablePhysics."
    assert (
        physics.adjointness_test(x) < 1e-4
    ), "Adjointness test failed for DecomposablePhysics with automatic A_adjoint."

    physics = dinv.physics.DecomposablePhysics(
        U=lambda x: x.mean(dim=1, keepdim=True), img_size=(3, 8, 8)
    )

    y = physics(x)
    x_adj = physics.A_adjoint(y)

    assert torch.allclose(
        physics.V(x), physics.V_adjoint(x)
    ), "V and V_adjoint should be identity if not provided."
    assert torch.allclose(physics.V(x), x), "V should be identity if not provided."
    assert x_adj.shape == x.shape, "A_adjoint shape mismatch for DecomposablePhysics."
    assert (
        physics.adjointness_test(x) < 1e-4
    ), "Adjointness test failed for DecomposablePhysics with automatic A_adjoint."


def test_separate_noise_models():
    physics1 = dinv.physics.Denoising()
    physics2 = dinv.physics.Denoising()
    assert id(physics1.noise_model) != id(
        physics2.noise_model
    ), "Expected distinct noise models for the distinct physics"
    assert isinstance(
        physics1.noise_model, dinv.physics.GaussianNoise
    ), f"Expected the default noise model to be GaussianNoise, got {type(physics1.noise_model).__name__}"
    sigma1 = physics1.noise_model.sigma
    sigma2 = physics2.noise_model.sigma
    sigma1_new = sigma2 + 1
    assert (
        sigma1_new != sigma2
    ), "Expected a standard deviation different from that of physics2"
    physics1.update(sigma=sigma1_new)
    assert (
        physics2.noise_model.sigma == sigma2
    ), "Expected physics2 to be unchanged after updating physics1"


def test_downsampling_default_filter_depreciation():
    with pytest.warns(
        UserWarning,
        match="deprecated",
    ):
        _ = dinv.physics.Downsampling()


@pytest.mark.parametrize("wavenumber", [21.55])
@pytest.mark.parametrize("contrast", [0.1, 1.0])
@pytest.mark.parametrize("wave_type", ["circular_wave", "plane_wave"])
def test_scattering_mie(device, wavenumber, contrast, wave_type):
    r"""
    This test uses the closed-form Mie theory solution for computing the total
    field of a single cylinder to validate our Scattering physics implementation.

    See https://opg.optica.org/oe/viewmedia.cfm?uri=oe-25-18-21786&html=true for more details.

    We limit the number of tests, since this is a rather long test
    """
    try:
        import scipy  # noqa: F401
    except ImportError:
        pytest.skip("Scipy is required for this test.")

    # skip if windows
    if os.name == "nt":
        pytest.skip("Scipy's special functions are not well supported on Windows.")

    wavenumber = torch.tensor([wavenumber])
    cylinder_contrast = contrast
    cylinder_radius = 0.25
    pixels = 64
    dtype = torch.complex128
    angles = 4
    radius_tx = 1.0

    transmitters, receivers = dinv.physics.scattering.circular_sensors(
        angles, radius=radius_tx, device=device
    )

    physics = dinv.physics.Scattering(
        img_width=pixels,
        device=device,
        background_wavenumber=wavenumber,
        wave_type=wave_type,
        transmitters=transmitters,
        receivers=receivers,
        verbose=True,
    )

    # test adjointness of the born sub-operator
    assert (
        physics.born_operator.adjointness_test(
            torch.ones((1, 1, pixels, pixels), device=device, dtype=dtype)
        ).abs()
        < 1e-4
    ), "Adjointness test failed for the Born sub-operator of the Scattering physics."

    n_coeffs = 55

    total_field_mie, incident_field_mie = dinv.physics.scattering.mie_theory(
        wavenumber,
        cylinder_radius,
        cylinder_contrast,
        pixels,
        wave_type=wave_type,
        angles=torch.linspace(0, 2 * torch.pi, angles + 1, device=device)[:-1],
        dtype=dtype,
        device=device,
        n_coeffs=n_coeffs,
        transmitter_radius=radius_tx,
    )

    # create cylinder contrast
    x = torch.zeros((pixels, pixels), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(
        torch.linspace(-0.5, 0.5, pixels, device=device),
        torch.linspace(-0.5, 0.5, pixels, device=device),
        indexing="ij",
    )
    r = torch.sqrt(xx**2 + yy**2)
    x[r <= cylinder_radius] = cylinder_contrast
    x = x.unsqueeze(0).unsqueeze(0)

    total_field = physics.compute_total_field(x)
    incident_field = physics.incident_field

    assert (
        incident_field - incident_field_mie
    ).abs().mean() < 1e-3, "theoretical and empirical incident fields do not match"
    assert (
        total_field - total_field_mie
    ).abs().mean() < 1e-1, "theoretical and empirical total fields do not match"


def test_squared_or_non_squared_norms(device):
    name = "fftdeblur"
    physics, imsize, _, dtype = find_operator(name, device)
    rng = torch.Generator(device)
    x = torch.randn(imsize, device=device, dtype=dtype, generator=rng).unsqueeze(0)
    sqnorm1 = physics.compute_sqnorm(x, max_iter=1000, tol=1e-9)
    norm = physics.compute_norm(x, max_iter=1000, tol=1e-9, squared=False)

    with pytest.warns(DeprecationWarning, match="compute_sqnorm"):
        sqnorm2 = physics.compute_norm(x, max_iter=1000, tol=1e-9, squared=True)

    assert torch.allclose(sqnorm1, sqnorm2, rtol=1e-4), "squared norms do not match"
    assert torch.allclose(sqnorm1, norm**2, rtol=1e-4), "norms do not match"


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_channels", [1, 3])
@pytest.mark.parametrize("img_size", [(32, 32), (33, 33), (32, 33)])
@pytest.mark.parametrize("patch_size", [8, 9, (8, 9)])
@pytest.mark.parametrize("stride", [4, 5, (4, 5)])
@pytest.mark.parametrize("psf_size", [(5, 5), (6, 6), (5, 6)])
@pytest.mark.parametrize("use_fft", [False, True])
def test_tiled_product_physics_adjointness(
    batch_size, n_channels, img_size, patch_size, psf_size, stride, use_fft, device
):
    from deepinv.physics.blur import TiledSpaceVaryingBlur

    x = torch.randn(batch_size, n_channels, *img_size).to(device)

    n_filters = TiledSpaceVaryingBlur.num_filters(
        img_size=img_size, patch_size=patch_size, stride=stride
    )
    h = torch.rand(1, n_channels, n_filters[0] * n_filters[1], *psf_size).to(device)

    physics = TiledSpaceVaryingBlur(
        filters=h,
        patch_size=patch_size,
        stride=stride,
        use_fft=use_fft,
        device=device,
    )

    Ax = physics.A(x)
    y = torch.randn_like(Ax)
    Aty = physics.A_adjoint(y)
    # Lower a bit the tolerence on Windows. It seems that there is a small numerical error on Windows
    is_windows = os.name == "nt"
    tol = 1e-2 if is_windows else 5e-3
    lhs = torch.sum(Ax * y)
    rhs = torch.sum(Aty * x)
    assert torch.allclose(lhs, rhs, rtol=tol, atol=5e-4)


# ---------------------------------------------------------------------------
# UltrasoundPlaneWave tests
# ---------------------------------------------------------------------------


def _picmus_like_config(dtype=torch.float32):
    """Return a small PICMUS-like config for ultrasound tests.

    Kept intentionally small so tests run quickly while still exercising every
    non-trivial code path (multiple angles, off-boresight element positions,
    IQ demodulation phase compensation).
    """
    n_elements = 32
    pitch = 3e-4  # 300 μm — L11-5v-ish
    n_angles = 7
    ele_x = torch.linspace(
        -pitch * (n_elements - 1) / 2, pitch * (n_elements - 1) / 2, n_elements
    ).to(dtype)
    ele_pos = torch.stack([ele_x, torch.zeros(n_elements, dtype=dtype)], dim=-1)
    import math as _math

    angles = torch.linspace(
        -_math.radians(16.0), _math.radians(16.0), steps=n_angles, dtype=dtype
    )
    # Explicit pixel_size so the test does not depend on the default
    # wavelength convention. Half-wavelength grid at fc = 5.208 MHz.
    lam = 1540.0 / 5.208e6
    return dict(
        img_size=(64, 48),
        angles=angles,
        element_positions=ele_pos,
        n_samples=384,
        sampling_frequency=20.832e6,
        demodulation_frequency=5.208e6,
        sound_speed=1540.0,
        pixel_size=(lam / 2.0, lam / 2.0),
        normalize=False,
        dtype=dtype,
    )


def _reference_cubdl_das(cfg, y_iq):
    """Self-contained CUBDL-style DAS reference (verbatim port of DAS_PW.forward).

    Implements Hyun et al.'s reference DAS from the CUBDL repo
    (``dperdios/... cubdl/das_torch.py``) in the plane-wave case:

      * ``delay_plane``: :math:`(x \\sin\\theta + z \\cos\\theta)/c`
      * ``delay_focus``: :math:`\\|(x,z) - (x_e, z_e)\\| / c`
      * ``grid_sample`` bilinear with ``align_corners=False`` on a
        ``(1, 2, 1, n_samples)`` IQ tensor
      * ``tshift = delays/fs - 2z/c`` phase compensation at ``fdemod``
      * coherent sum over transmit angles and receive elements

    :param cfg: kwargs used to build the :class:`UltrasoundPlaneWave` under test.
    :param y_iq: measurement tensor ``(B, 2, n_a, n_e, n_s)``.
    :return: DAS output ``(B, 2, Z, X)``.
    """
    import math as _math
    from torch.nn.functional import grid_sample

    B = y_iq.shape[0]
    Z, X = cfg["img_size"]
    angles = torch.as_tensor(cfg["angles"]).to(y_iq.dtype).to(y_iq.device)
    ele_pos = torch.as_tensor(cfg["element_positions"]).to(y_iq.dtype).to(y_iq.device)
    n_a = angles.shape[0]
    n_e = ele_pos.shape[0]
    n_s = cfg["n_samples"]
    fs = cfg["sampling_frequency"]
    c = cfg["sound_speed"]
    fdemod = cfg["demodulation_frequency"]

    # Build the same pixel grid as the physics under test (cfg supplies it).
    dz, dx = cfg["pixel_size"]
    x_center = 0.5 * (ele_pos[:, 0].min() + ele_pos[:, 0].max()).item()
    z0 = 0.0
    x0 = x_center - dx * (X - 1) / 2.0
    z_axis = z0 + dz * torch.arange(Z, dtype=y_iq.dtype, device=y_iq.device)
    x_axis = x0 + dx * torch.arange(X, dtype=y_iq.dtype, device=y_iq.device)
    zz, xx = torch.meshgrid(z_axis, x_axis, indexing="ij")
    grid = torch.stack([xx, zz], dim=-1).reshape(-1, 2)  # (Z*X, 2)
    xg = grid[:, 0]
    zg = grid[:, 1]

    # CUBDL delays (in meters, then samples).
    txdel_m = xg.unsqueeze(0) * torch.sin(angles).unsqueeze(-1) + zg.unsqueeze(
        0
    ) * torch.cos(angles).unsqueeze(
        -1
    )  # (n_a, Z*X)
    dxe = xg.unsqueeze(0) - ele_pos[:, 0].unsqueeze(1)
    dze = zg.unsqueeze(0) - ele_pos[:, 1].unsqueeze(1)
    rxdel_m = torch.hypot(dxe, dze)  # (n_e, Z*X)
    txdel = txdel_m * fs / c
    rxdel = rxdel_m * fs / c

    idas = torch.zeros(B, Z * X, dtype=y_iq.dtype, device=y_iq.device)
    qdas = torch.zeros(B, Z * X, dtype=y_iq.dtype, device=y_iq.device)
    for b in range(B):
        for k in range(n_a):
            for e in range(n_e):
                iq = torch.stack((y_iq[b, 0, k, e], y_iq[b, 1, k, e]), dim=0).view(
                    1, 2, 1, n_s
                )
                delays = txdel[k] + rxdel[e]  # (Z*X,)
                dgs = (delays.view(1, 1, -1, 1) * 2 + 1) / n_s - 1
                dgs = torch.cat((dgs, torch.zeros_like(dgs)), dim=-1)
                ifoc, qfoc = grid_sample(iq, dgs, align_corners=False).view(2, -1)
                tshift = delays / fs - grid[:, 1] * 2.0 / c
                theta = 2.0 * _math.pi * fdemod * tshift
                ir = ifoc * torch.cos(theta) - qfoc * torch.sin(theta)
                qr = qfoc * torch.cos(theta) + ifoc * torch.sin(theta)
                idas[b] = idas[b] + ir
                qdas[b] = qdas[b] + qr

    out = torch.stack([idas, qdas], dim=1).reshape(B, 2, Z, X)
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ultrasound_planewave_adjointness(dtype, device, rng):
    """Adjointness of UltrasoundPlaneWave in both real dtypes."""
    if device != torch.device("cpu") and dtype == torch.float64:
        pytest.skip("float64 GPU can be slow on some CI runners; CPU is enough.")
    cfg = _picmus_like_config(dtype=dtype)
    physics = dinv.physics.UltrasoundPlaneWave(**cfg, device=device)
    x = torch.randn(1, 2, *cfg["img_size"], dtype=dtype, device=device, generator=rng)
    err = physics.adjointness_test(x).abs().item()
    if dtype == torch.float64:
        assert err < 1e-8, f"adjointness fp64 error = {err}"
    else:
        assert err < 5e-3, f"adjointness fp32 error = {err}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ultrasound_planewave_das_cubdl_parity(dtype, device, rng):
    """Numerical parity of A_adjoint with a self-contained CUBDL DAS reference."""
    if device != torch.device("cpu") and dtype == torch.float64:
        pytest.skip("float64 GPU can be slow on some CI runners; CPU is enough.")
    cfg = _picmus_like_config(dtype=dtype)
    physics = dinv.physics.UltrasoundPlaneWave(**cfg, device=device)

    n_a = cfg["angles"].shape[0]
    n_e = cfg["element_positions"].shape[0]
    y = torch.randn(
        1,
        2,
        n_a,
        n_e,
        cfg["n_samples"],
        dtype=dtype,
        device=device,
        generator=rng,
    )
    x_ours = physics.A_adjoint(y)
    x_ref = _reference_cubdl_das(cfg, y)

    ref_scale = x_ref.abs().max().clamp(min=1e-12)
    err = (x_ours - x_ref).abs().max() / ref_scale
    if dtype == torch.float64:
        assert err.item() < 1e-8, f"CUBDL parity fp64 rel error = {err.item()}"
    else:
        assert err.item() < 5e-5, f"CUBDL parity fp32 rel error = {err.item()}"


def test_ultrasound_planewave_point_scatterer_localization(device):
    """A(δ_p) then A_adjoint(y) should peak near the scatterer location."""
    cfg = _picmus_like_config(dtype=torch.float32)
    physics = dinv.physics.UltrasoundPlaneWave(**cfg, device=device)
    Z, X = cfg["img_size"]
    zp, xp = Z // 2, X // 2
    x = torch.zeros(1, 2, Z, X, device=device)
    x[0, 0, zp, xp] = 1.0
    y = physics.A(x)
    x_das = physics.A_adjoint(y)
    envelope = torch.sqrt(x_das[0, 0] ** 2 + x_das[0, 1] ** 2)
    peak = torch.argmax(envelope.flatten()).item()
    zpk, xpk = peak // X, peak % X
    assert (
        abs(zpk - zp) <= 1 and abs(xpk - xp) <= 1
    ), f"peak at ({zpk},{xpk}) but scatterer at ({zp},{xp})"


@pytest.mark.parametrize("interp", ["nearest", "linear", "keys"])
def test_ultrasound_planewave_interp_adjointness(interp, device, rng):
    """All interpolation kernels preserve exact adjointness in fp64."""
    cfg = _picmus_like_config(dtype=torch.float64)
    physics = dinv.physics.UltrasoundPlaneWave(**cfg, interp=interp, device=device)
    x = torch.randn(
        1, 2, *cfg["img_size"], dtype=torch.float64, device=device, generator=rng
    )
    err = physics.adjointness_test(x).abs().item()
    assert err < 1e-8, f"interp={interp} adjointness fp64 error = {err}"


def test_ultrasound_planewave_transducer_impulse_response_adjointness(device, rng):
    """Transducer pulse-echo impulse response preserves exact mathematical adjointness in fp64."""
    cfg = _picmus_like_config(dtype=torch.float64)

    # Generate a simple 1D pulse tensor
    pulse = torch.randn(15, dtype=torch.float64, device=device)

    physics = dinv.physics.UltrasoundPlaneWave(
        **cfg,
        pulse=pulse,
        device=device,
    )
    # Check that h is registered and has energy 1.0
    assert physics.pulse_echo_ir is not None
    assert torch.allclose(
        torch.linalg.norm(physics.pulse_echo_ir), torch.tensor(1.0, dtype=torch.float64)
    )

    x = torch.randn(
        1, 2, *cfg["img_size"], dtype=torch.float64, device=device, generator=rng
    )
    err = physics.adjointness_test(x).abs().item()
    assert err < 1e-8, f"transducer impulse response adjointness fp64 error = {err}"


def test_ultrasound_planewave_nearest_matches_rounded_gather(device, rng):
    """Nearest-neighbor adjoint reads exactly the round(τ·fs)-th sample."""
    cfg = _picmus_like_config(dtype=torch.float64)
    physics = dinv.physics.UltrasoundPlaneWave(
        **cfg,
        interp="nearest",
        device=device,
    )
    n_a = cfg["angles"].shape[0]
    n_e = cfg["element_positions"].shape[0]
    y = torch.randn(
        1,
        2,
        n_a,
        n_e,
        cfg["n_samples"],
        dtype=torch.float64,
        device=device,
        generator=rng,
    )
    x_das = physics.A_adjoint(y)

    # Independent slow-but-explicit nearest-neighbor DAS.
    import math as _math

    Z, X = cfg["img_size"]
    grid = physics.pixel_grid.reshape(-1, 2)
    xg, zg = grid[:, 0], grid[:, 1]
    ele_pos = physics.element_positions
    angles = physics.angles
    fs = physics.fs
    c = physics.c
    fdemod = physics.demodulation_frequency
    y_iq = y[0, 0] + 1j * y[0, 1]  # (n_a, n_e, n_s) complex
    x_c = torch.zeros(Z * X, dtype=torch.complex128, device=device)
    for k in range(n_a):
        tau_tx = (xg * torch.sin(angles[k]) + zg * torch.cos(angles[k])) / c
        for e in range(n_e):
            dxe = xg - ele_pos[e, 0]
            dze = zg - ele_pos[e, 1]
            tau_rx = torch.hypot(dxe, dze) / c
            tau = tau_tx + tau_rx
            s = tau * fs
            idx = torch.round(s + 0.0).to(torch.long)  # torch rounds half-even
            # Match physics: floor(s + 0.5) — round half up.
            idx = torch.floor(s + 0.5).to(torch.long)
            valid = (idx >= 0) & (idx <= cfg["n_samples"] - 1)
            idx_c = idx.clamp(0, cfg["n_samples"] - 1)
            samples = y_iq[k, e, idx_c]
            samples = torch.where(valid, samples, torch.zeros_like(samples))
            phase = torch.exp(1j * 2 * _math.pi * fdemod * (tau - 2 * zg / c))
            x_c = x_c + samples * phase
    x_ref = torch.stack([x_c.real, x_c.imag], dim=0).reshape(2, Z, X)
    assert torch.allclose(x_das[0], x_ref, atol=1e-10, rtol=1e-10)


def test_ultrasound_planewave_keys_matches_kernel(device, rng):
    """Keys cubic gather matches an explicit 4-tap Keys-1981 kernel."""
    cfg = _picmus_like_config(dtype=torch.float64)
    # Small tensor so we can afford the reference loop.
    physics = dinv.physics.UltrasoundPlaneWave(
        **cfg,
        interp="keys",
        device=device,
    )
    y = torch.randn(
        1,
        2,
        cfg["angles"].shape[0],
        cfg["element_positions"].shape[0],
        cfg["n_samples"],
        dtype=torch.float64,
        device=device,
        generator=rng,
    )
    x_das = physics.A_adjoint(y)

    def keys_kernel(t):
        at = torch.abs(t)
        t2 = at * at
        t3 = t2 * at
        inner = 1.5 * t3 - 2.5 * t2 + 1.0
        outer = -0.5 * t3 + 2.5 * t2 - 4.0 * at + 2.0
        w = torch.where(at <= 1.0, inner, outer)
        return torch.where(at <= 2.0, w, torch.zeros_like(w))

    import math as _math

    Z, X = cfg["img_size"]
    grid = physics.pixel_grid.reshape(-1, 2)
    xg, zg = grid[:, 0], grid[:, 1]
    ele_pos = physics.element_positions
    angles = physics.angles
    fs, c, fdemod = physics.fs, physics.c, physics.demodulation_frequency
    n_s = cfg["n_samples"]
    y_iq = y[0, 0] + 1j * y[0, 1]
    x_c = torch.zeros(Z * X, dtype=torch.complex128, device=device)
    for k in range(angles.shape[0]):
        tau_tx = (xg * torch.sin(angles[k]) + zg * torch.cos(angles[k])) / c
        for e in range(ele_pos.shape[0]):
            dxe = xg - ele_pos[e, 0]
            dze = zg - ele_pos[e, 1]
            tau = tau_tx + torch.hypot(dxe, dze) / c
            s = tau * fs
            s_floor = torch.floor(s)
            acc = torch.zeros_like(s, dtype=torch.complex128)
            for off in (-1, 0, 1, 2):
                idx = (s_floor + off).to(torch.long)
                valid = (idx >= 0) & (idx <= n_s - 1)
                idx_c = idx.clamp(0, n_s - 1)
                w = keys_kernel(s - (s_floor + off))
                w = torch.where(valid, w, torch.zeros_like(w))
                acc = acc + y_iq[k, e, idx_c] * w
            phase = torch.exp(1j * 2 * _math.pi * fdemod * (tau - 2 * zg / c))
            x_c = x_c + acc * phase
    x_ref = torch.stack([x_c.real, x_c.imag], dim=0).reshape(2, Z, X)
    assert torch.allclose(x_das[0], x_ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("window", ["rect", "hann", "hamming", "tukey0.25"])
def test_ultrasound_planewave_rx_apod_adjointness(window, device, rng):
    """Windowed receive apodization preserves adjointness."""
    cfg = _picmus_like_config(dtype=torch.float64)
    physics = dinv.physics.UltrasoundPlaneWave(
        **cfg,
        f_number=1.75,
        rx_apod_window=window,
        device=device,
    )
    x = torch.randn(
        1, 2, *cfg["img_size"], dtype=torch.float64, device=device, generator=rng
    )
    err = physics.adjointness_test(x).abs().item()
    assert err < 1e-8, f"apod={window} adjointness fp64 error = {err}"


@pytest.mark.parametrize("window", ["hann", "hamming", "tukey0.25"])
def test_ultrasound_planewave_tx_apod_adjointness(window, device, rng):
    """Windowed transmit apodization preserves adjointness."""
    cfg = _picmus_like_config(dtype=torch.float64)
    physics = dinv.physics.UltrasoundPlaneWave(
        **cfg,
        tx_apod_window=window,
        device=device,
    )
    x = torch.randn(
        1, 2, *cfg["img_size"], dtype=torch.float64, device=device, generator=rng
    )
    err = physics.adjointness_test(x).abs().item()
    assert err < 1e-8, f"tx apod={window} adjointness fp64 error = {err}"


def test_ultrasound_planewave_select_angles(device, rng):
    """select_angles(indices) matches slicing the full measurement."""
    cfg = _picmus_like_config(dtype=torch.float32)
    physics = dinv.physics.UltrasoundPlaneWave(**cfg, device=device)
    x = torch.randn(1, 2, *cfg["img_size"], device=device, generator=rng)
    y_full = physics.A(x)
    indices = [0, 2, 5]
    physics_sub = physics.select_angles(indices)
    y_sub = physics_sub.A(x)
    # scatter_add is nondeterministic on CUDA — allow float32-eps drift.
    y_ref = y_full[:, :, indices]
    err = (y_sub - y_ref).abs().max() / y_ref.abs().max().clamp(min=1e-12)
    assert err.item() < 1e-5, f"select_angles relative error = {err.item()}"


# ---------------------------------------------------------------------------
# Native RF signal mode (signal_kind="rf")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ultrasound_planewave_rf_adjointness(dtype, device, rng):
    """RF-mode PW operator satisfies the dot-product adjointness test."""
    if device != torch.device("cpu") and dtype == torch.float64:
        pytest.skip("float64 GPU can be slow on some CI runners; CPU is enough.")
    cfg = _picmus_like_config(dtype=dtype)
    physics = dinv.physics.UltrasoundPlaneWave(**cfg, signal_kind="rf", device=device)
    # RF signals are single-channel real.
    x = torch.randn(1, 1, *cfg["img_size"], dtype=dtype, device=device, generator=rng)
    err = physics.adjointness_test(x).abs().item()
    if dtype == torch.float64:
        assert err < 1e-8, f"RF adjointness fp64 error = {err}"
    else:
        assert err < 5e-3, f"RF adjointness fp32 error = {err}"


def test_ultrasound_planewave_rf_shape(device):
    """RF-mode operator uses single-channel shapes throughout."""
    cfg = _picmus_like_config(dtype=torch.float32)
    physics = dinv.physics.UltrasoundPlaneWave(**cfg, signal_kind="rf", device=device)
    Z, X = cfg["img_size"]
    n_a = cfg["angles"].shape[0]
    n_e = cfg["element_positions"].shape[0]
    n_s = cfg["n_samples"]

    x = torch.randn(2, 1, Z, X, device=device)
    y = physics.A(x)
    assert y.shape == (2, 1, n_a, n_e, n_s), f"got {y.shape}"
    assert y.dtype == torch.float32

    x_das = physics.A_adjoint(y)
    assert x_das.shape == (2, 1, Z, X)

    # Wrong channel count should raise.
    with pytest.raises(ValueError, match="Expected image of shape"):
        physics.A(torch.randn(2, 2, Z, X, device=device))
    with pytest.raises(ValueError, match="Expected measurement of shape"):
        physics.A_adjoint(torch.randn(2, 2, n_a, n_e, n_s, device=device))


def test_ultrasound_planewave_rf_matches_iq_zero_imag(device, rng):
    """RF forward = IQ forward with fdemod=0 applied to (real, 0).

    With no baseband demodulation (fdemod=0) the IQ phase term collapses
    to 1; the operator then becomes real-linear and does not mix the I and Q
    channels. Placing an RF signal in channel 0 and leaving channel 1 = 0
    must reproduce the RF-mode output on channel 0, with channel 1 exactly 0.
    """
    dtype = torch.float64
    cfg = _picmus_like_config(dtype=dtype)
    physics_rf = dinv.physics.UltrasoundPlaneWave(
        **cfg, signal_kind="rf", device=device
    )
    physics_iq = dinv.physics.UltrasoundPlaneWave(
        **{**cfg, "demodulation_frequency": 0.0}, signal_kind="iq", device=device
    )

    x_rf = torch.randn(
        1, 1, *cfg["img_size"], dtype=dtype, device=device, generator=rng
    )
    x_iq = torch.cat([x_rf, torch.zeros_like(x_rf)], dim=1)  # (1, 2, Z, X)

    y_rf = physics_rf.A(x_rf)  # (1, 1, n_t, n_e, n_s)
    y_iq = physics_iq.A(x_iq)  # (1, 2, n_t, n_e, n_s)

    # I channel of IQ output must match RF output.
    assert torch.allclose(y_iq[:, 0:1], y_rf, atol=1e-12, rtol=1e-12)
    # Q channel must be exactly zero (no cross-mixing without carrier phase).
    assert y_iq[:, 1].abs().max().item() < 1e-12


def test_ultrasound_planewave_rf_point_scatterer(device):
    """RF forward + adjoint localizes a point scatterer within one pixel."""
    cfg = _picmus_like_config(dtype=torch.float32)
    physics = dinv.physics.UltrasoundPlaneWave(**cfg, signal_kind="rf", device=device)
    Z, X = cfg["img_size"]
    zp, xp = Z // 2, X // 2
    x = torch.zeros(1, 1, Z, X, device=device)
    x[0, 0, zp, xp] = 1.0
    y = physics.A(x)
    x_das = physics.A_adjoint(y)
    peak = torch.argmax(x_das[0, 0].flatten()).item()
    zpk, xpk = peak // X, peak % X
    assert abs(zpk - zp) <= 1 and abs(xpk - xp) <= 1


def test_ultrasound_planewave_normalize_unit_norm(device, rng):
    """normalize=True brings the squared operator norm to ~1."""
    cfg = _picmus_like_config(dtype=torch.float64)
    cfg.pop("normalize", None)
    physics = dinv.physics.UltrasoundPlaneWave(**cfg, normalize=True, device=device)
    x = torch.randn(
        1, 2, *cfg["img_size"], dtype=torch.float64, device=device, generator=rng
    )
    sqnorm = physics.compute_sqnorm(x, verbose=False).item()
    assert abs(sqnorm - 1.0) < 1e-2, f"||A||^2 = {sqnorm}, expected ~1"
    # Adjointness still holds under normalization.
    assert physics.adjointness_test(x).abs().item() < 1e-8


def test_ultrasound_planewave_normalize_none_emits_warning(device):
    """Unset normalize should emit a UserWarning (Tomography convention)."""
    cfg = _picmus_like_config(dtype=torch.float32)
    cfg.pop("normalize", None)
    with pytest.warns(UserWarning, match="normalize"):
        dinv.physics.UltrasoundPlaneWave(**cfg, device=device)


def test_ultrasound_planewave_invalid_sound_speed(device):
    """Zero or negative sound speed should raise a ValueError."""
    cfg = _picmus_like_config(dtype=torch.float32)
    cfg["sound_speed"] = 0.0
    with pytest.raises(ValueError, match="sound_speed"):
        dinv.physics.UltrasoundPlaneWave(**cfg, device=device)


def test_ultrasound_planewave_rf_select_transmits_preserves_kind(device, rng):
    """select_transmits round-trips signal_kind='rf'."""
    cfg = _picmus_like_config(dtype=torch.float32)
    physics = dinv.physics.UltrasoundPlaneWave(**cfg, signal_kind="rf", device=device)
    sub = physics.select_transmits([0, 2, 5])
    assert sub.signal_kind == "rf"
    assert sub.n_channels == 1
    x = torch.randn(1, 1, *cfg["img_size"], device=device, generator=rng)
    y_sub = sub.A(x)
    assert y_sub.shape[1] == 1
