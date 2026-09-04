from __future__ import annotations
import inspect
from pathlib import Path

import torch

from deepinv.models.base import Reconstructor
from deepinv.models.utils import load_state_dict_from_url
from deepinv.utils.mixins import MRIMixin
from deepinv.physics.mri import MultiCoilMRI, MRI


class DIRECTModel(Reconstructor, MRIMixin):
    r"""Pretrained DIRECT (NKI-AI) multi-coil MRI reconstruction models.

    Runs a model from `DIRECT <https://github.com/NKI-AI/direct>`_. Calgary-Campinas 12-coil brain models
    (from `<https://huggingface.co/NKI-AI/direct-calgary-campinas>`_, fixed 5x or 10x acc):

    - `jointicnet_5x` (or `_10x`)
    - `recurrentvarnet_5x` (or `_10x`)
    - `varnet_5x` (or `_10x`)
    - `conjgradnet_5x` (or `_10x`)
    - `iterdualnet_5x` (or `_10x`)
    - `kikinet_5x` (or `_10x`)
    - `lpdnet_5x` (or `_10x`)
    - `unet_5x` (or `_10x`)
    - `xpdnet_5x` (or `_10x`)
    - `multidomainnet` (downloaded from https://files.aiforoncology.nl/direct-project, repaired locally, uploaded to https://huggingface.co/Andrewwango/direct)

    UNIFORM multi-anatomy vSHARP (one model, from `<https://huggingface.co/NKI-AI/direct-uniform>`_, trained on
    multi-anatomy fastMRI + CMRxRecon, default 4x), pick the anatomy config:

    - `vsharp_brain` (or `vsharp_cardiac`, `vsharp_knee`, `vsharp_prostate`)

    The wrapped models handle the MRI physics and estimate coil maps themselves.

    .. note::
        deepinv uses centered FFTs but DIRECT uncentered, so we pre-shift `y` (a checkerboard modulation) into
        DIRECT's convention. The output scale is not preserved, so its intensity is proportional to but not equal to `y`.

    This model requires DIRECT >=2.2.0 and Python >=3.12. Install it with `pip install deepinv[direct]`.

    :param str model_name: model name, see list above.
    :param bool, str, Path pretrained: `True` or `download` downloads `model_name` weights to `models_dir` (skipped if already present). Or pass checkpoint path directly.
    :param str, Path models_dir: see above; defaults to torch hub cache (:func:`torch.hub.get_dir`).
    :param torch.device, str device: device.

    |sep|

    :Example:

    TODO
    """

    def __init__(
        self,
        model_name: str = "jointicnet_5x",
        pretrained: bool | str | Path = True,
        models_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__(device=device)

        try:
            from omegaconf import OmegaConf
            from direct.config.defaults import DefaultConfig
            from direct.environment import (
                load_models_into_environment_config,
                build_operators,
                initialize_models_from_config,
                setup_engine,
            )
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "DIRECT package not found. Please install it with `pip install deepinv[direct]` (requires Python >=3.12)."
            ) from e

        # (repo, weights file, config file) per model family. vSHARP shares one weights file across anatomies.
        if model_name.startswith("vsharp_"):
            repo, weights_file, cfg_file = (
                "NKI-AI/direct-uniform",
                "uniform_vsharp.pt",
                f"uniform_{model_name[len('vsharp_'):]}.yaml",
            )
        elif model_name.startswith("multidomainnet"):
            repo, weights_file, cfg_file = (
                "Andrewwango/direct",
                f"{model_name}.pt",
                f"{model_name}.yaml",
            )
        else:
            repo, weights_file, cfg_file = (
                "NKI-AI/direct-calgary-campinas",
                f"{model_name}.pt",
                f"{model_name}.yaml",
            )
        models_dir = (
            Path(models_dir) if models_dir is not None else Path(torch.hub.get_dir())
        )
        models_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = models_dir / cfg_file

        if pretrained:
            if isinstance(pretrained, (str, Path)) and str(pretrained) != "download":
                state = torch.load(pretrained, map_location=device, weights_only=False)
                cfg_path = Path(pretrained).parent / cfg_file
            else:
                base = f"https://huggingface.co/{repo}/resolve/main"
                state = load_state_dict_from_url(
                    f"{base}/{weights_file}",
                    model_dir=str(models_dir),
                    map_location=device,
                )
                if not cfg_path.exists():
                    torch.hub.download_url_to_file(f"{base}/{cfg_file}", str(cfg_path))

        # Build the DIRECT engine (model, operators, sensitivity model) from the config.
        file_cfg = OmegaConf.load(str(cfg_path))
        cfg = OmegaConf.structured(DefaultConfig)
        model_classes, models_cfg = load_models_into_environment_config(file_cfg)
        cfg.model = models_cfg.model
        del models_cfg["model"]
        cfg.additional_models = models_cfg
        cfg.physics = OmegaConf.merge(cfg.physics, file_cfg.physics)
        forward_operator, self.backward_operator = build_operators(cfg.physics)
        model, additional = initialize_models_from_config(
            cfg, model_classes, forward_operator, self.backward_operator, str(device)
        )
        self.engine = setup_engine(
            cfg,
            str(device),
            model,
            additional,
            forward_operator=forward_operator,
            backward_operator=self.backward_operator,
            mixed_precision=False,
        )
        self.engine.ndim = 2
        # Some engines (e.g. vSHARP) refine the sensitivity maps inside forward_function; others (e.g. JointICNet) expect pre-refined maps.
        self._sens_in_forward = "compute_sensitivity_map" in inspect.getsource(
            type(self.engine).forward_function
        )

        if pretrained:
            self.engine.model.load_state_dict(state["model"], strict=False)
            for name, module in self.engine.models.items():
                if name in state:
                    module.load_state_dict(state[name], strict=False)

        self.engine.model.to(device).eval()
        for m in self.engine.models.values():
            m.to(device).eval()
        self.to(device)

    def forward(
        self,
        y: torch.Tensor,
        physics: MultiCoilMRI | MRI,
        **kwargs,
    ) -> torch.Tensor:
        r"""Reconstruct image from k-space `y` and `physics`.

        :param torch.Tensor y: k-space of shape `(B,2,N,H,W)` for multicoil or `(B,2,H,W)` for singlecoil MRI.
        :param deepinv.physics.MultiCoilMRI, deepinv.physics.MRI physics: MRI physics with mask (coil maps ignored).
        """
        from direct.data.mri_transforms import (
            EstimateSensitivityMapModule,
            ComputeScalingFactorModule,
            NormalizeModule,
        )

        if isinstance(physics, MRI):
            y = y.unsqueeze(2).contiguous()
        elif not isinstance(physics, MultiCoilMRI):
            raise NotImplementedError("DIRECTModel supports only MRI or MultiCoilMRI.")

        y = y.moveaxis(1, -1).contiguous()  # BNHW2
        device = y.device
        mask = physics.mask[:, :1].bool().cpu()

        # Get ACS size from fully-sampled center
        h2, w2 = y.shape[2] // 2, y.shape[3] // 2

        def get_acs_size(centerline, centerloc):
            return 2 * min(
                (
                    int((centerline[:centerloc].flip(0) == 0).int().argmax())
                    if not centerline[:centerloc].all()
                    else centerloc
                ),
                (
                    int((centerline[centerloc:] == 0).int().argmax())
                    if not centerline[centerloc:].all()
                    else centerline.numel() - centerloc
                ),
            )

        acs_size_h = max(get_acs_size(mask[0, 0, :, w2], h2), 1) // 2
        acs_size_w = max(get_acs_size(mask[0, 0, h2], w2), 1) // 2

        acs_mask = torch.zeros(y.shape[0], 1, y.shape[2], y.shape[3], 1, dtype=y.dtype)
        acs_mask[
            :, :, h2 - acs_size_h : h2 + acs_size_h, w2 - acs_size_w : w2 + acs_size_w
        ] = 1.0

        # deepinv MRI uses centered FFTs. Models with uncentered operators (Calgary configs) need y pre-shifted into their
        # convention (a checkerboard modulation, i.e. a half-FOV image shift); centered ones (e.g. vSHARP) take y directly.
        # DIRECT's transforms also need CPU tensors.
        kspace = y.cpu()

        # DIRECT for Calgary use uncentered FFTs whereas for FastMRI it uses centered FFTs (as deepinv). Correct for uncentered:
        if not getattr(self.backward_operator, "keywords", {}).get("centered", True):
            kspace *= (
                (-1.0)
                ** (
                    torch.arange(y.shape[2])[:, None]
                    + torch.arange(y.shape[3])[None, :]
                )
            ).to(y.dtype)[None, None, :, :, None]

        # Forward pass
        sample = {
            "masked_kspace": kspace,
            "sampling_mask": mask.to(y.dtype).unsqueeze(-1),
            "acs_mask": acs_mask,
        }
        if kspace.shape[1] == 1:
            sample["sensitivity_map"] = torch.zeros_like(kspace).index_fill_(
                -1, torch.tensor([0]), 1.0
            )
        else:
            sample = EstimateSensitivityMapModule(
                kspace_key="masked_kspace",
                backward_operator=self.backward_operator,
                gaussian_sigma=0.7,
            )(sample)
        sample = ComputeScalingFactorModule(
            normalize_key="masked_kspace", percentile=0.99
        )(sample)
        sample = NormalizeModule(keys_to_normalize=["masked_kspace"])(sample)

        sample = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in sample.items()
        }

        with torch.no_grad():
            if not self._sens_in_forward:
                sample["sensitivity_map"] = self.engine.compute_sensitivity_map(
                    sample["sensitivity_map"]
                )
            x, _ = self.engine.forward_function(sample)

        if isinstance(x, (list, tuple)):  # vSHARP returns one image per unrolled step
            x = x[-1]

        x *= sample["scaling_factor"].view(-1, *([1] * (x.ndim - 1)))
        x = (
            torch.view_as_complex(x.contiguous())
            if x.dim() == 4 and x.shape[-1] == 2
            else x.to(torch.complex64)
        )
        return self.from_torch_complex(x).to(device)  # TODO sort out devices mess
