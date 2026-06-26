r"""
Ultrasound beamforming physics operators using the `zea <https://zea.readthedocs.io/>`_ backend.

Provides:

- :class:`UltrasoundBeamformingWithZea` — linear DAS beamforming, raw RF → beamformed IQ

Requirements:

- ``pip install zea``
- Set the Keras backend to PyTorch **before** any other imports:

.. code-block:: python

    import os
    os.environ["KERAS_BACKEND"] = "torch"
    import zea
"""

from __future__ import annotations

import importlib

import torch

from deepinv.physics.forward import LinearPhysics


class _UltrasoundZeaBase(LinearPhysics):
    r"""Common base for ultrasound physics operators implemented with zea.

    Handles pipeline construction, parameter preparation, and device management.
    Subclasses implement :meth:`A` for the specific forward model.
    """

    def __init__(
        self,
        pipeline,
        parameters,
        **kwargs,
    ):
        if importlib.util.find_spec("zea") is None:
            raise ModuleNotFoundError(
                "UltrasoundBeamformingWithZea requires the zea package.\n"
                "Install with:  pip install zea\n"
                "Set (before other imports):  os.environ['KERAS_BACKEND'] = 'torch'"
            )

        if "device" not in kwargs and pipeline.device is not None:
            kwargs["device"] = pipeline.device.replace("gpu", "cuda")

        self.zea_pipeline = pipeline
        self.zea_parameters = parameters
        self.zea_pipeline_inputs: dict = pipeline.prepare_parameters(parameters)

        super().__init__(**kwargs)

    def to(self, *args, **kwargs) -> "_UltrasoundZeaBase":
        r"""Move the operator to a new device.

        Updates the zea pipeline's device setting and moves all pre-computed
        scan parameter tensors (delays, grid coordinates, etc.) to the new device.
        """
        device: torch.device | None = None
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device = torch.device(arg)
                break
            elif isinstance(arg, int):
                device = torch.device("cuda", arg)
                break
        if device is None and "device" in kwargs:
            device = torch.device(kwargs["device"])

        if device is not None and hasattr(self, "zea_pipeline"):
            self.zea_pipeline.device = str(device)
            for key, value in list(self.zea_pipeline_inputs.items()):
                if isinstance(value, torch.Tensor):
                    self.zea_pipeline_inputs[key] = value.to(device)

        return super().to(*args, **kwargs)


class UltrasoundBeamformingWithZea(_UltrasoundZeaBase):
    r"""Ultrasound DAS beamforming physics operator using the
    `zea <https://zea.readthedocs.io/>`_ toolbox.

    Implements the linear forward operator :math:`A : x \mapsto y` where:

    - :math:`x` is the raw channel (RF) data with shape
      ``(batch, n_tx, n_ax, n_el, n_ch)``.
    - :math:`y` is the beamformed IQ/RF image with shape
      ``(batch, n_ch_out, grid_z, grid_x)`` (channel-first, deepinv convention).


    If the ``pipeline`` was created with a device (e.g. ``device="cuda:0"``)
    and no ``device`` keyword is passed here, that device is adopted
    automatically so the Physics and pipeline always stay on the same device.

    :param pipeline: zea beamforming pipeline that maps raw RF
        data to beamformed IQ/RF (excluding envelope detection /
        log-compression).
    :param parameters: Scan parameters used to derive pipeline
        inputs (delays, grid coordinates, etc.).
    :param tuple img_size: Size of :math:`x` *without* the batch dimension,
        e.g. ``(n_tx, n_ax, n_el, 1)``.  Required for the automatic adjoint.
    :param kwargs: Additional keyword arguments forwarded to
        :class:`deepinv.physics.LinearPhysics` (``device``, ``noise_model``,
        ``max_iter``, ``solver``, …).

    """

    def A(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward beamforming operator: raw RF data → beamformed IQ image.

        :param torch.Tensor x: Raw channel data, shape
            ``(batch, n_tx, n_ax, n_el, n_ch)``.
        :returns: Beamformed image, shape ``(batch, n_ch_out, grid_z, grid_x)``.
        """
        outputs = self.zea_pipeline(data=x, **self.zea_pipeline_inputs)
        # zea output is channel-last: (batch, grid_z, grid_x, n_ch_out)
        y = outputs["data"]
        # Permute to deepinv channel-first convention
        return y.permute(0, 3, 1, 2).contiguous()
