####
# Demo requirements
#
# This example depends on:
# - SIRF
# - deepinv
#
# Recommended setup
# Use the PETRIC2 / SIRF Docker image:
# https://github.com/SyneRBI/SIRF-SuperBuild/pkgs/container/sirf/versions?filters%5Bversion_type%5D=tagged
#
# Then install deepinv inside the container:
#     pip install deepinv
#
# Additional helper
# This demo also requires the temporary Python helper `torch.py` from:
# https://github.com/Imraj-Singh/SIRF/blob/torch/src/torch/torch.py
#
# In this example it is imported locally as:
#     from torch_tmp import sirf_to_torch, torch_to_sirf_
#
# Upstream status
# The corresponding SIRF pull request is:
# https://github.com/SyneRBI/SIRF/pull/1305
#
# Once that PR is merged, rebuilding SIRF should remove the need for the
# temporary local helper.

# Evangelos Papoutsellis, Imraj Singh, Kris Thielemans, Casper da Costa Luis


import os
import sys

import deepinv as dinv

import sirf.STIR as pet
from sirf.Utilities import examples_data_path

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_tmp import sirf_to_torch, torch_to_sirf_

pet.set_verbosity(0)
pet.AcquisitionData.set_storage_scheme("memory")

### Load thorax (SIRF DataContainer)
data_path = os.path.join(examples_data_path('PET'), 'thorax_single_slice')
image = pet.ImageData(os.path.join(data_path, 'emission.hv'))
attn_image = pet.ImageData(os.path.join(data_path, 'attenuation.hv'))
template = pet.AcquisitionData(os.path.join(data_path, 'template_sinogram.hs'))

### Simul attenuation and noise (Still SIRF DataContainer)
acq_model_for_attn = pet.AcquisitionModelUsingRayTracingMatrix()
asm_attn = pet.AcquisitionSensitivityModel(attn_image, acq_model_for_attn)
asm_attn.set_up(template)
attn_factors = asm_attn.forward(template.get_uniform_copy(1))
asm_attn = pet.AcquisitionSensitivityModel(attn_factors)
acq_model_full = pet.AcquisitionModelUsingRayTracingMatrix()
acq_model_full.set_acquisition_sensitivity(asm_attn)
acq_model_full.set_up(template,image)

acquired_data = acq_model_full.forward(image)
np.random.seed(10)
acquired_data.fill(np.random.poisson(acquired_data.as_array()))
background_term = acquired_data.get_uniform_copy(acquired_data.max()/10)

class PET_SIRF(dinv.physics.LinearPhysics):
    def __init__(
        self,
        img_template,
        acq_template,
        attenuation_image=None,
        attenuation_factors=None,
        additive_term=None,
        background_term=None,
        device="cpu",
        num_tangential_lors=10,
    ):
        super().__init__()
        self.device = device
        self.num_tangential_lors = num_tangential_lors

        self.img_template = img_template.clone()
        self.acq_template = acq_template.clone()
        self.work_img = img_template.clone()
        self.work_acq = acq_template.clone()

        self.attenuation_image = attenuation_image
        self.attenuation_factors = attenuation_factors
        self.additive_term = self._to_acq_data(additive_term)
        self.background_term = self._to_acq_data(background_term)

        self._create_sirf_models()

    def _new_acq_model(self):
        m = pet.AcquisitionModelUsingRayTracingMatrix()
        m.set_num_tangential_LORs(self.num_tangential_lors)
        return m

    def _to_acq_data(self, x):
        if x is None:
            return None
        if isinstance(x, pet.AcquisitionData):
            return x.clone()
        out = self.acq_template.get_uniform_copy(0.0)
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        out.fill(np.asarray(x, dtype=np.float32))
        return out

    def _build_asm(self):
        if self.attenuation_factors is not None:
            return pet.AcquisitionSensitivityModel(self._to_acq_data(self.attenuation_factors))

        if self.attenuation_image is None:
            return None

        attn_model = self._new_acq_model()
        attn_model.set_up(self.acq_template, self.attenuation_image)
        asm0 = pet.AcquisitionSensitivityModel(self.attenuation_image, attn_model)
        asm0.set_up(self.acq_template)
        attn_factors = asm0.forward(self.acq_template.get_uniform_copy(1.0))
        return pet.AcquisitionSensitivityModel(attn_factors)

    def _create_sirf_models(self):
        asm = self._build_asm()

        self.full_model = self._new_acq_model()
        if asm is not None:
            self.full_model.set_acquisition_sensitivity(asm)
        if self.additive_term is not None:
            self.full_model.set_additive_term(self.additive_term)
        if self.background_term is not None:
            self.full_model.set_background_term(self.background_term)
        self.full_model.set_up(self.acq_template, self.img_template)

        self.linear_model = self._new_acq_model()
        if asm is not None:
            self.linear_model.set_acquisition_sensitivity(asm)
        self.linear_model.set_up(self.acq_template, self.img_template)


    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x = self._squeeze_batch_if_needed(x, self.img_template.as_array().ndim)
        torch_to_sirf_(x.contiguous(), self.work_img)
        self.linear_model.direct(self.work_img, out=self.work_acq)
        return sirf_to_torch(self.work_acq, self.device)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        # y = self._squeeze_batch_if_needed(y, self.acq_template.as_array().ndim)
        torch_to_sirf_(y.contiguous(), self.work_acq)
        self.linear_model.adjoint(self.work_acq, out=self.work_img)
        return sirf_to_torch(self.work_img, self.device)

    def forward(self, x: torch.Tensor, apply_poisson=False, **kwargs) -> torch.Tensor:
        # x = self._squeeze_batch_if_needed(x, self.img_template.as_array().ndim)
        torch_to_sirf_(x.contiguous(), self.work_img)
        y_sirf = self.full_model.forward(self.work_img)   # <- FIX
        y = sirf_to_torch(y_sirf, self.device)
        if apply_poisson:
            y = torch.poisson(torch.clamp(y, min=0))
        return y

# background term
background_term = template.get_uniform_copy(acquired_data.max() / 10)

physics = PET_SIRF(
    img_template=image,
    acq_template=template,
    attenuation_image=attn_image,   
    background_term=background_term,  
    device="cpu",
    num_tangential_lors=10,
)

image_torch = torch.from_numpy(image.as_array())
acquired_data_torch = torch.from_numpy(acquired_data.as_array())

max_iter = 5
sol = physics.A_dagger(acquired_data_torch, solver="CG", max_iter=max_iter)
sol = torch.clamp(sol, min=0)

plt.imshow(sol.numpy()[0], cmap="inferno")
plt.show()        