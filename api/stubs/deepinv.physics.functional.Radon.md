# Radon

### *class* deepinv.physics.functional.Radon(in_size, theta=None, circle=False, parallel_computation=True, fan_beam=False, fan_parameters=None, dtype=torch.float, device=torch.device('cpu'))

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Sparse Radon transform operator.

* **Parameters:**
  * **in_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the size of the input image (assumed square).
  * **theta** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the angles at which the Radon transform is computed. Default is `torch.arange(180)`.
  * **circle** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, the input image is assumed to be a circle. Default is `False`.
  * **parallel_computation** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, all projections are performed in parallel. Requires more memory but is faster on GPUs.
  * **fan_beam** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, use fan beam geometry, if `False` use parallel beam
  * **fan_parameters** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – 

    Only used if fan_beam is `True`. Contains the parameters defining the scanning geometry. The dict should contain the keys:
    - ”pixel_spacing” defining the distance between two pixels in the image, default: 0.5 / (in_size)
    - ”source_radius” distance between the x-ray source and the rotation axis (middle of the image), default: 57.5
    - ”detector_radius” distance between the x-ray detector and the rotation axis (middle of the image), default: 57.5
    - ”n_detector_pixels” number of pixels of the detector, default: 258
    - ”detector_spacing” distance between two pixels on the detector, default: 0.077

    The default values are adapted from the geometry in Khalil *et al.*<sup>[1](#footcite-khalil2023hyperspectral)</sup>.
    where pixel spacing, source and detector radius and detector spacing are given in cm.
    Note that a to small value of n_detector_pixels\*detector_spacing can lead to severe circular artifacts in any reconstruction.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – the data type of the output. Default is torch.float.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – the device of the output. Default is torch.device(‘cpu’).

<hr />

* **References:**

* <a id='footcite-khalil2023hyperspectral'>**[1]**</a> Mohamad Khalil, Jan Kehres, and Wail Mustafa. Hyperspectral 2d fan-beam x-ray ct dataset of 5 materials. September 2023. Dataset. URL: [https://doi.org/10.5281/zenodo.8307932](https://doi.org/10.5281/zenodo.8307932), [doi:10.5281/zenodo.8307932](https://doi.org/10.5281/zenodo.8307932).

#### forward(x)

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the input image.
