# to_complex_denoiser

### deepinv.models.complex.to_complex_denoiser(denoiser, mode='real_imag')

Converts a denoiser with real inputs into the one with complex inputs.

Converts a denoiser with real inputs into one that accepts complex-valued inputs by applying the denoiser separately on the real and imaginary parts, or in the absolute value and phase parts.

* **Parameters:**
  * **denoiser** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – a denoiser which takes in real-valued inputs.
  * **mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – the mode by which the complex inputs are processed. Can be either `'real_imag'` or `'abs_angle'`.
* **Returns:**
  (torch.nn.Module) the denoiser which takes in complex-valued inputs.
* **Return type:**
  [*ComplexDenoiserWrapper*](https://deepinv.org/api/stubs/deepinv.models.ComplexDenoiserWrapper.html.md#deepinv.models.ComplexDenoiserWrapper)
