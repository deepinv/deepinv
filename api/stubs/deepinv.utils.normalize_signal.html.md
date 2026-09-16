# normalize_signal

### deepinv.utils.normalize_signal(inp, , mode, vmin=None, vmax=None)

Normalize a batch of signals between zero and one.

* **Parameters:**
  * **inp** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the input signal to normalize, it should be of shape `(B, *)`.
  * **mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – the normalization, either `'min_max'` for min-max normalization or `'clip'` for clipping.
    If `clip` is selected, the values of `vmin` and `vmax` are used as clipping bounds if provided,
    otherwise the default bounds of 0.0 and 1.0 are used.
    Note that min-max normalization of constant signals is ill-defined and here it amounts to mapping the constant
    value to the closest value between zero and one (which is equivalent to clipping).
* **Returns:**
  the normalized batch of signals.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

## Examples using `normalize_signal`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div>
<!-- thumbnail-parent-div-close --></div>
