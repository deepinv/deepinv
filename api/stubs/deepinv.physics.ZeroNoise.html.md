# ZeroNoise

### *class* deepinv.physics.ZeroNoise(\*args, \*\*kwargs)

Bases: [`NoiseModel`](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)

Zero noise model $y=x$, serves as a placeholder.

<hr />

* **Used in benchmarks:**

- [DIV2K Super Resolution 2x](https://deepinv.org/auto_benchmarks/div2k_super_resolution_2x.html.md#div2k-super-resolution-2x)
- [DIV2K Inpainting easy](https://deepinv.org/auto_benchmarks/div2k_inpainting_easy.html.md#div2k-inpainting-easy)

#### forward(x, \*args, \*\*kwargs)

Return the same input.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements.
* **Returns:**

<a id="sphx-glr-backref-deepinv-physics-zeronoise"></a>

## Examples using `ZeroNoise`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This examples shows you how to use DeepInverse with your own physics.">  <div class="sphx-glr-thumbnail-title">Bring your own physics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div>
<!-- thumbnail-parent-div-close --></div>
