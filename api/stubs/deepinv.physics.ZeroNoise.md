# ZeroNoise

### *class* deepinv.physics.ZeroNoise(\*args, \*\*kwargs)

Bases: [`NoiseModel`](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.md#deepinv.physics.NoiseModel)

Zero noise model $y=x$, serves as a placeholder.

<hr />

* **Used in benchmarks:**

- [DIV2K Inpainting easy](https://deepinv.org/auto_benchmarks/div2k_inpainting_easy.md#div2k-inpainting-easy)
- [DIV2K Super Resolution 2x](https://deepinv.org/auto_benchmarks/div2k_super_resolution_2x.md#div2k-super-resolution-2x)

#### forward(x, \*args, \*\*kwargs)

Return the same input.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements.
* **Returns:**

<a id="sphx-glr-backref-deepinv-physics-zeronoise"></a>

## Examples using `ZeroNoise`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This examples shows you how to use DeepInverse with your own physics.">![](auto_examples/basics/images/thumb/sphx_glr_demo_custom_physics_thumb.png)

[Bring your own physics](https://deepinv.org/auto_examples/basics/demo_custom_physics.md)

  <div class="sphx-glr-thumbnail-title">Bring your own physics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">![](auto_examples/basics/images/thumb/sphx_glr_demo_quickstart_thumb.png)

[5 minute quickstart tutorial](https://deepinv.org/auto_examples/basics/demo_quickstart.md)

  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div>
<!-- thumbnail-parent-div-close --></div>
