# ZeroFidelity

### *class* deepinv.optim.ZeroFidelity

Bases: [`DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)

Zero data fidelity term $\datafid{x}{y} = 0$.
This is used to remove the data fidelity term in the loss function.

#### fn(x, y, physics, \*args, \*\*kwargs)

This function returns zero for all inputs.

#### grad(x, y, physics, \*args, \*\*kwargs)

This function returns a zero image.

<a id="sphx-glr-backref-deepinv-optim-zerofidelity"></a>

## Examples using `ZeroFidelity`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div>
<!-- thumbnail-parent-div-close --></div>
