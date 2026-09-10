# MedianFilter

### *class* deepinv.models.MedianFilter(kernel_size=9, padding=0, same=True)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Median filter.

It computes the median value of a sliding window over the input tensor. The window is defined by the kernel size.

* **Parameters:**
  * **kernel_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – size of pooling kernel, int or 2-tuple
  * **padding** – pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
  * **same** – override padding and enforce same padding, boolean

<a id="sphx-glr-backref-deepinv-models-medianfilter"></a>

## Examples using `MedianFilter`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to fit deepinv.loss.metric.NIQE on a new dataset, and use it to evaluate denoiser performance.">  <div class="sphx-glr-thumbnail-title">Fitting NIQE on a custom dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator to accelerate the sampling.">  <div class="sphx-glr-thumbnail-title">Building your custom MCMC sampling algorithm.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
