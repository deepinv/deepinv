# Scale

### *class* deepinv.transform.Scale(\*args, factors=None, padding_mode='reflection', mode='bicubic', \*\*kwargs)

Bases: [`Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)

2D Scaling.

Resample the input image on a grid obtained using
an isotropic dilation, with random scale factor
and origin. By default, the input image is viewed
as periodic and the output image is effectively padded
by reflections. Additionally, resampling is performed
using bicubic interpolation.

See the paper Scanvic *et al.*<sup>[1](#footcite-scanvic2026scale)</sup> for more details.

Note each image in the batch is transformed independently.

* **Parameters:**
  * **factors** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – list of scale factors (default: [.75, .5])
  * **padding_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – padding mode for grid sampling
  * **mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – interpolation mode for grid sampling
  * **n_trans** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of transformed versions generated per input image.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator, if None, use torch.Generator(), defaults to None

<hr />

* **References:**

* <a id='footcite-scanvic2026scale'>**[1]**</a> Jérémy Scanvic, Mike Davies, Patrice Abry, and Julián Tachella. Scale-equivariant imaging: self-supervised learning for image super-resolution and deblurring. *IEEE Transactions on Computational Imaging*, 2026.

<a id="sphx-glr-backref-deepinv-transform-scale"></a>

## Examples using `Scale`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div>
<!-- thumbnail-parent-div-close --></div>
