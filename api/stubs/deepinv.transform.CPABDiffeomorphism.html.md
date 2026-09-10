# CPABDiffeomorphism

### *class* deepinv.transform.CPABDiffeomorphism(\*args, constant_batch=True, n_tesselation=3, zero_boundary=True, volume_perservation=True, override=True, device='cpu', \*\*kwargs)

Bases: [`Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)

Continuous Piecewise-Affine-based Diffeomorphism.

This requires the libcpab package which you can install from our [maintained fork](https://github.com/Andrewwango/libcpab)
using `pip install libcpab`.

Wraps CPAB from a modified version of the [original implementation](https://github.com/SkafteNicki/libcpab).
from Freifeld *et al.*<sup>[1](#footcite-freifeld2017transformations)</sup>.

These diffeomorphisms benefit from fast GPU-accelerated transform + fast inverse.

Generates `n_trans` randomly transformed versions.

See [`deepinv.transform.Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform) for further details and examples.

#### WARNING
This implementation does not allow using a `torch.Generator` to generate reproducible transformations.
You may be able to achieve reproducibility by using a global seed instead.

* **Parameters:**
  * **n_trans** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of transformed versions generated per input image.
  * **constant_batch** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – if `True`, all images in batch transformed with same params.
  * **n_tesselation** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of cells in tesselation in all dimensions.
    See `libcpab.Cpab` [docs](https://github.com/SkafteNicki/libcpab?tab=readme-ov-file#how-to-use) for more info.
  * **zero_boundary** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – see `libcpab.Cpab` docs.
  * **volume_perservation** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – see `libcpab.Cpab` docs.
  * **override** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – see `libcpab.Cpab` docs.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – torch device.

<hr />

* **References:**

* <a id='footcite-freifeld2017transformations'>**[1]**</a> Oren Freifeld, Søren Hauberg, Kayhan Batmanghelich, and Jonn W Fisher. Transformations based on continuous piecewise-affine velocity fields. *IEEE transactions on pattern analysis and machine intelligence*, 39(12):2496–2509, 2017.

<a id="sphx-glr-backref-deepinv-transform-cpabdiffeomorphism"></a>

## Examples using `CPABDiffeomorphism`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div>
<!-- thumbnail-parent-div-close --></div>
