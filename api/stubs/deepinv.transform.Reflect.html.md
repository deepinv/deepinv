# Reflect

### *class* deepinv.transform.Reflect(\*args, dim=(-2, -1), \*\*kwargs)

Bases: [`Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)

Reflect (flip) in random multiple axes.

Generates `n_trans` reflected images, each time subselecting axes from dim (without replacement).
Hence to transform through all group elements, set `n_trans` to `2**len(dim)` e.g `Reflect(dim=[-2, -1], n_trans=4)`

See [`deepinv.transform.Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform) for further details and examples.

* **Parameters:**
  * **dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – axis or axes on which to randomly select axes to reflect.
  * **n_trans** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of transformed versions generated per input image.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator, if None, use torch.Generator(), defaults to None

#### invert_params(params)

Invert the parameters for reflection transformations

* **Parameters:**
  **params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – transform parameters as dict
* **Return dict:**
  inverted parameters.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

<a id="sphx-glr-backref-deepinv-transform-reflect"></a>

## Examples using `Reflect`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div>
<!-- thumbnail-parent-div-close --></div>
