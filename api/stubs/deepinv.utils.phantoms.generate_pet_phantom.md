# generate_pet_phantom

### deepinv.utils.phantoms.generate_pet_phantom(img_shape, mu_value=0.01, add_spheres=True, add_inner_cylinder=True, r0=0.45, r1=0.28, oversampling_factor=4, device='cpu', return_labels=False)

Generate a 2D or 3D PET-like phantom, attenuation map, and optional segmentation labels.

The phantom mimics a simplified PET scanner body phantom consisting of:

- *Outer elliptical body*: a large elliptical cylinder (or ellipsoid in 3D)
  filled with uniform emission activity (value 1.0) and attenuation `mu_value`.
  Its transverse semi-axes are `r0` (along H) and `r1` (along W),
  expressed as fractions of the respective image size.
- *Inner cold rod* (optional, `add_inner_cylinder=True`): a small elliptical
  cylinder at the centre of the body with lower emission activity (value 0.25) and
  reduced attenuation (`mu_value / 3`), representing a low-activity insert.
- *Hot and cold spheres* (optional, `add_spheres=True`): four pairs of spheres
  placed at two axial offsets inside the body. *Hot* spheres (value 2.5) simulate
  lesion-like regions with elevated metabolic activity; *cold* spheres (value 0.25)
  simulate photopenic defects (e.g. cysts or necrotic tissue).

The function generates the phantom at `oversampling_factor` times the requested
resolution and then downsamples by averaging, reducing partial-volume effects at
the boundaries.

For 2D inputs (`img_shape = (H, W)`), the phantom is generated as a 3D volume
and the central axial slice is returned.

* **Parameters:**
  * **img_shape** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple)) – Shape of the output image `(H, W)` or volume `(D, H, W)`.
  * **mu_value** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Attenuation coefficient of the outer body (default: 0.01).
  * **add_spheres** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – If `True`, hot and cold spheres are added inside the body
    (default: `True`).
  * **add_inner_cylinder** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – If `True`, a low-activity inner rod is added at the
    centre of the body (default: `True`).
  * **r0** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Fractional semi-axis of the outer ellipse along H.
    Must satisfy `0 < r0 <= 0.5` so that the ellipse fits within the image.
    Default: 0.45.
  * **r1** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Fractional semi-axis of the outer ellipse along W.
    Must satisfy `0 < r1 <= 0.5` so that the
    ellipse fits within the image. If `add_spheres=True`, `r1` should be at least
    ~0.25 to ensure the off-centre spheres remain inside the body. Default: 0.28.
  * **oversampling_factor** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – Upsampling factor used during phantom generation to
    reduce partial-volume artefacts (default: 4).
  * **device** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *or* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – Device on which tensors are allocated (default:
    `"cpu"`).
  * **return_labels** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – If `True`, also return the segmentation labels (default: `False`).
* **Returns:**
  Tuple `(x_em, x_att)` or `(x_em, x_att, labels)` if `return_labels=True`.
  All tensors have shape `(1, 1, H, W)` for 2D inputs or `(1, 1, D, H, W)` for 3D inputs.
  `x_em` is the emission activity map and `x_att` is the attenuation map.
  `labels` is a segmentation map with labels 0 (background), 1 (plastic body),
  2 (lung/inner cold region), 3 (hot spheres), and 4 (cold spheres).
* **Return type:**
  [tuple](https://docs.python.org/3.12/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] or [tuple](https://docs.python.org/3.12/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]
* **Raises:**
  [**ValueError**](https://docs.python.org/3.12/library/exceptions.html#ValueError) – If `r0` or `r1` are outside the valid range `(0, 0.5]`.

<hr />

* **Example:**

```pycon
>>> from deepinv.utils.phantoms import generate_pet_phantom
>>> x_em, x_att, labels = generate_pet_phantom(img_shape=(32, 64, 64), return_labels=True)
>>> print(x_em.shape, x_att.shape)
torch.Size([1, 1, 32, 64, 64]) torch.Size([1, 1, 32, 64, 64])
```

## Examples using `generate_pet_phantom`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a PET scanner, simulate measurements with and without time-of-flight (ToF) information, and reconstruct an image from them.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet2d_thumb.png)

[Positron emission tomography (PET) in 2D](https://deepinv.org/auto_examples/physics/demo_pet2d.md)

  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a PET scanner, simulate measurements with and without time-of-flight (ToF) information, and reconstruct a volume from them.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet3d_thumb.png)

[Positron emission tomography (PET) in 3D](https://deepinv.org/auto_examples/physics/demo_pet3d.md)

  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div>
<!-- thumbnail-parent-div-close --></div>
