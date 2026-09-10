# generate_pet_phantom

### deepinv.utils.phantoms.generate_pet_phantom(img_shape, mu_value=0.01, add_spheres=True, add_inner_cylinder=True, r0=0.45, r1=0.28, oversampling_factor=4, device='cpu')

Generate a 2D or 3D PET-like phantom and its corresponding attenuation map.

The phantom mimics a simplified PET scanner body phantom consisting of:

- *Outer elliptical body*: a large elliptical cylinder (or ellipsoid in 3D)
  filled with uniform emission activity (value 1.0) and attenuation `mu_value`.
  Its semi-axes are `r0` (along the first spatial dimension) and `r1` (along
  the second spatial dimension), expressed as fractions of the respective image size.
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
  * **img_shape** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Shape of the output image `(H, W)` or volume `(D, H, W)`.
  * **mu_value** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Attenuation coefficient of the outer body (default: 0.01).
  * **add_spheres** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, hot and cold spheres are added inside the body
    (default: `True`).
  * **add_inner_cylinder** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, a low-activity inner rod is added at the
    centre of the body (default: `True`).
  * **r0** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Fractional semi-axis of the outer ellipse along the *first* spatial
    dimension (D or H for 3D/2D inputs). Must satisfy `0 < r0 <= 0.5` so that the
    ellipse fits within the image. Default: 0.45.
  * **r1** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Fractional semi-axis of the outer ellipse along the *second* spatial
    dimension (H or W for 3D/2D inputs). Must satisfy `0 < r1 <= 0.5` so that the
    ellipse fits within the image. If `add_spheres=True`, `r1` should be at least
    ~0.25 to ensure the off-centre spheres remain inside the body. Default: 0.28.
  * **oversampling_factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Upsampling factor used during phantom generation to
    reduce partial-volume artefacts (default: 4).
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *or* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – Device on which tensors are allocated (default:
    `"cpu"`).
* **Returns:**
  Tuple `(x_em, x_att)` where both tensors have shape `(1, 1, H, W)`
  for 2D inputs or `(1, 1, D, H, W)` for 3D inputs. `x_em` is the emission
  activity map and `x_att` is the attenuation map.
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]
* **Raises:**
  [**ValueError**](https://docs.python.org/3.9/library/exceptions.html#ValueError) – If `r0` or `r1` are outside the valid range `(0, 0.5]`.

<hr />

* **Example:**

```pycon
>>> from deepinv.utils.phantoms import generate_pet_phantom
>>> x_em, x_att = generate_pet_phantom(img_shape=(64, 64, 32))
>>> print(x_em.shape, x_att.shape)
torch.Size([1, 1, 64, 64, 32]) torch.Size([1, 1, 64, 64, 32])
```

## Examples using `generate_pet_phantom`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct an image from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct a volume from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div>
<!-- thumbnail-parent-div-close --></div>
