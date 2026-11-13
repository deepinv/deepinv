# generate_nema_iec_phantom

### deepinv.utils.phantoms.generate_nema_iec_phantom(size, activities=None, background=0.25, lung=0.0, normalize=True)

Generate a 2D or 3D NEMA IEC body phantom for PET/SPECT.

The phantom contains six spheres of diameters 10, 13, 17, 22, 28 and 37 mm arranged on a circle inside a rounded body with uniform background activity Daube-Witherspoon *et al.*<sup>[1](#footcite-daubewitherspoon2002pet)</sup>.
A central cylindrical insert represents the low-activity lung region.
Sphere activities can be set independently to model hot or cold regions relative to the background.

Inspired by David Sarrut’s implementation in [OpenGATE](https://github.com/OpenGATE/opengate/blob/master/opengate/contrib/phantoms/nemaiec.py).

<hr />

* **Example:**

```pycon
>>> import deepinv as dinv
>>> dinv.utils.generate_nema_iec_phantom(64).shape
torch.Size([64, 64])
>>> dinv.utils.generate_nema_iec_phantom((32, 64, 64)).shape
torch.Size([32, 64, 64])
```

* **Parameters:**
  * **size** ([*int*](https://docs.python.org/3.12/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int) *]*) – Square slice size through the sphere centres, or volume shape `(D, H, W)`.
  * **activities** ([*list*](https://docs.python.org/3.12/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.12/library/functions.html#float) *]*) – Six sphere activities, ordered by increasing diameter. Defaults to `[1.0] * 6`.
  * **background** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Background activity.
  * **lung** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Lung insert activity.
  * **normalize** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – Rescale maximum activity to 1 if `True`.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) Phantom of shape `(size, size)` or `(D, H, W)`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<hr />

* **References:**

* <a id='footcite-daubewitherspoon2002pet'>**[1]**</a> Margaret E. Daube-Witherspoon, Joel S. Karp, Michael E. Casey, Frank P. DiFilippo, Horace Hines, Gerd Muehllehner, Vilim Simcic, Charles W. Stearns, Lars-Eric Adam, Steve Kohlmyer, and Vesna Sossi. Pet performance measurements using the NEMA NU 2-2001 standard. *Journal of Nuclear Medicine*, 43(10):1398–1409, 2002. URL: [https://pubmed.ncbi.nlm.nih.gov/12368380/](https://pubmed.ncbi.nlm.nih.gov/12368380/).
