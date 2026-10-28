# NEMAIECPhantomDataset

### *class* deepinv.utils.phantoms.NEMAIECPhantomDataset(size=128, activities=None, background=0.25, lung=0.0, normalize=True, transform=None, use_dict_output=False)

Bases: [`Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)

Dataset for the NEMA IEC body phantom.

Each item contains one channel. A [`torch.utils.data.DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) adds
the batch dimension.

#### NOTE
See [`deepinv.utils.phantoms.generate_nema_iec_phantom()`](https://deepinv.org/api/stubs/deepinv.utils.phantoms.generate_nema_iec_phantom.md#deepinv.utils.phantoms.generate_nema_iec_phantom) for phantom geometry and generation details Daube-Witherspoon *et al.*<sup>[1](#footcite-daubewitherspoon2002pet)</sup>.

* **Parameters:**
  * **size** ([*int*](https://docs.python.org/3.12/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int) *]*) – Size of a square transverse slice, or volume shape `(D, H, W)`.
  * **activities** ([*list*](https://docs.python.org/3.12/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.12/library/functions.html#float) *]*) – List of 6 sphere activity levels, ordered by increasing diameter. Defaults to `[1.0] * 6`.
  * **background** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – activity level of the uniform background.
  * **lung** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – activity level of the central cold insert.
  * **normalize** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – if `True`, rescale so that the maximum value is 1.
  * **transform** (*Callable*) – Transformation to apply to the output image.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – Whether to return output as dict with key “x” or a bare Tensor (default: `False`).

<hr />

* **References:**

* <a id='footcite-daubewitherspoon2002pet'>**[1]**</a> Margaret E. Daube-Witherspoon, Joel S. Karp, Michael E. Casey, Frank P. DiFilippo, Horace Hines, Gerd Muehllehner, Vilim Simcic, Charles W. Stearns, Lars-Eric Adam, Steve Kohlmyer, and Vesna Sossi. Pet performance measurements using the NEMA NU 2-2001 standard. *Journal of Nuclear Medicine*, 43(10):1398–1409, 2002. URL: [https://pubmed.ncbi.nlm.nih.gov/12368380/](https://pubmed.ncbi.nlm.nih.gov/12368380/).
