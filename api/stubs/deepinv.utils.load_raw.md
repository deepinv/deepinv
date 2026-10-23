# load_raw

### deepinv.utils.load_raw(fname, dtype=torch.float32, \*\*kwargs)

Loads any RAW image file as a torch tensor along with a metadata dictionary.

Reads any proprietary raw files supported by `LibRaw <https://www.libraw.org>`
using `rawpy` (ie `.arw`, `.cr2`/`.cr3`, `.nef`, `.raf`,
`.orf`, `.rw2`,\`\`.pef\`\`, `.srw`,\`\`.dng\`\`…).
Returns the sensor image as a tensor of shape `(1, 1, H, W)`
along with a dict of ISP metadata.

#### NOTE
Only $2\times 2$ Bayer sensors are supported for now; non-Bayer CFAs
(e.g. X-Trans) will raise an error.

#### WARNING
Requires `rawpy` to be installed. Install it with `pip install rawpy`.

* **Parameters:**
  * **fname** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.12/library/pathlib.html#pathlib.Path)) – path to the raw file.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – dtype of the returned tensor.
* **Returns:**
  tuple `(y, meta)` where `y` is a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
  of shape `(1, 1, H, W)` and `meta` is a `dict` of ISP metadata
  with keys `cfa_pattern`, `cfa_colors`, `color_description`,
  `num_colors`, `black_level_per_channel`,
  `camera_white_level_per_channel`, `camera_whitebalance`,
  `daylight_whitebalance`, `color_matrix`, `rgb_xyz_matrix`,
  `tone_curve`, `raw_shape`, `visible_shape`, `crop_margins`,
  `pixel_aspect`, `flip` and `raw_type`.
* **Return type:**
  [tuple](https://docs.python.org/3.12/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [dict](https://docs.python.org/3.12/library/stdtypes.html#dict)]
