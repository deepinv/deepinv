# load_ismrmrd_raw

### deepinv.utils.load_ismrmrd_raw(filename, ifft_slice_dim=False)

Load ISMRMRD hdf5 raw Cartesian multi-coil MRI data.

Saves the acquired lines of an ISMRMRD `.h5` dataset onto a Cartesian k-space grid of zeros
as `(1, 2, N, D, H, W)` where N = num coils, D = slice/depth and `(H, W)` = 2D plane.

The D dim is detected automatically for 2D vs 3D:

- 3D acquisition: loads the entire 3D k-space, with `D = readout` and `(H, W) = (partition, phase-encode)`.
- 2D slice-based acquisition: loads the first slice only, with `D = 1` (singleton) and `(H, W) = (phase-encode, readout)`.

Code derived from [ISMRMRD](https://github.com/ismrmrd/ismrmrd-python) and [examples](https://github.com/ismrmrd/ismrmrd-python-tools).

The dataset requirers ismrmrd: `pip install ismrmrd`

Only the first average, contrast, phase, repetition and set is kept. Noise scans are dropped.

#### NOTE
Readout oversampling is removed by default, since scanners oversample the (fully-sampled) readout (typically 2x) to avoid frequency-encode aliasing,
which doubles the readout FOV and is redundant for reconstruction.

* **Parameters:**
  * **filename** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – path to the ISMRMRD `.h5` file.
  * **ifft_slice_dim** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, inverse FFT the slice/depth axis D so it is returned in image space, ready to index slices. Note for 2D it is ignored.
* **Returns:**
  real tensor of shape `(1, 2, N, D, H, W)`, where N = num coils.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
