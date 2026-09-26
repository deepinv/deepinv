# RandomPatchSampler

### *class* deepinv.datasets.RandomPatchSampler(x_dir=None, y_dir=None, patch_size=32, file_format='.npy', ch_axis=None, dtype=torch.float32, loader=None, use_dict_output=False)

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

Dataset for nD images that samples one random patch per image.

This dataset builds from one or two directories of nD images (must be of format `.npy`, `.nii(.gz)`, or `.b2nd`, if `loader` is not specified).
On each epoch, it returns a randomly sampled patch of fixed size from each volume.

#### WARNING
This loader uses torch’s random functionality. To ensure reproducibility, set the DataLoader’s `generator` with a fixed seed.

**Supported use cases:**

- Single-directory: provide only the ground-truth folder `x_dir` or measurement folder `y_dir` (returns patches from that directory).
- Paired-directory: provide both `x_dir` and `y_dir` (returns matched patches from both).

**Channel handling:**

- If `ch_axis=None`: a singleton channel dimension is added at axis 0.
- If `ch_axis=0`: images are assumed channel-first.
- If `ch_axis=-1`: images are assumed channel-last and transposed to channel-first.
- Patches are never extracted along the channel axis (patch size for that axis is ignored).

**Patch size handling:**

- Accepts either an integer (applied to all spatial dims) or a tuple.
- If `patch_size` is tuple, and `patch_size[i] == 1`, this is equivalent to slicing across axis i (singleton at axis i will be squeezed). This can be used to e.g. extract 2D slices from a 3D volume.
- If tuple length is one less than the image ndim, the channel axis is auto-filled with `None`.

**Randomness & reproducibility:**

- Patch coordinates are drawn with Python’s `random` module.
- To ensure deterministic behavior across workers, set the DataLoader’s
  `worker_init_fn` or `generator` according to the PyTorch reproducibility guidelines.

**Notes:**

- All images must have the same dimensionality.
- When both directories are provided, only files present in both are used.
- Shapes of each file are checked for consistency (spatial not smaller than `patch_size` + channels remain consistent across files).

* **Parameters:**
  * **x_dir** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – Optional path to folder of ground-truth images. Required if `y_dir` is not given.
  * **y_dir** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – Optional path to folder of measurement images. Required if `x_dir` is not given.
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Size of patches to extract. If int, applies the same size across all spatial dimensions (default `32`)
  * **file_format** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – File format to load. Other files are ignored (default `".npy"`).
  * **ch_axis** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* *None*) – Optional axis of the channel dimension. If None, a new singleton channel is added.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – Data type to use when loading the images.
  * **loader** (*Callable* *,* *None*) – Optional custom loader function. Must accept path and the keyword `as_memmap`, which will always be set to True. Must return an object that has shape attribute and returns a `np.ndarray` when sliced. If None, an internal loader is chosen based on `file_format`.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).
