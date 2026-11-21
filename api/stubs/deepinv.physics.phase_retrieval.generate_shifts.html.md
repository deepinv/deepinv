# generate_shifts

### deepinv.physics.phase_retrieval.generate_shifts(img_size, n_img=25, fov=None, overlap=None, probe_radius=None)

Generate probe shifts on a square grid across the image.

The scan is a square grid of `n_img` positions spanning `fov`.
Giving `overlap` instead sets `n_img` from the overlap and probe radius.

* **Parameters:**
  * **img_size** ([*Any*](https://docs.python.org/3.12/library/typing.html#typing.Any)) – Size of the image.
  * **n_img** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – Number of shifts (must be a perfect square). Ignored if
    `overlap` is given.
  * **fov** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – Field of view for shift computation.
  * **overlap** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Target linear overlap between neighbouring probes, in
    `[0, 1)`. The achieved overlap is at least this value.
  * **probe_radius** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – Probe radius in pixels, required when `overlap`
    is given.
* **Returns:**
  Shifts of shape `(n_img, 2)` in pixels, in `(row, column)` order.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
