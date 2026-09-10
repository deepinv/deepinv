# Kohler

### *class* deepinv.datasets.Kohler(root=None, frames='middle', ordering='printout_first', transform=None, download=False, use_dict_output=False)

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

Dataset for [Recording and Playback of Camera Shake](https://doi.org/10.1007/978-3-642-33786-4_3)

Published in Kohler *et al.*<sup>[1](#footcite-kohler2012recording)</sup>.

The dataset consists of blurry shots and sharp frames, each blurry shot
being associated with about 200 sharp frames. There are 48 blurry shots in
total, each associated to one of 4 printouts, and to one of 12 camera
trajectories inducing motion blur. Unlike certain deblurring datasets (e.g.
GOPRO) where the blurry images are synthesized from sharp images, the
blurry shots in the Köhler dataset are acquired with a real camera. It is
the movement of the camera during exposition that causes the blur. What we
call printouts are the 4 images that were printed out on paper and fixed to
a screen to serve as photographed subjects — all images in the dataset show
one of these 4 printouts.

The ground truth images are **not** the 4 images that were printed out.
Instead, they are the frames of videos taken in the same condition as for
the blurry shots. The reason behind this choice is to ensure the same
lightness for better comparison. In total, there are about 200 frames per
video, and equivalently by blurry shot. There is a lot of redundancy
between the frames as the camera barely moves between consecutive frames,
for this reason the implementation allows selecting a single frame as the
privileged ground truth. This enables using the tooling provided by
deepinv such as [`deepinv.test()`](https://deepinv.org/api/stubs/deepinv.test.html.md#deepinv.test) and which gives approximately the same
performance as comparing to all the frames. It is the parameter `frames`
that controls this behavior, when it is set to either `"first"`,
`"middle"`, `"last"`, or to a specific frame index (between 1 and 198). If
the user wants to compare against all the frames, e.g. to reproduce the
benchmarks of the original paper, they can do so by setting the parameter
`frames` to `"all"` or to a list of frame indices.

The dataset does not have a preferred ordering and this implementation
uses lexicographic ordering on the printout index (1 to 4) and the
trajectory index (1 to 12). The parameter `ordering` controls whether to
order by printout first `"printout_first"` or by trajectory first
`"trajectory_first"`. This enables accessing the 48 items using the standard
method `__getitem__` using an index between 0 and 47. The nonstandard
method `get_item` allows selecting one of them by printout and trajectory
index directly if needed.

* **Parameters:**
  * **frames** (*Union* *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[**Union* *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]* *]* *]*) – Can be the frame number, `"first"`, `"middle"`, `"last"`, or `"all"`. If a list is provided, the method will return a list of sharp frames.
  * **ordering** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Ordering of the dataset. Can be `"printout_first"` or `"trajectory_first"`.
  * **root** (*Union* *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path) *]*) – Root directory of the dataset.
  * **transform:** (*Callable*) – (optional)  A function used to transform both the blurry shots and the sharp frames.
  * **download** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Download the dataset.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<hr />

* **Examples:**
  Download the dataset and load one of its elements
  ```default
  from deepinv.datasets import Kohler
  dataset = Kohler(root="datasets/Kohler",
                   frames="middle",
                   ordering="printout_first",
                   download=True)
  # Usual interface
  sharp_frame, blurry_shot = dataset[0]
  print(sharp_frame.shape, blurry_shot.shape)
  # Convenience method to directly index the printouts and trajectories
  sharp_frame, blurry_shot = dataset.get_item(1, 1, frames="middle")
  print(sharp_frame.shape, blurry_shot.shape)
  ```

<hr />

* **References:**

* <a id='footcite-kohler2012recording'>**[1]**</a> Rolf Kohler, Michael Hirsch, Betty Mohler, Bernhard Schölkopf, and Stefan Harmeling. Recording and playback of camera shake: benchmarking blind deconvolution with a real-world database. In *Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part VII 12*, 27–40. Springer, 2012.

#### *classmethod* download(root=None, remove_finished=False)

Download the dataset.

* **Parameters:**
  * **root** (*Union* *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path) *]*) – Root directory of the dataset.
  * **remove_finished** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Remove the archives after extraction.

<hr />

* **Examples:**
  Download the dataset
  ```default
  from deepinv.datasets import Kohler
  Kohler.download("datasets/Kohler")
  ```

#### get_item(printout_index, trajectory_index, frames=None)

Get a sharp frame and a blurry shot from the dataset.

* **Parameters:**
  * **printout_index** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Index of the printout.
  * **trajectory_index** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Index of the trajectory.
  * **frames** (*Union* *[**None* *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[**Union* *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]* *]* *]*) – Can be the frame number, “first”, “middle”, “last”, or “all”. If a list is provided, the method will return a list of sharp frames. By default, it uses the value provided in the constructor.
* **Returns:**
  (torch.Tensor, Union[torch.Tensor, list[torch.Tensor]]) The sharp frame(s) and the blurry shot.
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

<hr />

* **Examples:**
  Get the first (middle) sharp frame and blurry shot
  ```default
  sharp_frame, blurry_shot = dataset.get_item(1, 1, frame="middle")
  ```

  Get the list of all sharp frames and the blurry shot
  ```default
  sharp_frames, blurry_shot = dataset.get_item(1, 1, frame="all")
  ```

  Query a list of specific frames and the blurry shot
  ```default
  sharp_frames, blurry_shot = dataset.get_item(1, 1, frame=[1, "middle", 199])
  ```

<a id="sphx-glr-backref-deepinv-datasets-kohler"></a>

## Examples using `Kohler`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates blind image deblurring using the pretrained kernel estimation network from the paper :footcitecarbajal2023blind. The network estimates spatially-varying blur kernels from a blurred image, which are then used in a space-varying blur physics model to reconstruct the sharp image using a non-blind deblurring algorithm.">  <div class="sphx-glr-thumbnail-title">Blind deblurring with kernel estimation network</div>
</div>
<!-- thumbnail-parent-div-close --></div>
