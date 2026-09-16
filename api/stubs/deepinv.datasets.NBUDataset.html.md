# NBUDataset

### *class* deepinv.datasets.NBUDataset(root_dir=None, satellite='gaofen-1', return_pan=False, transform_ms=None, transform_pan=None, download=False, use_dict_output=False)

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

NBU remote sensing multispectral satellite imagery dataset.

Returns `Cx256x256` multispectral (MS) satellite images of urban scenes from 6 different satellites.
with `C=4` for `"gaofen-1"`, `ikonos`, `quickbird`, `worldview-4` and `C=8` for the rest.

#### NOTE
When there are 4 channels, they correspond to the blue, green, red, and near-infrared bands.
When there are 8 channels, they correspond to the coastal, blue, green, yellow, red, red-edge, near-infrared 1,
and near-infrared 2 bands.
See [this](https://www.pgc.umn.edu/guides/delivery-docs/pgc-commercial-satellite-imagery-documentation/) for more details.

For pan-sharpening problems, you can return pan-sharpening measurements by using `return_pan=True`,
outputting a [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) of `(MS, PAN)` where `PAN` are 1024x1024 panchromatic images.

This dataset was compiled in Meng *et al.*<sup>[1](#footcite-meng2020large)</sup> and downloaded from [this drive](https://github.com/Lihui-Chen/Awesome-Pansharpening?tab=readme-ov-file#datasets).
We perform no other processing other than to take the “Urban” subset and provide each satellite’s data separately, which you can choose using the `satellite` argument:

- `"gaofen-1"`: 5 images
- `"ikonos"`: 60 images
- `"quickbird"`: 150 images
- `"worldview-2"`: 150 images
- `"worldview-3"`: 55 images
- `"worldview-4"`: 90 images

#### NOTE
Returns images as [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) normalized to 0-1 over the whole dataset.

See [Remote sensing with satellite images](https://deepinv.org/auto_examples/physics/demo_remote_sensing.html.md#sphx-glr-auto-examples-physics-demo-remote-sensing-py) for example using
this dataset with remote sensing inverse problems.

<hr />

* **Examples:**
  Instantiate dataset and download raw data from the Internet
  ```default
  from deepinv.datasets import NBUDataset
  dataset = NBUDataset(
      root_dir=".",            # root directory
      satellite="worldview-2", # choose satellite
      download=True,           # download dataset
      return_pan=True          # return panchromatic image too as pair (MS, PAN)
  )
  print(dataset.check_dataset_exists())
  print(len(dataset))
  ```
* **Parameters:**
  * **root_dir** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – NBU dataset root directory
  * **satellite** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – satellite name, choose from the options above, defaults to “gaofen-1”.
  * **return_pan** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, return panchromatic images as TensorList of (MS, PAN), if `False`, just return multispectral images.
  * **transform_ms** (*Callable*) – optional transform for multispectral images
  * **transform_pan** (*Callable*) – optional transform for panchromatic images
  * **download** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to download dataset
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<hr />

* **References:**

* <a id='footcite-meng2020large'>**[1]**</a> Xiangchao Meng, Yiming Xiong, Feng Shao, Huanfeng Shen, Weiwei Sun, Gang Yang, Qiangqiang Yuan, Randi Fu, and Hongyan Zhang. A large-scale benchmark data set for evaluating pansharpening performance: overview and implementation. *IEEE Geoscience and Remote Sensing Magazine*, 9(1):18–52, 2020.

#### check_dataset_exists()

Verify that the image folders exist and contain all the images.

`root_dir` should have the following structure:

```default
root_dir --- nbu --- <satellite> --- 1.mat
          |       |               |
          |       |               -- x.mat
          |       -- <satellite>
          -- xxx
```

<a id="sphx-glr-backref-deepinv-datasets-nbudataset"></a>

## Examples using `NBUDataset`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
