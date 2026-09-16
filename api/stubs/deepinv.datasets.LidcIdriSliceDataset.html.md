# LidcIdriSliceDataset

### *class* deepinv.datasets.LidcIdriSliceDataset(root=None, transform=None, hounsfield_units=False, use_dict_output=False)

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

Dataset for [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) that provides access to CT image slices.

Published in Armato III *et al.*<sup>[1](#footcite-armato2011lung)</sup>.

The Lung Image Database Consortium image collection (LIDC-IDRI) consists
<br/>
of diagnostic and lung cancer screening thoracic computed tomography (CT)
<br/>
scans with marked-up annotated lesions.
<br/>

#### WARNING
To download the raw dataset, you will need to install the [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images),
then download the manifest file (.tcia file) [here](https://www.cancerimagingarchive.net/collection/lidc-idri/), and open it by double clicking.

**Raw data file structure:**

```default
self.root --- LIDC-IDRI --- LICENCE
           |             -- LIDC-IDRI-0001 --- `STUDY_UID` --- `SERIES_UID` --- xxx.xml
           |             |                                                   -- 1-001.dcm
           |             -- LIDC-IDRI-1010                                   |
           |                                                                 -- 1-xxx.dcm
           -- metadata.csv
```

0) There are 1010 patients and a total of 1018 CT scans.
<br/>
1) Each CT scan is composed of 2d slices.
<br/>
2) Each slice is stored as a .dcm file
<br/>
3) This class gives access to one slice of a CT scan per data sample.
<br/>
4) Each slice is represented as an (512, 512) array.
<br/>
* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Root directory of dataset. Directory path from where we load and save the dataset.
  * **transform:** (*Callable*) – (optional)  A function/transform that takes in a data sample and returns a transformed version.
  * **hounsfield_units** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, convert pixel values to [Hounsfield Units (HU)](https://en.wikipedia.org/wiki/Hounsfield_scale). Default is `False`.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<hr />

* **Examples:**
  Instantiate dataset
  ```default
  import torch
  from deepinv.datasets import LidcIdriSliceDataset
  root = "/path/to/dataset/LIDC-IDRI"
  dataset = LidcIdriSliceDataset(root=root)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
  batch = next(iter(dataloader))
  print(batch.shape)
  ```

#### NOTE
This class requires the `pandas` and `pydicom` packages to be installed. Install them with `pip install pandas` and `pip install pydicom`.

<hr />

* **References:**

* <a id='footcite-armato2011lung'>**[1]**</a> Samuel G Armato III, Geoffrey McLennan, Luc Bidaut, Michael F McNitt-Gray, Charles R Meyer, Anthony P Reeves, Binsheng Zhao, Denise R Aberle, Claudia I Henschke, Eric A Hoffman, and others. The lung image database consortium (lidc) and image database resource initiative (idri): a completed reference database of lung nodules on ct scans. *Medical physics*, 38(2):915–931, 2011.

#### *class* SliceSampleIdentifier(slice_fname, scan_folder, patient_id)

Bases: [`NamedTuple`](https://docs.python.org/3.9/library/typing.html#typing.NamedTuple)

Data structure for identifying slices.

In LIDC-IDRI, there are 1010 patients.
Among them, 8 patients have each 2 CT scans.

* **Parameters:**
  * **slice_fname** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Filename of a dicom file containing 1 slice of the scan.
  * **scan_folder** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Path to all dicom files from the same scan.
  * **patient_id** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Foldername of one patient among the 1010.

#### patient_id *: [str](https://docs.python.org/3.9/library/stdtypes.html#str)*

Alias for field number 2

#### scan_folder *: [str](https://docs.python.org/3.9/library/stdtypes.html#str)*

Alias for field number 1

#### slice_fname *: [str](https://docs.python.org/3.9/library/stdtypes.html#str)*

Alias for field number 0
