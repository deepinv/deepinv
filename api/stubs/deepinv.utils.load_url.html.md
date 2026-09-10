# load_url

### deepinv.utils.load_url(url, \*\*kwargs)

Load URL to a buffer.

This can be used as the argument for other IO functions such as
[`deepinv.utils.load_torch()`](https://deepinv.org/api/stubs/deepinv.utils.load_torch.html.md#deepinv.utils.load_torch), [`deepinv.utils.load_np()`](https://deepinv.org/api/stubs/deepinv.utils.load_np.html.md#deepinv.utils.load_np) etc.
to load data directly from a URL.

Downloaded content is cached under [`deepinv.utils.get_cache_home()`](https://deepinv.org/api/stubs/deepinv.utils.get_cache_home.html.md#deepinv.utils.get_cache_home)
so repeated calls for the same URL do not hit the network again. The
cache layout mirrors the URL: a file fetched from
`https://huggingface.co/datasets/deepinv/images/resolve/main/celeba_example.jpg`
is stored at
`<cache_home>/url_cache/huggingface.co/datasets/deepinv/images/resolve/main/celeba_example.jpg`.
Two URLs that differ only in their query string share the same cache
entry — fine for the `?download=true` query used by HuggingFace.

The HTTP request uses a `(connect, read)` timeout of
`(10, 60)` seconds; a hung connection or stalled stream raises a
[`deepinv.utils.DownloadError`](https://deepinv.org/api/stubs/deepinv.utils.DownloadError.html.md#deepinv.utils.DownloadError) rather than blocking indefinitely.

* **Parameters:**
  **url** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – URL of the file to load
* **Returns:**
  `BytesIO` buffer.
* **Raises:**
  [**deepinv.utils.DownloadError**](https://deepinv.org/api/stubs/deepinv.utils.DownloadError.html.md#deepinv.utils.DownloadError) – if the file cannot be downloaded.
* **Return type:**
  [*BytesIO*](https://docs.python.org/3.9/library/io.html#io.BytesIO)

## Examples using `load_url`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various input/output functions provided by DeepInverse for handling medical and scientific imaging formats. We demonstrate loading and plotting from DICOM, NIfTI, ISMRMRD, PyTorch, NumPy and raster data sources.">  <div class="sphx-glr-thumbnail-title">Loading scientific images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div>
<!-- thumbnail-parent-div-close --></div>
