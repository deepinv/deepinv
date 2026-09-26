# Crop

### *class* deepinv.datasets.utils.Crop(size)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Torchvision-style transform to take crop in corner or any arbitrary place.

Expects tensor of shape (…, H, W).

* **Parameters:**
  **size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – if int or tuple of ints of length 2, crop in top left corner with size specified by tuple or square of size specified by int.
  If tuple of length 4, pass as (top, left, height, width) to torchvision crop function.
