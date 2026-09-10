# Rescale

### *class* deepinv.datasets.utils.Rescale(\*args, rescale_mode='min_max', \*\*kwargs)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Image value rescale torchvision-style transform.

The transform expects tensor of shape (…, H, W) and performs rescale over all dimensions (i.e. over all images in batch).

* **Parameters:**
  **rescale_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – rescale mode, either “min_max” or “clip”.

#### forward(x)

Rescale image.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – image tensor of shape (…, H, W)
