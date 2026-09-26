# ToComplex

### *class* deepinv.datasets.utils.ToComplex(\*args, \*\*kwargs)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Torchvision-style transform to add empty imaginary dimension to image.

Expects tensor of shape (…, H, W) and returns tensor of shape (…, 2, H, W).

#### forward(x)

Convert real image to complex image.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – image tensor of shape (…, H, W)
