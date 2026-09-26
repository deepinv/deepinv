# deepinv.utils.TensorList

### *class* deepinv.utils.TensorList(x)

Represents a list of [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) with different shapes.
It allows to sum, flatten, append, etc. lists of tensors seamlessly, in a
similar fashion to [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor).

* **Parameters:**
  **x** – a list of [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), a single [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) or a TensorList.
