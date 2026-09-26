# ICNN

### *class* deepinv.models.ICNN(in_channels=3, num_filters=64, kernel_dim=5, num_layers=10, strong_convexity=0.5, pos_weights=True, device='cpu', dim=2)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Convolutional Input Convex Neural Network (ICNN).

The network is built to be convex in its input.
The model is fully convolutional and thus can be applied to images of any size.

Based on the implementation from Tan *et al.*<sup>[1](#footcite-tan2023data)</sup>.

* **Parameters:**
  * **in_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of input channels.
  * **num_filters** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of hidden units.
  * **kernel_dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – dimension of the convolutional kernels.
  * **num_layers** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of layers.
  * **strong_convexity** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Strongly convex parameter.
  * **pos_weights** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to force positive weights in the forward pass.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device to put the model on.
  * **dim** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – Whether to build 2D or 3D network (if str, can be “2”, “2d”, “3D”, etc.)

<hr />

* **References:**

* <a id='footcite-tan2023data'>**[1]**</a> Hong Ye Tan, Subhadip Mukherjee, Junqi Tang, and Carola-Bibiane Schönlieb. Data-driven mirror descent with input-convex neural networks. *SIAM Journal on Mathematics of Data Science*, 5(2):558–587, 2023.

#### forward(x)

Calculate potential function of the ICNN.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor of shape `(B, C, H, W)`.

#### grad(x)

Calculate the gradient of the potential function.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor of shape `(B, C, H, W)`.
