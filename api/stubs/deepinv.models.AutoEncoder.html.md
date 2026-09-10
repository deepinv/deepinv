# AutoEncoder

### *class* deepinv.models.AutoEncoder(dim_input, dim_mid=1000, dim_hid=32, residual=True)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Simple fully connected autoencoder network.

Simple architecture that can be used for debugging or fast prototyping.

* **Parameters:**
  * **dim_input** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – total number of elements (pixels) of the input.
  * **dim_hid** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of features in intermediate layer.
  * **dim_hid** – latent space dimension.
  * **residual** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – use a residual connection between input and output.
