# PromptIR

### *class* deepinv.models.PromptIR(in_channels=3, out_channels=3, dim=48, num_blocks=(4, 6, 6, 8), num_refinement_blocks=4, heads=(1, 2, 4, 8), ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', decoder=True, device=None, pretrained='download')

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor), [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

PromptIR restoration model.

PromptIR is a blind restoration model that was proposed in Potlapalli *et al.*<sup>[1](#footcite-potlapalli2023promptir)</sup>.

The authors’ pretrained weights for in_channels=out_channels=3 can be downloaded via setting `pretrained='download'` (default).

* **Parameters:**
  * **in_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of channels of the input.
  * **out_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of channels of the output.
  * **dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – base dimension of the model.
  * **num_blocks** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – number of transformer blocks at each level of the encoder/decoder
  * **num_refinement_blocks** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of transformer blocks in the refinement module.
  * **heads** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – number of attention heads at each level of the encoder/decoder.
  * **ffn_expansion_factor** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – expansion factor of the feed-forward networks.
  * **bias** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use bias in the convolutional layers.
  * **LayerNorm_type** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – type of layer normalization to use (‘BiasFree’ or ‘WithBias’).
  * **decoder** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use the decoder with prompt generation blocks.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device to load the model on.
  * **pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – path to the pretrained weights or ‘download’ to download the authors’ weights.

<hr />

* **Example:**

```pycon
>>> import torch
>>> from deepinv.models import PromptIR
>>> model = PromptIR()
>>> x = torch.randn(1, 3, 256, 256)
>>> out = model(x)
>>> out.shape
torch.Size([1, 3, 256, 256])
```

<hr />

* **References:**

* <a id='footcite-potlapalli2023promptir'>**[1]**</a> Vaishnav Potlapalli, Syed Waqas Zamir, Salman H Khan, and Fahad Shahbaz Khan. Promptir: prompting for all-in-one image restoration. *Advances in Neural Information Processing Systems*, 36:71275–71293, 2023.

#### load_pretrained(checkpoint_path)

Load pretrained weights.

* **Parameters:**
  **checkpoint_path** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – path to the checkpoint or ‘download’ to download the authors’ weights.
