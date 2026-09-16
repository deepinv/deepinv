# DiscriminatorMetric

### *class* deepinv.loss.adversarial.DiscriminatorMetric(metric=None, real_label=1.0, fake_label=0.0, no_grad=False, device='cpu')

Bases: [`object`](https://docs.python.org/3.9/library/functions.html#object)

Generic GAN discriminator metric building block.

Compares discriminator output with labels depending on if the image should be real or not.

The loss function is composed following LSGAN from Mao *et al.*<sup>[1](#footcite-mao2017least)</sup>.

This can be overridden to provide any flavour of discriminator metric, e.g. NSGAN, WGAN, LSGAN etc.

See Lucic *et al.*<sup>[2](#footcite-lucic2018gans)</sup> for a comparison.

* **Parameters:**
  * **metric** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – loss with which to compare outputs, defaults to [`torch.nn.MSELoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)
  * **real_label** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – value for ideal real image, defaults to 1.
  * **fake_label** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – value for ideal fake image, defaults to 0.
  * **no_grad** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to no_grad the metric computation, defaults to `False`
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – torch device, defaults to `"cpu"`

<hr />

* **References:**

* <a id='footcite-mao2017least'>**[1]**</a> Xudong Mao, Qing Li, Haoran Xie, Raymond YK Lau, Zhen Wang, and Stephen Paul Smolley. Least squares generative adversarial networks. In *Proceedings of the IEEE international conference on computer vision*, 2794–2802. 2017.
* <a id='footcite-lucic2018gans'>**[2]**</a> Mario Lucic, Karol Kurach, Marcin Michalski, Sylvain Gelly, and Olivier Bousquet. Are gans created equal? a large-scale study. *Advances in neural information processing systems*, 2018.
