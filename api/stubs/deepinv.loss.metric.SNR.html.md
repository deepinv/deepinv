# SNR

### *class* deepinv.loss.metric.SNR(\*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

Compute the signal-to-noise ratio (SNR)

The signal-to-noise ratio (in dB) associated to a ground truth signal $x$ and a noisy estimate $\hat{x} = \inverse{y}$ is defined by

$$
\mathrm{SNR} = 10 \log_{10} \left( \frac{\|x\|_2^2}{\|x - y\|_2^2} \right).
$$

#### NOTE
The input is assumed to be batched and the SNR is computed for each element independently.

* **Parameters:**
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – The noisy signal.
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – The reference signal.
* **Returns:**
  (torch.Tensor) The SNR value in decibels (dB).
