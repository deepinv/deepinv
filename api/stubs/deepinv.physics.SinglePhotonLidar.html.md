# SinglePhotonLidar

### *class* deepinv.physics.SinglePhotonLidar(sigma=1.0, bins=50, device='cpu', rng=None)

Bases: [`Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)

Single photon lidar operator for depth ranging.

See Rapp *et al.*<sup>[1](#footcite-rapp2020advances)</sup> for a review of this imaging method.

The forward operator is given by

$$
y_{i,j,t} = \mathcal{P}(h(t-d_{i,j}) r_{i,j} + b_{i,j})

$$

where $\mathcal{P}$ is the Poisson noise model, $h(t)$ is a Gaussian impulse response function at
time $t$, $d_{i,j}$ is the depth of the scene at pixel $(i,j)$,
$r_{i,j}$ is the intensity of the scene at pixel $(i,j)$ and $b_{i,j}$ is the background noise
at pixel $(i,j)$.

For a pixel grid of size (H,W) and batch size B, the signals have size (B, 3, H, W), where the first channel
contains the depth of the scene $d$, the second channel contains the intensity of the scene $r$ and
the third channel contains the per pixel background noise levels $b$.

* **Parameters:**
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Standard deviation of the Gaussian impulse response function.
  * **bins** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of histogram bins per pixel.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device to use (gpu or cpu).
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – (optional) a pseudorandom random number generator for
    the Poisson noise model [`deepinv.physics.PoissonNoise`](https://deepinv.org/api/stubs/deepinv.physics.PoissonNoise.html.md#deepinv.physics.PoissonNoise)

<hr />

* **References:**

* <a id='footcite-rapp2020advances'>**[1]**</a> Joshua Rapp, Julián Tachella, Yoann Altmann, Stephen McLaughlin, and Vivek K Goyal. Advances in single-photon lidar for autonomous vehicles: working principles, challenges, and recent advances. *IEEE Signal Processing Magazine*, 37(4):62–71, 2020.

#### A(x, \*\*kwargs)

Applies the forward operator.

Input is of size (B, 3, H, W) and output is of size (B, bins, H, W)

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – tensor containing the depth, intensity and background noise levels.

#### A_dagger(y, \*\*kwargs)

Applies Matched filtering to find the peaks.

Input is of size (B, bins, H, W), output of size (B, 3, H, W).

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements

<a id="sphx-glr-backref-deepinv-physics-singlephotonlidar"></a>

## Examples using `SinglePhotonLidar`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we show how to use the deepinv.physics.SinglePhotonLidar forward model.">  <div class="sphx-glr-thumbnail-title">Single photon lidar operator for depth ranging.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
