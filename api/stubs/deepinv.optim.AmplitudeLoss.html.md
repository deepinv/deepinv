# AmplitudeLoss

### *class* deepinv.optim.AmplitudeLoss

Bases: [`DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)

Amplitude loss as the data fidelity term for [`deepinv.physics.PhaseRetrieval()`](https://deepinv.org/api/stubs/deepinv.physics.PhaseRetrieval.html.md#deepinv.physics.PhaseRetrieval) reconstrunction.

In this case, the data fidelity term is defined as

$$
f(x) = \sum_{i=1}^{m}{(\sqrt{|b_i x|^2}-\sqrt{y_i})^2},
$$

where $b_i$ is the i-th row of the linear operator $B$ of the phase retrieval class and $y_i$ is the i-th entry of the measurements, and $m$ is the number of measurements.
