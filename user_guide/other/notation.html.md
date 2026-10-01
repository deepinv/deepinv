# Math Notation

The documentation of `deepinv` uses a unified mathematical notation that is summarized in the following table:

#### List of mathematical symbols

| Symbol                                | Meaning                                                                                                                                  |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| $x\in\xset$                           | Underlying image or signal to reconstruct of $n$ elements.                                                                               |
| $y\in\yset$                           | Observed measurement vector of size $m$.                                                                                                 |
| $p(x)$                                | Distribution of images $x$ (often referred to as prior distribution).                                                                    |
| $p(y)$                                | Distribution of measurements $y$.                                                                                                        |
| $\forw{x}$                            | Deterministic mapping that captures the physics of the imaging system.                                                                   |
| $A^\top\colon\yset\to\xset$           | Adjoint of the measurement operator.                                                                                                     |
| $\noise{y}$                           | Stochastic mapping adding noise to measurements.                                                                                         |
| $\inverse{y}$                         | Reconstruction network that maps measurements to images $y\mapsto x$.                                                                    |
| $\denoiser{x}{\sigma}$                | Gaussian denoiser for noise of standard deviation $\sigma$.                                                                              |
| $\datafid{x}{y} = \distance{A(x)}{y}$ | Data fidelity term, enforcing measurement consistency $y\approx A(x)$, depending on the choice of the<br/>distance function (see below). |
| $\distance{u}{y}$                     | Distance function measuring the discrepancy between $u$ and $y$.<br/>It is linked to the noise model (likelihood).                       |
| $\reg{x}$                             | Regularization term that promotes plausible reconstructions. It is linked to $p(x)$.                                                     |
| $\lambda$                             | Hyperparameter controlling the amount of regularization.                                                                                 |
