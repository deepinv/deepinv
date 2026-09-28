# Optimizer tests

`test_algos.py` checks one PGD update against the L2 gradient and each
supported prior's proximal operator. It lists differentiable and
non-differentiable priors separately. `compat/test_algos.py` compares short
trajectories with the legacy PGD using representative priors and the supported
differentiable fidelities.

The shared problem fixture lives in `conftest.py`. The small
positive float64 images keep likelihood gradients well defined. The parent
`device` fixture runs the tests on CPU and any available accelerator. Wavelet
cases skip if `ptwt` or PyWavelets is unavailable.

PGD needs `data_fidelity.grad` and `prior.prox`. L1 and IndicatorL2 fidelities
are non-differentiable, so the fidelity compatibility tests omit them. A callable
DataFidelity case also checks the autograd gradient fallback. PnP, RED, and
ScorePrior need a denoiser parameter that optim_v2.PGD does not expose; PatchNR
needs its own model setup. The main prior cases cover the remaining usable priors,
including non-differentiable ones. Itoh uses SpatialUnwrapping physics, and the
stacked fidelity uses stacked measurements and physics.

Run with `python -m pytest deepinv/tests/optim_v2 -q`.
