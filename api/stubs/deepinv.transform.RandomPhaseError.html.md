# RandomPhaseError

### *class* deepinv.transform.RandomPhaseError(\*args, scale=0.2, \*\*kwargs)

Bases: [`Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)

Random phase error transform.

This transform is specific to MRI problems, and adds a phase error to k-space using:

$Ty=\exp(-i\phi_k)y$ where $\phi_k=\pi\alpha s_e$ if $k$ is an even index,
or $\phi_k=\pi\alpha s_o$ if odd, and where $\alpha$ is a scale parameter,
and $s_o,s_e\sim U(-1,1)$.

This transform is reproducible: for given param dict `se, so`, the transform is deterministic.

* **Parameters:**
  **scale** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – scale parameters $s_e$ and $s_o$ or range to pick randomly.
