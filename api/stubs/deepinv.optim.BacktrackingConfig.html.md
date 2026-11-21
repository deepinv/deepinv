# BacktrackingConfig

### *class* deepinv.optim.BacktrackingConfig(gamma=0.1, eta=0.9, max_iter=20)

Bases: [`object`](https://docs.python.org/3.12/library/functions.html#object)

Configuration parameters for backtracking line search on the stepsize.

* **Parameters:**
  * **gamma** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Armijo-like parameter (controls sufficient decrease).
  * **eta** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Step reduction factor (e.g. multiply step by eta on failure).
  * **max_iter** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – Maximum number of backtracking steps.

<a id="sphx-glr-backref-deepinv-optim-backtrackingconfig"></a>

## Examples using `BacktrackingConfig`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs projected phase from preprocessed near-field holograms acquired at several propagation distances, see e.g. huhn2022Fast for details on the physics.">![](auto_examples/physics/images/thumb/sphx_glr_demo_fresnel_phase_retrieval_thumb.png)

[Multi-distance Fresnel phase retrieval](https://deepinv.org/auto_examples/physics/demo_fresnel_phase_retrieval.html.md)

  <div class="sphx-glr-thumbnail-title">Multi-distance Fresnel phase retrieval</div>
</div>
<!-- thumbnail-parent-div-close --></div>
