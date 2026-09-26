# gStep

### *class* deepinv.optim.optim_iterators.gStep(g_first=False, \*\*kwargs)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Module for the single iteration steps on the prior term $\lambda \regname$.

* **Parameters:**
  * **g_first** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If True, the algorithm starts with a step on g and finishes with a step on f. Default: False.
  * **kwargs** – Additional keyword arguments.

<a id="sphx-glr-backref-deepinv-optim-optim-iterators-gstep"></a>

## Examples using `gStep`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to define your own optimization algorithm. For example, here, we implement the Primal-Dual Condat-Vu (CV) algorithm, and apply it for Single Pixel Camera reconstruction.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_custom_optim_thumb.png)

[PnP with custom optimization algorithm (Primal-Dual Condat-Vu)](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_custom_optim.html.md)

  <div class="sphx-glr-thumbnail-title">PnP with custom optimization algorithm (Primal-Dual Condat-Vu)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="where both the data fidelity and the prior are learned modules, distinct for each iterations.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_learned_primal_dual_thumb.png)

[Learned Primal-Dual algorithm for CT scan.](https://deepinv.org/auto_examples/unfolded/demo_learned_primal_dual.html.md)

  <div class="sphx-glr-thumbnail-title">Learned Primal-Dual algorithm for CT scan.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
