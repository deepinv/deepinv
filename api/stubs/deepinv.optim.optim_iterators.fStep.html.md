# fStep

### *class* deepinv.optim.optim_iterators.fStep(g_first=False, \*\*kwargs)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Module for the single iteration steps on the data-fidelity term $f$.

* **Parameters:**
  * **g_first** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If True, the algorithm starts with a step on g and finishes with a step on f. Default: False.
  * **kwargs** – Additional keyword arguments.

<a id="sphx-glr-backref-deepinv-optim-optim-iterators-fstep"></a>

## Examples using `fStep`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to define your own optimization algorithm. For example, here, we implement the Primal-Dual Condat-Vu (CV) algorithm, and apply it for Single Pixel Camera reconstruction.">  <div class="sphx-glr-thumbnail-title">PnP with custom optimization algorithm (Primal-Dual Condat-Vu)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="where both the data fidelity and the prior are learned modules, distinct for each iterations.">  <div class="sphx-glr-thumbnail-title">Learned Primal-Dual algorithm for CT scan.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
