# stack

### deepinv.physics.stack(\*physics)

Stacks multiple forward operators $A = \begin{bmatrix} A_1(x) \\ A_2(x) \\ \vdots \\ A_n(x) \end{bmatrix}$.

The measurements produced by the resulting model are [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) objects, where
each entry corresponds to the measurements of the corresponding operator.

* **Parameters:**
  **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Physics operators $A_i$ to be stacked.

## Examples using `stack`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Physics Operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div>
<!-- thumbnail-parent-div-close --></div>
