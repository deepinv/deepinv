# GLOWCouplingBlock

### *class* deepinv.optim.prior.GLOWCouplingBlock(dim, subnet, clamp=1.6)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

GLOW-style affine coupling block.

Each block performs two successive affine coupling steps on the input vector,
which is split into two halves $(x_1, x_2)$:

* Step 1 — a subnetwork acting on $x_1$ produces a pointwise scale $s_1$ and
  shift $t_1$ that are applied to $x_2$: $y_2 = x_2 \cdot \exp(s_1) + t_1$.
* Step 2 — a second subnetwork acting on $y_2$ produces $(s_2, t_2)$ that are
  applied to $x_1$ : $y_1 = x_1 \cdot \exp(s_2) + t_2$.

Both steps are exactly invertible, and their combined log-determinant of the
Jacobian is $1^{\top}(s_1 + s_2)$.  The scale outputs are soft-clamped via
$\text{clamp} \times \frac{2}{\pi} \text{arctan}(s / \text{clamp})$ to keep the log-determinant bounded and
training stable.

The two-step affine coupling structure follows Dinh *et al.*<sup>[1](#footcite-dinh2017density)</sup>, and
the soft-clamping of scales is introduced in Kingma and Dhariwal<sup>[2](#footcite-kingma2018glow)</sup>.

* **Parameters:**
  * **dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – total input/output dimension (will be split evenly).
  * **subnet** (*Callable*) – a callable `subnet(channels_in, channels_out) -> nn.Module`
    that constructs the subnetworks used inside each coupling step.
  * **clamp** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – soft-clamping magnitude for the log-scale outputs. Default is `1.6`.

<hr />

* **References:**

* <a id='footcite-dinh2017density'>**[1]**</a> Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. In *International Conference on Learning Representations*. 2017.
* <a id='footcite-kingma2018glow'>**[2]**</a> Durk P Kingma and Prafulla Dhariwal. Glow: generative flow with invertible 1x1 convolutions. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018. URL: [https://proceedings.neurips.cc/paper_files/paper/2018/file/d139db6a236200b21cc7f752979132d0-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2018/file/d139db6a236200b21cc7f752979132d0-Paper.pdf).

#### forward(x, rev=False)

Applies the coupling block in the forward or inverse direction.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape `(N, dim)`.
  * **rev** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, applies the inverse transformation. Default is `False`.
* **Returns:**
  tuple `(y, log_det)` where `y` ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) is the
  transformed tensor of shape `(N, dim)` and `log_det` ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor))
  is the log-determinant of the Jacobian of shape `(N,)`.
