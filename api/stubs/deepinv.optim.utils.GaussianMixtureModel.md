# GaussianMixtureModel

### *class* deepinv.optim.utils.GaussianMixtureModel(n_components, dimension, device='cpu')

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Gaussian mixture model including parameter estimation.

Implements a Gaussian Mixture Model, its negative log likelihood function and an EM algorithm
for parameter estimation.

* **Parameters:**
  * **n_components** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of components of the GMM
  * **dimension** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – data dimension
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – gpu or cpu.

#### classify(x, cov_regularization=False)

returns the index of the most likely component

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input data of shape batch_dimension x dimension
  * **cov_regularization** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether using regularized covariance matrices

#### component_log_likelihoods(x, cov_regularization=False)

returns a tensor containing the log likelihood values of x for each component

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input data of shape batch_dimension x dimension
  * **cov_regularization** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether using regularized covariance matrices

#### fit(dataloader, max_iters=100, stopping_criterion=None, data_init=True, cov_regularization=1e-5, verbose=False)

Batched Expectation Maximization algorithm for parameter estimation.

* **Parameters:**
  * **dataloader** ([*torch.utils.data.DataLoader*](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)) – containing the data
  * **max_iters** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations
  * **stopping_criterion** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – stop when objective decrease is smaller than this number.
    None for performing exactly max_iters iterations
  * **data_init** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – True for initialize mu by the first data points, False for using current values as initialization
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Output progress information in the console

#### forward(x)

evaluate negative log likelihood function

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input data of shape batch_dimension x dimension

#### get_cov()

get method for covariances

#### get_cov_inv_reg()

get method for covariances

#### get_weights()

get method for weights

#### load_state_dict(\*args, \*\*kwargs)

Override load_state_dict to maintain internal parameters.

#### set_cov(cov)

Sets the covariance parameters to cov and maintains their log-determinants and inverses

* **Parameters:**
  **cov** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – new covariance matrices in a n_components x dimension x dimension tensor

#### set_cov_reg(reg)

Sets covariance regularization parameter for evaluating
Needed for EPLL.

* **Parameters:**
  **reg** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – covariance regularization parameter

#### set_weights(weights)

sets weight parameter while ensuring non-negativity and summation to one

* **Parameters:**
  **weights** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – non-zero weight tensor of size n_components with non-negative entries
