- [x] Updating the doc with working examples
_ _ _ 
- 
	- [x] functional/ (Flo and Hai 02/04)
_ _ _ 
- 
	- [x] generators/base.py (Flo 03/04)
	- [x] generators/blur.py (Flo 03/04)
	- [x] generators/mri.py (Flo 03/04)
	- **Comment (Flo)**: I had to change the output of AccelerationMaskGenerator.step() because it was outputting 2 identical (repeated) channels per default, which was different the output shape announced in the doc, where channels was set to 1 by default. This should be verified with the person who coded this. (I kept the 2 channel return as a commented line)
- - [x] generators/noise.py (Flo 03/04)

- - [x] Double check by Hai that doc is okay

**Comment (Flo)** : Need to refactorize the doc, as right now individual physics generators appear both under the *Introduction/Generators/* section and in each *Forward operators/* section. I think Julian chose the second option, that we did not see with Hai on Tuesday.

_ _ _ 
- [x] Refactor doc physics (double check done)
- [ ] A tour of blur operators
- [ ] Check multiGPU class generator

- [ ] Coding unit test using pytest
  - [x] Check output types for generator (dict, device, dtype)
  - [x] Check all boundary conditions + colour + non square images for physics (adjoint)
  - [x] Check that generators return the right stuff
  - [x] Check GeneratorMixture (ce qu'a déjà fait Flo)
  - [ ] Check SpaceVaryingBlur
  - [ ] Check ProductConvolution
_ _ _ 
- [ ] Discussions
  - [ ] Move rotate, scale, ... to functional? -> would be better
  - [ ] Operator norm is different from 1 
  - [ ] Restructure the doc (in particular self-supervised, ...)
  - [ ] Add functions to Physics such as jvp, ... This would make it possible to play easily with nonlinear operators, tangent maps,...


    ../../examples/basics/demo_loading.py failed leaving traceback:

    Traceback (most recent call last):
      File "/media/data/Pierre/Works/PACKAGES/deepinv/examples/basics/demo_loading.py", line 191, in <module>
        model_new.load_state_dict(ckpt_state_dict)
      File "/home/pierre/anaconda3/envs/deepinv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 2153, in load_state_dict
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
    RuntimeError: Error(s) in loading state_dict for BaseUnfold:
    	Missing key(s) in state_dict: "init_params_algo.g_param.20", "init_params_algo.g_param.21", "init_params_algo.g_param.22", "init_params_algo.g_param.23", "init_params_algo.g_param.24", "init_params_algo.g_param.25", "init_params_algo.g_param.26", "init_params_algo.g_param.27", "init_params_algo.g_param.28", "init_params_algo.g_param.29", "init_params_algo.stepsize.20", "init_params_algo.stepsize.21", "init_params_algo.stepsize.22", "init_params_algo.stepsize.23", "init_params_algo.stepsize.24", "init_params_algo.stepsize.25", "init_params_algo.stepsize.26", "init_params_algo.stepsize.27", "init_params_algo.stepsize.28", "init_params_algo.stepsize.29", "params_algo.g_param.20", "params_algo.g_param.21", "params_algo.g_param.22", "params_algo.g_param.23", "params_algo.g_param.24", "params_algo.g_param.25", "params_algo.g_param.26", "params_algo.g_param.27", "params_algo.g_param.28", "params_algo.g_param.29", "params_algo.stepsize.20", "params_algo.stepsize.21", "params_algo.stepsize.22", "params_algo.stepsize.23", "params_algo.stepsize.24", "params_algo.stepsize.25", "params_algo.stepsize.26", "params_algo.stepsize.27", "params_algo.stepsize.28", "params_algo.stepsize.29". 

    ../../examples/patch-priors/demo_patch_priors_CT.py failed leaving traceback:

    Traceback (most recent call last):
      File "/media/data/Pierre/Works/PACKAGES/deepinv/examples/patch-priors/demo_patch_priors_CT.py", line 151, in <module>
        noise_model = LogPoissonNoise(mu=mu, N0=N0)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/media/data/Pierre/Works/PACKAGES/deepinv/deepinv/physics/noise.py", line 275, in __init__
        self.N0 = torch.nn.Parameter(torch.tensor(N0))
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/pierre/anaconda3/envs/deepinv/lib/python3.12/site-packages/torch/nn/parameter.py", line 40, in __new__
        return torch.Tensor._make_subclass(cls, data, requires_grad)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    RuntimeError: Only Tensors of floating point and complex dtype can require gradients

    ../../examples/unfolded/demo_DEQ.py failed leaving traceback:

    Traceback (most recent call last):
      File "/media/data/Pierre/Works/PACKAGES/deepinv/examples/unfolded/demo_DEQ.py", line 197, in <module>
        train(
      File "/media/data/Pierre/Works/PACKAGES/deepinv/deepinv/training_utils.py", line 829, in train
        trained_model = trainer.train(
                        ^^^^^^^^^^^^^^
      File "/media/data/Pierre/Works/PACKAGES/deepinv/deepinv/training_utils.py", line 582, in train
        self.step(
      File "/media/data/Pierre/Works/PACKAGES/deepinv/deepinv/training_utils.py", line 471, in step
        x_net, logs = self.compute_loss(physics_cur, x, y, train=train)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/media/data/Pierre/Works/PACKAGES/deepinv/deepinv/training_utils.py", line 413, in compute_loss
        loss_total.backward()  # Backward the total loss
        ^^^^^^^^^^^^^^^^^^^^^
      File "/home/pierre/anaconda3/envs/deepinv/lib/python3.12/site-packages/torch/_tensor.py", line 522, in backward
        torch.autograd.backward(
      File "/home/pierre/anaconda3/envs/deepinv/lib/python3.12/site-packages/torch/autograd/__init__.py", line 266, in backward
        Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
      File "/media/data/Pierre/Works/PACKAGES/deepinv/deepinv/unfolded/deep_equilibrium.py", line 112, in backward_hook
        g = backward_FP({"est": (grad,)}, None)[0]["est"][0]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/pierre/anaconda3/envs/deepinv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/pierre/anaconda3/envs/deepinv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
        return forward_call(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/media/data/Pierre/Works/PACKAGES/deepinv/deepinv/optim/fixed_point.py", line 245, in forward
        X = self.anderson_acceleration_step(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/media/data/Pierre/Works/PACKAGES/deepinv/deepinv/optim/fixed_point.py", line 182, in anderson_acceleration_step
        p = torch.linalg.solve(H[:, : m + 1, : m + 1], q[:, : m + 1])[
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgetrfBatched( handle, n, dA_array, ldda, ipiv_array, info_array, batchsize)`
