# get_freer_gpu

### deepinv.utils.get_freer_gpu(verbose=True, use_torch_api=True, hide_warnings=False)

Returns the GPU device with the most free memory.

Use in conjunction with `torch.cuda.is_available()`.

If `use_torch_api=True` then attempts to select GPU using only torch commands, otherwise
uses system driver to detect GPUs (via `nvidia-smi` command). The first method may be slower
but is more reliable as the former depends on environment settings.
If system method is chosen and fails, the call falls back to using torch commands and a warning
is printed. If no CUDA devices are detected, then `None` is returned.

* **Parameters:**
  * **verbose** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – print selected GPU index and memory
  * **use_torch_api** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – use torch commands if True, or Nvidia driver otherwise
  * **hide_warnings** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – suppress all warnings for all methods
* **Return torch.device device:**
  selected cuda device

#### WARNING
GPU indices in `nvidia-smi` may not match those in PyTorch if in your environment `CUDA_DEVICE_ORDER`
is not set to `PCI_BUS_ID`:
[https://discuss.pytorch.org/t/gpu-devices-nvidia-smi-and-cuda-get-device-name-output-appear-inconsistent/13150](https://discuss.pytorch.org/t/gpu-devices-nvidia-smi-and-cuda-get-device-name-output-appear-inconsistent/13150)
If the variable is not set or has different value, the call to will print a warning
(if not suppressed with `hide_warnings=True`) but will not change the device.
