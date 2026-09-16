<a id="multigpu"></a>

# Using Multiple GPUs

Since all deepinv building blocks inherit from [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) , they are compatible with torch data parallel
modules, either via [`torch.nn.DataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) or [`torch.nn.parallel.DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel).

For instance, one can simply write:

```default
import torch
import deepinv as dinv

backbone = dinv.models.DRUNet(pretrained=None, device=torch.device("cuda"))
model = dinv.models.ArtifactRemoval(backbone)
gpu_number = torch.cuda.device_count()  # number of GPUs to use
model = torch.nn.DataParallel(model, device_ids=list(range(gpu_number)))
```

which can seamlessly be combined with the default Trainer [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer).

Note however that it is recommended to use [`torch.nn.parallel.DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) instead of the former
when training on multiple GPUs. Among other drawbacks of the previous approach, it is not possible to set attributes of
a model within the forward pass, which is required for some deepinv models. In this case, the training loop needs to be
modified. We point the reader to the [PyTorch documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
to extend their training codes to the multi-gpu case.

## Distributed Reconstruction

DeepInverse also provides a simplified API for distributing DeepInverse objects across multiple devices and processes for reconstruction.
See the [user guide on distributed reconstruction](https://deepinv.org/user_guide/distributed/reconstruction.html.md#distributed-reconstruction) for more information.
