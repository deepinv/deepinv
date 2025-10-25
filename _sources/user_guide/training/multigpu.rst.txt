.. _multigpu:

Using Multiple GPUs
===================

Since all deepinv building blocks inherit from :class:`torch.nn.Module` , they are compatible with torch data parallel
modules, either via :class:`torch.nn.DataParallel` or :class:`torch.nn.parallel.DistributedDataParallel`.

For instance, one can simply write:

::

    import torch
    import deepinv as dinv

    backbone = dinv.models.DRUNet(pretrained=None, device=torch.device("cuda"))
    model = dinv.models.ArtifactRemoval(backbone)
    gpu_number = torch.cuda.device_count()  # number of GPUs to use
    model = torch.nn.DataParallel(model, device_ids=list(range(gpu_number)))


which can seamlessly be combined with the default Trainer :class:`deepinv.Trainer`.

Note however that it is recommended to use :class:`torch.nn.parallel.DistributedDataParallel` instead of the former
when training on multiple GPUs. Among other drawbacks of the previous approach, it is not possible to set attributes of
a model within the forward pass, which is required for some deepinv models. In this case, the training loop needs to be
modified. We point the reader to the `PyTorch documentation <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_
to extend their training codes to the multi-gpu case.
