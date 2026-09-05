import torch
from typing import Callable
from deepinv.datasets.fastmri import FastMRISliceDataset, MRISliceTransform


class CalgarySliceDataset(FastMRISliceDataset):
    # TODO: auto-download demo sample for MICCAI tutorial
    """Dataset for `Calgary-Campinas <https://sites.google.com/view/calgary-campinas-dataset>`_ 12-coil raw brain kspace.

    Loads Calgary `h5` volumes of shape `(num_slices, H, W, 2N)`, where slice dim is in image domain and `H,W` is kspace.
    The dataset loads and preprocesses all kspace slices per volume, of shape `(2, N, H, W)`. These are fully-sampled for train/val volumes and masked for the test set.

    Also computes the GT `x`, the magnitude root-sum-square reconstructions of shape `(1, H, W)`, or `torch.nan` for the masked test set.

    The dataset is loaded as tuples `(x, y, params)`, where
    `params` optionally contains the sampling `mask` and, if desired, estimated `coil_maps`.

    Calgary kspace uses the opposite centering convention to deepinv, so it is converted here (a half-FOV checkerboard shift)
    so that `y` works directly with :class:`deepinv.physics.MultiCoilMRI`.

    :param str, pathlib.Path root: path to the dataset.
    :param Callable transform: transform taking `(target, kspace)`, defaults to :class:`CalgarySliceTransform`.
    :param kwargs: passed to :class:`deepinv.datasets.FastMRISliceDataset` (e.g. `slice_index`, `filter_id`, metadata cache, `use_dict_output`).

    TODO example
    """

    def __init__(self, root, transform: Callable | None = None, **kwargs):
        super().__init__(
            root=root,
            transform=transform if transform is not None else CalgarySliceTransform(),
            **kwargs,
        )

    @staticmethod
    def _retrieve_metadata(fname):
        import h5py

        with h5py.File(fname, "r") as hf:
            num_slices, height, width, channels = hf["kspace"].shape
        return {
            "num_slices": num_slices,
            "height": height,
            "width": width,
            "coils": channels // 2,
        }

    def __getitem__(self, idx):  # TODO add val set masks?
        import h5py

        fname, slice_ind, metadata = self.samples[idx]

        with h5py.File(fname, "r") as hf:
            # (slices, H, W, 2N) interleaved real/imag -> (N, H, W) complex
            kspace = torch.view_as_complex(
                torch.from_numpy(
                    hf["kspace"][slice_ind]
                )  # slice dim already in image domain
                .unflatten(-1, (-1, 2))
                .contiguous()
            ).permute(2, 0, 1)

        # Pre-shift y because Calgary assumed uncentered FFT whereas deepinv MRI assumes FastMRI convention of centered FFT.
        H, W = kspace.shape[-2:]
        kspace = kspace * (
            (-1.0) ** (torch.arange(H)[:, None] + torch.arange(W)[None, :])
        ).to(kspace.real.dtype)

        kspace = self.from_torch_complex(kspace.unsqueeze(0)).squeeze(0)  # (2, N, H, W)
        target = self.rss(self.kspace_to_im(kspace.unsqueeze(0))).squeeze(0)
        params = {}

        if self.transform is not None:
            target, kspace, params = self.transform(
                target, kspace, seed=str(fname) + str(slice_ind), metadata=metadata
            )

        if self.use_dict_output:
            out = {}

            if target is not None:
                out["x"] = target

            out["y"] = kspace

            if params:
                out["params"] = params

            return out

        return (target if target is not None else torch.nan, kspace) + (
            (params,) if params else ()
        )


class CalgarySliceTransform(MRISliceTransform):
    """Extract params and estimate coil maps for Calgary raw data.

    To be used with :class:`CalgarySliceDataset`.
    """

    def __call__(self, target, kspace, seed=None, metadata=None, **kwargs):
        params = {
            "mask": (
                self.to_torch_complex(kspace.unsqueeze(0)).abs().sum(1) > 0
            ).float()
        }  # (1, H, W)
        if self.estimate_coil_maps:
            params["coil_maps"] = self.generate_maps(kspace, metadata=metadata)
        return target, kspace, params
