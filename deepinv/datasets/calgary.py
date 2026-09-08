import torch
from typing import Callable
from deepinv.datasets.fastmri import FastMRISliceDataset, MRISliceTransform


class CalgarySliceDataset(FastMRISliceDataset):
    """Dataset for `Calgary-Campinas <https://sites.google.com/view/calgary-campinas-dataset>`_ 12-coil raw brain kspace.

    Loads Calgary `h5` volumes of shape `(num_slices, H, W, 2N)`, where slice dim is in image domain and `H,W` is kspace.
    The dataset loads and preprocesses all kspace slices per volume, of shape `(2, N, H, W)`. These are fully-sampled for train/val volumes and masked for the test set.

    Also computes the GT `x`, the magnitude root-sum-square reconstructions of shape `(1, H, W)`, or `torch.nan` for the masked test set.

    The dataset is loaded as tuples `(x, y, params)`, where
    `params` optionally contains the sampling `mask` and, if desired, estimated `coil_maps`.

    .. note::
        The test set comes already masked, which :class:`deepinv.datasets.CalgarySliceTransform` estimates. For the validation set, the data
        is fully-sampled. You can simulate masked data using precomputed Poisson-disk masks as follows ::

            mask_file = f"R{acceleration}_{y.shape[-2]}x{y.shape[-1]}.npy"
            torch.hub.download_url_to_file(f"https://huggingface.co/datasets/NKI-AI/direct-mri-masks/resolve/main/calgary_campinas_masks/{mask_file.name}", str(mask_file))
            masks = np.load(mask_file) # (100, H, W) bool
            mask = torch.from_numpy(masks[0]).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
            y *= mask

    Calgary kspace uses the opposite centering convention to deepinv, so it is converted here (a half-FOV checkerboard shift)
    so that `y` works directly with :class:`deepinv.physics.MultiCoilMRI`.

    :param str, pathlib.Path root: path to the dataset.
    :param Callable transform: transform taking `(target, kspace)`, defaults to :class:`deepinv.datasets.CalgarySliceTransform`.
    :param kwargs: passed to :class:`deepinv.datasets.FastMRISliceDataset` (e.g. `slice_index`, `filter_id`, metadata cache, `use_dict_output`).

    |sep|

    :Examples:

        Download a Calgary test volume (see :ref:`sphx_glr_auto_examples_models_demo_mri_pretrained.py`) and load its middle slice:

        >>> import deepinv as dinv  # doctest: +SKIP
        >>> from deepinv.datasets import CalgarySliceDataset, download_archive  # doctest: +SKIP
        >>> root = dinv.utils.get_cache_home() / "calgary"  # doctest: +SKIP
        >>> download_archive(dinv.utils.get_image_url("demo_calgary_test_e13991s3_P01536.7.h5"), root / "vol.h5")  # doctest: +SKIP
        >>> x, y, params = CalgarySliceDataset(root, slice_index="middle")[0]  # doctest: +SKIP
        >>> y.shape  # (2, N, H, W) multicoil k-space  # doctest: +SKIP
        torch.Size([2, 12, 218, 170])
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

    def __getitem__(self, idx):
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

    To be used with :class:`deepinv.datasets.CalgarySliceDataset`.
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
