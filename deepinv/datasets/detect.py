from __future__ import annotations
from pathlib import Path
import torch
from deepinv.datasets.base import ImageDataset
from deepinv.datasets.utils import download_archive
from deepinv.utils.io import load_tiff


class DeteCTDataset(ImageDataset):
    """2DeteCT dataset of 2D Computed Tomography acquisitions.

    The dataset was acquired by :footcite:t:`kiss20232detect` and used for benchmarking CT reconstruction algorithms in :footcite:t:`kiss2025benchmarking`.
    The data is industrial CT projection data (i.e. sinograms) of various materials acquired using a proprietary scanner from `Centrum Wiskunde & Informatica <https://www.cwi.nl/en/>`_.
    The samples contain materials resembling the attenuation of human anatomy; see :footcite:t:`kiss20232detect` for more details.

    The projections (shape `(1,n_angles,956)`) are preprocessed (flat/dark-corrected, log-transformed, all in PyTorch) following `LION <https://github.com/CambridgeCIA/LION>`_
    such that the setup matches exactly :footcite:t:`kiss2025benchmarking`, such that the dataset can be used to compare DeepInverse image reconstruction methods
    with the values reported in :footcite:t:`kiss2025benchmarking`.

    Each sample is scanned 3 times: `mode1`, `mode2` and `mode3`. See below for their usage.

    "Ground truth" `x` are also provided as iterative recons using all angles, of shape `(1,1024,1024)`.

    To download the data from `Zenodo <https://doi.org/10.5281/zenodo.8014758>`_,
    use :func:`download_dataset <deepinv.datasets.DeteCTDataset.download_dataset>`, which extracts each archive into
    the ``2DeteCT_slicesXXXX-YYYY`` (+ ``_RecSeg``) subfolders in root. Note: for the test set, you only need to download slices ``4001-5000``, i.e. do
    `dinv.datasets.DeteCTDataset.download_dataset(root='/path/to/2DeteCT', blocks='test')`.

    :param str, pathlib.Path root: root dir, should contain subfolders named `2DeteCT_slicesXXXX-YYYY` (+ `_RecSeg`)
    :param str problem: benchmarking problem from 2DeteCT.
      - `full`: `mode2` acquired data (3600 projections)
      - `sparse_view`: `mode2` acquired data then evenly subsampled
      - `limited_angle`: `mode2` acquired data then limited angles taken
      - `low_dose`: `mode1` acquired data (3W instead of 90W)
      - `beam_hardening`: `mode3` acquired data (acquired without a filter, leading to beam-hardening)

    :param int n_angles: kept projections for sparse_view/limited_angle, defaults to 3600 (i.e. all angles).
    :param str slice_ids: `all` (default, every slice found from 1-5000), `train`/`val`/`test` (LION 3930/550/470 sample split), or `ood` (out-of-distribution slices 5521-6370).
    :param bool use_dict_output: whether to return output as dict with keys "x", "y", "params" instead of tuple (default `False`).

    Example:

    TODO using sample slide + recon on HF
    """

    def __init__(
        self,
        root: str | Path,
        problem: str = "full",
        n_angles: int = 3600,
        slice_ids: int = "all",
        use_dict_output: bool = False,
    ):
        super().__init__(use_dict_output=use_dict_output)
        self.root = Path(root)
        self.problem, self.n_angles = problem, n_angles
        self.mode = {"low_dose": "mode1", "beam_hardening": "mode3"}.get(
            problem, "mode2"
        )

        lo, hi = {
            "all": (1, 5000),
            "train": (1, 3930),
            "val": (3931, 4480),
            "test": (4531, 5000),
            "ood": (5521, 6370),
        }[slice_ids]

        self.slices = sorted(
            int(p.name[5:])
            for p in self.root.glob("2DeteCT_slices*/slice[0-9]*")
            if p.is_dir()
            and not p.parent.name.endswith("_RecSeg")
            and lo <= int(p.name[5:]) <= hi
        )

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        slice_num = self.slices[i]
        if slice_num >= 5521:  # OOD set
            block = "2DeteCT_slicesOOD"
        else:
            block_start = (slice_num - 1) // 1000 * 1000 + 1  # e.g. 1, 1001, ..., 4001
            block = f"2DeteCT_slices{block_start}-{block_start + 999}"
        stem = f"slice{slice_num:05d}"

        data_dir = self.root / block / stem / self.mode

        sino = load_tiff(data_dir / "sinogram.tif")[:, :, :-1]  # (1, 1, 3600, 1912)
        dark = load_tiff(data_dir / "dark.tif")  # (1, 1, 1, 1912)
        flat = 0.5 * (
            load_tiff(data_dir / "flat1.tif") + load_tiff(data_dir / "flat2.tif")
        )

        if slice_num < 2830 or 5520 < slice_num < 5871:

            def detector_shift(a):
                out = torch.empty_like(a)
                out[..., :-1] = a[..., 1:]
                out[..., -1] = 2 * a[..., -1] - a[..., -2]
                return out

            sino, flat, dark = (
                detector_shift(sino),
                detector_shift(flat),
                detector_shift(dark),
            )

        # Bin detector pixels
        sino = sino[..., 0::2] + sino[..., 1::2]  # (1, 1, 3600, 956)
        dark = dark[..., 0::2] + dark[..., 1::2]  # (1, 1, 1, 956)
        flat = flat[..., 0::2] + flat[..., 1::2]

        # Detector corrections:
        sino = (sino - dark) / (flat - dark)  # flat/dark-field correction
        sino = -sino.clip(min=1e-6).log()  # Beer-Lambert
        sino = sino.flip(dims=(-1,))  # flip detector

        if self.problem == "sparse_view":
            sino = sino[:, :, :: 3600 // self.n_angles]  # (1, 1, n_angles, 956)
        elif self.problem == "limited_angle":
            sino = sino[:, :, : self.n_angles]  # (1, 1, n_angles, 956)

        y = sino.squeeze(0).contiguous().float()

        x = (
            load_tiff(
                self.root / (block + "_RecSeg") / stem / "mode2" / "reconstruction.tif"
            )
            .squeeze(0)
            .contiguous()
            .float()
        )

        return {"x": x, "y": y} if self.use_dict_output else (x, y)

    @staticmethod
    def download_dataset(
        root: str | Path, blocks: str | list = "all", force_download: bool = False
    ) -> None:  # pragma: no cover
        """Download and extract the 2DeteCT archives from Zenodo into ``root``.

        Each block's raw data and reference reconstructions (RecSeg) are extracted into the
        ``2DeteCT_slicesXXXX-YYYY`` (+ ``_RecSeg``) subfolders expected by the dataset.

        .. warning::
            The archives are very large (up to ~34GB each); ``blocks="all"`` needs several hundred GB of disk.

        :param str, pathlib.Path root: dir to download into (same ``root`` passed to init).
        :param str, list blocks: which slice ranges to download: ``"all"`` (slices 1-5000),
            ``"test"`` (only slices 4001-5000, i.e. the benchmark test set), ``"ood"`` (out-of-distribution slices 5521-6370),
            or a list of ranges from ``["1-1000", "1001-2000", "2001-3000", "3001-4000", "4001-5000", "OOD"]``.
        :param bool force_download: re-download even if the archive already exists.
        """
        root = Path(root)
        if isinstance(blocks, str):
            blocks = {
                "all": ["1-1000", "1001-2000", "2001-3000", "3001-4000", "4001-5000"],
                "test": ["4001-5000"],
                "ood": ["OOD"],
            }.get(blocks, [blocks])

        ZENODO_RECORDS = {
            #  range         data       recseg
            "1-1000": ("8014758", "8017583"),
            "1001-2000": ("8014766", "8017604"),
            "2001-3000": ("8014787", "8017612"),
            "3001-4000": ("8014829", "8017618"),
            "4001-5000": ("8014874", "8017624"),
            "OOD": ("8014907", "8017653"),
        }

        for block in blocks:
            data_id, recseg_id = ZENODO_RECORDS[block]
            for record_id, folder in (
                (data_id, f"2DeteCT_slices{block}"),
                (recseg_id, f"2DeteCT_slices{block}_RecSeg"),
            ):
                download_archive(
                    url=f"https://zenodo.org/records/{record_id}/files/{folder}.zip?download=1",
                    save_path=root / folder / f"{folder}.zip",
                    extract=True,
                    force_download=force_download,
                )

    @staticmethod
    def get_astra_geometry(problem: str = "full", n_angles: int = None) -> tuple:
        """Get astra object geometry and project geometry for 2DeteCT setup.

        Construct geometry objects to pass to :class:`deepinv.physics.TomographyWithAstra`
        in order to test physics-conditioned algorithms on the 2DeteCT benchmark.

        The object geometry values and fan-beam projection geometry values are taken from `LION <https://github.com/CambridgeCIA/LION>`_.

        The projection geometry is defined as conebeam with one detector row.

        Usage ::

            import deepinv as dinv
            obj_geom, proj_geom = dinv.datasets.DeteCTDataset.get_astra_geometry()
            physics = dinv.physics.TomographyWithAstra(
                object_geometry=obj_geom,
                projection_geometry=proj_geom,
                is_2d=True, # important
                normalize=True,
                device=device,
                noise_model=dinv.physics.PoissonGaussianNoise(),
            )

        :param str problem: 2DeteCT benchmark problem, either "sparse_view" or "limited_angle", for how to undersample angles.
        :param n_angles: for sparse_view or limited_angle, how many angles.
        :return tuple: obj_geom dict, proj_geom dict
        """
        import astra

        obj_geom = astra.create_vol_geom(1024, 1024, 1, -513, 511, -513, 511, -0.5, 0.5)

        det_pix = 2 * 0.0748  # binned detector pixel in mm
        fov = det_pix * 956 * 431.019989 / 529.000488  # field-of-view width in mm
        scale = 1024 / fov  # rescale such that recon grid has unit voxels
        sod = 431.019989 * scale  # source-origin distance
        sdd = 529.000488 * scale  # source-detector distance
        det_pix *= scale

        angles = -torch.linspace(0, 2 * torch.pi, 3600 + 1)[:-1] + torch.pi

        if problem == "sparse_view":
            angles = angles[:: 3600 // n_angles]
        elif problem == "limited_angle":
            angles = angles[:n_angles]

        proj_geom = astra.create_proj_geom(
            "cone",
            det_pix,
            det_pix,
            1,
            956,
            angles.numpy(),
            sod,
            sdd - sod,
        )

        return obj_geom, proj_geom
