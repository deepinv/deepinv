import shutil, os
import math
from typing import NamedTuple, Sequence, Mapping
from pathlib import Path
import PIL
from PIL.Image import Image as PIL_Image
import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np

from deepinv.datasets import (
    DIV2K,
    Urban100HR,
    Set14HR,
    CBSD68,
    LsdirHR,
    FMD,
    Kohler,
    FastMRISliceDataset,
    SimpleFastMRISliceDataset,
    MRISliceTransform,
    CMRxReconSliceDataset,
    NBUDataset,
    LidcIdriSliceDataset,
    Flickr2kHR,
    ImageDataset,
    generate_dataset,
    HDF5Dataset,
    TensorDataset,
    ImageFolder,
    SKMTEASliceDataset,
)
from deepinv.datasets.utils import (
    download_archive,
    loadmat,
    Crop,
    Rescale,
    ToComplex,
)
from deepinv.datasets.base import check_dataset
from deepinv.utils.demo import get_image_url
from deepinv.physics.mri import MultiCoilMRI, MRI, DynamicMRI
from deepinv.physics.generator import (
    GaussianMaskGenerator,
    BernoulliSplittingMaskGenerator,
)
from deepinv.physics.inpainting import Inpainting
from deepinv.physics.forward import Physics
from deepinv.utils.tensorlist import TensorList
from deepinv.loss.metric import PSNR
from deepinv.training import Trainer, test as trainer_test
from deepinv.tests.dummy import DummyModel

from unittest.mock import patch
import io


def get_dummy_pil_png_image():
    """Generates a dummy PIL image for testing."""
    im = PIL.Image.new("RGB", (128, 128), color=(0, 0, 0))
    buffer = io.BytesIO()
    im.save(buffer, format="PNG")
    buffer.seek(0)
    return PIL.PngImagePlugin.PngImageFile(buffer)


def check_dataset_format(
    dataset: Dataset,
    length: int = None,
    dtype: type = None,
    shape: tuple = None,
    allow_non_tensor: bool = False,
    skip_check: bool = False,
):
    """Check dataset format is correct.

    :param torch.utils.data.Dataset dataset: input dataset
    :param int length: intended dataset length.
    :param type dtype: intended dtype of returned batch.
    :param tuple shape: intended shape of returned batch, if it has the shape attribute.
    :param bool allow_non_tensor: if `True`, allow non tensors e.g. PIL Image and numpy ndarray to be returned.
    :param bool skip_check: skip ImageDataset checks.
    """
    if not skip_check:
        check_dataset(dataset, allow_non_tensor=allow_non_tensor)

    if dtype in (
        Tensor,
        np.ndarray,
        int,
        float,
        str,
        dict,
        list,
        tuple,  # but not "tuple_of_pils", because that is not collatable
        bytes,
        Mapping,
        NamedTuple,
        Sequence,
    ):  # from https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.default_collate

        # Define dataloader with random data sample
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(
                dataset, [torch.randint(0, len(dataset), (1,)).item()]
            )
        )
        _ = next(iter(dataloader))

        if not skip_check:
            # Check trainer compatible with dataset
            model = DummyModel()
            physics = Physics()
            try:
                _ = Trainer(
                    model,
                    physics,
                    optimizer=None,
                    train_dataloader=dataloader,
                    online_measurements=True,
                    save_path=None,
                    compare_no_learning=False,
                    metrics=[],
                ).setup_train(train=True)

                # We must switch any physics calculations as the data being checked here can be arbitrary
                # e.g. ints, which is currently not supported by PyTorch https://github.com/pytorch/pytorch/issues/58734
                _ = trainer_test(
                    model,
                    dataloader,
                    physics,
                    online_measurements=True,
                    compare_no_learning=False,
                    metrics=[],
                )

            except ValueError as e:
                # We may be checking paired unsup dataset, in which case training is ok to fail
                if "Online measurements can't be used if x is all NaN" not in str(e):
                    raise

    if length is not None:
        assert (
            len(dataset) == length
        ), f"Dataset should be length {length} but got {len(dataset)}."

    # The below tests are for datasets that return images only (and not tuples)
    if dtype is not None:
        if dtype == "tuple_of_pils":
            # This is a workaround for Python not having ability to check a variable is a `tuple[xxx]`.
            assert all(isinstance(d, PIL_Image) for d in dataset[0])
        else:
            assert isinstance(
                dataset[0], dtype
            ), f"Dataset should return data of type {dtype} but got type {type(dataset[0])}."

    if shape is not None:
        assert (
            dataset[0].shape == shape
        ), f"Dataset should return data of shape {shape} but got shape {dataset[0].shape}"


class MyDataset(ImageDataset):
    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return 1

    def __getitem__(self, _):
        return self.batch


def test_base_dataset():
    x, y, params = Tensor([0]), Tensor([0]), {"a": Tensor([0])}
    bad = "hello"
    check_dataset(MyDataset(x))
    check_dataset(MyDataset([x, y]))
    check_dataset(MyDataset([torch.nan, y]))
    check_dataset(MyDataset([x, y, params]))
    check_dataset(MyDataset([torch.nan, y, params]))
    check_dataset(MyDataset([torch.nan, params]))

    for bad_dataset_input in (
        torch.nan,
        [bad, y],
        [x, bad],
        [bad, y, params],
        [x, bad, params],
        [x, bad, params],
        [x, y, {1: 2}],
        [x, x, x, params],
        [x, params, y],
        bad,
        [x],
    ):
        with pytest.raises(RuntimeError):
            check_dataset(MyDataset(bad_dataset_input))


@pytest.mark.parametrize("physgen", [None, "mask"])
def test_hdfdataset(physgen):
    img_size = (1, 4, 4)
    dataset = MyDataset(torch.zeros(1, *img_size))
    physics = Inpainting(img_size, mask=0.5)
    physics_generator = (
        None if physgen is None else BernoulliSplittingMaskGenerator(img_size, 0.5)
    )
    path = generate_dataset(
        dataset,
        physics,
        save_dir="temp",
        batch_size=1,
        physics_generator=physics_generator,
    )
    dataset = HDF5Dataset(path, load_physics_generator_params=True)
    check_dataset_format(dataset, length=1, dtype=tuple, allow_non_tensor=False)
    dataset.close()
    assert dataset.hd5 is None


def test_tensordataset():
    x, y, params = (
        torch.zeros(1, 3, 4, 4),
        torch.zeros(1, 3, 4, 4),
        {"a": torch.zeros(1, 3, 4, 4)},
    )
    bad = np.zeros((1, 3, 4, 4))
    _ = TensorDataset(x=x)
    _ = TensorDataset(x=x, y=y)
    _ = TensorDataset(y=y)
    _ = TensorDataset(x=x, y=y, params=params)
    _ = TensorDataset(x=x, params=params)
    dataset = TensorDataset(y=y, params=params)
    assert math.isnan(
        dataset[0][0]
    ), "Dataset return tuple's first element must be NaN or single-element NaN tensor."

    for bad_dataset_input in (
        {},
        {"x": bad},
        {"y": bad},
        {"x": x, "y": torch.cat([y, y])},  # Batch size mismatch
    ):
        with pytest.raises(ValueError):
            _ = TensorDataset(**bad_dataset_input)


def get_transforms(transform_name, shape):
    if transform_name == "Crop":
        return Crop((shape[-2] // 2, shape[-1] // 2)), (
            *shape[:-2],
            shape[-2] // 2,
            shape[-1] // 2,
        )
    elif transform_name == "rescale":
        return Rescale(), shape
    elif transform_name == "tocomplex":
        return ToComplex(), (*shape[:2], 2, *shape[2:])
    else:
        raise ValueError("Invalid transform_name.")


@pytest.mark.parametrize("transform_name", ["Crop", "rescale", "tocomplex"])
def test_transforms(transform_name):
    transform, shape = get_transforms(transform_name, (1, 1, 8, 8))
    x = torch.rand(1, 1, 8, 8)
    y = transform(x)
    assert y.shape == shape


@pytest.fixture
def download_div2k():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "DIV2K"

    # Download div2K raw dataset
    DIV2K(tmp_data_dir, mode="val", download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


def test_load_div2k_dataset(download_div2k):
    """Check that DIV2K/DIV2K_train_HR contains 800 PIL images."""
    for totensor in [ToTensor(), None]:
        check_dataset_format(
            DIV2K(download_div2k, mode="val", download=False, transform=totensor),
            length=100,
            dtype=Tensor if totensor else PIL_Image,
            allow_non_tensor=not totensor,
        )


@pytest.fixture
def download_urban100():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "Urban100"

    # Download Urban100 raw dataset
    Urban100HR(tmp_data_dir, download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


def test_load_urban100_dataset(download_urban100):
    """Check that dataset contains 100 PIL images."""
    for totensor in [ToTensor(), None]:
        check_dataset_format(
            Urban100HR(download_urban100, download=False, transform=totensor),
            length=100,
            dtype=Tensor if totensor else PIL_Image,
            allow_non_tensor=not totensor,
        )


@pytest.fixture
def download_set14():
    """Downloads dataset for tests and removes it after test executions."""
    if not os.environ.get("DEEPINV_MOCK_TESTS", False):
        tmp_data_dir = "Set14"

        # Download Set14 raw dataset
        Set14HR(tmp_data_dir, download=True)

        # This will return control to the test function
        yield tmp_data_dir

        # After the test function complete, any code after the yield statement will run
        shutil.rmtree(tmp_data_dir)
    else:
        with (
            patch.object(Set14HR, "check_dataset_exists", return_value=True),
            patch.object(
                Path,
                "glob",
                side_effect=lambda p: (
                    [] if p[-3:] != "png" else [f"{i}_HR.png" for i in range(1, 15)]
                ),
            ),  # Only patch globbing pngs
            patch.object(PIL.Image, "open", return_value=get_dummy_pil_png_image()),
        ):
            yield "/dummy"


def test_load_set14_dataset(download_set14):
    """Check that dataset contains 14 PIL images."""
    for totensor in [
        ToTensor(),
    ]:
        check_dataset_format(
            Set14HR(download_set14, download=False, transform=totensor),
            length=14,
            dtype=Tensor if totensor else PIL_Image,
            allow_non_tensor=not totensor,
        )


@pytest.fixture
def download_flickr2khr():
    """Download or mock Flickr2kHR before testing"""
    if not os.environ.get("DEEPINV_MOCK_TESTS", False):
        tmp_data_dir = "Flickr2kHR"

        # Download Flickr raw dataset
        Flickr2kHR(tmp_data_dir, download=True)

        # This will return control to the test function
        yield tmp_data_dir

        # After the test function complete, any code after the yield statement will run
        shutil.rmtree(tmp_data_dir)
    else:
        with (
            patch.object(Flickr2kHR, "check_dataset_exists", return_value=True),
            patch.object(
                Path,
                "glob",
                side_effect=lambda p: (
                    [] if p[-3:] != "png" else [f"{i}_HR.png" for i in range(1, 101)]
                ),
            ),  # Only patch globbing pngs
            patch.object(PIL.Image, "open", return_value=get_dummy_pil_png_image()),
        ):
            yield "/dummy"


def test_load_Flickr2kHR_dataset(download_flickr2khr):
    """Test the dataset"""
    for totensor in [ToTensor(), None]:
        check_dataset_format(
            Flickr2kHR(download_flickr2khr, download=False, transform=totensor),
            length=100,
            dtype=Tensor if totensor else PIL_Image,
            allow_non_tensor=not totensor,
        )


@pytest.fixture
def download_cbsd68(download=True):
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "CBSD68"

    # Download CBSD raw dataset from huggingface
    try:
        CBSD68(tmp_data_dir, download=download)
    except ImportError:
        download = False

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    if download:
        shutil.rmtree(tmp_data_dir)


def test_load_cbsd68_dataset(download_cbsd68):
    """Check that dataset contains 68 PIL images."""

    pytest.importorskip(
        "datasets",
        reason="This test requires datasets. It should be "
        "installed with `pip install datasets`",
    )
    for totensor in [ToTensor(), None]:
        check_dataset_format(
            CBSD68(download_cbsd68, download=False, transform=totensor),
            length=68,
            dtype=Tensor if totensor else PIL_Image,
            allow_non_tensor=not totensor,
        )


@pytest.fixture
def download_Kohler():
    """Download the Köhler dataset before a test and remove it after completion."""
    if not os.environ.get("DEEPINV_MOCK_TESTS", False):
        root = "Kohler"
        Kohler.download(root)

        # Return the control flow to the test function
        yield root

        # Clean up the created directory
        shutil.rmtree(root)
    else:
        with patch.object(PIL.Image, "open", return_value=get_dummy_pil_png_image()):
            yield "/dummy"


@pytest.mark.parametrize("frames", ["middle", "first", "last", "all", 0, -1])
@pytest.mark.parametrize("ordering", ["printout_first", "trajectory_first"])
def test_load_Kohler_dataset(download_Kohler, frames, ordering):
    """Check that the Köhler dataset contains 48 PIL images."""
    root = download_Kohler

    for totensor in [ToTensor(), None]:
        dataset = Kohler(
            root=root,
            frames=frames,
            ordering=ordering,
            transform=totensor,
            download=False,
        )

        check_dataset_format(
            dataset,
            length=48,
            dtype=(
                tuple if totensor else None
            ),  # when no Transform, this is a tuple of list of PILs which is too complicated
            allow_non_tensor=not totensor,
            skip_check=True,
        )

    data_points = [dataset[0], dataset.get_item(1, 1, frames)]

    for sharp_frame, blurry_shot in data_points:
        if frames != "all":
            assert (
                type(sharp_frame) == PIL.PngImagePlugin.PngImageFile
            ), "The sharp frame is unexpectedly not a PIL image."
        else:
            assert isinstance(
                sharp_frame, list
            ), "The sharp frames are unexpectedly not a list."

        assert (
            type(blurry_shot) == PIL.PngImagePlugin.PngImageFile
        ), "The blurry frame is unexpectedly not a PIL image."


@pytest.fixture
def download_lsdir():
    """Downloads dataset for tests and removes it after test executions."""
    if not os.environ.get("DEEPINV_MOCK_TESTS", False):
        tmp_data_dir = "LSDIR"

        # Download LSDIR raw dataset
        LsdirHR(tmp_data_dir, mode="val", download=True)

        # This will return control to the test function
        yield tmp_data_dir

        # After the test function complete, any code after the yield statement will run
        shutil.rmtree(tmp_data_dir)
    else:
        mocker = lambda p: (
            [] if p[-3:] != "png" else [f"{i}.png" for i in range(1, 251)]
        )
        with (
            # Only patch globbing pngs
            patch.object(Path, "glob", side_effect=mocker),
            patch.object(os, "listdir", return_value=True),
            patch.object(os.path, "isdir", return_value=True),
            patch.object(PIL.Image, "open", return_value=get_dummy_pil_png_image()),
        ):
            yield "/dummy"


def test_load_lsdir_dataset(download_lsdir):
    """Check that dataset contains 250 PIL images."""
    for totensor in [ToTensor(), None]:
        check_dataset_format(
            LsdirHR(download_lsdir, mode="val", transform=totensor, download=False),
            length=250,
            dtype=Tensor if totensor else PIL_Image,
            allow_non_tensor=not totensor,
        )


@pytest.fixture
def download_fmd():
    """Downloads dataset for tests and removes it after test executions."""
    if not os.environ.get("DEEPINV_MOCK_TESTS", False):
        tmp_data_dir = "FMD"

        # indicates which subsets we want to download
        types = ["TwoPhoton_BPAE_R"]

        # Download FMD raw dataset
        FMD(tmp_data_dir, img_types=types, download=True)

        # This will return control to the test function
        yield tmp_data_dir

        # After the test function complete, any code after the yield statement will run
        shutil.rmtree(tmp_data_dir)
    else:
        with (
            patch.object(
                os, "listdir", return_value=[f"{i}.png" for i in range(1, 51)]
            ),
            patch.object(PIL.Image, "open", return_value=get_dummy_pil_png_image()),
        ):
            yield "/dummy"


def test_load_fmd_dataset(download_fmd):
    """Check that dataset contains 5000 noisy PIL images with its ground truths."""
    for totensor in [ToTensor(), None]:
        check_dataset_format(
            FMD(
                download_fmd,
                img_types=["TwoPhoton_BPAE_R"],
                transform=totensor,
                target_transform=totensor,
                download=False,
            ),
            length=5000,
            dtype=tuple if totensor else "tuple_of_pils",
            allow_non_tensor=not totensor,
        )


@pytest.fixture
def mock_lidc_idri():
    """Mock the LIDC-IDRI dataset"""
    if os.environ.get("DEEPINV_MOCK_TESTS", False):
        import pandas as pd
        import pydicom

        data = [["CT", f"Dummy_ID_{i}", f"/dummy/Scan{i}"] for i in range(1, 1019)]
        dummy_df = pd.DataFrame(
            data, columns=["Modality", "Subject ID", "File Location"]
        )
        # Generated using pydicomgenerator
        # https://github.com/sjoerdk/dicomgenerator
        dummy_dicom = pydicom.dcmread(
            os.path.join(os.path.dirname(__file__), "dicomgenerator_dummy.dcm")
        )

        # NOTE: dicomgenerator_dummy.dcm lacks a TransferSyntaxUID attribute.
        # We monkey patch it to make the test work.
        dummy_dicom.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        # NOTE: In lidc_idri, dcmread is imported from pydicom and stored to a variable.
        # This means that it cannot be mocked by patching pydicom.dcmread. Instead,
        # we patch the variable from the lidc_module directly.
        with (
            patch.object(os.path, "isdir", return_value=True),
            patch.object(os.path, "exists", return_value=True),
            patch.object(pd, "read_csv", return_value=dummy_df),
            patch.object(os, "listdir", return_value=["Slice1.dcm", "Slice2.dcm"]),
            # We use patch instead of patch.object to avoid cluttering the namespace.
            patch("deepinv.datasets.lidc_idri.dcmread", return_value=dummy_dicom),
        ):
            yield "/dummy"
    else:
        pytest.skip(
            "LIDC-IDRI dataset cannot be downloaded automatically and is not available for testing."
        )


# NOTE: The LIDC-IDRI needs to be downloaded manually.
@pytest.mark.parametrize("hounsfield_units", [False, True])
def test_load_lidc_idri_dataset(mock_lidc_idri, hounsfield_units):
    """Test the LIDC-IDRI dataset."""
    for totensor in [ToTensor(), None]:
        check_dataset_format(
            LidcIdriSliceDataset(
                root=mock_lidc_idri,
                transform=totensor,
                hounsfield_units=hounsfield_units,
            ),
            length=2036,
            dtype=Tensor if totensor else np.ndarray,
            allow_non_tensor=not totensor,
        )


@pytest.fixture
def download_nbu():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "NBU"

    # Download Urban100 raw dataset
    NBUDataset(tmp_data_dir, satellite="gaofen-1", download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


def test_load_nbu_dataset(download_nbu):
    """Check that dataset correct length and type."""
    dataset = NBUDataset(download_nbu, satellite="gaofen-1", download=False)
    check_dataset_format(dataset, length=5, dtype=Tensor, shape=(4, 256, 256))
    assert torch.all(
        (0 <= dataset[0]) & (dataset[0] <= 1)
    ), "Dataset image should be Tensor between 0-1."

    # Check pan band
    check_dataset_format(
        NBUDataset(download_nbu, satellite="gaofen-1", download=False, return_pan=True),
        length=5,
        dtype=TensorList,
        shape=[(4, 256, 256), (1, 1024, 1024)],
    )

    # Test ImageFolder with globs
    dataset = ImageFolder(
        download_nbu,
        x_path="nbu/gaofen-1/MS_256/*.mat",
        transform=ToTensor(),
        loader=lambda f: loadmat(f)["imgMS"],
    )
    check_dataset_format(dataset, length=5, dtype=Tensor, shape=(4, 256, 256))

    dataset = ImageFolder(
        download_nbu,
        y_path="nbu/gaofen-1/MS_256/*.mat",
        transform=ToTensor(),
        loader=lambda f: loadmat(f)["imgMS"],
    )
    check_dataset_format(dataset, length=5, dtype=tuple, allow_non_tensor=True)
    x, y = dataset[0]
    assert math.isnan(x) and y.shape == (4, 256, 256)


@pytest.fixture
def download_simplefastmri():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "fastmri"

    # Download simple FastMRI slice dataset
    SimpleFastMRISliceDataset(tmp_data_dir, download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


def test_SimpleFastMRISliceDataset(download_simplefastmri):
    dataset = SimpleFastMRISliceDataset(
        root_dir=download_simplefastmri,
        anatomy="knee",
        train=True,
        train_percent=1.0,
        download=False,
    )
    check_dataset_format(dataset, length=2, dtype=Tensor, shape=(2, 320, 320))
    assert not torch.all(dataset[0] == dataset[1])


@pytest.fixture
def download_fastmri():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "fastmri"
    file_name = "demo_fastmri_brain_multicoil.h5"

    # Download single FastMRI volume
    os.makedirs(tmp_data_dir, exist_ok=True)
    url = get_image_url(file_name)
    download_archive(url, f"{tmp_data_dir}/{file_name}")

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


def test_FastMRISliceDataset(download_fastmri):
    # Raw data shape
    kspace_shape = (512, 213)
    n_coils = 4
    n_slices = 16
    img_size = (213, 213)

    # Clean data shape
    rss_shape = (320, 320)

    data_dir = download_fastmri

    # Test metadata caching
    _ = FastMRISliceDataset(
        root=data_dir,
        slice_index="all",
        save_metadata_to_cache=True,
        metadata_cache_file="fastmrislicedataset_cache.pkl",
    )

    # Test data shapes
    dataset = FastMRISliceDataset(
        root=data_dir,
        slice_index="all",
        load_metadata_from_cache=True,
        metadata_cache_file="fastmrislicedataset_cache.pkl",
    )
    check_dataset_format(dataset, length=n_slices, dtype=tuple, shape=None)

    target1, kspace1 = dataset[0]
    target2, kspace2 = dataset[1]

    assert target1.shape == (1, *img_size)
    assert kspace1.shape == (2, n_coils, *kspace_shape)
    assert not torch.all(target1 == target2)
    assert not torch.all(kspace1 == kspace2)

    # Test compatible with MultiCoilMRI
    physics = MultiCoilMRI(
        mask=torch.ones(kspace_shape),
        coil_maps=torch.ones(kspace_shape, dtype=torch.complex64),
        img_size=img_size,
    )
    rss1 = physics.A_adjoint(kspace1.unsqueeze(0), rss=True, crop=True)
    assert torch.allclose(target1.unsqueeze(0), rss1)

    # Test singlecoil MRI mag works
    physics = MRI(mask=torch.ones(kspace_shape), img_size=img_size)
    mag1 = physics.A_adjoint(kspace1.unsqueeze(0)[:, :, 0], mag=True, crop=True)
    assert target1.unsqueeze(0).shape == mag1.shape

    # Test save simple dataset
    subset = dataset.save_simple_dataset(f"{download_fastmri}/temp_simple.pt")
    check_dataset_format(subset, length=n_slices, dtype=Tensor, shape=(2, *rss_shape))

    # Test slicing returns correct num of slices
    def num_slices(slice_index):
        return len(
            FastMRISliceDataset(
                root=data_dir,
                slice_index=slice_index,
                load_metadata_from_cache=True,
                metadata_cache_file="fastmrislicedataset_cache.pkl",
            ).samples
        )

    assert (
        num_slices("all"),
        num_slices("middle"),
        num_slices("middle+1"),
        num_slices(0),
        num_slices([0, 1]),
        num_slices("random"),
    ) == (n_slices, 1, 3, 1, 2, 1)

    # Test raw data transform for estimating maps and generating masks, and test ACS
    dataset = FastMRISliceDataset(
        root=data_dir,
        transform=MRISliceTransform(
            mask_generator=GaussianMaskGenerator(kspace_shape, acc=4),
            estimate_coil_maps=True,
        ),
        load_metadata_from_cache=True,
        metadata_cache_file="fastmrislicedataset_cache.pkl",
    )
    x, y, params = dataset[0]
    assert torch.all(y * params["mask"] == y)
    assert 0.24 < params["mask"].mean() < 0.26
    assert params["coil_maps"].shape == (n_coils, *kspace_shape)
    assert dataset.transform.get_acs() == 17  # ACS via mask generator

    # Test prewhitening and normalising
    dataset = FastMRISliceDataset(
        root=data_dir,
        transform=MRISliceTransform(
            acs=11,  # set manually as fully-sampled data has no ACS metadata
            prewhiten=True,
            normalize=True,
        ),
        load_metadata_from_cache=True,
        metadata_cache_file="fastmrislicedataset_cache.pkl",
    )
    assert dataset.transform.get_acs() == 11
    assert 1 < dataset[0][1].max() < 100  # normalized
    # TODO test prewhitening

    # Test filter_id in FastMRI init
    assert (
        len(
            FastMRISliceDataset(
                root=data_dir,
                filter_id=lambda s: "brain" in str(s.fname) and s.slice_ind < 3,
                load_metadata_from_cache=True,
                metadata_cache_file="fastmrislicedataset_cache.pkl",
            )
        )
        == 3
    )


@pytest.fixture
def download_CMRxRecon():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "CMRxRecon"

    # Download single CMRxRecon volume
    os.makedirs(tmp_data_dir, exist_ok=True)
    download_archive(
        get_image_url("CMRxRecon.zip"), f"{tmp_data_dir}/CMRxRecon.zip", extract=True
    )

    # This will return control to the test function
    yield f"{tmp_data_dir}/CMRxRecon"

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


def test_CMRxReconSliceDataset(download_CMRxRecon):

    img_size = (12, 512, 256)

    physics_generator = GaussianMaskGenerator(img_size)

    data_dir = download_CMRxRecon

    # Test metadata caching
    _ = CMRxReconSliceDataset(
        root=data_dir,
        save_metadata_to_cache=True,
        metadata_cache_file="cmrxreconslicedataset_cache.pkl",
        mask_dir=None,
        apply_mask=False,
    )

    # Test data shapes
    dataset = CMRxReconSliceDataset(
        root=data_dir,
        load_metadata_from_cache=True,
        metadata_cache_file="cmrxreconslicedataset_cache.pkl",
        mask_generator=physics_generator,
        mask_dir=None,
        apply_mask=True,
    )

    check_dataset_format(dataset, length=3, dtype=tuple)

    target1, kspace1, params1 = dataset[0]
    target2, kspace2, params2 = dataset[1]

    assert target1.shape == kspace1.shape == (2, *img_size)
    assert not torch.all(target1 == target2)
    assert not torch.all(kspace1 == kspace2)
    assert not torch.all(params1["mask"] == params2["mask"])
    assert torch.all(kspace1 * params1["mask"] == kspace1)  # kspace already masked
    assert (
        0.1 < params1["mask"].mean() < 0.26
    )  # masked has correct acc (< 0.25 due to padding)

    # Test reproducibility
    _, _, params1_again = dataset[0]
    assert torch.all(params1_again["mask"] == params1["mask"])

    # Loaded kspace is directly compatible with deepinv physics
    physics = DynamicMRI(img_size=img_size)
    kspace1_dinv = physics(
        target1.unsqueeze(0), mask=params1["mask"].unsqueeze(0)
    ).squeeze(0)
    assert torch.all(kspace1 == kspace1_dinv)

    # Test loading mask
    dataset = CMRxReconSliceDataset(
        root=data_dir,
        load_metadata_from_cache=True,
        metadata_cache_file="cmrxreconslicedataset_cache.pkl",
        apply_mask=True,
    )
    target1, kspace1, params1 = dataset[0]
    assert torch.all(kspace1 * params1["mask"] == kspace1)  # kspace already masked
    assert (
        0.1 < params1["mask"].mean() < 0.26
    )  # masked has correct acc (< 0.25 due to padding)

    # Test no apply mask
    dataset = CMRxReconSliceDataset(
        root=data_dir,
        load_metadata_from_cache=True,
        metadata_cache_file="cmrxreconslicedataset_cache.pkl",
        apply_mask=False,
    )
    target1, kspace1 = dataset[0]
    assert (kspace1 == 0).sum() == 0


@pytest.fixture
def download_SKMTEA():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "SKMTEA"
    file_name = "SKMTEA_tiny_2_slice.h5"

    # Download tiny SKMTEA volume
    os.makedirs(tmp_data_dir, exist_ok=True)
    url = get_image_url(file_name)
    download_archive(url, f"{tmp_data_dir}/{file_name}")

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


def test_SKMTEASliceDataset(download_SKMTEA, device):

    n_coils, img_size = 8, (512, 160)

    data_dir = download_SKMTEA

    # Test metadata caching
    dataset = SKMTEASliceDataset(
        root=data_dir,
        save_metadata_to_cache=True,
    )
    assert len(dataset) == 2

    # Test data shapes and dtypes
    dataset = SKMTEASliceDataset(
        root=data_dir,
        load_metadata_from_cache=True,
    )
    assert len(dataset) == 2
    x, y, params = next(iter(DataLoader(dataset)))
    assert x.shape == (1, 2, *img_size)
    assert y.shape == (1, 2, n_coils, *img_size)
    assert params["mask"].shape == (1, 1, *img_size)
    assert params["coil_maps"].shape == (1, n_coils, *img_size)

    assert x.dtype == y.dtype == params["mask"].dtype == torch.float32
    assert params["coil_maps"].dtype == torch.complex64

    # Test physics compatible
    physics = MultiCoilMRI(**params, device=device)
    y2 = physics(x.to(device)).detach().cpu()
    assert PSNR(max_pixel=None, complex_abs=True)(y2, y) > 40

    # Test filter_id
    assert (
        len(
            SKMTEASliceDataset(
                root=data_dir,
                load_metadata_from_cache=True,
                filter_id=lambda s: s.slice_ind == 1,
            )
        )
        == 1
    )
