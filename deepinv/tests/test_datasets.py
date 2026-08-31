import shutil, os
import sys
import math
from contextlib import contextmanager
from typing import NamedTuple, Sequence, Mapping
from pathlib import Path
import PIL
from PIL.Image import Image as PIL_Image
import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, CenterCrop
from deepinv.loss import Metric
import numpy as np
import h5py

import deepinv as dinv
from deepinv.datasets import (
    BrainWebPET,
    BrainWebMRI,
    DIV2K,
    Urban100HR,
    Set14HR,
    CBSD68,
    BSDS500,
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
    RandomPatchSampler,
)
from deepinv.datasets.base import check_dataset, batch_as_dict
from deepinv.datasets.utils import (
    download_archive,
    extract_zipfile,
    extract_tarball,
    extract_rarfile,
    Crop,
    Rescale,
    ToComplex,
)
from deepinv.utils.io import load_mat
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

from unittest.mock import MagicMock, patch
import io


def get_dummy_pil_png_image():
    """Generates a dummy PIL image for testing."""
    im = PIL.Image.new("RGB", (128, 128), color=(0, 0, 0))
    buffer = io.BytesIO()
    im.save(buffer, format="PNG")
    buffer.seek(0)
    return PIL.PngImagePlugin.PngImageFile(buffer)


@contextmanager
def dataset_output_context(use_dict_output):
    """Capture the deprecation warning for the legacy tuple output format."""
    if use_dict_output:
        yield
    else:
        with pytest.warns(
            DeprecationWarning, match="tuple format for dataset outputs is deprecated"
        ):
            yield


def image_output_type(
    use_dict_output, transform, *, paired=False, untransformed_type=PIL_Image
):
    """Return the expected sample type for an image dataset configuration."""
    if use_dict_output:
        return dict if transform is not None else "dict_of_pils"
    if paired:
        return tuple if transform is not None else "tuple_of_pils"
    return Tensor if transform is not None else untransformed_type


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
        with dataset_output_context(use_dict_output=dataset.use_dict_output):
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
                with dataset_output_context(use_dict_output=dataset.use_dict_output):
                    _ = Trainer(
                        model,
                        physics,
                        optimizer=None,
                        train_dataloader=dataloader,
                        online_measurements=True,
                        save_path=None,
                        compare_no_learning=False,
                        metrics=None,
                    ).setup_train(train=True)

                class DummyMetric(Metric):
                    def __init__(self):
                        super().__init__("dummy")

                    def forward(self, x_net, x, **kwargs):
                        return torch.tensor(0.0, device=x.device)

                # We must switch any physics calculations as the data being checked here can be arbitrary
                # e.g. ints, which is currently not supported by PyTorch https://github.com/pytorch/pytorch/issues/58734
                with dataset_output_context(use_dict_output=dataset.use_dict_output):
                    _ = trainer_test(
                        model,
                        dataloader,
                        physics,
                        online_measurements=True,
                        compare_no_learning=False,
                        metrics=DummyMetric(),
                    )

            except ValueError as e:
                # We may be checking paired unsup dataset, in which case training is ok to fail
                if (
                    "Dataloader must return ground truth `x` for online measurements."
                    not in str(e)
                ):
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
        elif dtype == "dict_of_pils":
            # This is a workaround for Python not having ability to check a variable is a `dict[str, PIL_Image]`.
            sample = dataset[0]
            images = [sample[k] for k in ("x", "y") if k in sample]
            assert images and all(isinstance(image, PIL_Image) for image in images)
        elif dtype == "dict_of_tensorlists":
            sample = dataset[0]
            tensorlists = [sample[k] for k in ("x", "y") if k in sample]
            assert tensorlists and all(isinstance(tl, TensorList) for tl in tensorlists)
        else:
            assert isinstance(
                dataset[0], dtype
            ), f"Dataset should return data of type {dtype} but got type {type(dataset[0])}."

    if shape is not None:
        batch = dataset[0]
        batch = batch_as_dict(batch)

        assert (
            batch["x"].shape == shape
        ), f"Dataset should return data of shape {shape} but got shape {batch['x'].shape}"


class MyDataset(ImageDataset):
    def __init__(self, batch, use_dict_output: bool = False):
        self.batch = batch
        super().__init__(use_dict_output=use_dict_output)

    def __len__(self):
        return 1

    def __getitem__(self, _):
        return self.batch


def test_base_dataset():
    x, y, params = Tensor([0]), Tensor([0]), {"a": Tensor([0])}
    bad = "hello"
    with dataset_output_context(use_dict_output=False):
        check_dataset(MyDataset(x))
        check_dataset(MyDataset([x, y]))
        check_dataset(MyDataset([torch.nan, y]))
        check_dataset(MyDataset([x, y, params]))
        check_dataset(MyDataset([torch.nan, y, params]))
        check_dataset(MyDataset([torch.nan, params]))

    # dict-shaped batches (use_dict_output=True)
    check_dataset(MyDataset({"x": x}, use_dict_output=True))
    check_dataset(MyDataset({"x": x, "y": y}, use_dict_output=True))
    check_dataset(MyDataset({"y": y}, use_dict_output=True))
    check_dataset(MyDataset({"x": x, "y": y, "params": params}, use_dict_output=True))
    check_dataset(MyDataset({"y": y, "params": params}, use_dict_output=True))

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
            check_dataset(MyDataset(bad_dataset_input, use_dict_output=True))

    for bad_dict_input in (
        {"params": params},  # neither x nor y
        {"x": bad},
        {"y": bad},
        {"x": x, "y": y, "params": {1: 2}},
    ):
        with pytest.raises(RuntimeError):
            check_dataset(MyDataset(bad_dict_input, use_dict_output=True))


SPLIT_NAMES = ["train", "test", "val", "dummy"]


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("split", [*SPLIT_NAMES, None])
@pytest.mark.parametrize("with_transform", [False, True])
@pytest.mark.parametrize("load_physics_generator_params", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("complex_dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("length", [10])
@pytest.mark.parametrize("with_params", [False, True])
@pytest.mark.parametrize("unsupervised", [False, True])
@pytest.mark.parametrize("close", [False, True])
@pytest.mark.parametrize("stack_size", [1, 2, 3])
@pytest.mark.parametrize("use_dict_output", [True, False])
def test_hdf5dataset(
    tmpdir,
    train,
    split,
    with_transform,
    load_physics_generator_params,
    dtype,
    complex_dtype,
    complex_data,
    length,
    with_params,
    unsupervised,
    close,
    stack_size,
    use_dict_output,
):
    path = f"{tmpdir}/dummy.h5"

    if with_transform:
        transform = MagicMock(side_effect=lambda x: x)
    else:
        transform = None

    # Populate a HDF5 file
    with h5py.File(path, "w") as f:
        # The stacked attribute is expected to be present only if stack_size > 1,
        # even though stacked = 1 could be meaningful.
        if stack_size > 1:
            f.attrs["stacked"] = stack_size

        def populate_dummy_data(field_name, *, value: int):
            data_dtype = complex_dtype if complex_data else dtype
            data = torch.full((length, 1, 4, 4), value, dtype=data_dtype)
            data = data.numpy()
            f.create_dataset(
                field_name,
                shape=data.shape,
                data=data,
                dtype=data.dtype,
            )

        # Every tensor has a constant value, which is distinct for different
        # splits and fields (x, y, and params indiscriminately). It allows
        # identification of each tensor to detect possible mismatches. The
        # value is defined as:
        # value = split_index * 3 + field_index
        # where field_index = 0 for x, 1 for y, and 2 for params.
        # Note that different ys in a stack share the same value and that
        # different params share the same value as well.
        for idx, split_name in enumerate(SPLIT_NAMES):
            if not unsupervised:
                populate_dummy_data(f"x_{split_name}", value=idx * 3 + 0)

            for stack_idx in range(stack_size):
                subfield_suffix = f"{stack_idx}" if stack_size > 1 else ""
                populate_dummy_data(
                    f"y{subfield_suffix}_{split_name}", value=idx * 3 + 1
                )

            if with_params:
                param_names = ["kernel"]
                # We test that y0 is loaded as a parameter if and only if the
                # measurements are not marked as stacked.
                if stack_size == 1:
                    param_names.append("y0")
                for param_name in param_names:
                    field_name = f"{param_name}_{split_name}"
                    populate_dummy_data(f"{param_name}_{split_name}", value=idx * 3 + 2)

    with dataset_output_context(use_dict_output):
        dataset = HDF5Dataset(
            path,
            train=train,
            split=split,
            transform=transform,
            load_physics_generator_params=load_physics_generator_params,
            dtype=dtype,
            complex_dtype=complex_dtype,
            use_dict_output=use_dict_output,
        )

    # Test HDF5Dataset.__len__
    assert (
        len(dataset) == length
    ), f"Dataset length should be {length} but got {len(dataset)}."

    # Test HDF5Dataset.__getitem__
    idx = 0
    entry = dataset[idx]
    entry = batch_as_dict(entry)

    x, y, params = entry.get("x", None), entry["y"], entry.get("params", {})

    if not unsupervised:
        assert "x" in entry, "Supervised dataset should return x."
    else:
        assert "x" not in entry, "Unsupervised dataset should not return x."

    assert "y" in entry, "Dataset should return y."

    if load_physics_generator_params and with_params:
        assert (
            "params" in entry
        ), "Dataset should return params when load_physics_generator_params is True and dataset contains params entries."
    else:
        assert (
            "params" not in entry
        ), "Dataset should not return params when load_physics_generator_params is False or dataset does not contain params entries."

    # Make the case disjunction at the start to simplify the logic
    split_name = split if split is not None else ("train" if train else "test")
    expected_value_x = SPLIT_NAMES.index(split_name) * 3 + 0
    expected_value_y = SPLIT_NAMES.index(split_name) * 3 + 1
    expected_value_params = SPLIT_NAMES.index(split_name) * 3 + 2

    data_dtype = complex_dtype if complex_data else dtype

    if x is not None:
        assert torch.allclose(
            x,
            torch.full((1, 4, 4), expected_value_x, dtype=data_dtype),
        ), f"Dataset x tensor has incorrect values."

    if stack_size == 1:
        expected_type_y = torch.Tensor
        ys = [y]
    else:
        expected_type_y = TensorList
        ys = y.x

    assert isinstance(y, expected_type_y), f"Dataset y has incorrect type."
    for y_el in ys:
        assert torch.allclose(
            y_el,
            torch.full((1, 4, 4), expected_value_y, dtype=data_dtype),
        ), f"Dataset y tensor has incorrect values."

    if with_params and load_physics_generator_params:
        assert "kernel" in params, "Params should contain kernel."

        if stack_size > 1:
            assert (
                "y0" not in params
            ), "Params should not contain y0 (stacked measurements)."
            expected_num_params = 1
        else:
            assert (
                "y0" in params
            ), "Params might contain y0 if the measurements are not stacked."
            expected_num_params = 2

        assert (
            len(params) == expected_num_params
        ), f"Params should contain {expected_num_params} tensors but got {len(params)}."

        assert torch.allclose(
            params["kernel"],
            torch.full((1, 4, 4), expected_value_params, dtype=data_dtype),
        ), f"Dataset params tensor has incorrect values."

    if transform is not None:
        assert transform.called == (
            not unsupervised
        ), "Transform should be called if and only if it is supervised."

    # Test HDF5Dataset.unsupervised
    # Verify that it is deprecated properly
    with pytest.warns(
        DeprecationWarning, match="The attribute 'unsupervised' is deprecated"
    ) as record:
        # Verify that it gives the right value
        assert (
            dataset.unsupervised == unsupervised
        ), "Dataset supervision label mismatch."

    # Test HDF5Dataset.close
    if close:
        dataset.close()
        # Reading should fail after closing
        with pytest.raises(ValueError):
            _ = dataset[idx]


@pytest.mark.parametrize("physgen", [None, "mask"])
@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize("supervised", [True, False])
@pytest.mark.parametrize("use_dict_output", [True, False])
def test_hdf5dataset_generate_dataset(
    tmpdir, physgen, stacked, supervised, use_dict_output
):
    img_size = (1, 4, 4)
    with dataset_output_context(use_dict_output):
        train_dataset = MyDataset(
            (
                {"x": torch.zeros(1, *img_size)}
                if use_dict_output
                else torch.zeros(1, *img_size)
            ),
            use_dict_output=use_dict_output,
        )
        test_dataset = MyDataset(
            (
                {"x": torch.zeros(1, *img_size)}
                if use_dict_output
                else torch.zeros(1, *img_size)
            ),
            use_dict_output=use_dict_output,
        )

    base_physics = Inpainting(img_size, mask=0.5)
    if stacked:
        physics = dinv.physics.stack(
            base_physics,
            Inpainting(img_size, mask=0.5),
        )
    else:
        physics = base_physics

    physics_generator = (
        None if physgen is None else BernoulliSplittingMaskGenerator(img_size, 0.5)
    )

    path = generate_dataset(
        train_dataset,
        physics,
        save_dir=str(tmpdir),
        batch_size=1,
        physics_generator=physics_generator,
        supervised=supervised,
        test_dataset=test_dataset,
    )

    with dataset_output_context(use_dict_output):
        train_ds = HDF5Dataset(
            path,
            split="train",
            load_physics_generator_params=True,
            use_dict_output=use_dict_output,
        )
    check_dataset_format(
        train_ds,
        length=1,
        dtype=None if stacked else (dict if use_dict_output else tuple),
        allow_non_tensor=False,
    )
    batch = train_ds[0]
    batch = batch_as_dict(batch)
    y_train, params_train = batch["y"], batch.get("params", {})

    if supervised:
        assert "x" in batch, "Supervised train split should have x."
    else:
        assert "x" not in batch, "Unsupervised train split should not have x."

    if stacked:
        assert isinstance(
            y_train, TensorList
        ), "Stacked physics should return TensorList."
        assert len(y_train) == 2, "Stacked measurements should have two elements."
    else:
        assert isinstance(
            y_train, torch.Tensor
        ), "Unstacked physics should return Tensor."

    if physgen is None:
        assert (
            len(params_train) == 0
        ), "Params should be empty when no generator is used."
    else:
        assert "mask" in params_train, "Params should contain mask when generator used."
        assert params_train["mask"].shape == img_size

    train_ds.close()
    assert train_ds.hd5 is None

    with dataset_output_context(use_dict_output):
        test_ds = HDF5Dataset(
            path,
            split="test",
            load_physics_generator_params=True,
            use_dict_output=use_dict_output,
        )

    # check_dataset_format runs a Trainer with `online_measurements=True` so ground-truth `x` is required
    if supervised:
        check_dataset_format(
            test_ds,
            length=1,
            dtype=None if stacked else (dict if use_dict_output else tuple),
            allow_non_tensor=False,
        )
    batch = test_ds[0]
    batch = batch_as_dict(batch)
    y_test, params_test = batch["y"], batch.get("params", {})

    assert "x" in batch, "Test split should have x."

    if stacked:
        assert isinstance(
            y_test, TensorList
        ), "Stacked physics should return TensorList."
        assert len(y_test) == 2, "Stacked measurements should have two elements."
    else:
        assert isinstance(
            y_test, torch.Tensor
        ), "Unstacked physics should return Tensor."

    if physgen is None:
        assert (
            len(params_test) == 0
        ), "Params should be empty when no generator is used."
    else:
        assert "mask" in params_test, "Params should contain mask when generator used."
        assert params_test["mask"].shape == img_size

    test_ds.close()
    assert test_ds.hd5 is None


@pytest.mark.parametrize("use_dict_output", [True, False])
def test_generate_dataset(tmp_path, use_dict_output):
    tmp_data_dir = str(tmp_path / "set14")
    with dataset_output_context(use_dict_output):
        # Dataset returns PIL images, no cropping so different sizes
        ds = Set14HR(tmp_data_dir, download=True, use_dict_output=use_dict_output)

        physics = dinv.physics.Denoising(
            noise_model=dinv.physics.GaussianNoise(sigma=0.1)
        )
        with pytest.raises(
            RuntimeError,
            match="generate_dataset expects dataset to return elements of same shape",
        ):
            _ = generate_dataset(
                train_dataset=ds,
                batch_size=4,
                physics=physics,
                device="cpu",
                save_dir="measurements",
            )
        # Test that no error is raised when we add crop
        ds = Set14HR(
            tmp_data_dir,
            transform=CenterCrop(32),
            use_dict_output=use_dict_output,
        )
        hdf_path = generate_dataset(
            train_dataset=ds,
            batch_size=1,
            physics=physics,
            device="cpu",
            save_dir="measurements",
            dataset_filename="generate_dataset_test",
        )
        from torchvision.transforms import ToTensor

        hdf_ds = HDF5Dataset(hdf_path, use_dict_output=use_dict_output)
        for sample_hdf, sample in zip(hdf_ds, ds, strict=True):
            sample = batch_as_dict(sample)
            sample = ToTensor()(sample["x"])
            sample_hdf = batch_as_dict(sample_hdf)["x"]
            assert sample_hdf.equal(
                sample
            ), "Ground-truth from HDF5 does not match original dataset, despite going through the same preprocessing."
        hdf_ds.hd5.close()
        shutil.rmtree(tmp_data_dir)
        os.remove(hdf_path)


@pytest.mark.parametrize("use_dict_output", [True, False])
def test_tensordataset(use_dict_output):
    x, y, params = (
        torch.zeros(1, 3, 4, 4),
        torch.zeros(1, 3, 4, 4),
        {"a": torch.zeros(1, 3, 4, 4)},
    )
    bad = np.zeros((1, 3, 4, 4))

    with dataset_output_context(use_dict_output=False):
        _ = TensorDataset(x=x)
        _ = TensorDataset(x=x, y=y)
        _ = TensorDataset(y=y)
        _ = TensorDataset(x=x, y=y, params=params)
        _ = TensorDataset(x=x, params=params)

    with dataset_output_context(use_dict_output):
        dataset = TensorDataset(y=y, params=params, use_dict_output=use_dict_output)

    if use_dict_output:
        assert set(dataset[0]) == {"y", "params"}
        assert dataset[0]["y"].shape == (3, 4, 4)
        assert dataset[0]["params"]["a"].shape == (3, 4, 4)
    else:
        assert isinstance(dataset[0], tuple) and len(dataset[0]) == 3
        assert math.isnan(
            dataset[0][0]
        ), "Dataset return tuple's first element must be NaN or single-element NaN tensor."
        assert dataset[0][1].shape == (3, 4, 4)
        assert dataset[0][2]["a"].shape == (3, 4, 4)

    for bad_dataset_input in (
        {},
        {"x": bad},
        {"y": bad},
        {"x": x, "y": torch.cat([y, y])},  # Batch size mismatch
    ):
        with pytest.raises(ValueError), dataset_output_context(use_dict_output):
            _ = TensorDataset(use_dict_output=use_dict_output, **bad_dataset_input)


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
def download_div2k(tmp_path):
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = str(tmp_path / "DIV2K")

    # Download div2K raw dataset
    with pytest.warns(DeprecationWarning, match="use_dict_output=True"):
        DIV2K(tmp_data_dir, mode="val", download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


@pytest.mark.parametrize("use_dict_output", [True, False])
def test_load_div2k_dataset(download_div2k, use_dict_output):
    """Check that DIV2K/DIV2K_train_HR contains 800 PIL images."""
    for totensor in [ToTensor(), None]:

        with dataset_output_context(use_dict_output):
            dtype = image_output_type(
                use_dict_output, totensor, paired=False, untransformed_type=PIL_Image
            )
            check_dataset_format(
                DIV2K(
                    download_div2k,
                    mode="val",
                    download=False,
                    transform=totensor,
                    use_dict_output=use_dict_output,
                ),
                length=100,
                dtype=dtype,
                allow_non_tensor=not totensor,
            )


@pytest.fixture
def download_urban100(tmp_path):
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = str(tmp_path / "Urban100")

    # Download Urban100 raw dataset
    with pytest.warns(DeprecationWarning, match="use_dict_output=True"):
        Urban100HR(tmp_data_dir, download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


@pytest.mark.parametrize("use_dict_output", [True, False])
def test_load_urban100_dataset(download_urban100, use_dict_output):
    """Check that dataset contains 100 PIL images."""
    for totensor in [ToTensor(), None]:
        with dataset_output_context(use_dict_output):
            dtype = image_output_type(
                use_dict_output, totensor, paired=False, untransformed_type=PIL_Image
            )
            check_dataset_format(
                Urban100HR(
                    download_urban100,
                    download=False,
                    transform=totensor,
                    use_dict_output=use_dict_output,
                ),
                length=100,
                dtype=dtype,
                allow_non_tensor=not totensor,
            )


@pytest.fixture
def download_set14(tmp_path):
    """Downloads dataset for tests and removes it after test executions."""
    if not os.environ.get("DEEPINV_MOCK_TESTS", False):
        tmp_data_dir = str(tmp_path / "Set14")

        # Download Set14 raw dataset
        with pytest.warns(DeprecationWarning, match="use_dict_output=True"):
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


@pytest.mark.parametrize("use_dict_output", [True, False])
def test_load_set14_dataset(download_set14, use_dict_output):
    """Check that dataset contains 14 PIL images."""
    for totensor in [ToTensor(), None]:
        with dataset_output_context(use_dict_output):
            dtype = image_output_type(
                use_dict_output, totensor, paired=False, untransformed_type=PIL_Image
            )
            check_dataset_format(
                Set14HR(
                    download_set14,
                    download=False,
                    transform=totensor,
                    use_dict_output=use_dict_output,
                ),
                length=14,
                dtype=dtype,
                allow_non_tensor=not totensor,
            )


@pytest.fixture
def download_flickr2khr(tmp_path):
    """Download or mock Flickr2kHR before testing"""
    if not os.environ.get("DEEPINV_MOCK_TESTS", False):
        tmp_data_dir = str(tmp_path / "Flickr2kHR")

        # Download Flickr raw dataset
        with pytest.warns(DeprecationWarning, match="use_dict_output=True"):
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


@pytest.mark.parametrize("use_dict_output", [True, False])
def test_load_Flickr2kHR_dataset(download_flickr2khr, use_dict_output):
    """Test the dataset"""
    for totensor in [ToTensor(), None]:
        with dataset_output_context(use_dict_output):
            dtype = image_output_type(
                use_dict_output, totensor, paired=False, untransformed_type=PIL_Image
            )
            check_dataset_format(
                Flickr2kHR(
                    download_flickr2khr,
                    download=False,
                    transform=totensor,
                    use_dict_output=use_dict_output,
                ),
                length=100,
                dtype=dtype,
                allow_non_tensor=not totensor,
            )


@pytest.fixture
def download_cbsd68(tmp_path, download=True):
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = str(tmp_path / "CBSD68")

    # Download CBSD raw dataset from huggingface
    try:
        with pytest.warns(DeprecationWarning, match="use_dict_output=True"):
            CBSD68(tmp_data_dir, download=download)
    except ImportError:
        download = False

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    if download:
        shutil.rmtree(tmp_data_dir)


@pytest.mark.parametrize("use_dict_output", [True, False])
def test_load_cbsd68_dataset(download_cbsd68, use_dict_output):
    """Check that dataset contains 68 PIL images."""

    pytest.importorskip(
        "datasets",
        reason="This test requires datasets. It should be "
        "installed with `pip install datasets`",
    )
    for totensor in [ToTensor(), None]:
        with dataset_output_context(use_dict_output):
            dtype = image_output_type(
                use_dict_output, totensor, paired=False, untransformed_type=PIL_Image
            )
            check_dataset_format(
                CBSD68(
                    download_cbsd68,
                    download=False,
                    transform=totensor,
                    use_dict_output=use_dict_output,
                ),
                length=68,
                dtype=dtype,
                allow_non_tensor=not totensor,
            )


@pytest.fixture
def download_bsds500(tmp_path, download=True):
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = str(tmp_path / "BSDS500")

    # Download BSDS500 raw dataset from github
    try:
        with pytest.warns(DeprecationWarning, match="use_dict_output=True"):
            BSDS500(tmp_data_dir, download=download)
    except ImportError:
        download = False

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    if download:
        shutil.rmtree(tmp_data_dir)


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("totensor", [True, False])
@pytest.mark.parametrize("rotate", [True, False])
@pytest.mark.parametrize("use_dict_output", [True, False])
def test_load_bsds500_dataset(
    download_bsds500, train, totensor, rotate, use_dict_output
):
    """Check that dataset contains 400 + 100 PIL images."""
    totensor = ToTensor() if totensor else None
    with dataset_output_context(use_dict_output):
        dtype = image_output_type(
            use_dict_output, totensor, paired=False, untransformed_type=PIL_Image
        )
        check_dataset_format(
            BSDS500(
                download_bsds500,
                download=False,
                transform=totensor,
                train=train,
                rotate=rotate,
                use_dict_output=use_dict_output,
            ),
            length=400 if train else 100,
            dtype=dtype,
            allow_non_tensor=not totensor,
        )


@pytest.fixture
def download_Kohler(tmp_path):
    """Download the Köhler dataset before a test and remove it after completion."""
    if not os.environ.get("DEEPINV_MOCK_TESTS", False):
        root = str(tmp_path / "Kohler")
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
@pytest.mark.parametrize("use_dict_output", [False])
def test_load_Kohler_dataset(download_Kohler, frames, ordering, use_dict_output):
    """Check that the Köhler dataset contains 48 PIL images."""
    root = download_Kohler

    for totensor in [ToTensor(), None]:
        with dataset_output_context(use_dict_output):
            dataset = Kohler(
                root=root,
                frames=frames,
                ordering=ordering,
                transform=totensor,
                download=False,
                use_dict_output=use_dict_output,
            )

        check_dataset_format(
            dataset,
            length=48,
            dtype=(
                dict if use_dict_output and totensor else tuple if totensor else None
            ),  # when no Transform, this is a tuple of list of PILs which is too complicated
            allow_non_tensor=not totensor,
            skip_check=True,
        )

    batch = dataset[0]
    batch = batch_as_dict(batch)
    x, y = batch["x"], batch["y"]
    data_points = [(x, y), dataset.get_item(1, 1, frames)]

    # totensor is None
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
def download_lsdir(tmp_path):
    """Downloads dataset for tests and removes it after test executions."""
    if not os.environ.get("DEEPINV_MOCK_TESTS", False):
        tmp_data_dir = str(tmp_path / "LSDIR")

        # Download LSDIR raw dataset
        with pytest.warns(DeprecationWarning, match="use_dict_output=True"):
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


@pytest.mark.parametrize("use_dict_output", [True, False])
def test_load_lsdir_dataset(download_lsdir, use_dict_output):
    """Check that dataset contains 250 PIL images."""
    for totensor in [ToTensor(), None]:
        with dataset_output_context(use_dict_output):
            dtype = image_output_type(
                use_dict_output, totensor, paired=False, untransformed_type=PIL_Image
            )
            check_dataset_format(
                LsdirHR(
                    download_lsdir,
                    mode="val",
                    transform=totensor,
                    download=False,
                    use_dict_output=use_dict_output,
                ),
                length=250,
                dtype=dtype,
                allow_non_tensor=not totensor,
            )


@pytest.fixture
def download_fmd(tmp_path):
    """Downloads dataset for tests and removes it after test executions."""
    if not os.environ.get("DEEPINV_MOCK_TESTS", False):
        tmp_data_dir = str(tmp_path / "FMD")

        # indicates which subsets we want to download
        types = ["TwoPhoton_BPAE_R"]

        # Download FMD raw dataset
        with pytest.warns(DeprecationWarning, match="use_dict_output=True"):
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


@pytest.mark.parametrize("use_dict_output", [True, False])
def test_load_fmd_dataset(download_fmd, use_dict_output):
    """Check that dataset contains 5000 noisy PIL images with its ground truths."""
    for totensor in [ToTensor(), None]:
        with dataset_output_context(use_dict_output):
            dtype = image_output_type(
                use_dict_output, totensor, paired=True, untransformed_type=PIL_Image
            )
            check_dataset_format(
                FMD(
                    download_fmd,
                    img_types=["TwoPhoton_BPAE_R"],
                    transform=totensor,
                    target_transform=totensor,
                    download=False,
                    use_dict_output=use_dict_output,
                ),
                length=5000,
                dtype=dtype,
                allow_non_tensor=not totensor,
            )


@pytest.fixture
def mock_lidc_idri():
    """Mock the LIDC-IDRI dataset"""
    if os.environ.get("DEEPINV_MOCK_TESTS", False):
        pytest.importorskip(
            "pandas",
            reason="This test requires pandas. It should be "
            "installed with `pip install pandas`",
        )
        pytest.importorskip(
            "pydicom",
            reason="This test requires pydicom. It should be "
            "installed with `pip install pydicom`",
        )
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
            patch("pydicom.dcmread", return_value=dummy_dicom),
        ):
            yield "/dummy"
    else:
        pytest.skip(
            "LIDC-IDRI dataset cannot be downloaded automatically and is not available for testing."
        )


# NOTE: The LIDC-IDRI needs to be downloaded manually.
@pytest.mark.parametrize("hounsfield_units", [False, True])
@pytest.mark.parametrize("use_dict_output", [True, False])
def test_load_lidc_idri_dataset(mock_lidc_idri, hounsfield_units, use_dict_output):
    """Test the LIDC-IDRI dataset."""

    for totensor in [ToTensor(), None]:
        with dataset_output_context(use_dict_output):
            check_dataset_format(
                LidcIdriSliceDataset(
                    root=mock_lidc_idri,
                    transform=totensor,
                    hounsfield_units=hounsfield_units,
                    use_dict_output=use_dict_output,
                ),
                length=2036,
                dtype=(dict if use_dict_output else Tensor if totensor else np.ndarray),
                allow_non_tensor=not totensor,
            )


@pytest.fixture
def download_nbu(tmp_path):
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = str(tmp_path / "NBU")

    # Download Urban100 raw dataset
    with pytest.warns(DeprecationWarning, match="use_dict_output=True"):
        NBUDataset(tmp_data_dir, satellite="gaofen-1", download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


@pytest.mark.parametrize("use_dict_output", [True, False])
def test_load_nbu_dataset(download_nbu, use_dict_output):
    """Check that dataset correct length and type."""
    pytest.importorskip(
        "scipy",
        reason="This test requires scipy. It should be "
        "installed with `pip install "
        "scipy`",
    )

    with dataset_output_context(use_dict_output):
        dataset = NBUDataset(
            download_nbu,
            satellite="gaofen-1",
            download=False,
            use_dict_output=use_dict_output,
        )

        check_dataset_format(
            dataset,
            length=5,
            dtype=dict if use_dict_output else Tensor,
            shape=(4, 256, 256),
        )
        batch = dataset[0]
        batch = batch_as_dict(batch)
        assert torch.all(
            (0 <= batch["x"]) & (batch["x"] <= 1)
        ), "Dataset image should be Tensor between 0-1."

        # Check pan band
        check_dataset_format(
            NBUDataset(
                download_nbu,
                satellite="gaofen-1",
                download=False,
                return_pan=True,
                use_dict_output=use_dict_output,
            ),
            length=5,
            dtype="dict_of_tensorlists" if use_dict_output else TensorList,
            shape=[(4, 256, 256), (1, 1024, 1024)],
        )

        # Test ImageFolder with globs
        dataset = ImageFolder(
            download_nbu,
            x_path="gaofen-1/MS_256/*.mat",
            transform=ToTensor(),
            loader=lambda f: load_mat(f)["imgMS"],
            use_dict_output=use_dict_output,
        )
        check_dataset_format(
            dataset,
            length=5,
            dtype=dict if use_dict_output else Tensor,
            shape=(4, 256, 256),
        )

        dataset = ImageFolder(
            download_nbu,
            y_path="gaofen-1/MS_256/*.mat",
            transform=ToTensor(),
            loader=lambda f: load_mat(f)["imgMS"],
            use_dict_output=use_dict_output,
        )
        check_dataset_format(
            dataset,
            length=5,
            dtype=dict if use_dict_output else tuple,
            allow_non_tensor=True,
        )

    batch = dataset[0]
    batch = batch_as_dict(batch)
    y = batch["y"]

    assert "x" not in batch
    assert y.shape == (4, 256, 256)
    assert "params" not in batch, "Params should be empty when no generator is used."


@pytest.fixture
def download_simplefastmri(tmp_path):
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = str(tmp_path / "fastmri")

    # Download simple FastMRI slice dataset
    with pytest.warns(DeprecationWarning, match="use_dict_output=True"):
        SimpleFastMRISliceDataset(tmp_data_dir, download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


@pytest.mark.parametrize("use_dict_output", [True, False])
def test_SimpleFastMRISliceDataset(download_simplefastmri, use_dict_output):
    with dataset_output_context(use_dict_output):
        dataset = SimpleFastMRISliceDataset(
            root_dir=download_simplefastmri,
            anatomy="knee",
            train=True,
            train_percent=1.0,
            download=False,
            use_dict_output=use_dict_output,
        )
    check_dataset_format(
        dataset,
        length=2,
        dtype=dict if use_dict_output else Tensor,
        shape=(2, 320, 320),
    )
    batch0, batch1 = dataset[0], dataset[1]
    batch0, batch1 = batch_as_dict(batch0), batch_as_dict(batch1)
    assert not torch.all(batch0["x"] == batch1["x"])


@pytest.fixture
def download_fastmri(tmp_path):
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = str(tmp_path / "fastmri")
    file_name = "demo_fastmri_brain_multicoil.h5"

    # Download single FastMRI volume
    os.makedirs(tmp_data_dir, exist_ok=True)
    url = get_image_url(file_name)
    download_archive(url, f"{tmp_data_dir}/{file_name}")

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


@pytest.mark.parametrize("use_dict_output", [True, False])
def test_FastMRISliceDataset(download_fastmri, use_dict_output):
    pytest.importorskip(
        "sigpy",
        reason="This test requires sigpy. It should be "
        "installed with `pip install "
        "sigpy`",
    )
    # Raw data shape
    kspace_shape = (512, 213)
    n_coils = 4
    n_slices = 16
    img_size = (213, 213)

    # Clean data shape
    rss_shape = (320, 320)

    data_dir = download_fastmri

    # Test metadata caching
    with dataset_output_context(use_dict_output):
        _ = FastMRISliceDataset(
            root=data_dir,
            slice_index="all",
            save_metadata_to_cache=True,
            metadata_cache_file="fastmrislicedataset_cache.pkl",
            use_dict_output=use_dict_output,
        )

        # Test data shapes
        dataset = FastMRISliceDataset(
            root=data_dir,
            slice_index="all",
            load_metadata_from_cache=True,
            metadata_cache_file="fastmrislicedataset_cache.pkl",
            use_dict_output=use_dict_output,
        )
    check_dataset_format(
        dataset, length=n_slices, dtype=dict if use_dict_output else tuple, shape=None
    )

    batch1 = dataset[0]
    batch1 = batch_as_dict(batch1)
    target1, kspace1 = batch1["x"], batch1["y"]

    batch2 = dataset[1]
    batch2 = batch_as_dict(batch2)
    target2, kspace2 = batch2["x"], batch2["y"]

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
    rss1 = physics.A_adjoint(kspace1.unsqueeze(0), rss=True)
    rss1 = physics.crop(rss1, shape=target1.shape)
    assert torch.allclose(target1.unsqueeze(0), rss1)

    # Test singlecoil MRI mag works
    physics = MRI(mask=torch.ones(kspace_shape), img_size=img_size)
    mag1 = physics.A_adjoint(kspace1.unsqueeze(0)[:, :, 0], mag=True)
    mag1 = physics.crop(mag1, shape=target1.shape)
    assert target1.unsqueeze(0).shape == mag1.shape

    # Test save simple dataset
    with dataset_output_context(use_dict_output):
        subset = dataset.save_simple_dataset(f"{download_fastmri}/temp_simple.pt")
    check_dataset_format(
        subset,
        length=n_slices,
        dtype=dict if use_dict_output else Tensor,
        shape=(2, *rss_shape),
    )

    # Test slicing returns correct num of slices
    def num_slices(slice_index):
        with dataset_output_context(use_dict_output):
            return len(
                FastMRISliceDataset(
                    root=data_dir,
                    slice_index=slice_index,
                    load_metadata_from_cache=True,
                    metadata_cache_file="fastmrislicedataset_cache.pkl",
                    use_dict_output=use_dict_output,
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
    with dataset_output_context(use_dict_output):
        dataset = FastMRISliceDataset(
            root=data_dir,
            transform=MRISliceTransform(
                mask_generator=GaussianMaskGenerator(kspace_shape, acc=4),
                estimate_coil_maps=True,
            ),
            load_metadata_from_cache=True,
            metadata_cache_file="fastmrislicedataset_cache.pkl",
            use_dict_output=use_dict_output,
        )

    batch = dataset[0]
    batch = batch_as_dict(batch)
    y, params = batch["y"], batch["params"]

    assert torch.all(y * params["mask"] == y)
    assert 0.24 < params["mask"].mean() < 0.26
    assert params["coil_maps"].shape == (n_coils, *kspace_shape)
    assert dataset.transform.get_acs() == 17  # ACS via mask generator

    physics_estim = MultiCoilMRI(**params)
    x0 = torch.randn(1, 2, *kspace_shape)
    assert physics_estim.adjointness_test(x0) < 1e-3
    assert (
        torch.abs(
            physics.compute_sqnorm(x0, max_iter=1000, tol=1e-6, verbose=False) - 1.0
        )
        < 1e-3
    )

    # Test prewhitening and normalising
    with dataset_output_context(use_dict_output):
        dataset = FastMRISliceDataset(
            root=data_dir,
            transform=MRISliceTransform(
                acs=11,  # set manually as fully-sampled data has no ACS metadata
                prewhiten=True,
                normalize=True,
            ),
            load_metadata_from_cache=True,
            metadata_cache_file="fastmrislicedataset_cache.pkl",
            use_dict_output=use_dict_output,
        )

        assert dataset.transform.get_acs() == 11
        if use_dict_output:
            assert 1 < dataset[0]["y"].max() < 100  # normalized
        else:
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
                    use_dict_output=use_dict_output,
                )
            )
            == 3
        )


@pytest.fixture
def download_CMRxRecon(tmp_path):
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = str(tmp_path / "CMRxRecon")

    # Download single CMRxRecon volume
    os.makedirs(tmp_data_dir, exist_ok=True)
    download_archive(
        get_image_url("CMRxRecon.zip"), f"{tmp_data_dir}/CMRxRecon.zip", extract=True
    )

    # This will return control to the test function
    yield f"{tmp_data_dir}/CMRxRecon"

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


@pytest.mark.parametrize("use_dict_output", [True, False])
def test_CMRxReconSliceDataset(download_CMRxRecon, use_dict_output):
    pytest.importorskip(
        "sigpy",
        reason="This test requires sigpy. It should be "
        "installed with `pip install sigpy`",
    )

    img_size = (12, 512, 256)

    physics_generator = GaussianMaskGenerator(img_size)

    data_dir = download_CMRxRecon

    def make_dataset(**kwargs):
        with dataset_output_context(use_dict_output):
            return CMRxReconSliceDataset(
                root=data_dir, use_dict_output=use_dict_output, **kwargs
            )

    # Test metadata caching
    _ = make_dataset(
        save_metadata_to_cache=True,
        metadata_cache_file="cmrxreconslicedataset_cache.pkl",
        mask_dir=None,
        apply_mask=False,
    )

    # Test data shapes
    dataset = make_dataset(
        load_metadata_from_cache=True,
        metadata_cache_file="cmrxreconslicedataset_cache.pkl",
        mask_generator=physics_generator,
        mask_dir=None,
        apply_mask=True,
    )

    check_dataset_format(dataset, length=3, dtype=dict if use_dict_output else tuple)

    batch1 = dataset[0]
    batch1 = batch_as_dict(batch1)
    batch2 = dataset[1]
    batch2 = batch_as_dict(batch2)

    target1, kspace1, params1 = batch1["x"], batch1["y"], batch1["params"]
    target2, kspace2, params2 = batch2["x"], batch2["y"], batch2["params"]

    assert target1.shape == kspace1.shape == (2, *img_size)
    assert not torch.all(target1 == target2)
    assert not torch.all(kspace1 == kspace2)
    assert not torch.all(params1["mask"] == params2["mask"])
    assert torch.all(kspace1 * params1["mask"] == kspace1)  # kspace already masked
    assert (
        0.1 < params1["mask"].mean() < 0.26
    )  # masked has correct acc (< 0.25 due to padding)

    # Test reproducibility
    batch1_again = dataset[0]
    batch1_again = batch_as_dict(batch1_again)
    params1_again = batch1_again["params"]
    assert torch.all(params1_again["mask"] == params1["mask"])

    # Loaded kspace is directly compatible with deepinv physics
    physics = DynamicMRI(img_size=img_size)
    kspace1_dinv = physics(
        target1.unsqueeze(0), mask=params1["mask"].unsqueeze(0)
    ).squeeze(0)
    assert torch.all(kspace1 == kspace1_dinv)

    # Test loading mask
    dataset = make_dataset(
        load_metadata_from_cache=True,
        metadata_cache_file="cmrxreconslicedataset_cache.pkl",
        apply_mask=True,
    )
    batch1 = dataset[0]
    batch1 = batch_as_dict(batch1)
    kspace1, params1 = batch1["y"], batch1["params"]
    assert torch.all(kspace1 * params1["mask"] == kspace1)  # kspace already masked
    assert (
        0.1 < params1["mask"].mean() < 0.26
    )  # masked has correct acc (< 0.25 due to padding)

    # Test no apply mask
    dataset = make_dataset(
        load_metadata_from_cache=True,
        metadata_cache_file="cmrxreconslicedataset_cache.pkl",
        apply_mask=False,
    )
    batch1 = dataset[0]
    batch1 = batch_as_dict(batch1)
    kspace1 = batch1["y"]
    assert (kspace1 == 0).sum() == 0


@pytest.fixture
def download_SKMTEA(tmp_path):
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = str(tmp_path / "SKMTEA")
    file_name = "SKMTEA_tiny_2_slice.h5"

    # Download tiny SKMTEA volume
    os.makedirs(tmp_data_dir, exist_ok=True)
    url = get_image_url(file_name)
    download_archive(url, f"{tmp_data_dir}/{file_name}")

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


@pytest.mark.parametrize("use_dict_output", [True, False])
def test_SKMTEASliceDataset(download_SKMTEA, device, use_dict_output):
    n_coils, img_size = 8, (512, 160)

    data_dir = download_SKMTEA

    def make_dataset(**kwargs):
        with dataset_output_context(use_dict_output):
            return SKMTEASliceDataset(
                root=data_dir, use_dict_output=use_dict_output, **kwargs
            )

    # Test metadata caching
    dataset = make_dataset(
        save_metadata_to_cache=True,
    )
    assert len(dataset) == 2

    # Test data shapes and dtypes
    dataset = make_dataset(
        load_metadata_from_cache=True,
    )
    assert len(dataset) == 2

    batch = next(iter(DataLoader(dataset)))
    batch = batch_as_dict(batch)
    x, y, params = batch["x"], batch["y"], batch["params"]
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
            make_dataset(
                load_metadata_from_cache=True,
                filter_id=lambda s: s.slice_ind == 1,
            )
        )
        == 1
    )


@pytest.fixture
def make_data(tmp_path, request):
    """Minimal synthetic datasets for 3D (.npy/.b2nd/.nii.gz), 2D+channels (C,H,W & H,W,C), and >3D no-channels (H,W,D,T)."""
    pytest.importorskip(
        "nibabel",
        reason="This test requires nibabel. It should be "
        "installed with `pip install nibabel`",
    )
    pytest.importorskip(
        "blosc2",
        reason="This test requires blosc2. It should be "
        "installed with `pip install blosc2`",
    )
    import nibabel as nib
    import blosc2

    root = tmp_path
    cases = []

    # 3D volumes
    shape3d = (40, 40, 40)
    fmt = getattr(request, "param")
    dx = root / f"{fmt.strip('.').replace('.', '_')}_x"
    dy = root / f"{fmt.strip('.').replace('.', '_')}_y"
    dx.mkdir()
    dy.mkdir()
    for i in range(2):
        vol = np.random.normal(size=shape3d)
        if fmt == ".npy":
            np.save(dx / f"{i}.npy", vol)
            np.save(dy / f"{i}.npy", vol)
        elif fmt == ".b2nd":
            blosc2.asarray(np.ascontiguousarray(vol), urlpath=str(dx / f"{i}.b2nd"))
            blosc2.asarray(np.ascontiguousarray(vol), urlpath=str(dy / f"{i}.b2nd"))
        elif fmt == ".nii.gz":
            nib.save(nib.Nifti1Image(vol, np.eye(4)), str(dx / f"{i}.nii.gz"))
            nib.save(nib.Nifti1Image(vol, np.eye(4)), str(dy / f"{i}.nii.gz"))
        elif fmt == ".pt":
            torch.save(torch.from_numpy(vol), str(dx / f"{i}.pt"))
            torch.save(torch.from_numpy(vol), str(dy / f"{i}.pt"))
        else:  # pragma: no cover
            raise ValueError(f"Unsupported fmt: {fmt}")
    if fmt == ".pt":

        def torch_loader(path, **args):
            return torch.load(path)

    cases.append(
        dict(
            name=f"3d-{fmt}",
            x=str(dx),
            y=str(dy),
            fmt=fmt,
            ch_axis=None,
            patch=(32, 32, 32),
            expected=(1, 32, 32, 32),
            loader=torch_loader if fmt == ".pt" else None,
        )
    )

    if fmt == ".npy":
        # 2D with channels
        C = 3
        # (C,H,W)
        d0x = root / "npy_2d_ch0_x"
        d0y = root / "npy_2d_ch0_y"
        d0x.mkdir()
        d0y.mkdir()
        # (H,W,C)
        dmx = root / "npy_2d_chm1_x"
        dmy = root / "npy_2d_chm1_y"
        dmx.mkdir()
        dmy.mkdir()
        for i in range(2):
            np.save(d0x / f"{i}.npy", np.random.normal(size=(C, 48, 48)))
            np.save(d0y / f"{i}.npy", np.random.normal(size=(C, 48, 48)))
            np.save(dmx / f"{i}.npy", np.random.normal(size=(48, 48, C)))
            np.save(dmy / f"{i}.npy", np.random.normal(size=(48, 48, C)))

        cases += [
            dict(
                name="2d-ch0",
                x=str(d0x),
                y=str(d0y),
                fmt=".npy",
                ch_axis=0,
                patch=(16, 16),
                expected=(3, 16, 16),
            ),
            dict(
                name="2d-chm1",
                x=str(dmx),
                y=str(dmy),
                fmt=".npy",
                ch_axis=-1,
                patch=(16, 16),
                expected=(3, 16, 16),
            ),
        ]

        # (H, D, W, T) (or general 4D case)
        d4x = root / "npy_4d_noch_x"
        d4y = root / "npy_4d_noch_y"
        d4x.mkdir()
        d4y.mkdir()
        for i in range(2):
            np.save(d4x / f"{i}.npy", np.random.normal(size=(36, 36, 20, 4)))
            np.save(d4y / f"{i}.npy", np.random.normal(size=(36, 36, 20, 4)))
        cases.append(
            dict(
                name="4d-noch",
                x=str(d4x),
                y=str(d4y),
                fmt=".npy",
                ch_axis=None,
                patch=(12, 12, 10, 2),
                expected=(1, 12, 12, 10, 2),
            )
        )
    yield cases


@pytest.mark.parametrize(
    "make_data", [".npy", ".b2nd", ".nii.gz", ".pt"], indirect=True
)
@pytest.mark.parametrize("use_dict_output", [True, False])
def test_RandomPatchSampler(make_data, use_dict_output):
    # (i) formats on 3D, (ii) 2D&channels, (iii) 4D no-channels
    for c in make_data:
        # x-only
        with dataset_output_context(use_dict_output):
            ds = RandomPatchSampler(
                x_dir=c["x"],
                patch_size=c["patch"],
                file_format=c["fmt"],
                ch_axis=c["ch_axis"],
                loader=c.get("loader", None),
                use_dict_output=use_dict_output,
            )
        assert len(ds) == 2

        batch = next(iter(ds))
        batch = batch_as_dict(batch)
        assert (
            batch["x"].shape == (1,) + tuple(c["patch"])
            if c["ch_axis"] is None
            else (c["expected"])
        )
        with dataset_output_context(use_dict_output):
            ds = RandomPatchSampler(
                x_dir=c["x"],
                y_dir=c["y"],
                patch_size=c["patch"],
                file_format=c["fmt"],
                ch_axis=c["ch_axis"],
                loader=c.get("loader", None),
                use_dict_output=use_dict_output,
            )
        batch = next(iter(ds))
        batch = batch_as_dict(batch)
        x, y = batch["x"], batch["y"]
        assert x.shape == c["expected"]
        assert y.shape == c["expected"]
        assert "params" not in batch

    # check if x is nan behaviour happens
    c0 = make_data[0]
    with dataset_output_context(use_dict_output):
        ds = RandomPatchSampler(
            y_dir=c0["y"],
            patch_size=c0["patch"],
            file_format=c0["fmt"],
            ch_axis=c0["ch_axis"],
            loader=c0.get("loader", None),
            use_dict_output=use_dict_output,
        )
    assert len(ds) == 2

    batch = next(iter(ds))
    batch = batch_as_dict(batch)
    assert "x" not in batch
    assert "y" in batch
    assert "params" not in batch


@pytest.mark.parametrize("lesion_diameters", [None, [15, 7]])
@pytest.mark.parametrize("use_dict_output", [True, False])
def test_brainweb_pet(tmp_path, lesion_diameters, use_dict_output):
    brainweb = pytest.importorskip("brainweb")

    class RandomFDG(brainweb.FDG):
        greyMatter = lambda: 120.0

    with dataset_output_context(use_dict_output):
        dataset = BrainWebPET(
            root=tmp_path,
            subject_ids=4,
            pet_class=RandomFDG,
            contrast=["T1", "T2"],
            random_degradations_kwargs={
                "petNoise": 0.0,
                "t1Noise": 0.0,
                "t2Noise": 0.0,
                "petSigma": 0.0,
                "t1Sigma": 0.0,
                "t2Sigma": 0.0,
            },
            lesion_diameters=lesion_diameters,
            lesion_kwargs={"intensity": [1000, 2000], "blur": [0, 0], "thresh": 30},
            seed=0,
            use_dict_output=use_dict_output,
        )

    batch = dataset[0]
    batch = batch_as_dict(batch)
    emission, params = batch["x"], batch["params"]

    assert len(dataset) == 1
    assert emission.shape == params["attenuation"].shape == params["t1"].shape
    assert emission.shape == params["t2"].shape
    assert emission.dtype == torch.float32
    pet_class = dataset.brainweb_kwargs["PetClass"]
    assert issubclass(pet_class, RandomFDG)
    assert pet_class.greyMatter == 120.0
    assert ("lesion_mask" in params) == bool(lesion_diameters)
    if lesion_diameters:
        assert torch.unique(params["lesion_mask"]).tolist() == [0, 1, 2]


@pytest.mark.parametrize("use_dict_output", [True, False])
def test_brainweb_mri(tmp_path, use_dict_output):
    pytest.importorskip("brainweb_dl")
    default_dataset = BrainWebMRI(root=tmp_path, use_dict_output=True)
    assert default_dataset.subject_ids == [4, 5, 6, 18, 20, 38, *range(41, 55)]

    with dataset_output_context(use_dict_output):
        dataset = BrainWebMRI(
            root=tmp_path,
            subject_ids=4,
            transform=lambda x: x / x.max(),
            use_dict_output=use_dict_output,
        )
        batch = dataset[0]
        batch = batch_as_dict(batch)
        volume = batch["x"]

        assert len(dataset) == 1
        assert volume.shape == (1, 181, 256, 256)
        assert volume.dtype == torch.float32
        assert volume.min() == 0
        assert volume.max() == 1

        cached_dataset = BrainWebMRI(
            root=tmp_path,
            subject_ids=4,
            download=False,
            use_dict_output=use_dict_output,
        )
        batch = cached_dataset[0]
        batch = batch_as_dict(batch)
        assert batch["x"].shape == (1, 181, 256, 256)

        for subject_id, contrast, filename in [
            (4, "T1", "subject04_t1w.nii.gz"),
            (4, "T2", "brainweb_s04_fuzzy.nii.gz"),
        ]:
            with pytest.raises(FileNotFoundError, match=filename):
                BrainWebMRI(
                    root=tmp_path / "missing",
                    subject_ids=subject_id,
                    contrast=contrast,
                    download=False,
                    use_dict_output=use_dict_output,
                )[0]


@pytest.mark.parametrize("kind", ["zipfile", "tarball", "rarfile"])
def test_extract_archive(tmp_path, kind):
    mocker = MagicMock()
    mocker.__enter__.return_value = mocker
    getattr(mocker, "getmembers" if kind == "tarball" else "infolist").return_value = [
        "a.txt",
        "b.txt",
    ]

    if kind == "zipfile":
        with patch(
            "deepinv.datasets.utils.zipfile.ZipFile", return_value=mocker
        ) as cls:
            extract_zipfile("archive.zip", tmp_path)
        cls.assert_called_once_with("archive.zip", "r")

    elif kind == "tarball":
        with patch("deepinv.datasets.utils.tarfile.open", return_value=mocker) as fn:
            extract_tarball("archive.tar", tmp_path)
        fn.assert_called_once_with("archive.tar", "r:*")

    elif kind == "rarfile":
        mock_module = MagicMock()
        mock_module.RarFile.return_value = mocker
        with patch.dict(sys.modules, {"rarfile": mock_module}):
            extract_rarfile("archive.rar", tmp_path)
        mock_module.RarFile.assert_called_once_with("archive.rar")

    assert mocker.extract.call_count == 2
