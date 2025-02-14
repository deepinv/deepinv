from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
)
import os
import torch
import numpy as np

error_import = None
try:
    import pandas as pd
except:
    error_import = ImportError(
        "pandas is not available. Please install the pandas package with `pip install pandas`."
    )
try:
    from pydicom import dcmread
except:
    error_import = ImportError(
        "dicom is not available. Please install the dicom package with `pip install dicom`."
    )


class LidcIdriSliceDataset(torch.utils.data.Dataset):
    """Dataset for `LIDC-IDRI <https://www.cancerimagingarchive.net/collection/lidc-idri/>`_ that provides access to CT image slices.

    | The Lung Image Database Consortium image collection (LIDC-IDRI) consists
    | of diagnostic and lung cancer screening thoracic computed tomography (CT)
    | scans with marked-up annotated lesions.

    .. warning::
        To download the raw dataset, you will need to install the `NBIA Data Retriever <https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images>`_,
        then download the manifest file (.tcia file)`here <https://www.cancerimagingarchive.net/collection/lidc-idri/>`_, and open it by double clicking.


    **Raw data file structure:** ::

        self.root --- LIDC-IDRI --- LICENCE
                   |             -- LIDC-IDRI-0001 --- `STUDY_UID` --- `SERIES_UID` --- xxx.xml
                   |             |                                                   -- 1-001.dcm
                   |             -- LIDC-IDRI-1010                                   |
                   |                                                                 -- 1-xxx.dcm
                   -- metadata.csv

    | 0) There are 1010 patients and a total of 1018 CT scans.
    | 1) Each CT scan is composed of 2d slices.
    | 2) Each slice is stored as a .dcm file
    | 3) This class gives access to one slice of a CT scan per data sample.
    | 4) Each slice is represented as an (512, 512) array.

    :param str root: Root directory of dataset. Directory path from where we load and save the dataset.
    :param Callable transform:: (optional)  A function/transform that takes in a data sample and returns a transformed version.

    |sep|

    :Examples:

        Instantiate dataset ::

            import torch
            from deepinv.datasets import LidcIdriSliceDataset
            root = "/path/to/dataset/LIDC-IDRI"
            dataset = LidcIdriSliceDataset(root=root)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
            batch = next(iter(dataloader))
            print(batch.shape)

    """

    class SliceSampleIdentifier(NamedTuple):
        """Data structure for identifying slices.

        In LIDC-IDRI, there are 1010 patients.
        Among them, 8 patients have each 2 CT scans.

        :param str slice_fname: Filename of a dicom file containing 1 slice of the scan.
        :param str scan_folder: Path to all dicom files from the same scan.
        :param str patient_id: Foldername of one patient among the 1010.
        """

        slice_fname: str
        scan_folder: str
        patient_id: str

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if error_import is not None and isinstance(error_import, ImportError):
            raise error_import

        self.root = root
        self.transform = transform

        ### LOAD CSV to find CT scan folder paths --------------------------------------

        csv_path = os.path.join(root, "metadata.csv")
        # check that root is a folder
        if not os.path.isdir(root):
            raise ValueError(
                f"The `root` folder doesn't exist. Please set `root` properly. Current value `{root}`."
            )
        # check that root folder contains "metadata.csv"
        if not os.path.exists(csv_path):
            raise ValueError(
                f"{csv_path} doesn't exist. Please set `root` properly. Current value `{root}`."
            )

        df = pd.read_csv(csv_path, index_col=False)
        # Get only CT scans
        filtered_df = df[df["Modality"] == "CT"]
        # Sort by Patient ID
        sorted_filtered_df = filtered_df.sort_values(by="Subject ID", ascending=True)

        ### LOAD SLICE SAMPLE INFO -----------------------------------------------------

        self.sample_identifiers = []
        n_scans = len(sorted_filtered_df)
        for i in range(n_scans):
            patient_id = sorted_filtered_df.iloc[i]["Subject ID"]
            scan_folder_path = sorted_filtered_df.iloc[i]["Download Timestamp"]

            # replace WINDOWS path separator into the curent system path separator
            scan_folder_path = scan_folder_path.replace("\\", os.sep)
            # replace POSIX path separator into the current system path separator
            scan_folder_path = scan_folder_path.replace("/", os.sep)
            # Normalize path : https://www.geeksforgeeks.org/python-os-path-normpath-method/
            scan_folder_path = os.path.normpath(scan_folder_path)
            # relative path -> absolute path
            scan_folder_fullpath = os.path.join(root, scan_folder_path)

            slice_list = os.listdir(scan_folder_fullpath)
            slice_list.sort()
            for fname in slice_list:
                if fname.endswith(".dcm"):
                    self.sample_identifiers.append(
                        self.SliceSampleIdentifier(
                            fname, scan_folder_fullpath, patient_id
                        )
                    )

    def __len__(self) -> int:
        return len(self.sample_identifiers)

    def __getitem__(self, idx: int) -> Any:
        slice_fname, scan_folder_path, _ = self.sample_identifiers[idx]
        slice_path = os.path.join(scan_folder_path, slice_fname)

        # type: numpy.ndarray
        # dtype: int16
        # shape: (512, 512)
        # TODO : check impact of the conversion to np.int16
        slice_array = dcmread(slice_path).pixel_array.astype(np.int16)

        if self.transform is not None:
            slice_array = self.transform(slice_array)

        return slice_array
