from typing import Any, Callable, Optional, Union, List, Dict, Tuple
from pathlib import Path
import os
from natsort import natsorted
from tqdm import tqdm

from numpy import ndarray
from torch import Tensor

from deepinv.datasets.fastmri import FastMRISliceDataset
from deepinv.datasets.utils import loadmat
from deepinv.physics.mri import MRIMixin
from deepinv.physics.generator.mri import BaseMaskGenerator

class CMRxReconSliceDataset(FastMRISliceDataset, MRIMixin):

    def __init__(
        self,
        root: Union[str, Path],
        data_folders: List[str] = ['SingleCoil/Cine/FullSample'],
        load_metadata_from_cache: bool = False,
        save_metadata_to_cache: bool = False,
        metadata_cache_file: Union[str, Path] = "dataset_cache.pkl",
        transform: Optional[Callable] = None,
        mask_generator: Optional[BaseMaskGenerator] = None,
    ):
        self.root = Path(root)
        self.data_dirs = natsorted([self.root / d for d in data_folders])
        self.transform = self.norm_pad_transform if transform is None else transform
        self.mask_generator = mask_generator

        if not all([os.path.isdir(d) for d in self.data_dirs]):
            raise ValueError(
                f"One or more data folder does not exist. Please set `root` and `data_folders` properly. Current values {self.data_dirs}."
            )
                
        with self.metadata_cache_manager(self.root, [], metadata_cache_file, load_metadata_from_cache, save_metadata_to_cache) as sample_identifiers:
            for fname in tqdm(natsorted([d.rglob("**/*.mat") for d in self.data_dirs])):
                if fname.endswith("_mask.mat"):
                    # TODO add this into regex above
                    continue
                metadata = self._retrieve_metadata(fname)
                for slice_ind in range(metadata["num_slices"]):
                    self.sample_identifiers.append(self.SliceSampleFileIdentifier(fname, slice_ind, metadata))
            
            self.sample_identifiers = sample_identifiers

    @staticmethod
    def _loadmat(fname: Union[str, Path, os.PathLike]) -> ndarray:
        return next(v for k, v in loadmat(fname).items() if not k.startswith('__'))

    @staticmethod
    def _retrieve_metadata(fname: Union[str, Path, os.PathLike]) -> Dict[str, Any]:
        """Open file and retrieve metadata
        
        Metadata includes width, height, slices, coils (if multicoil) and timeframes.

        :param Union[str, Path, os.PathLike] fname: filename to open
        :return: metadata dict of key-value pairs.
        """
        shape = CMRxReconSliceDataset._loadmat(fname).shape #WH(N)DT
        return {
            "width": shape[0], #W
            "height": shape[1], #H
            "slices": shape[-2], #D (depth)
            "timeframes": shape[-1], #T
        } | {
            "coils": shape[2], #N (coils)
        } if len(shape) == 5 else {}

    def __getitem__(self, i: int) -> Tuple[Tensor]:
        fname, slice_index, metadata = self.sample_identifiers[i]
        kspace = self._loadmat(fname) # shape WH(N)DT
        kspace = kspace[..., slice_index, :] # shape WH(N)T

        if len(kspace.shape) == 5:
            kspace = kspace[:, :, 0] # shape WHT

        if self.mask_generator is None:
            mask = self._loadmat(fname)
        else:
            mask = self.mask_generator.step(
                seed=seed(fname + str(slice_index)),
                img_size=kspace.shape[:2]
            )

        return self.transform(kspace, mask)

    def to_tensor(self, data: ndarray) -> Tensor:
        return torch.from_numpy(np.stack((data.real, data.imag), axis=-1))

    def norm_pad_transform(self, kspace: ndarray, mask: ndarray) -> Tuple[Tensor]:
        target = self.to_tensor(kspace) # shape WHTC        
        input_kspace = self.to_tensor(kspace) # shape WHTC
        mask = self.to_tensor(mask if len(mask.shape) > 2 else np.expand_dims(mask, axis=-1)) # shape WH(1 or T)
        # mask shape [w, h, 1 or t]

        #! 2. Convert to image
        target_image = ifftnc(target)
        # target_image shape [w, h, t, ch]
        input_image = ifftnc(input_kspace)
        # input_image shape [w, h, t, ch]
            
        #! 3. Padding
        padded_input, padded_mask = pad_size_tensor(input_image, mask, self.padding_size)
        # print(padded_input.shape, padded_mask.shape)
        # padded_input shape [512, 256, t, ch]
        # padded_mask shape [512, 256, 1 or t]
        padded_target, _ = pad_size_tensor(target_image, mask, self.padding_size)
        # print(padded_target.shape)
        # padded_target shape [512, 256, t, ch]
        
        # data is already masked
        
        #! 6. Apply time window
        seed = None if not self.use_seed else str(tuple(map(ord, fname)))
        start_time = random.Random(seed).randint(0, 12 - self.time_window)
        
        padded_input = padded_input[:, :, start_time:start_time + self.time_window, :]
        #padded_input_kspace = padded_input_kspace[:, :, start_time:start_time + self.time_window, :]
        padded_target = padded_target[:, :, start_time:start_time + self.time_window, :]
        
        #! 7. normalization
        if self.normalize:
            padded_input, mean, std = normalize_instance(padded_input, eps=1e-11)
            padded_target = normalize(padded_target, mean, std, eps=1e-11)
        else:
            mean, std = 0, 1
        
        #! 4. Apply fft to get related k-space
        padded_input_kspace = fftnc(padded_input)
        # padded_input_kspace shape [512, 256, t, ch]

        #! 5. Apply mask
        if self.apply_mask:
            padded_input_kspace *= padded_mask[..., None]
            padded_input_kspace += 0.0 # remove signs of zeros

        image = padded_input.permute(3, 2, 0, 1)  # HWTC->CTWH
        target = padded_target.permute(3, 2, 0, 1)  # HWTC->CTWH
        kspace = padded_input_kspace.permute(3, 2, 0, 1)
        mask = padded_mask[..., None].permute(3, 2, 0, 1).float()

        if self.noise_level is not None and self.noise_level > 0:
            kspace = GaussianNoise(sigma=self.noise_level, rng=self.generator)(kspace) * mask

        return target, kspace, mask






def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)

def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std

def pad_size_tensor(data: torch.Tensor, mask: torch.Tensor, desired_shape: Tuple[int, int]= (512, 256)): 
    # shape: [batch, width, height, t, channels]
    w_, h_ = desired_shape #desired shapes
    h, w = data.shape[-3], data.shape[-4] #actual original shapes #TODO dims -3 and -4
    h_pad, w_pad = (h_-h), (w_-w)
    
    if data.dim() == 4:
        pad = (0, 0,
               0, 0,
                h_pad//2, h_pad//2,
                w_pad//2, w_pad//2
            )        
    elif data.dim() == 5:
        pad = (0, 0,
               0, 0,
               0, 0,
                h_pad // 2, h_pad // 2,
                w_pad // 2, w_pad // 2,
            )

    pad_4_mask = (0, 0,
                h_pad // 2, h_pad // 2,
                w_pad // 2, w_pad // 2,
                
                )
    
    data_padded = F.pad(data, pad, mode='constant', value=0)
    mask_padded = F.pad(mask, pad_4_mask, mode='constant', value=0)
    
    # print("pad_size_tensor data_padded: ", data_padded.size())
    # print("pad_size_tensor mask_padded: ", mask_padded.size())
    
    return data_padded, mask_padded

def crop_to_depad(data, metadata):
    
    ori_height, ori_width = metadata['height'], metadata['width']    
    # print(ori_height, ori_width)
    data = data.permute(0, 3, 4, 2, 1)
    w_crop = (data.shape[-1] - ori_width) // 2
    h_crop = (data.shape[-2] - ori_height) // 2
    
    data = torchvision.transforms.functional.crop(data, h_crop, w_crop, ori_height, ori_width)    

    return data.permute(0, 4, 3, 1, 2)

def new_crop(data, metadata):
    ori_height, ori_width = metadata['height'], metadata['width']
    data = data.permute(0, 1, 3, 2)
    w_crop = (data.shape[-1] - ori_width) // 2
    h_crop = (data.shape[-2] - ori_height) // 2

    data = torchvision.transforms.functional.crop(data, h_crop, w_crop, ori_height, ori_width)

    return data.permute(0, 1, 3, 2)