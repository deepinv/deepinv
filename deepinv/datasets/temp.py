import os
import pickle
import warnings
from contextlib import contextmanager
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path


@contextmanager
def metadata_cache_manager(root, metadata_cache_file, load_metadata_from_cache, save_metadata_to_cache):
    sample_identifiers = defaultdict(list)

    if load_metadata_from_cache and os.path.exists(metadata_cache_file):
        with open(metadata_cache_file, "rb") as f:
            dataset_cache = pickle.load(f)
            if dataset_cache.get(root) is None:
                raise ValueError(
                    "`metadata_cache_file` doesn't contain the metadata. Please "
                    + "either deactivate `load_dataset_from_cache` OR set `metadata_cache_file` properly."
                )
            print(f"Using dataset cache from {metadata_cache_file}.")
            sample_identifiers = dataset_cache[root]
        yield sample_identifiers

    else:
        # Warn if cache file is missing
        if load_metadata_from_cache and not os.path.exists(metadata_cache_file):
            warnings.warn(
                f"Couldn't find dataset cache at {metadata_cache_file}. Loading dataset from scratch."
            )

        yield sample_identifiers  # Let the calling code populate sample_identifiers

        # Save the updated metadata to cache
        if save_metadata_to_cache:
            dataset_cache = {root: sample_identifiers}
            print(f"Saving dataset cache to {metadata_cache_file}.")
            with open(metadata_cache_file, "wb") as cache_f:
                pickle.dump(dataset_cache, cache_f)


# Usage
class Dataset:
    def __init__(self, root, metadata_cache_file, load_metadata_from_cache, save_metadata_to_cache):
        self.root = root
        self.metadata_cache_file = metadata_cache_file
        self.load_metadata_from_cache = load_metadata_from_cache
        self.save_metadata_to_cache = save_metadata_to_cache
        self.sample_identifiers = defaultdict(list)

    def _retrieve_metadata(self, fname):
        # Dummy implementation for metadata retrieval
        return {"num_slices": 5}  # Replace with actual logic

    def load_data(self):
        all_fnames = sorted(list(Path(self.root).iterdir()))
        with metadata_cache_manager(
            self.root, self.metadata_cache_file, self.load_metadata_from_cache, self.save_metadata_to_cache
        ) as sample_identifiers:
            for fname in tqdm(all_fnames):
                metadata = self._retrieve_metadata(fname)
                for slice_ind in range(metadata["num_slices"]):
                    sample_identifiers[str(fname)].append(
                        self.SliceSampleFileIdentifier(fname, slice_ind, metadata)
                    )
            self.sample_identifiers = sample_identifiers
