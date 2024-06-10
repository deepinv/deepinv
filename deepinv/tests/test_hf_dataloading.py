from datasets import load_dataset, load_from_disk


DATA_DIR = "DRUNET_preprocessed"

# download from Internet
# https://huggingface.co/datasets/deepinv/drunet_dataset
dataset = load_dataset("deepinv/drunet_dataset")

# save it to disk, which is useful to avoid downloading again
dataset.save_to_disk(DATA_DIR)

# load from disk (useless here, as we already have the dataset in memory)
dataset = load_from_disk(DATA_DIR)
    