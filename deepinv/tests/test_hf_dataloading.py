from torch.utils.data import DataLoader, IterableDataset

from torchvision import transforms

import datasets as hf_datasets

class HFDataset(IterableDataset):
    r"""
    Creates an iteratble dataset from a Hugging Face dataset to enable streaming.
    """
    def __init__(self, hf_dataset, transforms=None, key='png'):
        self.hf_dataset = hf_dataset
        self.transform = transforms
        self.key = key

    def __iter__(self):
        for sample in self.hf_dataset:
            if self.transform:
                out = self.transform(sample[self.key])
            else:
                out = sample[self.key]
            yield out


def test_hf_dataloading():
    dataset = hf_datasets.load_dataset("deepinv/drunet_dataset", streaming=True)

    img_size = 32
    transforms_loader = transforms.Compose([transforms.CenterCrop(img_size),
                                            transforms.ToTensor()])

    hf_dataset = HFDataset(dataset['train'], transforms=transforms_loader)

    dataloader = DataLoader(hf_dataset, batch_size=5)

    for batch in dataloader:
        break

    assert batch.shape == (5, 3, img_size, img_size)


    