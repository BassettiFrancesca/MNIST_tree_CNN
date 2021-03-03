from torch.utils.data import Dataset


class Subset(Dataset):

    def __init__(self, dataset, indices):
        self._dataset = dataset
        self.indices = indices
        self.old_dataset = old_dataset

    def __getitem__(self, idx):
        return self._dataset[idx]

    def get_item(self, idx):
        return self.old_dataset.get_item(self.indices[idx])

    def __len__(self):
        return len(self._dataset)


class Subset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
