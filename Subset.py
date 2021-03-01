from torch.utils.data import Dataset


class Subset(Dataset):

    def __init__(self, dataset, old_dataset, indices):
        self._dataset = dataset
        self.indices = indices
        self.old_dataset = old_dataset

    def __getitem__(self, idx):
        return self._dataset[idx]

    def get_item(self, idx):
        return self.old_dataset.get_item(self.indices[idx])

    def __len__(self):
        return len(self._dataset)
