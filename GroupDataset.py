from torch.utils.data import Dataset


class GroupDataset(Dataset):

    def __init__(self, dataset, groups):
        self._dataset = dataset
        self.groups = groups

    def __getitem__(self, idx):
        (x, y) = self._dataset[idx]
        for (c, group) in enumerate(self.groups):
            if y in group:
                return x, c

    def get_item(self, idx):
        return self._dataset[idx]

    def __len__(self):
        return len(self._dataset)
