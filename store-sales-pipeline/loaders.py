import torch
from torch.utils.data import DataLoader


class TimeSeriesDataLoader:
    def __init__(self, dataset, batch_size, shuffle=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last

        self.loader = DataLoader(dataset=self.dataset,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle,
                                 num_workers=self.num_workers,
                                 drop_last=self.drop_last)

    def __iter__(self):
        for batch_idx, (data, target) in enumerate(self.loader):
            yield data.float(), target.float()

    def __len__(self):
        return len(self.loader)