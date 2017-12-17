from torch.utils.data import Dataset

class NLIDataset(Dataset):
    def __init__(self, data, labels):
        assert len(data[0]) == len(labels)
        assert len(data[0]) == len(data[1])
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, key):
        return self.data[0][key], self.data[1][key], self.labels[key]


class BatchedNLIDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pairs = self.pairs[idx]
        tokens1 = [pair[0] for pair in pairs]
        tokens2 = [pair[1] for pair in pairs]
        labels = [pair[2] for pair in pairs]
        return tokens1, tokens2, labels