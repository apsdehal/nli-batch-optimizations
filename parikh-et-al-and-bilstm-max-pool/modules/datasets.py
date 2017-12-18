from torch.utils.data import Dataset


class NLIDataset(Dataset):

    def __init__(self, sentences1, sentences2, sizes1, sizes2, labels):
        """
        :param sentences1: A 2D numpy array with sentences (the first in each
            pair) composed of token indices
        :param sentences2: Same as above for the second sentence in each pair
        :param sizes1: A 1D numpy array with the size of each sentence in the
            first group. Sentences should be filled with the
            PADDING token after that point
        :param sizes2: Same as above
        :param labels: 1D numpy array with labels as integers
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.sizes1 = sizes1
        self.sizes2 = sizes2
        self.labels = labels
        self.num_items = len(sentences1)

    def shuffle_data(self):
        """
        Shuffle all data using the same random sequence.
        :return:
        """
        shuffle_arrays(self.sentences1, self.sentences2,
                       self.sizes1, self.sizes2, self.labels)

    def get_batch(self, from_, to):
        """
        Return an NLIDataset object with the subset of the data contained in
        the given interval. Note that the actual number of items may be less
        than (`to` - `from_`) if there are not enough of them.
        :param from_: which position to start from
        :param to: which position to end
        :return: an NLIDataset object
        """
        if from_ == 0 and to >= self.num_items:
            return self

        subset = NLIDataset(self.sentences1[from_:to],
                            self.sentences2[from_:to],
                            self.sizes1[from_:to],
                            self.sizes2[from_:to],
                            self.labels[from_:to])
        return subset

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        return self.sentences1[idx], self.sentences2[idx], self.labels[idx]


class BatchedNLIDataset(Dataset):

    def __init__(self, pairs, word_dict, sizes1, sizes2, label_dict):
        self.pairs = pairs
        self.label_dict = label_dict
        self.word_dict = word_dict
        self.max_len1 = sizes1
        self.max_len2 = sizes2

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pairs = self.pairs[idx]
        tokens1 = [pair[0] for pair in pairs]
        tokens2 = [pair[1] for pair in pairs]
        sentences1, sizes1 = utils._convert_pairs_to_indices(tokens1,
                                                             self.word_dict,
                                                             self.max_len1)
        sentences2, sizes2 = utils._convert_pairs_to_indices(tokens2,
                                                             self.word_dict,
                                                             self.max_len2)
        if self.label_dict is not None:
            labels = utils.convert_labels(pairs, self.label_dict)
        else:
            labels = None

        return sentences1, sentences2, labels
