import numpy as np
import shutil
import torch
import os
import nltk
import time

from torch import nn
from collections import Counter
from torch.utils.data.dataloader import default_collate
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


# Some code is taken from
# https://github.com/erickrf/multiffn-nli
class Utils:
    UNKNOWN = '**UNK**'
    PADDING = '**PAD**'
    # it's called "GO" but actually serves as a null alignment
    GO = '**GO**'

    def __init__(self):
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def tokenize_english(self, text):
        return self.tokenizer.tokenize(text)

    def tokenize_corpus(self, pairs):
        tokenized_pairs = []
        for sent1, sent2, label in pairs:
            tokens1 = self.tokenize_english(sent1)
            tokens2 = self.tokenize_english(sent2)
            tokenized_pairs.append((tokens1, tokens2, label))

        return tokenized_pairs

    def count_corpus_tokens(self, pairs):
        """
        Examine all pairs ans extracts all tokens from both text
        and hypothesis.
        :param pairs: a list of tuples (sent1, sent2, relation) with tokenized
            sentences
        :return: a Counter of lowercase tokens
        """
        c = Counter()
        for sent1, sent2, _ in pairs:
            c.update(t.lower() for t in sent1)
            c.update(t.lower() for t in sent2)

        return c

    def shuffle_arrays(self, *arrays):
        rng_state = np.random.get_state()
        for array in arrays:
            np.random.shuffle(array)
            np.random.set_state(rng_state)

    def create_label_dict(self, params, pairs):
        """
        Return a dictionary mapping the labels found in `pairs` to numbers
        :param pairs: a list of tuples (_, _, label), with label as a string
        :return: a dict
        """
        labels = set(pair[2] for pair in pairs)
        mapping = zip(labels, range(len(labels)))
        params.nr_classes = len(labels)
        return dict(mapping)

    def convert_labels(self, pairs, label_map):
        """
        Return a numpy array representing the labels in `pairs`
        :param pairs: a list of tuples (_, _, label), with label as a string
        :param label_map: dictionary mapping label strings to numbers
        :return: a numpy array
        """
        return np.array([label_map[pair[2]] for pair in pairs], dtype=np.int32)

    def create_dataset(self, pairs, word_dict, label_dict=None,
                       max_len1=None, max_len2=None):
        """
        Generate and return a NLIDataset object for storing the data in numpy
        format.
        :param pairs: list of tokenized tuples (sent1, sent2, label)
        :param word_dict: a dictionary mapping words to indices
        :param label_dict: a dictionary mapping labels to numbers. If None,
            labels are ignored.
        :param max_len1: the maximum length that arrays for sentence 1
            should have (i.e., time steps for an LSTM). If None, it
            is computed from the data.
        :param max_len2: same as max_len1 for sentence 2
        :return: RTEDataset
        """
        tokens1 = [pair[0] for pair in pairs]
        tokens2 = [pair[1] for pair in pairs]
        sentences1, sizes1 = self._convert_pairs_to_indices(tokens1, word_dict,
                                                            max_len1)
        sentences2, sizes2 = self._convert_pairs_to_indices(tokens2, word_dict,
                                                            max_len2)
        if label_dict is not None:
            labels = self.convert_labels(pairs, label_dict)
        else:
            labels = None

        return NLIDataset(sentences1, sentences2, sizes1, sizes2, labels)

    def collate_batch(self, batch, params):
        if params.use_optimizations:
            premise = np.array([k for bat in batch for k in bat[0]],
                               dtype=np.float)
            hypo = np.array([k for bat in batch for k in bat[1]],
                            dtype=np.float)
            labels = np.array([k for bat in batch for k in bat[2]],
                              dtype=np.float)
            return torch.from_numpy(premise), torch.from_numpy(hypo), \
                torch.from_numpy(labels)
        else:
            return default_collate(batch)

    def _convert_pairs_to_indices(self, sentences, word_dict, max_len=None,
                                  use_null=True):
        sizes = np.array([len(sent) for sent in sentences])
        if use_null:
            sizes += 1
            if max_len is not None:
                max_len += 1

        if max_len is None:
            max_len = sizes.max()

        shape = (len(sentences), max_len)
        array = np.full(shape, word_dict[self.PADDING], dtype=np.int32)

        for i, sent in enumerate(sentences):
            words = []

            if len(sent) <= max_len - 1:
                words = sent
            else:
                idx = 0
                while len(words) < max_len - 1:
                    words.append(sent[idx])
                    idx += 1

            indices = [word_dict[token] for token in words]

            if use_null:
                indices = [word_dict[self.GO]] + indices

            array[i, :len(indices)] = indices

        return array, sizes

    def load_parameters(self, dirname):
        filename = os.path.join(dirname, 'model-params.json')
        with open(filename, 'rb') as f:
            data = json.load(f)

        return data

    def get_sentence_sizes(self, pairs):
        sizes1 = np.array([len(pair[0]) for pair in pairs])
        sizes2 = np.array([len(pair[1]) for pair in pairs])
        return (sizes1, sizes2)

    def get_max_sentence_sizes(self, pairs1, pairs2):
        train_sizes1, train_sizes2 = self.get_sentence_sizes(pairs1)
        valid_sizes1, valid_sizes2 = self.get_sentence_sizes(pairs2)
        train_max1 = max(train_sizes1)
        valid_max1 = max(valid_sizes1)
        max_size1 = max(train_max1, valid_max1)
        train_max2 = max(train_sizes2)
        valid_max2 = max(valid_sizes2)
        max_size2 = max(train_max2, valid_max2)

        return max_size1, max_size2

    def normalize_embeddings(self, embeddings):
        """
        Normalize the embeddings to have norm 1.
        :param embeddings: 2-d numpy array
        :return: normalized embeddings
        """
        # normalize embeddings
        norms = np.linalg.norm(embeddings.numpy(), axis=1).reshape((-1, 1))
        embeddings = torch.from_numpy(embeddings.numpy() / norms)
        return embeddings

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        epoch = state['epoch']
        print("=> Saving model to %s" % filename)
        if epoch % 50 == 0 and epoch != 0:
            shutil.copyfile(filename, 'model_' + str(epoch) + '.pth.tar')
        if is_best:
            print("=> The model just saved has performed best on" +
                  "validation set till now")
            shutil.copyfile(filename, 'model_best.pth.tar')

        return filename

    def load_checkpoint(self, resume):
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
            return checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            return None

    def get_time_hhmmss(self, start=None):
        """
        Calculates time since `start` and formats as a string.
        """
        if start is None:
            return time.strftime("%Y/%m/%d %H:%M:%S")
        end = time.time()
        m, s = divmod(end - start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str

    def init_weights(self, model):
        # As mentioned in paper
        mean = 0
        stddev = 0.01
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, stddev)

    def get_embedded_mask(self, embedded):
        return (embedded != 1).float()


utils = Utils()
