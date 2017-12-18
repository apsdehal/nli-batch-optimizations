import numpy as np
import os
import torch
import json
import nltk

from torch import nn
from torchtext import vocab
from collections import defaultdict

from modules.Utils import utils


class IO:
    UNKNOWN = '**UNK**'
    PADDING = '**PAD**'
    # it's called "GO" but actually serves as a null alignment
    GO = '**GO**'

    def _generate_random_vector(self, size):
        """
        Generate a random vector from a uniform distribution between
        -0.1 and 0.1.
        """
        return np.random.uniform(-0.1, 0.1, size)

    def write_extra_embeddings(self, embeddings, dirname):
        """
        Write the extra embeddings (for unknown, padding and null)
        to a numpy file. They are assumed to be the first three in
        the embeddings model.
        """
        path = os.path.join(dirname, 'extra-embeddings.npy')
        torch.save(embeddings[:3], path)

    def load_embeddings(self, params, normalize=True, generate=True):
        glove = vocab.GloVe(name='6B', dim=params.embedding_dim)
        wordlist, embeddings = glove.stoi, glove.vectors

        mapping = zip(wordlist, range(3, len(wordlist) + 3))

        # always map OOV words to 0
        wd = defaultdict(int, mapping)
        wd[self.UNKNOWN] = 0
        wd[self.PADDING] = 1
        wd[self.GO] = 2

        if generate:
            vector_size = embeddings.shape[1]
            extra = torch.FloatTensor([
                self._generate_random_vector(vector_size),
                self._generate_random_vector(vector_size),
                self._generate_random_vector(vector_size)])
            self.write_extra_embeddings(extra, params.save_loc)

        else:
            path = os.path.join(params.save_loc, 'extra-embeddings.npy')
            extra = torch.load(path)

        embeddings = torch.cat((extra, embeddings), 0)

        print('Embeddings have shape {}'.format(embeddings.shape))
        if normalize:
            embeddings = utils.normalize_embeddings(embeddings)

        nn_embedding = nn.Embedding(embeddings.shape[0],
                                    embeddings.shape[1])

        nn_embedding.weight.data.copy_(embeddings)

        # Fix weights for training
        nn_embedding.weight.requires_grad = False

        return wd, nn_embedding

    def read_corpus(self, filename, lowercase):
        print('Reading data from %s' % filename)
        # we are only interested in the actual sentences + gold label
        # the corpus files has a few more things
        useful_data = []
        # the Multinli corpus has one JSON object per line
        with open(filename, 'rb') as f:
            for line in f:
                line = line.decode('utf-8')
                if lowercase:
                    line = line.lower()
                data = json.loads(line)
                if data['gold_label'] == '-':
                    # ignore items without a gold label
                    continue

                sentence1_parse = data['sentence1_parse']
                sentence2_parse = data['sentence2_parse']
                label = data['gold_label']

                tree1 = nltk.Tree.fromstring(sentence1_parse)
                tree2 = nltk.Tree.fromstring(sentence2_parse)
                tokens1 = tree1.leaves()
                tokens2 = tree2.leaves()
                t = (tokens1, tokens2, label)
                useful_data.append(t)

        return useful_data

    def read_corpus_batched(self, filename, lowercase):
        print('Reading data from %s' % filename)
        # we are only interested in the actual sentences + gold label
        # the corpus files has a few more things
        useful_data = []
        done = dict()
        # the Multinli corpus has one JSON object per line
        with open(filename, 'rb') as f:
            for line in f:
                line = line.decode('utf-8')
                if lowercase:
                    line = line.lower()
                data = json.loads(line)
                if data['gold_label'] == '-':
                    # ignore items without a gold label
                    continue
                prompt_id = data['promptid']

                if prompt_id not in done:
                    done[prompt_id] = len(useful_data)
                    useful_data.append([])

                sentence1_parse = data['sentence1_parse']
                sentence2_parse = data['sentence2_parse']
                label = data['gold_label']

                tree1 = nltk.Tree.fromstring(sentence1_parse)
                tree2 = nltk.Tree.fromstring(sentence2_parse)
                tokens1 = tree1.leaves()
                tokens2 = tree2.leaves()
                t = (tokens1, tokens2, label)
                useful_data[done[prompt_id]].append(t)

        return useful_data


io = IO()
