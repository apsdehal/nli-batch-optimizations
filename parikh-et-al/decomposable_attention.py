import numpy as np
import torch.nn as nn


def build_model(vectors, shape, settings):
    max_length, nr_hidden, nr_class = shape

    global embedding
    embedding = nn.Embedding(vectors.shape(0), vectors.shape(1))
    embedding.weight.data.copy_(vectors)

    # Fix weights for training
    embedding.weight.requires_grad = False

    return DecomposableAttention(shape, settings)


class DecomposableAttention(nn.Module):
    def __init__(self, shape, settings):
        global embedding
        self.embedding = embedding
        if settings['gru_encode']:
            self.encoder = BiRNNEncoder(max_length, nr_hidden,
                                        dropout=settings['dropout'])

        self.attender = Attention(max_length, nr_hidden,
                                  dropout=settings['dropout'])
        self.align = SoftAlignment(max_length, nr_hidden)

        self.compare = Comparator(max_length, nr_hidden,
                                  dropout=settings['dropout'])
        self.entail = Entailment(nr_hidden, nr_class,
                                 dropout=settings['dropout'])

    def forward(self, premise, hypo):
        premise = self.embedding(premise)
        hypo = self.embedding(premise)

        if settings['gru_encode']:
            premise = self.encoder(premise)
            hypo = self.encoder(hypo)

        attention = self.attender(premise, hypo)

        align1 = self.align(hypo, attention)
        align2 = self.align(premise, attention)

        feats1 = self.compare(premise, align1)
        feats2 = self.compare(hypo, align2)

        scores = entail(feats1, feats2)

        return scores
