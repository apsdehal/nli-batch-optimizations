import numpy as np
import torch.nn as nn
import torch

from modules.TimeDistributed import TimeDistributed

PROJECTION_DIM = 200


def build_model(vectors, shape, settings):

    global embedding
    # Later include max_norm=1, norm_type=2 when pytorch release happens
    embedding = nn.Embedding(vectors.shape[0],
                             vectors.shape[1])

    embedding.weight.data.copy_(torch.from_numpy(vectors))

    # Fix weights for training
    embedding.weight.requires_grad = False

    return DecomposableAttention(shape, settings)


def init_weights(model):
    # As mentioned in paper
    mean = 0
    stddev = 0.01
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean, stddev)


def get_embedded_mask(embedded):
    return (embedded != 0).float()


class DecomposableAttention(nn.Module):
    def __init__(self, shape, settings):
        super(DecomposableAttention, self).__init__()

        global embedding

        self.embedding = embedding
        self.settings = settings
        self.max_length, self.nr_hidden, self.nr_class = shape
        self.projection = nn.Linear(self.nr_hidden, PROJECTION_DIM)

        settings['nr_hidden'] = PROJECTION_DIM
        self.nr_hidden = PROJECTION_DIM

        if settings['gru_encode']:
            self.encoder = BiRNNEncoder(self.max_length, self.nr_hidden,
                                        dropout=settings['dropout'])

        self.intra_sentence_attender = Attention(self.max_length,
                                                 self.nr_hidden,
                                                 dropout=settings['dropout'])

        self.intra_align = SoftAlignment(self.max_length, self.nr_hidden)
        self.intra_align_project = nn.Linear(self.nr_hidden * 2,
                                             PROJECTION_DIM)

        self.attender = Attention(self.max_length, self.nr_hidden,
                                  dropout=settings['dropout'])

        self.align = SoftAlignment(self.max_length, self.nr_hidden)

        self.compare = Comparison(self.max_length, self.nr_hidden,
                                  dropout=settings['dropout'])
        self.aggregate = Aggregate(self.nr_hidden, self.nr_class,
                                   dropout=settings['dropout'])

        init_weights(self)

    def forward(self, premise, hypo):
        premise = self.embedding(premise)
        hypo = self.embedding(hypo)

        premise = self.projection(premise)
        hypo = self.projection(hypo)

        premise_mask = get_embedded_mask(premise)
        hypo_mask = get_embedded_mask(hypo)

        if self.settings['gru_encode']:
            premise = self.encoder(premise)
            hypo = self.encoder(hypo)

        # Intra Sentence Attention
        premise = self.intra_attention(premise)
        hypo = self.intra_attention(hypo)

        premise = self.intra_align_project(premise)
        hypo = self.intra_align_project(hypo)

        projected_premise = self.attender(premise)
        projected_hypo = self.attender(hypo)

        att_ji = projected_hypo.bmm(projected_premise.permute(0, 2, 1))

        # Shape: batch_length * max_length * max_length
        att_ij = att_ji.permute(0, 1, 2)

        aligned_hypo = self.align(hypo, att_ij)
        aligned_premise = self.align(premise, att_ij, transpose=True)

        premise_compare_input = torch.cat([premise, aligned_premise], dim=-1)
        hypo_compare_input = torch.cat([hypo, aligned_hypo], dim=-1)

        compared_premise = self.compare(premise_compare_input)
        compared_premise = compared_premise * premise_mask
        # Shape: (batch_size, compare_dim)
        compared_premise = compared_premise.sum(dim=1)

        compared_hypo = self.compare(hypo_compare_input)
        compared_hypo = compared_hypo * hypo_mask
        # Shape: (batch_size, compare_dim)
        compared_hypo = compared_hypo.sum(dim=1)

        aggregate_input = torch.cat([compared_premise, compared_hypo], dim=-1)
        scores = self.aggregate(aggregate_input)

        return scores

    def intra_attention(self, sentence):
        projected_sentence = self.intra_sentence_attender(sentence)
        intra_att_ji = projected_sentence.bmm(projected_sentence.permute(0, 2,
                                                                         1))
        intra_att_ij = intra_att_ji.permute(0, 1, 2)
        aligned_sentence = self.intra_align(sentence, intra_att_ij)
        sentence = torch.cat([sentence, aligned_sentence], dim=-1)

        return sentence


class BiRNNEncoder(nn.Module):
    def __init__(self, max_length, nr_hidden, dropout=0.0):
        super(BiRNNEncoder, self).__init__()

        self.nr_hidden = nr_hidden

        self.fully_connected = nn.Sequential(
            nn.Linear(nr_hidden * 2, nr_hidden),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(nr_hidden, nr_hidden, max_length,
                            dropout=dropout,
                            bidirectional=True,
                            batch_first=True)
        self.fully_connected = TimeDistributed(self.fully_connected)
        self.dropout = nn.Dropout(p=dropout)

        init_weights(self)

    def forward(self, input):
        output, _ = self.lstm(input)
        return self.dropout(self.fully_connected(output))


class Attention(nn.Module):
    def __init__(self, max_length, nr_hidden, dropout=0.0):
        super(Attention, self).__init__()

        self.model = TimeDistributed(nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(nr_hidden, nr_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(nr_hidden, nr_hidden),
            nn.ReLU()
        ))

        init_weights(self.model)

    def forward(self, sentence):
        return self.model(sentence)


class SoftAlignment(nn.Module):
    def __init__(self, max_length, nr_hidden):
        super(SoftAlignment, self).__init__()

        self.max_length = max_length
        self.nr_hidden = nr_hidden

    def forward(self, sentence, attention_matrix, transpose=False):
        if transpose:
            attention_matrix = attention_matrix.permute(0, 2, 1)

        exponents = torch.exp(attention_matrix -
                              torch.max(attention_matrix,
                                        dim=-1,
                                        keepdim=True)[0])

        summation = torch.sum(exponents, dim=-1, keepdim=True)

        attention_weights = exponents / summation

        return attention_weights.bmm(sentence)


class Comparison(nn.Module):
    def __init__(self, max_length, nr_hidden, dropout=0):
        super(Comparison, self).__init__()

        self.max_length = max_length
        self.nr_hidden = nr_hidden
        self.model = TimeDistributed(nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(nr_hidden * 2, nr_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(nr_hidden, nr_hidden),
            nn.ReLU()
        ))
        init_weights(self.model)

    def forward(self, concatenated_aligned_sentence):
        return self.model(concatenated_aligned_sentence)


class Aggregate(nn.Module):
    def __init__(self, nr_hidden, nr_out, dropout=0.0):
        super(Aggregate, self).__init__()

        self.model = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(nr_hidden * 2, nr_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(nr_hidden, nr_out)
        )

        init_weights(self.model)

    def forward(self, input):
        return self.model(input)
