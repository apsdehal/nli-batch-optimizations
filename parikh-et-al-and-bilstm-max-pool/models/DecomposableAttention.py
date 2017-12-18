import torch

from torch import nn

from modules.BiRNNEncoder import BiRNNEncoder
from modules.Attention import Attention
from modules.SoftAlignment import SoftAlignment
from modules.Comparison import Comparison
from modules.Aggregate import Aggregate
from modules.Utils import utils


class DecomposableAttention(nn.Module):
    def __init__(self, settings, embeddings):
        super(DecomposableAttention, self).__init__()

        if isinstance(settings, argparse.Namespace):
            settings = vars(settings)

        self.embedding = embeddings
        self.settings = settings
        self.max_length = settings['max_len']
        self.hidden_dim = settings['hidden_dim']
        self.nr_classes = settings['nr_classes']

        if settings['hidden_dim'] != settings['projection_dim']:
            self.projection = nn.Linear(self.hidden_dim,
                                        settings['projection_dim'])
        else:
            self.projection = lambda x: x

        settings['hidden_dim'] = settings['projection_dim']
        self.hidden_dim = settings['hidden_dim']

        if settings['gru_encode']:
            self.encoder = BiRNNEncoder(self.max_length, self.hidden_dim,
                                        dropout=settings['dropout'])
        if settings['use_intra_attention']:
            self.intra_sentence_attender = \
                Attention(self.max_length, self.hidden_dim,
                          dropout=settings['dropout'])

            self.intra_align = SoftAlignment(self.max_length, self.hidden_dim)
            self.intra_align_project = nn.Linear(self.hidden_dim * 2,
                                                 self.hidden_dim)

        self.attender = Attention(self.max_length, self.hidden_dim,
                                  dropout=settings['dropout'])

        self.align = SoftAlignment(self.max_length, self.hidden_dim)

        self.compare = Comparison(self.max_length, self.hidden_dim,
                                  dropout=settings['dropout'])
        self.aggregate = Aggregate(self.hidden_dim, self.nr_classes,
                                   dropout=settings['dropout'])

        utils.init_weights(self)

    def forward(self, premise, hypo):
        premise_mask = utils.get_embedded_mask(premise)
        hypo_mask = utils.get_embedded_mask(hypo)

        premise = self.embedding(premise)
        hypo = self.embedding(hypo)

        premise = self.projection(premise)
        hypo = self.projection(hypo)

        if self.settings['gru_encode']:
            premise = self.encoder(premise)
            hypo = self.encoder(hypo)

        if self.settings['use_intra_attention']:
            # Intra Sentence Attention
            premise = self.intra_attention(premise, premise_mask)
            hypo = self.intra_attention(hypo, hypo_mask)

            premise = self.intra_align_project(premise)
            hypo = self.intra_align_project(hypo)

        projected_premise = self.attender(premise)
        projected_hypo = self.attender(hypo)

        att_ji = projected_hypo.bmm(projected_premise.permute(0, 2, 1))

        # Shape: batch_length * max_length * max_length
        att_ij = att_ji.permute(0, 2, 1)

        aligned_hypo = self.align(hypo, hypo_mask, att_ij)
        aligned_premise = self.align(premise, premise_mask, att_ij,
                                     transpose=True)

        premise_compare_input = torch.cat([premise, aligned_hypo], dim=-1)
        hypo_compare_input = torch.cat([hypo, aligned_premise], dim=-1)

        compared_premise = self.compare(premise_compare_input)
        compared_premise = compared_premise * premise_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_premise = compared_premise.sum(dim=1)

        compared_hypo = self.compare(hypo_compare_input)
        compared_hypo = compared_hypo * hypo_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_hypo = compared_hypo.sum(dim=1)

        aggregate_input = torch.cat([compared_premise, compared_hypo], dim=-1)
        scores = self.aggregate(aggregate_input)

        return scores

    def intra_attention(self, sentence, sentence_mask):
        projected_sentence = self.intra_sentence_attender(sentence)
        intra_att_ji = projected_sentence.bmm(projected_sentence.permute(0, 2,
                                                                         1))
        intra_att_ij = intra_att_ji.permute(0, 2, 1)
        aligned_sentence = self.intra_align(sentence, sentence_mask,
                                            intra_att_ij)
        sentence = torch.cat([sentence, aligned_sentence], dim=-1)

        return sentence
