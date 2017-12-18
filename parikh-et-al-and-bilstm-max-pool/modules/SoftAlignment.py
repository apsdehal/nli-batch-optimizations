import torch

from torch import nn


class SoftAlignment(nn.Module):
    def __init__(self, max_length, nr_hidden):
        super(SoftAlignment, self).__init__()

        self.max_length = max_length
        self.nr_hidden = nr_hidden

    def forward(self, sentence, mask, attention_matrix, transpose=False):
        if transpose:
            attention_matrix = attention_matrix.permute(0, 2, 1)

        exponents = torch.exp(attention_matrix -
                              torch.max(attention_matrix,
                                        dim=-1,
                                        keepdim=True)[0])

        exponents = exponents

        summation = torch.sum(exponents, dim=-1, keepdim=True)

        attention_weights = exponents / summation

        attention_weights = attention_weights * mask.unsqueeze(-1)

        attention_weights = attention_weights / \
            (attention_weights.sum(-1, keepdim=True) + 1e-13)

        return attention_weights.bmm(sentence)
