import torch

from torch import nn
from modules.Utils import utils


class BiLSTMMaxPooling(nn.Module):
    def __init__(self, params, embedding):
        super(BiLSTMMaxPooling, self).__init__()

        self.embedding_dim = embedding.embedding_dim

        self.max_length = params.max_len
        self.nr_hidden = params.hidden_dim
        self.nr_classes = params.nr_classes

        self.bilstm = nn.LSTM(self.embedding_dim, self.nr_hidden,
                              num_layers=1,
                              batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout2d(p=params.dropout)

        self.linear_1 = nn.Linear(8 * self.nr_hidden, self.nr_hidden)
        self.linear_2 = nn.Linear(9 * self.nr_hidden, self.nr_hidden)
        self.linear_3 = nn.Linear(self.nr_hidden, self.nr_classes)

        # Glove embedding
        self.embedding = embedding

        utils.init_weights(self)

    def forward(self, premise, hypo):
        # B x T => B x T x E
        premise = self.embedding(premise)
        hypo = self.embedding(hypo)

        # Encode with bilstm and take all of the hidden states
        # for each t belonging to seq_len
        # B x T x E => B x T x 2H
        premise_output, _ = self.bilstm(premise)

        # Take max pool along seq_len
        # B x T x 2H => B x 2H
        premise_encoded, _ = premise_output.max(dim=1)

        premise_encoded = self.dropout(premise_encoded)

        hypo_output, _ = self.bilstm(hypo)
        hypo_encoded, _ = hypo_output.max(dim=1)

        hypo_encoded = self.dropout(hypo_encoded)

        # B x 2H => B x 8H
        lin_input = torch.cat([premise_encoded, hypo_encoded,
                               premise_encoded - hypo_encoded,
                               torch.mul(premise_encoded, hypo_encoded)],
                              dim=1)

        # B x 8H => B x H
        lin_output = self.linear_1(lin_input)
        lin_output = nn.functional.relu(lin_output)

        lin_output = self.dropout(lin_output)

        # Skip connections, B x H => B x 9H
        lin_input_2 = torch.cat([lin_input, lin_output], dim=1)

        # B x 9H => B x H
        lin_output_2 = self.linear_2(lin_input_2)
        lin_output_2 = nn.functional.relu(lin_output_2)

        lin_output = self.dropout(lin_output_2)

        # B x H => B x 3
        return self.linear_3(lin_output_2)
