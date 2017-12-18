from torch import nn
from modules.TimeDistributed import TimeDistributed
from modules.Utils import utils


class Comparison(nn.Module):
    def __init__(self, max_length, nr_hidden, dropout=0):
        super(Comparison, self).__init__()

        self.max_length = max_length
        self.nr_hidden = nr_hidden
        self.model = TimeDistributed(nn.Sequential(
            nn.Linear(nr_hidden * 2, nr_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(nr_hidden, nr_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        ))
        utils.init_weights(self.model)

    def forward(self, concatenated_aligned_sentence):
        return self.model(concatenated_aligned_sentence)
