from torch import nn
from .modules.TimeDistributed import TimeDistributed
from .modules.Utils import utils


class Attention(nn.Module):
    def __init__(self, max_length, nr_hidden, dropout=0.0):
        super(Attention, self).__init__()

        self.model = TimeDistributed(nn.Sequential(
            nn.Linear(nr_hidden, nr_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(nr_hidden, nr_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        ))

        utils.init_weights(self.model)

    def forward(self, sentence):
        return self.model(sentence)
