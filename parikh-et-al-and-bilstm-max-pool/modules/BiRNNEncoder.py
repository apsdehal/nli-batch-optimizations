from torch import nn

from modules.TimeDistributed import TimeDistributed
from modules.Utils import utils


class BiRNNEncoder(nn.Module):
    def __init__(self, max_length, nr_hidden, dropout=0.0):
        super(BiRNNEncoder, self).__init__()

        self.nr_hidden = nr_hidden

        self.fully_connected = nn.Sequential(
            nn.Linear(nr_hidden * 2, nr_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(nr_hidden, nr_hidden, max_length,
                            dropout=dropout,
                            bidirectional=True,
                            batch_first=True)
        self.fully_connected = TimeDistributed(self.fully_connected)
        self.dropout = nn.Dropout(p=dropout)

        utils.init_weights(self)

    def forward(self, input):
        output, _ = self.lstm(input)
        return self.dropout(self.fully_connected(output))
