from torch import nn
from modules.Utils import utils


class Aggregate(nn.Module):
    def __init__(self, nr_hidden, nr_out, dropout=0.0):
        super(Aggregate, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(nr_hidden * 2, nr_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(nr_hidden, nr_out)
        )

        utils.init_weights(self.model)

    def forward(self, input):
        return self.model(input)
