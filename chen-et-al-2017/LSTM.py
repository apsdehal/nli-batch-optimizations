from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
class LSTM(nn.Module):
    def __init__(self, nin, hidden_size):
        super(LSTM, self).__init__()
        if torch.cuda.is_available():
            self.linear_f = nn.Linear(nin + hidden_size, hidden_size).cuda()
            self.linear_i = nn.Linear(nin + hidden_size, hidden_size).cuda()
            self.linear_ctilde = nn.Linear(nin + hidden_size, hidden_size).cuda()
            self.linear_o = nn.Linear(nin + hidden_size, hidden_size).cuda()

        else:
            self.linear_f = nn.Linear(nin + hidden_size, hidden_size)
            self.linear_i = nn.Linear(nin + hidden_size, hidden_size)
            self.linear_ctilde = nn.Linear(nin + hidden_size, hidden_size)
            self.linear_o = nn.Linear(nin + hidden_size, hidden_size)

        self.hidden_size = hidden_size
        self.init_weights()

    def forward(self, x, mask):
        hidden, c = self.init_hidden(x.size(1))

        def step(emb, hid, c_t_old, mask_cur):
            combined = torch.cat((hid, emb), 1)

            f = F.sigmoid(self.linear_f(combined))
            i = F.sigmoid(self.linear_i(combined))
            o = F.sigmoid(self.linear_o(combined))
            c_tilde = F.tanh(self.linear_ctilde(combined))

            c_t = f * c_t_old + i * c_tilde
            c_t = mask_cur[:, None] * c_t + (1. - mask_cur)[:, None] * c_t_old

            hid_new = o * F.tanh(c_t)
            hid_new = mask_cur[:, None] * hid_new + (1. - mask_cur)[:, None] * hid

            return hid_new, c_t, i

        h_hist = []
        i_hist = []
        for i in range(x.size(0)):
            hidden, c, i = step(x[i].squeeze(), hidden, c, mask[i])
            h_hist.append(hidden[None, :, :])
            i_hist.append(i[None, :, :])

        return torch.cat(h_hist), torch.cat(i_hist)

    def init_hidden(self, bat_size):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(bat_size, self.hidden_size).cuda())
            c0 = Variable(torch.zeros(bat_size, self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(bat_size, self.hidden_size))
            c0 = Variable(torch.zeros(bat_size, self.hidden_size))
        return h0, c0

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.linear_f, self.linear_i, self.linear_ctilde, self.linear_o]

        for layer in lin_layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.fill_(0)

