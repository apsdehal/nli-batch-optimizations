
# coding: utf-8

# In[4]:


import torch
from data_iterator import TextIterator
import numpy
from torch.autograd import Variable


# In[5]:


def prepare_data(seqs_x, seqs_y, labels, maxlen=None):
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        new_labels = []
        for l_x, s_x, l_y, s_y, l in zip(lengths_x, seqs_x, lengths_y, seqs_y, labels):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
                new_labels.append(l)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y
        labels = new_labels

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x)
    maxlen_y = numpy.max(lengths_y)

    if torch.cuda.is_available():
        x = torch.zeros(maxlen_x, n_samples).long().cuda()
        y = torch.zeros(maxlen_y, n_samples).long().cuda()
        x_mask = torch.zeros(maxlen_x, n_samples).cuda()
        y_mask = torch.zeros(maxlen_y, n_samples).cuda()
        l = torch.zeros(n_samples,).long().cuda()
    else:
        x = torch.zeros(maxlen_x, n_samples).long()
        y = torch.zeros(maxlen_y, n_samples).long()
        x_mask = torch.zeros(maxlen_x, n_samples)
        y_mask = torch.zeros(maxlen_y, n_samples)
        l = torch.zeros(n_samples,).long()
   
    for idx, [s_x, s_y, ll] in enumerate(zip(seqs_x, seqs_y, labels)):
        if torch.cuda.is_available():
            x[:lengths_x[idx], idx] = torch.Tensor(s_x).cuda()
            x_mask[:lengths_x[idx], idx] = 1.
            y[:lengths_y[idx], idx] = torch.Tensor(s_y).cuda()
            y_mask[:lengths_y[idx], idx] = 1.
            l[idx] = int(ll)
        else:
            x[:lengths_x[idx], idx] = torch.Tensor(s_x)
            x_mask[:lengths_x[idx], idx] = 1.
            y[:lengths_y[idx], idx] = torch.Tensor(s_y)
            y_mask[:lengths_y[idx], idx] = 1.
            l[idx] = int(ll)

    return Variable(x), Variable(x_mask), Variable(y), Variable(y_mask), Variable(l)


# In[ ]:




