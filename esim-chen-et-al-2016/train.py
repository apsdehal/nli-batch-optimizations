import logging
from utils import prepare_data
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import pickle as pkl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import ESIM
from data_iterator import TextIterator
import time
import os
import torch.optim as optim

logger = logging.getLogger(__name__)

dim_word=300 # word vector dimensionality
dim=300  # the number of GRU units
encoder='lstm'  # encoder model
decoder='lstm' # decoder model
patience=6 # early stopping patience
max_epochs=20
finish_after=10000000  # finish after this many updates
decay_c=0.  # L2 regularization penalty
clip_c=10.0  # gradient clipping threshold
lrate=0.00008 # learning rate
n_words=42394  # vocabulary size
maxlen=100  # maximum length of the description
optimizer_spec='adadelta'
batch_size=32
valid_batch_size=32
saveto=''
dispFreq=10000
validFreq=1000
saveFreq=1  # save the parameters after every saveFreq updates
use_dropout=True
reload_=True
verbose=False  # print verbose information for debug but slow speed
datasets=[]
valid_datasets=[]
test_datasets=[]
dictionary=''
embedding='data/glove.840B.300d.txt'  # pretrain embedding file, such as word2vec, GLOVE



def return_embeddings(embedding, vocabulary_size, embedding_dim, worddicts):

    word_embeddings = np.zeros((vocabulary_size, dim_word))
    with open(embedding, 'r') as f:
        for line in f:
            words=line.split()
            word = words[0]
            vector = words[1:]
            len_vec = len(vector)
            if(len_vec>300):
                diff = len_vec-300
                word = word.join(vector[:diff])
                vector = vector[diff:]
            if word in worddicts and worddicts[word] < vocabulary_size:
                vector = [float(x) for x in vector]
                word_embeddings[worddicts[word], :] = vector[0:300]
    return word_embeddings
### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
logging.basicConfig(filename='log' + str(time.time()) + '.log', level=logging.DEBUG)

print 'Loading data'
train_set = TextIterator('word_sequence/premise_multinli_1.0_train.txt',
                         'word_sequence/hypothesis_multinli_1.0_train.txt',
                         'word_sequence/label_multinli_1.0_train.txt',
                         'word_sequence/vocab_cased.pkl',
                         n_words=n_words,
                         batch_size=batch_size)
valid_set = TextIterator('word_sequence/premise_multinli_1.0_dev_matched.txt',
                         'word_sequence/hypothesis_multinli_1.0_dev_matched.txt',
                         'word_sequence/label_multinli_1.0_dev_matched.txt',
                         'word_sequence/vocab_cased.pkl',
                         n_words=n_words,
                         batch_size=valid_batch_size,
                         shuffle=False)

test_set = TextIterator('word_sequence/premise_multinli_1.0_dev_mismatched.txt',
                        'word_sequence/hypothesis_multinli_1.0_dev_mismatched.txt',
                        'word_sequence/label_multinli_1.0_dev_mismatched.txt',
                        'word_sequence/vocab_cased.pkl',
                        n_words=n_words,
                        batch_size=valid_batch_size,
                        shuffle=False)
with open('word_sequence/vocab_cased.pkl', 'rb') as f:
    worddicts = pkl.load(f)
emb = return_embeddings(embedding=embedding, vocabulary_size=n_words, embedding_dim=dim_word, worddicts=worddicts)

from model import ESIM


def checkpoint(epoch):
    model_out_path = "modelt_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    logger.debug("Checkpoint saved to {}".format(model_out_path))


def checkpoint_valid(epoch):
    model_out_path = "model_valid_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    logger.debug("Checkpoint saved to {}".format(model_out_path))


if reload_ and os.path.exists(saveto):
    print 'Reload options'
    with open('%s' % saveto, 'rb') as f:
        model = torch.load(saveto)
else:
    model = ESIM(batch_size=batch_size, dim_hidden=dim, embedding_dim=dim_word, embeddings=emb, vocab_size=n_words).cuda()
def pred_acc(iterator):
    model.eval()
    valid_acc = 0
    n_done = 0
    num_times = 10

    for x1, x2, y in iterator:
        n_done += len(x1)
        tp1 = time.time()
        premise, premise_mask, hypothesis, hypothesis_mask, l = prepare_data(x1, x2, y, maxlen)
        tp2 = time.time()

        outputs = model(premise, premise_mask, hypothesis, hypothesis_mask, l)
        valid_acc += (outputs.max(1)[1] == l).sum().data[0]
    valid_acc = 1.0 * valid_acc / n_done
    return valid_acc

print 'Optimization'
optimizer = optim.Adam(model.parameters(), lr=lrate)
ce_loss = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    batch_idx=0
    for p, h, l in train_set:
        x1, x1_mask, x2, x2_mask, y = prepare_data(p, h, l, maxlen=maxlen)
        optimizer.zero_grad()
        outputs = model(x1, x1_mask, x2, x2_mask, y)
        loss = ce_loss(outputs, y)
        loss.backward()
        if clip_c > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip_c)
        optimizer.step()
        if batch_idx % dispFreq == 0:
            logger.debug('Train Epoch: {}\tIteration: {}\tLoss: {:.6f}'.format(
                epoch, batch_idx, loss.data[0]))
        batch_idx+=1



best_acc=0.
bad_ctr=0
for epoch in range(max_epochs):
    train(epoch)
    acc=pred_acc(valid_set)
    logger.debug('Epoch ' + str(epoch) + ' Valid Accuracy = ' + str(acc))
    if acc>best_acc:
        best_acc=acc
        checkpoint_valid(epoch)
    else:
        bad_ctr+=1
    if bad_ctr==patience:
        lrate*=0.5
        optimizer = optim.Adam(model.parameters(), lr=lrate)
        bad_ctr=0
    if epoch % saveFreq ==0:
        checkpoint(epoch)
