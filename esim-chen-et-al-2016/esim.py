# coding: utf-8

# In[1]:


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

# In[2]:


def return_embeddings(embedding, vocabulary_size, embedding_dim, worddicts):
    loaded_embeddings = np.zeros((vocabulary_size, embedding_dim))
    with open(embedding, 'r') as f:
        for line in f:
            tmp = line.split()
            word = tmp[0]
            vector = tmp[1:]
            if word in worddicts and worddicts[word] < vocabulary_size:
                loaded_embeddings[worddicts[word], :] = vector
    return loaded_embeddings


# In[3]:


logger = logging.getLogger(__name__)


# In[4]:


def train(
    dim_word=300,  # word vector dimensionality
    dim=100,  # the number of GRU units
    encoder='lstm',  # encoder model
    decoder='lstm',  # decoder model
    patience=10,  # early stopping patience
    max_epochs=10,
    finish_after=10000000,  # finish after this many updates
    decay_c=0.,  # L2 regularization penalty
    clip_c=10.0,  # gradient clipping threshold
    lrate=0.01,  # learning rate
    n_words=100000,  # vocabulary size
    maxlen=100,  # maximum length of the description
    optimizer_spec='adadelta',
    batch_size=16,
    valid_batch_size=16,
    saveto='model.pth',
    dispFreq=100,
    validFreq=1000,
    saveFreq=1000,  # save the parameters after every saveFreq updates
    use_dropout=False,
    reload_=False,
    verbose=False,  # print verbose information for debug but slow speed
    datasets=[],
    valid_datasets=[],
    test_datasets=[],
    dictionary='',
    embedding='data/glove.840B.300d.txt',  # pretrain embedding file, such as word2vec, GLOVE
):
    # logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
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
    worddicts_r = dict()
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk
    emb = return_embeddings(embedding=embedding, vocabulary_size=n_words, embedding_dim=dim_word, worddicts=worddicts_r)

    def checkpoint(epoch):
        model_out_path = "modelt_epoch_{}.pth".format(epoch)
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
    def checkpoint_valid(epoch):
        model_out_path = "model_valid_{}.pth".format(epoch)
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
    if reload_ and os.path.exists(saveto):
        print 'Reload options'
        with open('%s' % saveto, 'rb') as f:
            model = torch.load(saveto)
    else:
        model = ESIM(batch_size=batch_size, dim_hidden=dim, embedding_dim=dim_word, embeddings=emb, vocab_size=n_words)

    # logger.debug((model_options))

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
    if optimizer_spec == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    else:
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lrate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=6)
    ce_loss = nn.CrossEntropyLoss()
    history_errs = []
    # reload history

    bad_counter = 0

    if validFreq == -1:
        validFreq = len(train_set[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train_set[0]) / batch_size

    uidx = 0
    estop = False
    prev_losses = []
    best_epoch_num = 0
    lr_change_list = []
    wait_counter = 0
    wait_N = 6
    best_acc = 0.
    fc1=[]
    fc2=[]
    fc_op=[]
    le=[]
    ler=[]
    ld=[]
    ldr=[]
    for eidx in xrange(max_epochs):
        n_samples = 0
        total_epoch_loss = 0
        for p, h, l in train_set:
            model.train()
            n_samples += len(p)

            uidx += 1
            x1, x1_mask, x2, x2_mask, y = prepare_data(p, h, l, maxlen=maxlen)
            '''if x1 is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue'''
            model.zero_grad()
            optimizer.zero_grad()
            ud_start = time.time()
            outputs = model(x1, x1_mask, x2, x2_mask, y)
            loss = ce_loss(outputs, y)
            total_epoch_loss += loss.data[0]
            loss.backward()
            if clip_c > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip_c)
            optimizer.step()
            print(optimizer.state_dict())
            '''if len(fc1)==0:
                fc1=model.fc1.weight.data.cpu().numpy()
                fc2 = model.fc2.weight.data.cpu().numpy()
                fc_op= model.fc_op.weight.data.cpu().numpy()
                le = model.LSTM_encoder.weight_ih_l0.data.cpu().numpy()
                ler = model.LSTM_encoder_rev.weight_ih_l0.data.cpu().numpy()
                ld = model.LSTM_decoder.weight_ih_l0.data.cpu().numpy()
                ldr = model.LSTM_decoder_rev.weight_ih_l0.data.cpu().numpy()
            else:
                d=fc1-model.fc1.weight.data.cpu().numpy()
                logger.debug("change in fc1 max min: "+str(np.max(d.flatten()))+str(np.min(d.flatten())))
                fc1=model.fc1.weight.data.cpu().numpy()
                d = fc2 - model.fc2.weight.data.cpu().numpy()
                logger.debug("change in fc2 max min: " +str(np.max(d.flatten()))+str(np.min(d.flatten())))
                fc2=model.fc2.weight.data.cpu().numpy()
                d = fc_op - model.fc_op.weight.data.cpu().numpy()
                logger.debug("change in fc_op max min: "+str(np.max(d.flatten()))+str(np.min(d.flatten())))
                fc_op=model.fc_op.weight.data.cpu().numpy()
                d = le - model.LSTM_encoder.weight_ih_l0.data.cpu().numpy()
                logger.debug("change in le max min: " +str(np.max(d.flatten()))+str(np.min(d.flatten())))
                le = model.LSTM_encoder.weight_ih_l0.data.cpu().numpy()
                d = ler - model.LSTM_encoder_rev.weight_ih_l0.data.cpu().numpy()
                logger.debug("change in ler max min: " +str(np.max(d.flatten()))+str(np.min(d.flatten())))
                ler = model.LSTM_encoder_rev.weight_ih_l0.data.cpu().numpy()
                d = ld - model.LSTM_decoder.weight_ih_l0.data.cpu().numpy()
                logger.debug("change in ld max min: " +str(np.max(d.flatten()))+str(np.min(d.flatten())))
                ld = model.LSTM_decoder.weight_ih_l0.data.cpu().numpy()
                d = ldr - model.LSTM_decoder_rev.weight_ih_l0.data.cpu().numpy()
                logger.debug("change in ldr max min: " +str(np.max(d.flatten()))+str(np.min(d.flatten())))
                ldr = model.LSTM_decoder_rev.weight_ih_l0.data.cpu().numpy()'''

            ud = time.time() - ud_start
        if eidx % saveFreq == 0:
            logger.debug("Saving at iteration: " + str(uidx))
            checkpoint(eidx)
        '''scheduler.step(total_epoch_loss)
        if(total_epoch_loss >= prev_losses[-wait_N:].max()):
            #lrate*=0.5
            bad_counter+=1
            if(bad_counter==patience):
                logger.debug("Early stopping!!")
                break'''
        logger.debug("Epoch: " + str(eidx) + " Loss: " + str(total_epoch_loss))
        if (len(prev_losses[-patience:]) > patience):
            if (total_epoch_loss >= max(prev_losses[-patience:])):
                logger.debug("Early stopping!!")
                break
        prev_losses.append(total_epoch_loss)
        t_acc = pred_acc(train_set)
        logger.debug('Epoch ' + str(eidx) + ' Train Accuracy = ' + str(t_acc))
        valid_acc = pred_acc(valid_set)
        logger.debug('Epoch ' + str(eidx) + ' Valid Accuracy = ' + str(valid_acc))
        test_acc = pred_acc(test_set)
        logger.debug('Epoch ' + str(eidx) + ' Test Accuracy = ' + str(test_acc))
        if (valid_acc > best_acc):
            best_acc = valid_acc
            checkpoint_valid(eidx)

        # logger.debug("Updating epoch "+str(eidx)+" with loss "+str(total_epoch_loss))
    '''with open('record.csv', 'w') as f:
        f.write(str(best_epoch_num) + '\n')
        f.write(','.join(map(str,lr_change_list)) + '\n')
        f.write(','.join(map(str,valid_acc_record)) + '\n')'''


# In[6]:


train(

    reload_=True,
    dim_word=300,
    dim=300,
    patience=7,
    n_words=42394,
    decay_c=0.,
    clip_c=10.,
    lrate=0.0004,
    optimizer_spec='adam',
    maxlen=100,
    batch_size=32,
    valid_batch_size=32,
    dispFreq=100,
    validFreq=int(549367 / 32 + 1),
    saveFreq=10,
    use_dropout=True,
    verbose=False,
)

