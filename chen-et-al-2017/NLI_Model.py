from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from LSTM import LSTM

class NLI(nn.Module):
    def __init__(self, dim_word, char_nout, dim_char_emb, word_embeddings_file, worddict, num_words, dim_hidden):
        super(NLI, self).__init__()
        self.dim_word = dim_word
        self.char_nout = char_nout
        self.dim_char_emb = dim_char_emb
        self.char_k_cols = dim_char_emb
        self.char_k_rows = [1, 3, 5]
        self.hidden_size = dim_hidden
        self.word_embeddings = self.create_word_embeddings(word_embeddings_file, worddict, num_words, dim_word)

        dim_emb = dim_word + 3 * char_nout
        self.alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

        self.LSTM1 = LSTM(dim_emb, dim_hidden)
        self.LSTM1_rev = LSTM(dim_emb, dim_hidden)
        self.LSTM2 = LSTM(dim_emb + 2 * dim_hidden, dim_hidden)
        self.LSTM2_rev = LSTM(dim_emb + 2 * dim_hidden, dim_hidden)
        self.LSTM3 = LSTM(dim_emb + 2 * dim_hidden, dim_hidden)
        self.LSTM3_rev = LSTM(dim_emb + 2 * dim_hidden, dim_hidden)

        if torch.cuda.is_available():
            print('CUDA available')
            self.drop = nn.Dropout(p=0.1).cuda()
            self.conv1 = nn.Conv2d(1, self.char_nout, (self.char_k_rows[0], self.char_k_cols)).cuda()
            self.conv2 = nn.Conv2d(1, self.char_nout, (self.char_k_rows[1], self.char_k_cols)).cuda()
            self.conv3 = nn.Conv2d(1, self.char_nout, (self.char_k_rows[2], self.char_k_cols)).cuda()
            self.Charemb = nn.Embedding(len(self.alphabet) + 1, self.dim_char_emb, padding_idx=0).cuda()
            self.Linear1 = nn.Linear(24 * dim_hidden, dim_hidden).cuda()
            self.Linear2 = nn.Linear(25 * dim_hidden, dim_hidden).cuda()
            self.Linear3 = nn.Linear(dim_hidden, 3).cuda()
        else:
            self.Charemb = nn.Embedding(len(self.alphabet) + 1, self.dim_char_emb, padding_idx=0)
            self.Linear1 = nn.Linear(24 * dim_hidden, dim_hidden)
            self.Linear2 = nn.Linear(25 * dim_hidden, dim_hidden)
            self.Linear3 = nn.Linear(dim_hidden, 3)
            self.drop = nn.Dropout(p=0.1)
            self.conv1 = nn.Conv2d(1, self.char_nout, (self.char_k_rows[0], self.char_k_cols))
            self.conv2 = nn.Conv2d(1, self.char_nout, (self.char_k_rows[1], self.char_k_cols))
            self.conv3 = nn.Conv2d(1, self.char_nout, (self.char_k_rows[2], self.char_k_cols))

        self.init_weights(dim_hidden)

    def create_word_embeddings(self, file_name, worddicts, num_words, dim_word):
        if torch.cuda.is_available():
            word_embeddings = Variable(torch.zeros(num_words, dim_word).cuda())
        else:
            word_embeddings = Variable(torch.zeros(num_words, dim_word))
        word_embeddings.data.normal_(0, 0.01)
        with open(file_name, 'r') as f:
            for line in f:
                tmp = line.split()
                word = tmp[0]
                vector = tmp[1:]
                len_vec = len(vector)

                if (len_vec > 300):
                    diff = len_vec - 300
                    word = word.join(vector[:diff])
                    vector = vector[diff:]

                if word in worddicts and worddicts[word] < num_words:
                    vector = [float(x) for x in vector]
                    if torch.cuda.is_available():
                        word_embeddings[worddicts[word], :] = torch.FloatTensor(vector[0:300]).cuda()
                    else:
                        word_embeddings[worddicts[word], :] = torch.FloatTensor(vector[0:300])

        return word_embeddings

    def forward(self, premise, char_premise, premise_mask, char_premise_mask, hypothesis, char_hypothesis,
                hypothesis_mask, char_hypothesis_mask, l):
        # premise = number of words * number of samples. Also hypothesis = number of words * number of samples
        n_timesteps_premise = premise.size(0)
        n_timesteps_hypothesis = hypothesis.size(0)
        n_samples = premise.size(1)

        premise_char_vector = self.compute_character_embeddings(char_premise, n_timesteps_premise, n_samples,
                                                                char_premise_mask)
        hypothesis_char_vector = self.compute_character_embeddings(char_hypothesis, n_timesteps_hypothesis, n_samples,
                                                                   char_hypothesis_mask)

        premise_word_emb = self.word_embeddings[premise.view(-1)].view(n_timesteps_premise, n_samples, self.dim_word)
        hypothesis_word_emb = self.word_embeddings[hypothesis.view(-1)].view(n_timesteps_hypothesis, n_samples,
                                                                             self.dim_word)

        hypothesis_emb = torch.cat((hypothesis_word_emb, hypothesis_char_vector), 2)
        hypothesis_emb = self.drop(hypothesis_emb)

        premise_emb = torch.cat((premise_word_emb, premise_char_vector), 2)
        premise_emb = self.drop(premise_emb)

        premise_seq, premise_rev_seq = self.sequence_encoder(premise_emb, premise_mask)
        hypothesis_seq, hypothesis_rev_seq = self.sequence_encoder(hypothesis_emb, hypothesis_mask)

        premise_comp = self.make_composite_vector(premise_seq, premise_rev_seq, premise_mask)
        hypothesis_comp = self.make_composite_vector(hypothesis_seq, hypothesis_rev_seq, hypothesis_mask)

        logit_0 = torch.cat(
            (premise_comp, hypothesis_comp, torch.abs(premise_comp - hypothesis_comp), premise_comp * hypothesis_comp),
            1)
        logit = F.relu(self.Linear1(logit_0))
        logit = self.drop(logit)

        logit = torch.cat((logit_0, logit), 1)

        logit = F.relu(self.Linear2(logit))
        logit = self.drop(logit)

        logit = self.Linear3(logit)

        return logit

    def sequence_encoder(self, emb, mask):
        reverse_emb = self.reverseTensor(emb)
        reverse_mask = self.reverseTensor(mask)

        #  LSTM1
        seq1 = self.LSTM1(emb, mask)
        seq_reverse1 = self.LSTM1_rev(reverse_emb, reverse_mask)

        inp_seq2 = torch.cat((seq1[0], self.reverseTensor(seq_reverse1[0])), len(seq1[0].size()) - 1)
        inp_seq2 = torch.cat((inp_seq2, emb), 2)
        reverse_inp_seq2 = self.reverseTensor(inp_seq2)

        #  LSTM2
        seq2 = self.LSTM2(inp_seq2, mask)
        seq_reverse2 = self.LSTM2_rev(reverse_inp_seq2, reverse_mask)

        inp_seq3 = torch.cat((seq2[0], self.reverseTensor(seq_reverse2[0])), len(seq2[0].size()) - 1)
        inp_seq3 = torch.cat((inp_seq3, emb), 2)
        reverse_inp_seq3 = self.reverseTensor(inp_seq3)

        #  LSTM3
        seq3 = self.LSTM3(inp_seq3, mask)
        seq_reverse3 = self.LSTM3_rev(reverse_inp_seq3, reverse_mask)

        return seq3, seq_reverse3

    def make_composite_vector(self, seq, seq_rev, mask):
        output = torch.cat((seq[0], self.reverseTensor(seq_rev[0])), len(seq[0].size()) - 1)

        gate = torch.cat((seq[1], self.reverseTensor(seq_rev[1])), len(seq[1].size()) - 1)
        gate = gate.norm(2, 2)
        output_masked = output * mask[:, :, None]
        gate_masked = gate[:, :, None] * mask[:, :, None]

        mean = (output_masked).sum(0) / (mask.sum(0)[:, None])
        maxi = (output_masked).max(0)[0]
        gate_2 = (output * gate_masked).sum(0) / ((gate_masked).sum(0))
        rep = torch.cat((mean, maxi, gate_2), 1)
        return rep

    def reverseTensor(self, tensor):
        idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
        if torch.cuda.is_available():
            idx = Variable(torch.LongTensor(idx).cuda())
        else:
            idx = Variable(torch.LongTensor(idx))
        inverted_tensor = tensor.index_select(0, idx)
        return inverted_tensor

    def compute_character_embeddings(self, chars_word, n_timesteps, num_samples, char_mask):
        emb_char = self.Charemb(chars_word.view(-1)).view(n_timesteps, num_samples, chars_word.size(2),
                                                          self.dim_char_emb)
        emb_char = emb_char * char_mask[:, :, :, None]
        emb_char_inp = emb_char.view(n_timesteps * num_samples, 1, chars_word.size(2), self.dim_char_emb)

        char_level_emb1 = self.apply_filter_and_get_char_embedding(self.conv1, emb_char_inp, num_samples, n_timesteps)
        char_level_emb2 = self.apply_filter_and_get_char_embedding(self.conv2, emb_char_inp, num_samples, n_timesteps)
        char_level_emb3 = self.apply_filter_and_get_char_embedding(self.conv3, emb_char_inp, num_samples, n_timesteps)

        emb_chars = [char_level_emb1, char_level_emb2, char_level_emb3]
        emb_char = torch.cat(emb_chars, 2)
        return emb_char

    def apply_filter_and_get_char_embedding(self, conv, emb_char_inp, n_samples, n_timesteps):
        emb_char = conv(emb_char_inp)
        emb_char = F.relu(emb_char)
        emb_char = emb_char.view(n_timesteps * n_samples, self.char_nout, emb_char.size(2))
        emb_char = emb_char.max(2)[0]
        emb_char = emb_char.view(n_timesteps, n_samples, self.char_nout)
        return emb_char

    def init_weights(self, dim):
        initrange = 0.1
        self.init_convs(self.conv1, self.char_k_rows[0])
        self.init_convs(self.conv2, self.char_k_rows[1])
        self.init_convs(self.conv3, self.char_k_rows[2])

        self.Charemb.weight.data.uniform_(-initrange, initrange)
        self.Linear1.weight.data.uniform_(-initrange, initrange)
        self.Linear1.bias.data.fill_(0)
        self.Linear2.weight.data.uniform_(-initrange, initrange)
        self.Linear2.bias.data.fill_(0)
        self.Linear3.weight.data.uniform_(-initrange, initrange)
        self.Linear3.bias.data.fill_(0)

    def init_convs(self, conv, char_k_row):
        w_bound = math.sqrt(3 * char_k_row * self.char_k_cols)
        conv.weight.data.uniform_(-1.0 / w_bound, 1.0 / w_bound)
        conv.bias.data.fill_(0)
