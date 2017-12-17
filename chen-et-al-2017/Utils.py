import torch
import os
from torch.utils.data import DataLoader
from Datasets import NLIDataset
from Datasets import BatchedNLIDataset
from Settings import batch_processing
from Settings import maxlen
from Settings import num_words
import pickle as pkl


with open('word_sequence/vocab_cased.pkl', 'rb') as f:
    worddicts = pkl.load(f)
worddicts_r = dict()

for kk, vv in worddicts.items():
    worddicts_r[vv] = kk

def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(resume):
    print(resume)
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
        return checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(resume))
        return None


def read_multinli(path):
    premise = []
    hypo = []
    labels = []
    dic = {'entailment': '0', 'neutral': '1', 'contradiction': '2'}

    with open(path) as file_data:
        file_data.readline()
        for line in file_data:

            sents = line.strip().split('\t')
            if sents[0] is '-':
                continue

            words_in = sents[1].strip().split(' ')
            words_in_premise = [x for x in words_in if x not in ('(', ')')]
            premise.append(words_in_premise)

            words_in = sents[2].strip().split(' ')
            words_in_hypo = [x for x in words_in if x not in ('(', ')')]
            hypo.append(words_in_hypo)

            label = dic[sents[0]]
            labels.append(label)
    return premise, hypo, labels


def read_multinliForBatchProcessing(path):
    dic = {'entailment': '0', 'neutral': '1', 'contradiction': '2'}
    done = {}
    useful_data = []

    with open(path) as file_data:
        file_data.readline()
        for line in file_data:

            sents = line.strip().split('\t')
            if sents[0] is '-':
                continue

            sents = line.strip().split('\t')
            if sents[0] is '-':
                continue

            words_in = sents[1].strip().split(' ')
            words_in_premise = [x for x in words_in if x not in ('(', ')')]  # premise list

            words_in = sents[2].strip().split(' ')
            words_in_hypo = [x for x in words_in if x not in ('(', ')')]  # hypo list

            label = dic[sents[0].strip()]  # label
            prompt_id = sents[7].strip()  # data identifier

            if prompt_id not in done:
                done[prompt_id] = len(useful_data)
                useful_data.append([])

            t = (words_in_premise, words_in_hypo, label)
            useful_data[done[prompt_id]].append(t)

    return useful_data


def formatDataToIndexRepresentation(source_buffer, target_buffer, label_buffer):
    source = []
    target = []
    for ss in source_buffer:
        ss.insert(0, '_BOS_')
        ss.append('_EOS_')
        ss = [worddicts[w] if w in worddicts else 1
              for w in ss]
        if num_words > 0:
            ss = [w if w < num_words else 1 for w in ss]
        source.append(ss)

    for tt in target_buffer:
        tt.insert(0, '_BOS_')
        tt.append('_EOS_')
        tt = [worddicts[w] if w in worddicts else 1
              for w in tt]
        if num_words > 0:
            tt = [w if w < num_words else 1 for w in tt]
        target.append(tt)

    return source, target, label_buffer


def formatDataToIndexRepresentationForBatchProcessing(data):
    new_data = []
    for dataPoint in data:  # [(p1,h1,l1),(p2,h2,l2)]
        new_list = []
        for recordTuple in dataPoint:  # (pi,hi,li)
            ss = recordTuple[0]
            ss.insert(0, '_BOS_')
            ss.append('_EOS_')
            ss = [worddicts[w] if w in worddicts else 1
                  for w in ss]
            if num_words > 0:
                ss = [w if w < num_words else 1 for w in ss]

            tt = recordTuple[1]
            tt.insert(0, '_BOS_')
            tt.append('_EOS_')
            tt = [worddicts[w] if w in worddicts else 1
                  for w in tt]
            if num_words > 0:
                tt = [w if w < num_words else 1 for w in tt]
            label = recordTuple[2]

            new_list.append((ss, tt, label))
        new_data.append(new_list)

    return new_data


def str2list(s):
    alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    l = len(s)
    ans = []
    for i in range(0, l):
        a = alphabet.find(s[i])
        if a >= 0:
            ans.append(a)
        else:
            ans.append(0)
    return ans


def prepare_data(batch):
    seqs_x = []
    seqs_y = []
    labels = []

    if batch_processing:
        for datum in batch:
            premise_list = datum[0]
            hyp_list = datum[1]
            labels_list = datum[2]
            for pre, hyp, lab in zip(premise_list, hyp_list, labels_list):
                seqs_x.append(pre)
                seqs_y.append(hyp)
                labels.append(lab)
    else:
        for datum in batch:
            seqs_x.append(datum[0])
            seqs_y.append(datum[1])
            labels.append(datum[2])

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
            return None

    max_char_len_x = 0
    max_char_len_y = 0
    seqs_x_char = []
    l_seqs_x_char = []
    seqs_y_char = []
    l_seqs_y_char = []

    for idx, [s_x, s_y, s_l] in enumerate(zip(seqs_x, seqs_y, labels)):
        temp_seqs_x_char = []
        temp_l_seqs_x_char = []
        temp_seqs_y_char = []
        temp_l_seqs_y_char = []
        for w_x in s_x:
            word = worddicts_r[w_x]
            word_list = str2list(word)
            l_word_list = len(word_list)
            temp_seqs_x_char.append(word_list)
            temp_l_seqs_x_char.append(l_word_list)
            if l_word_list >= max_char_len_x:
                max_char_len_x = l_word_list
        for w_y in s_y:
            word = worddicts_r[w_y]
            word_list = str2list(word)
            l_word_list = len(word_list)
            temp_seqs_y_char.append(word_list)
            temp_l_seqs_y_char.append(l_word_list)
            if l_word_list >= max_char_len_y:
                max_char_len_y = l_word_list

        seqs_x_char.append(temp_seqs_x_char)
        l_seqs_x_char.append(temp_l_seqs_x_char)
        seqs_y_char.append(temp_seqs_y_char)
        l_seqs_y_char.append(temp_l_seqs_y_char)

    n_samples = len(seqs_x)
    maxlen_x = max(lengths_x)
    maxlen_y = max(lengths_y)
    if torch.cuda.is_available():
        x = torch.zeros(maxlen_x, n_samples).long().cuda()
        y = torch.zeros(maxlen_y, n_samples).long().cuda()
        x_mask = torch.zeros(maxlen_x, n_samples).cuda()
        y_mask = torch.zeros(maxlen_y, n_samples).cuda()
        l = torch.zeros(n_samples, ).long().cuda()
        char_x = torch.zeros(maxlen_x, n_samples, max_char_len_x).long().cuda()
        char_x_mask = torch.zeros(maxlen_x, n_samples, max_char_len_x).cuda()
        char_y = torch.zeros(maxlen_y, n_samples, max_char_len_y).long().cuda()
        char_y_mask = torch.zeros(maxlen_y, n_samples, max_char_len_y).cuda()
    else:
        x = torch.zeros(maxlen_x, n_samples).long()
        y = torch.zeros(maxlen_y, n_samples).long()
        x_mask = torch.zeros(maxlen_x, n_samples)
        y_mask = torch.zeros(maxlen_y, n_samples)
        l = torch.zeros(n_samples, ).long()
        char_x = torch.zeros(maxlen_x, n_samples, max_char_len_x).long()
        char_x_mask = torch.zeros(maxlen_x, n_samples, max_char_len_x)
        char_y = torch.zeros(maxlen_y, n_samples, max_char_len_y).long()
        char_y_mask = torch.zeros(maxlen_y, n_samples, max_char_len_y)

    for idx, [s_x, s_y, ll] in enumerate(zip(seqs_x, seqs_y, labels)):
        if torch.cuda.is_available():
            x[:lengths_x[idx], idx] = torch.Tensor(s_x).cuda()
            x_mask[:lengths_x[idx], idx] = 1.
            y[:lengths_y[idx], idx] = torch.Tensor(s_y).cuda()
            y_mask[:lengths_y[idx], idx] = 1.
            l[idx] = int(ll)

            for j in range(0, lengths_x[idx]):
                char_x[j, idx, :l_seqs_x_char[idx][j]] = torch.Tensor(seqs_x_char[idx][j]).cuda()
                char_x_mask[j, idx, :l_seqs_x_char[idx][j]] = 1.
            for j in range(0, lengths_y[idx]):
                char_y[j, idx, :l_seqs_y_char[idx][j]] = torch.Tensor(seqs_y_char[idx][j]).cuda()
                char_y_mask[j, idx, :l_seqs_y_char[idx][j]] = 1.
        else:
            x[:lengths_x[idx], idx] = torch.Tensor(s_x)
            x_mask[:lengths_x[idx], idx] = 1.
            y[:lengths_y[idx], idx] = torch.Tensor(s_y)
            y_mask[:lengths_y[idx], idx] = 1.
            l[idx] = int(ll)

            for j in range(0, lengths_x[idx]):
                char_x[j, idx, :l_seqs_x_char[idx][j]] = torch.Tensor(seqs_x_char[idx][j])
                char_x_mask[j, idx, :l_seqs_x_char[idx][j]] = 1.
            for j in range(0, lengths_y[idx]):
                char_y[j, idx, :l_seqs_y_char[idx][j]] = torch.Tensor(seqs_y_char[idx][j])
                char_y_mask[j, idx, :l_seqs_y_char[idx][j]] = 1.

    return [x, x_mask, char_x, char_x_mask, y, y_mask, char_y, char_y_mask, l]


def load_data(train_loc, dev_loc, test_loc, batch_size):
    train_premise_text, train_hypo_text, train_labels = read_multinli(train_loc)
    dev_premise_text, dev_hypo_text, dev_labels = read_multinli(dev_loc)
    test_premise_text, test_hypo_text, test_labels = read_multinli(test_loc)

    train_premise, train_hypo, train_labels = formatDataToIndexRepresentation(train_premise_text, train_hypo_text,
                                                                              train_labels)
    dev_premise, dev_hypo, dev_labels = formatDataToIndexRepresentation(dev_premise_text, dev_hypo_text,
                                                                        dev_labels)
    test_premise, test_hypo, test_labels = formatDataToIndexRepresentation(test_premise_text, test_hypo_text,
                                                                           test_labels)

    nli_train = NLIDataset((train_premise, train_hypo), train_labels)
    nli_dev = NLIDataset((dev_premise, dev_hypo), dev_labels)
    nli_test = NLIDataset((test_premise, test_hypo), test_labels)

    train_loader = DataLoader(dataset=nli_train,
                              shuffle=True,
                              collate_fn=prepare_data,
                              batch_size=batch_size)
    dev_loader = DataLoader(dataset=nli_dev,
                            shuffle=False,
                            collate_fn=prepare_data,
                            batch_size=batch_size)
    test_loader = DataLoader(dataset=nli_test,
                             shuffle=False,
                             collate_fn=prepare_data,
                             batch_size=batch_size)
    return train_loader, dev_loader, test_loader


def load_data_for_batch_processing(train_loc, dev_loc, test_loc, new_batch_size):
    train_data = read_multinliForBatchProcessing(train_loc)
    dev_data = read_multinliForBatchProcessing(dev_loc)
    test_data = read_multinliForBatchProcessing(test_loc)

    train_indexed_data = formatDataToIndexRepresentationForBatchProcessing(train_data)
    dev_indexed_data = formatDataToIndexRepresentationForBatchProcessing(dev_data)
    test_indexed_data = formatDataToIndexRepresentationForBatchProcessing(test_data)

    nli_train = BatchedNLIDataset(train_indexed_data)
    nli_dev = BatchedNLIDataset(dev_indexed_data)
    nli_test = BatchedNLIDataset(test_indexed_data)

    train_loader = DataLoader(dataset=nli_train,
                              shuffle=True,
                              collate_fn=prepare_data,
                              batch_size=new_batch_size)
    dev_loader = DataLoader(dataset=nli_dev,
                            shuffle=False,
                            collate_fn=prepare_data,
                            batch_size=new_batch_size)
    test_loader = DataLoader(dataset=nli_test,
                             shuffle=False,
                             collate_fn=prepare_data,
                             batch_size=new_batch_size)
    return train_loader, dev_loader, test_loader
