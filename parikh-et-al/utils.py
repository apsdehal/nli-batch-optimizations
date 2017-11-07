import numpy as np
import jsonlines
import json
import torch
import os
import shutil

LABELS = {'entailment': 0, 'contradiction': 1, 'neutral': 2}


def read_multinli(path):
    premise = []
    hypo = []
    labels = []

    with jsonlines.open(path) as file_:
        for eg in file_:
            label = eg['gold_label']
            if label == '-':
                continue
            premise.append(eg['sentence1'])
            hypo.append(eg['sentence2'])
            labels.append(LABELS[label])
    return premise, hypo, labels


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    epoch = state['epoch']
    if epoch % 50 == 0 and epoch != 0:
        shutil.copyfile(filename, 'model_' + str(epoch) + '.pth.tar')
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
