import numpy as np
import jsonlines
import json

LABELS = {'entailment': 0, 'contradiction': 1, 'neutral': 2}


# def to_categorical(y, num_classes):
#     """ 1-hot encodes a tensor """
#     return np.eye(num_classes, dtype='uint8')[y]


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
    # return premise, hypo, to_categorical(np.asarray(labels, dtype='int32'), 3)
    return premise, hypo, labels
