import spacy
import torch

from pathlib import Path
from utils import read_multinli
from torch.autograd import Variable

from decomposable_attention import build_model
from spacy_hook import get_embeddings, get_word_ids


DATA_PATH = '../data/sample.jsonl'


def main():
    sample_path = Path.cwd() / DATA_PATH

    sample_premise, sample_hypo, sample_labels = read_multinli(sample_path)
    nlp = spacy.load('en')

    shape = (20, 300, 3)
    settings = {
        'lr': 0.001,
        'batch_size': 3,
        'dropout': 0.2,
        'nr_epoch': 5,
        'tree_truncate': False,
        'gru_encode': False
    }
    model = build_model(get_embeddings(nlp.vocab), shape, settings)

    data = []
    for texts in (sample_premise, sample_hypo):
        data.append(get_word_ids(
                    list(nlp.pipe(texts, n_threads=1, batch_size=3)),
                    max_length=20,
                    rnn_encode=False,
                    tree_truncate=False))

    model(Variable(torch.from_numpy(data[0]).long()),
          Variable(torch.from_numpy(data[1]).long()))


if __name__ == '__main__':
    main()
