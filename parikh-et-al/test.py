import spacy

from pathlib import Path
from utils import read_multinli

from decomposable_attention import build_model
from spacy_hook import get_embeddings, get_word_ids

DATA_PATH = '../data/sample.jsonl'


def main():
    sample_path = Path.cwd() / DATA_PATH

    sample_premise, sample_hypo, sample_labels = read_multinli(sample_path)
    nlp = spacy.load('en')

    model = build_model(get_embeddings(nlp.vocab), shape, settings)

    data = []
    for texts in (sample_premise, sample_hypo):
        data.append(get_word_ids(
                    list(nlp.pipe(texts, n_threads=1, batch_size=3)),
                    max_length=20,
                    rnn_encode=False,
                    tree_truncate=False))

    model(data[0], data[1])


if __name__ == '__main__':
    main()
