import numpy
import numpy.random
import json
from spacy.tokens.span import Span

from utils import load_checkpoint
from decomposable_attention import build_model


cuda_available = torch.cuda.is_available()


class PyTorchSimilarityShim(object):
    @classmethod
    def load(cls, path, nlp, shape, settings, get_features=None):
        if get_features is None:
            get_features = get_word_ids
        embeddings = get_embeddings(nlp.vocab)
        model = build_model(embeddings, shape, settings)
        checkpoint = load_checkpoint(settings['resume'])

        model.load_state_dict(checkpoint['state_dict'])

        return cls(model,
                   get_features=get_features,
                   max_length=settings['max_length'])

    def __init__(self, model, get_features=None, max_length=100):
        self.model = model

        if cuda_available:
            self.model = self.model.cuda()

        self.model.eval()
        self.get_features = get_features
        self.max_length = max_length

    def __call__(self, doc):
        doc.user_hooks['similarity'] = self.predict
        doc.user_span_hooks['similarity'] = self.predict

    def predict(self, doc1, doc2):
        x1 = self.get_features([doc1], max_length=self.max_length,
                               tree_truncate=True)
        x2 = self.get_features([doc2], max_length=self.max_length,
                               tree_truncate=True)

        x1 = Variable(torch.from_numpy(x1))
        x2 = Variable(torch.from_numpy(x2))

        if cuda_available:
            x1 = x1.cuda()
            x2 = x2.cuda()

        scores = self.model.predict(torch.cat([x1, x2], dim=-1))
        return scores[0]


def get_embeddings(vocab, nr_unk=100):
    nr_vector = max(lex.rank for lex in vocab) + 1
    vectors = numpy.zeros((nr_vector + nr_unk + 2,
                          vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank + 1] = lex.vector / lex.vector_norm
    return vectors


def get_word_ids(docs, rnn_encode=False, tree_truncate=False,
                 max_length=100, nr_unk=100):
    Xs = numpy.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        if tree_truncate:
            if isinstance(doc, Span):
                queue = [doc.root]
            else:
                queue = [sent.root for sent in doc.sents]
        else:
            queue = list(doc)
        words = []
        while len(words) <= max_length and queue:
            word = queue.pop(0)
            if rnn_encode or (not word.is_punct and not word.is_space):
                words.append(word)
            if tree_truncate:
                queue.extend(list(word.lefts))
                queue.extend(list(word.rights))
        words.sort()
        for j, token in enumerate(words):
            if token.has_vector:
                Xs[i, j] = token.rank + 1
            else:
                Xs[i, j] = (token.shape % (nr_unk - 1)) + 2
            j += 1
            if j >= max_length:
                break
        else:
            Xs[i, len(words)] = 1
    return Xs


def create_similarity_pipeline(nlp, shape, settings):
    return [
        nlp.tagger,
        nlp.entity,
        nlp.parser,
        PyTorchSimilarityShim.load(nlp.path / 'similarity',
                                   nlp,
                                   shape,
                                   settings)
    ]
