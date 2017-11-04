import plac
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import ujson as json
import logging
import spacy


from spacy_hook import get_embeddings, get_word_ids
from spacy_hook import create_similarity_pipeline

from decomposable_attention_pytorch import build_model
try:
    import cPickle as pickle
except ImportError:
    import pickle


log = logging.getlogger(__name__)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


class NLIDataset(Dataset):
    def __init__(self, data, labels):
        assert len(self.data[0]) == len(self.labels)
        assert len(self.data[0]) == len(self.data[1])
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return (self.data[0][key], self.data[1][key]), self.labels[key]


def train(train_loc, dev_loc, shape, settings):
    data = read_multinli(train_loc)
    train_premise_text, train_hypo_text, train_labels = data
    dev_premise_text, dev_hypo_text, dev_labels = read_multinli(dev_loc)

    log.debug("Loading spaCy")
    nlp = space.load('en')

    model = build_model(get_embeddings(nlp.vocab), shape, settings)

    data = []
    for texts in (train_premise_text, train_hypo_text, dev_premise_text,
                  dev_hypo_text):
        data.append(get_word_ids(
                    list(nlp.pipe(texts, n_threads=20, batch_size=20000)),
                    max_length=shape[0],
                    rnn_encode=settings['gru_encode'],
                    tree_truncate=settings['tree_truncate']))

    train_premise, train_hypo, dev_premise, dev_hypo = data
    nli_train = NLIDataset((train_premise, train_hypo), train_labels)
    nli_dev = NLIDataset((dev_premise, dev_hypo), dev_labels)

    train_loader = DataLoader(dataset=nli_train,
                              shuffle=True,
                              batch_size=settings['batch_size'])
    dev_loader = DataLoader(dataset=nli_train,
                            shuffle=False,
                            batch_size=settings['batch_size'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=settings['lr'])

    for epoch in range(num_epochs):
        for i, (premise, hypo, labels) in enumerate(train_loader):
            premise_batch, hypo_batch = Variable(premise), Variable(hypo)
            label_batch = Variable(labels)
            optimizer.zero_grad()
            output = model(premise_batch, hypo_batch)
            loss = criterion(output, label_batch.float())
            loss.backward()
            optimizer.step()

            if (i + 1) % (settings['batch_size'] * 4) == 0:
                train_acc = test_model(train_loader, model)
                vali_acc = test_model(val_loader, model)
                print('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4},' +
                      'Train Acc: {5}, Validation Acc:{6}'
                      .format(epoch + 1,
                              num_epochs,
                              i + 1,
                              len(nli_train) // settings['batch_size'],
                              loss.data[0],
                              train_acc,
                              val_acc))


def test_model(loader, model):
    model.eval()
    correct = 0
    total = 0

    for premise, hypo, labels in loader:
        premise_batch, hypo_batch = Variable(premise), Variable(hypo)
        label_batch = Variable(labels)
        output = model(premise_batch, hypo_batch)
        # Need to complete this later
        print(output)
    model.train()


def evaluate(dev_loc):
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)
    nlp = spacy.load('en',
                     create_pipeline=create_similarity_pipeline)
    total = 0.
    correct = 0.
    for text1, text2, label in zip(dev_texts1, dev_texts2, dev_labels):
        doc1 = nlp(text1)
        doc2 = nlp(text2)
        sim = doc1.similarity(doc2)
        if sim.argmax() == label.argmax():
            correct += 1
        total += 1
    return correct, total


def demo():
    nlp = spacy.load('en',
                     create_pipeline=create_similarity_pipeline)
    doc1 = nlp(u'What were the best crime fiction books in 2016?')
    doc2 = nlp(u'What should I read that was published last year?' +
               ' I like crime stories.')
    log.info(doc1)
    log.info(doc2)
    log.info("Similarity %f" % doc1.similarity(doc2))


LABELS = {'entailment': 0, 'contradiction': 1, 'neutral': 2}


def read_multinli(path):
    premise = []
    hypo = []
    labels = []

    with path.open() as file_:
        for line in file_:
            eg = json.loads(line)
            label = eg['gold_label']
            if label == '-':
                continue
            texts1.append(eg['sentence1'])
            texts2.append(eg['sentence2'])
            labels.append(LABELS[label])
    return premise, hypo, to_categorical(np.asarray(labels, dtype='int32'), 3)


@plac.annotations(
    mode=("Mode to execute", "positional", None, str,
          ["train", "evaluate", "demo"]),
    train_loc=("Path to training data", "positional", None, Path),
    dev_loc=("Path to development data", "positional", None, Path),
    max_length=("Length to truncate sentences", "option", "L", int),
    nr_hidden=("Number of hidden units", "option", "H", int),
    dropout=("Dropout level", "option", "d", float),
    learn_rate=("Learning rate", "option", "e", float),
    batch_size=("Batch size for neural network training", "option", "b", int),
    nr_epoch=("Number of training epochs", "option", "i", int),
    tree_truncate=("Truncate sentences by tree distance", "flag", "T", bool),
    gru_encode=("Encode sentences with bidirectional GRU", "flag", "E", bool),
)
def main(mode, train_loc, dev_loc,
         tree_truncate=False,
         gru_encode=False,
         max_length=100,
         nr_hidden=100,
         dropout=0.2,
         learn_rate=0.001,
         batch_size=100,
         nr_epoch=5):
    shape = (max_length, nr_hidden, 3)
    settings = {
        'lr': learn_rate,
        'dropout': dropout,
        'batch_size': batch_size,
        'nr_epoch': nr_epoch,
        'tree_truncate': tree_truncate,
        'gru_encode': gru_encode
    }
    if mode == 'train':
        train(train_loc, dev_loc, shape, settings)
    elif mode == 'evaluate':
        correct, total = evaluate(dev_loc)
        print(correct, '/', total, correct / total)
    else:
        demo()


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s',
                        level=config['LOG_LEVEL'])
    plac.call(main)
