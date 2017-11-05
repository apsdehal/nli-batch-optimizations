import plac
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import ujson as json
import logging
import spacy


from utils import read_multinli, save_checkpoint, load_checkpoint
from spacy_hook import get_embeddings, get_word_ids
from spacy_hook import create_similarity_pipeline
from config import config

from decomposable_attention import build_model


log = logging.getLogger(__name__)

cuda_available = torch.cuda.is_available()


class NLIDataset(Dataset):
    def __init__(self, data, labels):
        assert len(data[0]) == len(labels)
        assert len(data[0]) == len(data[1])
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, key):
        return self.data[0][key], self.data[1][key], self.labels[key]


def load_data(nlp, train_loc, dev_loc, shape, settings):
    data = read_multinli(train_loc)
    train_premise_text, train_hypo_text, train_labels = data
    dev_premise_text, dev_hypo_text, dev_labels = read_multinli(dev_loc)

    log.debug("Generating word vectors")
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

    log.debug("Creating dataloaders")
    train_loader = DataLoader(dataset=nli_train,
                              shuffle=True,
                              batch_size=settings['batch_size'])
    dev_loader = DataLoader(dataset=nli_train,
                            shuffle=False,
                            batch_size=settings['batch_size'])
    return train_loader, dev_loader


def train(train_loc, dev_loc, shape, settings):
    log.debug("Loading spaCy")
    nlp = spacy.load('en')
    log.debug("spaCy loaded")

    log.debug("Building model")
    model = build_model(get_embeddings(nlp.vocab), shape, settings)
    log.debug("Model built")

    train_loader, dev_loader = load_data(nlp,
                                         train_loc,
                                         dev_loc,
                                         shape,
                                         settings)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()),
                           lr=settings['lr'])

    start_epoch = 0
    best_prec = 0
    if settings['resume']:
        checkpoint = load_checkpoint(settings['resume'])
        if checkpoint is not None:
            start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    is_best = False

    if cuda_available:
        model = model.cuda()

    model.train()
    log.info("Starting training")
    for epoch in range(start_epoch, settings['num_epochs']):
        for i, (premise, hypo, labels) in enumerate(train_loader):
            premise_batch = Variable(premise.long())
            hypo_batch = Variable(hypo.long())
            label_batch = Variable(labels)

            if cuda_available:
                premise_batch = premise_batch.cuda()
                hypo_batch = hypo_batch.cuda()
                label_batch = label_batch.cuda()

            optimizer.zero_grad()
            output = model(premise_batch, hypo_batch)
            loss = criterion(output, label_batch.long())
            loss.backward()
            optimizer.step()

            if (i + 1) % (settings['batch_size'] * 4) == 0:
                train_acc = test_model(train_loader, model)
                vali_acc = test_model(dev_loader, model)
                log.info('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4},' +
                         'Train Acc: {5}, Validation Acc:{6}'
                         .format(epoch + 1,
                                 num_epochs,
                                 i + 1,
                                 len(nli_train) // settings['batch_size'],
                                 loss.data[0],
                                 train_acc,
                                 val_acc))
        val_acc = test_model(dev_loader, model)

        if val_acc > best_prec:
            best_prec = val_acc
            is_best = True

        # TODO remove embeddings from state_dict before saving
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        is_best = False


def test_model(loader, model):
    model.eval()
    correct = 0
    total = 0

    for premise, hypo, labels in loader:
        premise_batch = Variable(premise.long())
        hypo_batch = Variable(hypo.long())
        labels_batch = Variable(labels.long())

        if cuda_available:
            premise_batch = premise_batch.cuda()
            hypo_batch = hypo_batch.cuda()
            label_batch = label_batch.cuda()

        output = model(premise_batch, hypo_batch)
        total += 1
        correct += (labels_batch == output.max(1)[1]).data.numpy().sum()
    model.train()

    return correct / total * 100


def evaluate(dev_loc, shape, settings):
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)
    nlp = spacy.load('en',
                     create_pipeline=lambda nlp:
                         create_similarity_pipeline(nlp, shape, settings))

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
    num_epochs=("Number of training epochs", "option", "i", int),
    tree_truncate=("Truncate sentences by tree distance", "flag", "T", bool),
    gru_encode=("Encode sentences with bidirectional GRU", "flag", "E", bool),
    resume=("Resume training", "option", "r", Path),
)
def main(mode, train_loc, dev_loc,
         tree_truncate=False,
         gru_encode=False,
         max_length=100,
         nr_hidden=300,
         dropout=0.2,
         learn_rate=0.001,
         batch_size=100,
         num_epochs=5,
         resume=None):
    shape = (max_length, nr_hidden, 3)
    settings = {
        'lr': learn_rate,
        'dropout': dropout,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'tree_truncate': tree_truncate,
        'gru_encode': gru_encode,
        'resume': resume
    }
    if mode == 'train':
        train(train_loc, dev_loc, shape, settings)
    elif mode == 'evaluate':
        correct, total = evaluate(dev_loc, shape, settings)
        print(correct, '/', total, correct / total)
    else:
        demo()


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s',
                        level=config['LOG_LEVEL'])
    plac.call(main)
