import plac
import torch
import argparse
import logging


from config import config
from pathlib import Path
from modules.IO import io
from modules.Utils import utils, BatchedNLIDataset
from trainer import Trainer


def train(params, load_only=False):
    wd, embeddings = io.load_embeddings(params)

    dev_pairs = io.read_corpus(params.dev_matched_file, True)
    label_dict = utils.create_label_dict(params, dev_pairs)
    dev_data = utils.create_dataset(dev_pairs, wd, label_dict,
                                    max_len1=params.max_len,
                                    max_len2=params.max_len)

    dev_mismatched_pairs = io.read_corpus(params.dev_mismatched_file, True)
    dev_mismatched_data = utils.create_dataset(
        dev_mismatched_pairs, wd, label_dict, max_len1=params.max_len,
        max_len2=params.max_len)

    if params.use_optimizations:
        train_pairs = io.read_corpus_batched(params.train_file, True)
        train_data = BatchedNLIDataset(train_pairs, wd, params.max_len,
                                       params.max_len, label_dict)
    else:
        train_pairs = io.read_corpus(params.train_file, True)
        train_data = utils.create_dataset(
            train_pairs, wd, label_dict, max_len1=params.max_len,
            max_len2=params.max_len)

    del train_pairs
    del dev_pairs
    del dev_mismatched_pairs

    trainer = Trainer(params, train_data, dev_data,
                      dev_mismatched_data, embeddings)
    trainer.load(params.model)

    if not load_only:
        trainer.train()
    return trainer


@plac.annotations(
    train_file=("Path to training data", "positional", None, Path),
    dev_matched_file=("Path to matched development data", "positional", None,
                      Path),
    dev_mismatched_file=("Path to mismatched development data", "positional",
                         None, Path),
    max_len=("Length to truncate sentences", "option", "L", int),
    hidden_dim=("Number of hidden units", "option", "H", int),
    dropout=("Dropout level", "option", "d", float),
    lr=("Learning rate", "option", "l", float),
    batch_size=("Batch size for neural network training", "option", "b", int),
    epochs=("Number of training epochs", "option", "e", int),
    gru_encode=("Encode sentences with bidirectional GRU", "flag", "E", bool),
    extra_debug=("Whether to provide extra debugging information and" +
                 "log gradient histograms", "flag", "D", bool),
    resume=("Resume training, required path to checkpoint file", "option",
            "r", str),
    model=("Which model to train, parikh/bilstm_max", "option",
           "m", str),
    save_loc=("Save location for files", "option",
              "s", str),
    log_dir=("Log directory for extra debug gradients", "option",
             "g", str),
    seed=("Seed for training", "option", "S", int),
    embedding_dim=("Dimensions for embedding", "option", "P", int),
    patience=("Patience for early stopping", "option", "p", int),
    use_optimizations=("Whether to use batch optimizations", "flag", "O",
                       bool),
    use_intra_attention=("Whether to use intra attention in Parikh et. al.",
                         "flag", "I", bool)
)
def main(train_file, dev_matched_file,
         dev_mismatched_file,
         gru_encode=False,
         max_len=100,
         hidden_dim=300,
         dropout=0.2,
         lr=0.0001,
         seed=7,
         patience=20,
         model="decomposable",
         use_intra_attention=False,
         use_optimizations=False,
         embedding_dim=200,
         extra_debug=False,
         batch_size=100,
         epochs=300,
         save_loc=".",
         resume=None,
         log_dir=None):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    settings = {
        'train_file': train_file,
        'dev_matched_file': dev_matched_file,
        'dev_mismatched_file': dev_mismatched_file,
        'lr': lr,
        'max_len': max_len,
        'dropout': dropout,
        'batch_size': batch_size,
        'hidden_dim': hidden_dim,
        'epochs': epochs,
        'gru_encode': gru_encode,
        'resume': resume,
        'model': model,
        'use_optimizations': use_optimizations,
        'use_intra_attention': use_intra_attention,
        'patience': patience,
        'embedding_dim': embedding_dim,
        'extra_debug': extra_debug,
        'save_loc': save_loc,
        'log_dir': log_dir
    }

    params = argparse.Namespace()
    params_dict = vars(params)
    params_dict.update(settings)

    train(params)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s',
                        level=config['LOG_LEVEL'])
    plac.call(main)
