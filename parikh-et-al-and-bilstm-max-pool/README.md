# Decomposable Attention and BiLSTM with Max Pooling models for NLI

This folder contains implementations for Parikh et. al. ([Link](https://arxiv.org/abs/1606.01933)) decomposable attention and bilstm with max-pooling models for Natural Language Inference.

## Running

Use the instruction in main folder or `pip install -r requirements.txt`. This should install all of the requirements.

Find the usage in the usage section below.

Sample command for running Parikh model:

`python main.py ../data/sample_train.jsonl ../data/sample_matched.jsonl ../data/sample_mismatched.jsonl --epochs 300  --patience 30 -I -O` where `-I` is for enabling intra-sentence attention and `-O` is for enabling batch augmentations. For more options see usage section below.

Sample command for running BiLSTM model: 
`python main.py ../data/sample_train.jsonl ../data/sample_matched.jsonl ../data/sample_mismatched.jsonl  --patience 30 --model bilstm_max --dropout 0.5 --embedding-dim 300 --epochs 300` where the options are self explanatory.

Use `--resume <pretrained_model_path>` option to load a pretrained model.

## Model Architecture

### Parikh et. al. 2016

This model can decomposed into three steps: (1) Attend (2) Compare (3) Aggregate. See the pictorial representations of steps below (taken from the paper). For more description, refer to the paper above:

![Parikh](https://i.imgur.com/OpRwKdX.png)

### BiLSTM with max-pooling

This model is inspired from recent findings in the paper [Conneau et. al. 2017](https://arxiv.org/abs/1705.02364). The basic model architecture is as below:

![BiLSTM](https://i.imgur.com/QOy40Zp.png)

We generate the following vector from premise and hypothesis encoded vector __`u`__ and __`v`__.

![Final vector](https://i.imgur.com/TPp5ErF.png)

This final vector is passed through three linear (fully connected) layer classifier to generate probabilities of 3 classes via softmax.

## Pretrained Models

Parikh et. al. 2016: [Model Link](https://drive.google.com/file/d/1jyvzJdYAglIKdtDKDpqKakBbTxnu6UUB/view?usp=sharing)

BiLSTM with Max Pooling: [Model Link](https://drive.google.com/file/d/1lKLT6ghhXbefUd0MBEVHORGkfzp49PHA/view?usp=sharing)

## Usage
```
usage: main.py [-h] [-E] [-L 100] [-H 300] [-d 0.2] [-l 0.001] [-S 7] [-p 20]
               [-m decomposable] [-I] [-O] [-P 200] [-D] [-b 100] [-e 5]
               [-s .] [-r None] [-g None]
               train_file dev_matched_file dev_mismatched_file

positional arguments:
  train_file            Path to training data
  dev_matched_file      Path to matched development data
  dev_mismatched_file   Path to mismatched development data

optional arguments:
  -h, --help            show this help message and exit
  -E, --gru-encode      Encode sentences with bidirectional GRU
  -L 100, --max-len 100
                        Length to truncate sentences
  -H 300, --hidden-dim 300
                        Number of hidden units
  -d 0.2, --dropout 0.2
                        Dropout level
  -l 0.001, --lr 0.001  Learning rate
  -S 7, --seed 7        Seed for training
  -p 20, --patience 20  Patience for early stopping
  -m decomposable, --model decomposable
                        Which model to train, parikh/bilstm_max
  -I, --use-intra-attention
                        Whether to use intra attention in Parikh et. al.
  -O, --use-optimizations
                        Whether to use batch optimizations
  -P 200, --embedding-dim 200
                        Dimensions for embedding
  -D, --extra-debug     Whether to provide extra debugging information andlog
                        gradient histograms
  -b 100, --batch-size 100
                        Batch size for neural network training
  -e 5, --epochs 5      Number of training epochs
  -s ., --save-loc .    Save location for files
  -r None, --resume None
                        Resume training, required path to checkpoint file
  -g None, --log-dir None
                        Log directory for extra debug gradients
````
