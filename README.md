# NLI Batch Optimization

Batch optimizations for NLI. Project done for DS-GA 1011. 

TLDR: A method for accelerating convergence speeds in NLI task


## Long Description

This project presents a method to organize mini-batches for training models on Natural Language Inference (NLI) task in order to accelerate convergence speeds and in some cases better accuracy. Techniques such as batch normalization and neural data filter which augment and adapt the varying mini-batches often result in accelerating training time. Inspired by these techniques we propose a method for creating mini-batches for NLI task. We test our proposed method on different models published for solving NLI. We fine tune these models and compare convergence rate and validation accuracy achieved to gather quantitative data to support our claim. Experiments show that using our method can accelerate the convergence speed of NLI models as is seen in case ESIM, Parikh et. al., Chen et. al. and others by 2-5 times. We also present fine-tuned results of several models that were trained on SNLI for MultiNLI.

## Models tested

- Parikh et. al. 2016 (Decomposable Attentional Model for NLI) [Paper Link](https://arxiv.org/abs/1606.01933) [Folder](https://github.com/apsdehal/nli-batch-optimizations/tree/master/parikh-et-al-and-bilstm-max-pool)
- ESIM (Chen et. al. 2016) [Paper Link](https://arxiv.org/abs/1609.06038) [Folder](https://github.com/apsdehal/nli-batch-optimizations/tree/master/esim-chen-et-al-2016)
- Chen et. al. 2017a [Paper Link](https://arxiv.org/abs/1708.01353) [Folder](https://github.com/apsdehal/nli-batch-optimizations/tree/master/chen-et-al-2017)
- BiLSTM with Max Pooling [Folder](https://github.com/apsdehal/nli-batch-optimizations/tree/master/parikh-et-al-and-bilstm-max-pool)


## Installation

- In this project, run `conda env create`
- `source activate nlu-project`

This will install all of the dependencies needed to run the project

Alternatively, you can use `pip install -r requirements.txt` to install all of the requirements.

## Running and Training

Check respective model's folder's README to run and train that model.

## Pretrained Models

Check each model's respective folder to find link to pretrained models. 

## Results

Following table shows the results obtained:

![Results](https://i.imgur.com/etEUOvZ.png)

Gradients values for hidden state matrix of BiLSTM:

<img src="https://i.imgur.com/45E70Mh.png" width="40%"/>

## LICENSE

MIT
