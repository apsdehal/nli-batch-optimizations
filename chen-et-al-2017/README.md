# Recurrent Neural Network-Based Sentence Encoder with Gated Attention for Natural Language Inference

Chen et. al. 2017 [Link](https://arxiv.org/abs/1708.01353)

Requirements:
pytorch 0.2.0_3 => (if using NYU Prince cluster,  module load pytorch/python3.5/0.2.0_3)
numpy/1.13.1 => (if using NYU Prince cluster, module load numpy/python3.5/intel/1.13.1)

The preprocessed vocab file is compiled and put here: https://drive.google.com/file/d/11Pekk1IKvy-GDJmtw2Ew_3v7myE9WoFg/view?usp=sharing

To train the model run:
$ python3 Train.py

Trained models:
1) Normal: https://drive.google.com/drive/folders/1bCyMP91pY0OcF_dH2Lv2fMn-ham1bDbZ?usp=sharing

2) Batch Optimized: https://drive.google.com/drive/folders/1hIDoaWD8Rgt6ECTzVQzLqBfN1qizHgfE?usp=sharing