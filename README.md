# Disentangled Graph Collaborative Filtering
This is our Tensorflow implementation for the paper:

>Xiang Wang, Hongye Jin, An Zhang, Xiangnan He, Tong Xu, and Tat-Seng Chua (2020). Disentangled Graph Collaborative Filtering, [Paper in arXiv](https://arxiv.org/abs/2007.01764). In SIGIR'20, Xi'an, China, July 25-30, 2020.

Author: Dr. Xiang Wang (xiangwang at u.nus.edu)

## Introduction
Disentangled Graph Collaborative Filtering (DGCF) is an explainable recommendation framework, which is equipped with (1) dynamic routing mechanism of capsule networks, to refine the strengths of user-item interactions in intent-aware graphs, (2) embedding propagation mechanism of graph neural networks, to distill the pertinent information from higher-order connectivity, and (3) distance correlation of independence modeling, to ensure the independence among intents. As such, we explicitly disentangle the hidden intents of users in the representation learning.

## Environment Requirement
We recommend to run this code in GPUs. The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow_gpu == 1.14.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1

'''
Created on Apr , 2021
Pytorch Implementation of Disentangled Graph Collaborative Filtering (DGCF) model in:
Wang Xiang et al. Disentangled Graph Collaborative Filtering. In SIGIR 2020.
Note that: This implementation is based on the codes of NGCF.
@author: Xiang Wang (xiangwang@u.nus.edu)
@author: Jisu Rho (jsroh1013@gmail.com)
'''
