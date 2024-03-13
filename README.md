# FLEX-block

FLEXblock is an implementation of blockchain functionality for simulating descentralized federated learning experiments. It is intended to extend the [FLEXible](https://github.com/FLEXible-FL/FLEXible) framework.

## Details
This repository includes both:
- An interface for creating your own blockchain architectures and working with them in a FLEX experiment.
- Some blockchain architectures to be used directly out of the box.

### Folder structure
- *flexBlock/pool*: contains the interface for creating blockchain architectures and some implementations of it.
- *flexBlock/blockchain*: contains the interface for creating blockchains and block to be used with architectures and some implementations of it.
- *notebooks*: some explanatory notebooks showing how to implement a custom blockchain architecture and working with the library.
- *tests*: contains tests for the features present in this library.

### Explanatory notebooks
- *cifar-10.ipynb*: Shows how to train a model in the cifar 10 dataset with a blockchain architecture.
- *Proof of federated Learning MNIST.ipynb*: Explains and shows how to use the Proof Of Federated Learning blockchain architecture for training a simple model in the popular MNIST dataset.
- *Custom blockchain.ipynb*: Shows how to create your own blockchain architecture and how to later use it in any kind of FLEX experiment.

## Features

The library implements an interface for creating your own blockchain architectures for FLEX experiments. Still, we offer some architectures out of the box.

|  Architecture |  Description  | Citation |
|----------|:-----------------------------------:|------:|
| Proof of Work (PoW) | The most classical consensus mechanism used in blockchain where miners race into solving a hash puzzle. In this architecture the sets of miners(aggregators) and clients is disjoint. | [Federated Learning Meets Blockchain in Edge Computing: Opportunities and Challenges](https://ieeexplore.ieee.org/document/9403374) |
| Proof of Stake (PoS) | This network uses a consensus mechanism where miners have some reputation/tokens called stake and they are picked randomly according to their stake. In this architecture the sets of clients and miners are the same. | [Robust Blockchained Federated Learning with Model Validation and Proof-of-Stake Inspired Consensus](https://arxiv.org/abs/2101.03300) |
| Proof of Federated Learning (PoFL) | A very popular consensus mechanism for blockchain enabled federated learning. Based on Proof of Useful Work miners race to get the model with the most accuracy on a given dataset. In this architecure the sets of clients and miners is disjoint. | [Proof of federated learning: A novel energy-recycling consensus algorithm](https://ieeexplore.ieee.org/abstract/document/9347812) |


## Installation

In order to install this repo locally:

``
    pip install -e .
``

FLEX-block is available on the PyPi repository and can be easily installed using pip:
