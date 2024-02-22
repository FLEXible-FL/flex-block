# FLEX-block

FLEXblock is an implementation of blockchain functionality for simulating descentralized federated learning experiments. It is intended to extend the [FLEXible](https://github.com/FLEXible-FL/FLEXible) framework.

## Features

The library implements an interface for creating your own blockchain architectures for FLEX experiments. Still, we offer some architectures out of the box.

|  Architecture |  Description  | Citation |
|----------|:-----------------------------------:|------:|
| Proof of Work (PoW) | The most classical consensus mechanism used in blockchain where miners race into solving a hash puzzle. In this architecture the sets of miners(aggregators) and clients is disjoint. |  |
| Proof of Stake (PoS) | This network uses a consensus mechanism where miners have some reputation/tokens called stake and they are picked randomly according to their stake. In this architecture the sets of clients and miners are the same. | |
| Proof of Federated Learning (PoFL) | A very popular consensus mechanism for blockchain enabled federated learning. Based on Proof of Useful Work miners race to get the model with the most accuracy on a given dataset. In this architecure the sets of clients and miners is disjoint. | [Proof of federated learning: A novel energy-recycling consensus algorithm](https://ieeexplore.ieee.org/abstract/document/9347812) |


## Installation

In order to install this repo locally:

``
    pip install -e .
``

FLEX-block is available on the PyPi repository and can be easily installed using pip:
