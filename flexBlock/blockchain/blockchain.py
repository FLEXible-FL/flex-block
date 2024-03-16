"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha256
from typing import Generic, List, TypeVar

import numpy as np


@dataclass
class Block(ABC):
    """
    Abstract base class for a block in a blockchain.

    Attributes
    ----------
        _previous_hash (str): The hash of the previous block in the blockchain.
        weights: The weights associated with the block.
        hash (str): The hash of the current block.
        timestamp (float): The timestamp of when the block was created.
    """

    @abstractmethod
    def __init__(self, weights):
        """
        Initialize a new instance of the Block class.

        Args:
        ----
            weights: The weights associated with the block.
        """
        self._previous_hash = None
        self.weights = weights
        self.hash = None
        self.timestamp = time.time_ns()

    @abstractmethod
    def compute_hash(self):
        """
        Compute the hash of the block.
        """
        pass

    @classmethod
    def hash_weights(cls, weights) -> str:
        """
        Compute the hash of the weights. This function is an utils for implementing the compute_hash method.
        It currently supports numpy arrays, lists, pytorch tensors, tensorflow tensors and lists of the previous elements.

        Args:
        ----
            weights: The weights to be hashed.

        Returns:
        -------
            str: The hash of the weights.
        """
        try:
            import torch

            if isinstance(weights, torch.Tensor):
                return cls.hash_weights(weights.cpu().numpy())
        except ImportError:
            pass

        if isinstance(weights, np.ndarray):
            return sha256(weights.tobytes()).hexdigest()

        if isinstance(weights, list):
            return "".join([cls.hash_weights(w) for w in weights])

        return sha256(bytes(weights)).hexdigest()


@dataclass
class BlockPoW(Block):
    """
    Represents a block in a Proof of Work (PoW) blockchain.

    Attributes
    ----------
        weights (list): The weights of the block.
        target_zeroes (int): The number of leading zeroes required in the hash.

    Methods
    -------
        compute_hash: Computes the hash of the block.
    """

    def __init__(self, weights):
        super().__init__(weights)
        self.target_zeroes = 3

    def compute_hash(self, nonce=None):
        """
        Compute the hash of the block.

        Args:
        ----
            nonce (int, optional): The nonce value to be included in the hash computation.

        Returns:
        -------
            str: The computed hash of the block.
        """
        hashed_weights = self.hash_weights(self.weights)
        if nonce:
            hashed_weights += str(nonce)
        return sha256((self._previous_hash + hashed_weights).encode()).hexdigest()


@dataclass
class BlockPoFL(Block):
    """
    Represents a Proof-of-Federated-Learning (PoFL) block in the blockchain.
    """

    def __init__(self, weights):
        super().__init__(weights)

    def compute_hash(self):
        """
        Compute the hash of the block by hashing the weights.

        Returns
        -------
            str: The computed hash of the block.
        """
        return self.hash_weights(self.weights)


@dataclass
class BlockPoS(Block):
    """
    Represents a Proof-of-Stake (PoS) block in the blockchain.

    Args:
    ----
        weights (list): The weights associated with the block.

    Attributes:
    ----------
        weights (list): The weights associated with the block.

    Methods:
    -------
        compute_hash: Computes the hash of the block.

    """

    def __init__(self, weights):
        super().__init__(weights)

    def compute_hash(self):
        """
        Compute the hash of the block.

        Returns
        -------
            str: The computed hash value.

        """
        hashed_weights = self.hash_weights(self.weights)
        return sha256((self._previous_hash + hashed_weights).encode()).hexdigest()


_BlockType = TypeVar("_BlockType", bound=Block)


class Blockchain(ABC, Generic[_BlockType]):
    """
    A class representing a blockchain.

    Attributes
    ----------
        chain (List[_BlockType]): The list of blocks in the blockchain.

    Methods
    -------
        __init__(self, genesis_block: _BlockType, *args, **kwargs): Initializes the blockchain with a genesis block.
        add_block(self, block: _BlockType): Adds a new block to the blockchain.
        get_last_block(self) -> _BlockType: Returns the last block in the blockchain.
    """

    @abstractmethod
    def __init__(self, genesis_block: _BlockType, *args, **kwargs):
        """
        Initialize the blockchain with a genesis block.

        Args:
        ----
            genesis_block (_BlockType): The genesis block of the blockchain.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        genesis_block._previous_hash = "0"
        genesis_block.hash = genesis_block.compute_hash()
        self.chain: List[_BlockType] = [genesis_block]

    def add_block(self, block: _BlockType):
        """
        Add a new block to the blockchain.

        Args:
            block (_BlockType): The block to be added to the blockchain.
        """
        previous_hash = self.chain[-1].hash if len(self.chain) else str(random.random())
        block._previous_hash = previous_hash
        block.hash = block.compute_hash()

        self.chain.append(block)

    def get_last_block(self) -> _BlockType:
        """
        Return the last block in the blockchain.

        Returns
        -------
            _BlockType: The last block in the blockchain.
        """
        return self.chain[-1]


class BlockchainPow(Blockchain[BlockPoW]):
    """
    A class representing a proof-of-work blockchain.

    Attributes
    ----------
        genesis_block (BlockPoW): The genesis block of the blockchain.

    Methods
    -------
        __init__(self, genesis_block: BlockPoW, *args, **kwargs): Initializes a new instance of the `BlockchainPow` class.
        add_block(self, block): Adds a new block to the blockchain.
        _adjust_difficulty(self): Adjusts the difficulty of mining new blocks based on the time taken to mine the last block.
    """

    def __init__(self, genesis_block: BlockPoW, *args, **kwargs):
        super().__init__(genesis_block)

    def add_block(self, block):
        super().add_block(block)

        if len(self.chain) >= 2:
            self._adjust_difficulty()
        else:
            self.chain[-1].target_zeroes = 3

    def _adjust_difficulty(self):
        last_block = self.chain[-1]
        previous_block = self.chain[-2]
        # If the last block took more than 10 seconds to mine, decrease difficulty
        diff_time = float(last_block.timestamp - previous_block.timestamp) / 1e9
        if diff_time < 1:
            self.chain[-1].target_zeroes = previous_block.target_zeroes + 1
        else:
            self.chain[-1].target_zeroes = previous_block.target_zeroes - 1

        if self.chain[-1].target_zeroes < 0:
            self.chain[-1].target_zeroes = 1
        if self.chain[-1].target_zeroes > 64:
            self.chain[-1].target_zeroes = 64


class BlockchainPoFL(Blockchain[BlockPoFL]):
    """
    A Proof-of-Federated-Learning Blockchain implementation.

    Args:
    ----
        genesis_block (BlockPoFL): The genesis block of the blockchain.

    Attributes:
    ----------
        chain (List[BlockPoFL]): The list of blocks in the blockchain.

    """

    def __init__(self, genesis_block: BlockPoFL, *args, **kwargs):
        super().__init__(genesis_block)

    def add_block(self, block):
        super().add_block(block)


class BlockchainPoS(Blockchain[BlockPoS]):
    """
    A class representing a Proof-of-Stake (PoS) blockchain.

    Attributes:
    ----------
        genesis_block (BlockPoS): The genesis block of the blockchain.
        seconds_to_mine (int): The number of seconds it takes to mine a block.

    Args:
    ----
        genesis_block (BlockPoS): The genesis block of the blockchain.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    """

    def __init__(self, genesis_block: BlockPoS, *args, **kwargs):
        super().__init__(genesis_block)
        self.seconds_to_mine = kwargs.get("seconds_to_mine", 10)

    def add_block(self, block):
        super().add_block(block)
        time.sleep(self.seconds_to_mine)
