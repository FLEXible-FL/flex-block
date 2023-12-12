import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha256
from typing import Generic, List, TypeVar


@dataclass
class Block(ABC):
    @abstractmethod
    def __init__(self, weights):
        self._previous_hash = None
        self.weights = weights
        self.hash = None
        self.timestamp = time.time_ns()

    @abstractmethod
    def compute_hash(self):
        pass


@dataclass
class BlockPoW(Block):
    def __init__(self, weights):
        super().__init__(weights)
        # Default value
        self.target_zeroes = 3

    def compute_hash(self, nonce=None):
        hashed_weights = sha256(bytes(self.weights)).hexdigest()
        if nonce:
            hashed_weights += str(nonce)
        return sha256((self._previous_hash + hashed_weights).encode()).hexdigest()


@dataclass
class BlockPoFL(Block):
    def __init__(self, weights):
        super().__init__(weights)

    def compute_hash(self):
        hashed_weights = sha256(bytes(self.weights)).hexdigest()
        return hashed_weights


@dataclass
class BlockPoS(Block):
    def __init__(self, weights):
        super().__init__(weights)

    def compute_hash(self):
        # hashed_weights = sha256(bytes(self.weights)).hexdigest()
        hashed_weights = sha256(bytes("hello world lol".encode())).hexdigest()
        return sha256((self._previous_hash + hashed_weights).encode()).hexdigest()


_BlockType = TypeVar("_BlockType", bound=Block)


class Blockchain(ABC, Generic[_BlockType]):
    @abstractmethod
    def __init__(self, genesis_block: _BlockType, *args, **kwargs):
        genesis_block._previous_hash = "0"
        genesis_block.hash = genesis_block.compute_hash()
        self.chain: List[_BlockType] = [genesis_block]

    def add_block(self, block: _BlockType):
        previous_hash = self.chain[-1].hash if len(self.chain) else str(random.random())
        block._previous_hash = previous_hash
        block.hash = block.compute_hash()

        self.chain.append(block)

    def get_last_block(self):
        return self.chain[-1]


class BlockchainPow(Blockchain[BlockPoW]):
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
    def __init__(self, genesis_block: BlockPoFL, *args, **kwargs):
        super().__init__(genesis_block)

    def add_block(self, block):
        super().add_block(block)


class BlockchainPoS(Blockchain[BlockPoS]):
    def __init__(self, genesis_block: BlockPoS, *args, **kwargs):
        super().__init__(genesis_block)
        self.seconds_to_mine = kwargs.get("seconds_to_mine", 10)

    def add_block(self, block):
        super().add_block(block)
        time.sleep(self.seconds_to_mine)
