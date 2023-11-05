from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import List
from hashlib import sha256
import random
import time


@dataclass
class _Block(ABC):
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
class BlockPoW(_Block):
    def __init__(self, weights):
        super().__init__(weights)
        self.target_zeroes = None

    def compute_hash(self, nonce=None):
        hashed_weights = sha256(bytes(self.weights)).hexdigest()
        if nonce:
            hashed_weights += str(nonce)
        return sha256((self._previous_hash + hashed_weights).encode()).hexdigest()


@dataclass
class BlockPoFL(_Block):
    def __init__(self, weights):
        super().__init__(weights)
        self.target_err = None

    def compute_hash(self):
        hashed_weights = sha256(bytes(self.weights)).hexdigest()
        return sha256((self._previous_hash + hashed_weights).encode()).hexdigest()


@dataclass
class BlockPoS(_Block):
    def __init__(self, weights):
        super().__init__(weights)

    def compute_hash(self):
        hashed_weights = sha256(bytes(self.weights)).hexdigest()
        return sha256((self._previous_hash + hashed_weights).encode()).hexdigest()


class _Blockchain(ABC):
    @abstractmethod
    def __init__(self, _BlockType: type):
        self.chain: List[_BlockType] = []
        self._block_type = _BlockType

    @abstractmethod
    def add_block(self, block):
        previous_hash = self.chain[-1].hash if len(self.chain) else str(random.random())
        block._previous_hash = previous_hash
        block.hash = block.compute_hash()

        self.chain.append(block)

    def get_last_block(self):
        return self.chain[-1]


class BlockchainPow(_Blockchain):
    def __init__(self):
        super().__init__(BlockPoW)

    def add_block(self, block: BlockPoW):
        super().add_block(block)
        assert isinstance(self.chain[-1], BlockPoW)

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


class BlockchainPoFL(_Blockchain):
    def __init__(self, seconds_to_mine, max_err, min_err):
        super().__init__(BlockPoFL)
        self.max_err = max_err if max_err else 0.5
        self.min_err = min_err if min_err else 0.2
        self.seconds_to_mine = seconds_to_mine if seconds_to_mine else 10

    def add_block(self, block: BlockPoFL):
        super().add_block(block)
        assert isinstance(self.chain[-1], BlockPoFL)

        if len(self.chain) >= 2:
            self._adjust_difficulty()
        else:
            self.chain[-1].target_err = self.max_err

    def _adjust_difficulty(self):
        last_block = self.chain[-1]
        previous_block = self.chain[-2]

        # If the last block took more than expected to mine, decrease difficulty
        diff_time = float(last_block.timestamp - previous_block.timestamp) / 1e8
        if diff_time < self.seconds_to_mine:
            self.chain[-1].target_err = previous_block.target_err - 0.05
        else:
            self.chain[-1].target_err = previous_block.target_err + 0.05

        if self.chain[-1].target_err > self.max_err:
            self.chain[-1].target_err = self.max_err
        if self.chain[-1].target_err < self.min_err:
            self.chain[-1].target_err = self.min_err


class BlockchainPoS(_Blockchain):
    def __init__(self, seconds_to_mine):
        super().__init__(BlockPoS)
        self.seconds_to_mine = seconds_to_mine if seconds_to_mine else 10

    def add_block(self, block: BlockPoS):
        super().add_block(block)
        assert isinstance(self.chain[-1], BlockPoS)

        time.sleep(self.seconds_to_mine)
