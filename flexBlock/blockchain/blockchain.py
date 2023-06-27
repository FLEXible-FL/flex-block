from dataclasses import dataclass
from typing import List


class Blockchain:
    def __init__(self):
        self.chain: List[Block] = []

    def add_block(self, block):
        self.chain.append(block)


@dataclass
class Block:
    def __init__(self, sender_id, weights):
        self.sender_id = sender_id
        self.weights = weights
