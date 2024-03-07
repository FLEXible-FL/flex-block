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
import unittest
import pytest

from flexBlock.blockchain import Block, Blockchain


class CustomBlock(Block):
    def __init__(self, weights, hash="CustomBlock"):
        super().__init__(weights)
        self.hash_computed = False
        self.name = hash

    def compute_hash(self):
        self.hash_computed = True
        return self.name


class CustomBlockchain(Blockchain[CustomBlock]):
    def __init__(self, genesis_block: CustomBlock, *args, **kwargs):
        super().__init__(genesis_block, *args, **kwargs)


@pytest.fixture(name="genesis_block")
def fixture_genesis_block():
    return CustomBlock([0, 0, 0])


@pytest.fixture(name="blockchain")
def fixture_blockchain(genesis_block):
    return CustomBlockchain(genesis_block)


class TestBlockchain(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_blockchain(self, blockchain):
        self._blockchain: CustomBlockchain = blockchain

    def test_when_block_is_added_then_hash_is_computed(self):
        name = "test_blockchain"
        new_block = CustomBlock([1, 1, 1], hash=name)
        self._blockchain.add_block(new_block)
        assert self._blockchain.get_last_block().weights == [
            1,
            1,
            1,
        ], "Weights are not equal"
        assert self._blockchain.get_last_block().hash_computed, "Hash is not computed"
        assert self._blockchain.get_last_block().hash == name, "Hash is not stored"
        assert (
            self._blockchain.chain[-1]._previous_hash == self._blockchain.chain[0].hash
        ), "previous hash is not equal to the hash of the previous block"
