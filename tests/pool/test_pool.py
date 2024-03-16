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
from unittest.mock import MagicMock
from typing import Dict, Hashable

import numpy as np
import pytest
from flex.data import Dataset, FedDataset
from flex.model import FlexModel
from flex.pool import FlexPool, init_server_model

from flexBlock.blockchain import Block, Blockchain
from flexBlock.pool import BlockchainPool, PoolConfig

WEIGHTS_LENGTH = 5


@init_server_model
def init_server_func():
    model = FlexModel()
    model["weights"] = [np.random.rand(WEIGHTS_LENGTH)]
    model["model"] = [np.random.rand(WEIGHTS_LENGTH)]
    return model


@pytest.fixture(name="flp")
def fixture_flex_pool() -> FlexPool:
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd = Dataset.from_array(X_data, y_data)
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd1 = Dataset.from_array(X_data, y_data)
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd2 = Dataset.from_array(X_data, y_data)
    fed_dataset = FedDataset({"client_1": fcd, "client_2": fcd1, "client_3": fcd2})
    return FlexPool.p2p_pool(fed_dataset=fed_dataset, init_func=init_server_func)


class CustomBlock(Block):
    def __init__(self, weights, flag=False):
        super().__init__(weights)
        self.flag = flag

    def compute_hash(self):
        return "0"


GENESIS_BLOCK = CustomBlock([0, 0, 0])


class CustomBlockchain(Blockchain[CustomBlock]):
    def __init__(self, genesis_block: CustomBlock):
        super().__init__(genesis_block)


class TestBlockchainPool(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_flex_pool(self, flp: FlexPool):
        self._flp: FlexPool = flp

    def _create_custom_pool(self, config=None):
        flp = self._flp

        class CustomPool(BlockchainPool[CustomBlockchain]):
            def __init__(self) -> None:
                bc = CustomBlockchain(GENESIS_BLOCK)
                pool = flp
                if config is None:
                    self.initialize_pool(bc, pool)
                else:
                    self.initialize_pool(bc, pool, config)

            def pack_block(self, weights):
                return CustomBlock(weights)

            def consensus_mechanism(
                self, miners: Dict[Hashable, FlexModel], **kwargs
            ) -> Hashable | None:
                miner = list(miners.keys())[0]
                miners[miner]["aggregated_weights"] = []
                return miner

        return CustomPool()

    def test_when_using_getter_then_pool_getters_are_used(self):
        pool = self._create_custom_pool()

        assert pool.aggregators == pool._pool.aggregators
        assert pool.clients == pool._pool.clients
        assert pool.servers == pool._pool.servers

    def test_when_gossip_then_all_miners_have_all_weights(self):
        pool = self._create_custom_pool()
        w = []
        pool._gossip()
        for miner in pool.aggregators._models.values():
            w.append(miner["weights"])
        assert len(w) == 3
        assert all(
            np.sum(np.equal(w_, x_)) == 3 * WEIGHTS_LENGTH for w_ in w for x_ in w
        ), "Miners have different weights"

    def test_when_gossip_to_miner_then_miner_has_all_weights(self):
        pool = self._create_custom_pool()
        first_miner = list(pool.aggregators._models.keys())[0]
        pool._gossip_to_miner(first_miner)
        w = pool.aggregators._models[first_miner]["weights"]
        assert len(w) == 3
        for i, miner in pool.aggregators._models.items():
            if i != first_miner:
                assert (
                    any(np.sum(np.equal(x, miner["weights"])) for x in w)
                    != WEIGHTS_LENGTH
                ), "Miner do not have the miners weights"

                assert len(miner["weights"]) < len(
                    w
                ), "Miner has more weights than it should"

    def test_when_agg_before_consensus_and_no_gossip_after_then_agg_only_happens_once(
        self,
    ):
        agg_operator = MagicMock()
        set_weights = MagicMock()
        config = PoolConfig(
            gossip_before_consensus=False,
            aggregate_before_consensus=True,
            gossip_on_agg=False,
        )
        pool = self._create_custom_pool(config=config)
        pool.servers.map(init_server_func)
        pool.aggregate(agg_function=agg_operator, set_weights=set_weights)
        assert agg_operator.call_count == len(pool.aggregators)
        assert set_weights.call_count == len(pool.aggregators)

    def test_when_agg_before_consensus_and_gossip_after_then_agg_happens_once_per_miner_and_after_gossip(
        self,
    ):
        agg_operator = MagicMock()
        set_weights = MagicMock()
        config = PoolConfig(
            gossip_before_consensus=False,
            aggregate_before_consensus=True,
            gossip_on_agg=True,
        )
        pool = self._create_custom_pool(config=config)
        pool.servers.map(init_server_func)
        pool.aggregate(agg_function=agg_operator, set_weights=set_weights)
        assert agg_operator.call_count == len(pool.aggregators) + 1
        assert set_weights.call_count == len(pool.aggregators) + 1

    def test_when_no_miner_is_selected_then_aggregation_stops(
        self,
    ):
        agg_operator = MagicMock()
        set_weights = MagicMock()
        config = PoolConfig(
            gossip_before_consensus=False,
            aggregate_before_consensus=False,
            gossip_on_agg=False,
        )
        pool = self._create_custom_pool(config=config)
        pool.consensus_mechanism = MagicMock(return_value=None)
        pool.servers.map(init_server_func)
        result = pool.aggregate(agg_function=agg_operator, set_weights=set_weights)
        assert not result
        pool.consensus_mechanism.assert_called_once()
        agg_operator.assert_not_called()
        set_weights.assert_not_called()

    def test_when_agg_is_done_then_pack_block_is_called_and_appended(
        self,
    ):
        agg_operator = MagicMock()
        set_weights = MagicMock()
        pool = self._create_custom_pool()
        blocks_in_chain = len(pool.blockchain.chain)
        pool.pack_block = MagicMock(return_value=CustomBlock([0, 0, 0]))
        pool.servers.map(init_server_func)
        pool.aggregate(agg_function=agg_operator, set_weights=set_weights)
        assert pool.pack_block.call_count == 1
        assert len(pool.blockchain.chain) == blocks_in_chain + 1
