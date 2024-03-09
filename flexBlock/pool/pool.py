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
from __future__ import annotations

import functools
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Hashable, Optional, TypeVar

from flex.actors.actors import FlexActors, FlexRole
from flex.data import Dataset, FedDataset
from flex.model.model import FlexModel
from flex.pool import FlexPool

from flexBlock.blockchain.blockchain import (Blockchain, BlockchainPoFL,
                                             BlockchainPoS, BlockchainPow,
                                             BlockPoFL, BlockPoS, BlockPoW)
from flexBlock.common import CLIENT_CONNECTIONS, DEBUG
from flexBlock.pool.utils import create_miners

_STAKE_BLOCKFED_TAG = "stake"
_INITIAL_STAKE = 32

_BlockchainType = TypeVar("_BlockchainType", bound=Blockchain)


@dataclass(frozen=True)
class PoolConfig:
    gossip_before_agg: bool = True
    gossip_on_agg: bool = True
    aggregate_before_agg: bool = False


_default_pool_config = PoolConfig()


class BlockchainPool(ABC, Generic[_BlockchainType]):
    _pool: FlexPool
    _blockchain: _BlockchainType
    _config: PoolConfig

    def initialize_pool(
        self,
        blockchain: _BlockchainType,
        pool: FlexPool,
        config: PoolConfig = _default_pool_config,
        **kwargs,
    ):
        self._pool = pool
        self._blockchain = blockchain
        self._config = config

    @property
    def blockchain(self):
        return self._blockchain

    @functools.cached_property
    def actor_ids(self):
        return self._pool.actor_ids

    @functools.cached_property
    def clients(self):
        """Property to get all the clients available in a pool.

        Returns:
            FlexPool: Pool containing all the clients from a pool
        """
        return self._pool.clients

    @functools.cached_property
    def aggregators(self):
        """Property to get all the aggregator available in a pool.

        Returns:
            FlexPool: Pool containing all the aggregators from a pool
        """
        return self._pool.aggregators

    @functools.cached_property
    def servers(self):
        """Property to get all the servers available in a pool.

        Returns:
            FlexPool: Pool containing all the servers from a pool
        """
        return self._pool.servers

    def _gossip(self):
        """Gossiping mechanism for the pool. The miners will share the weights of their clients with each other."""
        miners = self._pool.aggregators
        total_weights = [
            weight for miner in miners._models.values() for weight in miner["weights"]
        ]

        for key in miners._models.keys():
            miners._models[key]["weights"] = total_weights

    def _gossip_to_miner(self, selected_miner):
        """Gossiping mechanism for the pool. Only the selected miner will get all the weights. This is done for efficency where a lot
        of miners are present in the pool (i.e. Proof of Stake)"""
        miners = self._pool.aggregators
        total_weights = [
            weight for miner in miners._models.values() for weight in miner["weights"]
        ]

        miners._models[selected_miner]["weights"] = total_weights

    def aggregate(
        self,
        agg_function: Callable,
        set_weights: Callable,
        **kwargs,
    ):
        if self._config.gossip_before_agg:
            self._gossip()

        if self._config.aggregate_before_agg:
            for v in self.aggregators._models.values():
                # aggregate weights and set them to the aggregator's model
                # We also need to save a copy of weights to restore them later (agg_function removes them)
                if len(v["weights"]) == 0:
                    continue  # The miner did not collect any weights

                weights = deepcopy(v["weights"])
                agg_function(v, None)
                v["weights"] = weights
                agg_pool = self.aggregators.select(
                    lambda actor_id, _: actor_id == v.actor_id
                )

                agg_pool.map(set_weights, agg_pool)

        selected_miner = self.consensus_mechanism(
            miners=self._pool.aggregators._models, **kwargs
        )

        if selected_miner is None:
            # We are not going to need the weights, we will pick them again in the next round
            for v in self.aggregators._models.values():
                v["weights"] = []

            return False

        gossip_after_consensus = self._config.gossip_on_agg and not self._config.gossip_before_agg
        if gossip_after_consensus:
            self._gossip_to_miner(selected_miner)

        selected_model = self.aggregators._models[selected_miner]

        if not self._config.aggregate_before_agg or gossip_after_consensus:
            agg_function(selected_model, None)
            # Set model of the selected miner
            agg_pool = self.aggregators.select(
                lambda actor_id, _: actor_id == selected_model.actor_id
            )
            agg_pool.map(set_weights, agg_pool)
        
        weights = deepcopy(selected_model["aggregated_weights"])
        self._blockchain.add_block(self.pack_block(weights=weights))

        for v in self.aggregators._models.values():
            if v.actor_id == selected_model.actor_id:
                continue

            v["weights"] = []
            v["aggregated_weights"] = []
            v["model"] = deepcopy(selected_model["model"])

        return True

    @abstractmethod
    def pack_block(self, weights):
        pass

    @abstractmethod
    def consensus_mechanism(
        self, miners: Dict[Hashable, FlexModel], **kwargs
    ) -> Optional[Hashable]:
        pass

    def __len__(self):
        return len(self._pool)


class PoWBlockchainPool(BlockchainPool):
    def __init__(
        self,
        fed_dataset: FedDataset,
        init_func: Callable,
        number_of_miners: int,
        **kwargs,
    ):
        if number_of_miners < 1:
            raise ValueError("The number of nodes must be at least 1")

        # Split clients between miners
        actors = FlexActors(
            {actor_id: FlexRole.client for actor_id in fed_dataset.keys()}
        )

        actors, models = create_miners(actors, number_of_miners, CLIENT_CONNECTIONS)

        # Create pool and initialize servers
        pool = FlexPool(
            flex_data=fed_dataset,
            flex_actors=actors,
            flex_models=models,
        )

        pool.servers.map(init_func, **kwargs)

        bc = (
            BlockchainPow(BlockPoW([]), **kwargs)
            if "blockchain" not in kwargs
            else kwargs["blockchain"]
        )

        config = PoolConfig(
            gossip_before_agg=False, gossip_on_agg=True, aggregate_before_agg=False
        )

        self.initialize_pool(bc, pool, config, **kwargs)

    def pack_block(self, weights):
        return BlockPoW(weights=weights)

    def consensus_mechanism(self, miners, **kwargs):
        miner_keys = list(miners.keys())
        previous_block = self.blockchain.get_last_block()
        selected_miner_index = 0
        nonce = 0

        while True:
            computed_hash = previous_block.compute_hash(nonce=nonce)
            if (
                computed_hash[: previous_block.target_zeroes]
                == "0" * previous_block.target_zeroes
            ):
                break

            nonce += 1
            selected_miner_index += 1
            selected_miner_index %= len(miner_keys)

        if DEBUG >= 1:
            print(
                f"[PoW] selected miner: {miner_keys[selected_miner_index]:10} nonce: {nonce}"
            )
        return miner_keys[selected_miner_index]


class PoFLBlockchainPool(BlockchainPool):
    def __init__(
        self,
        fed_dataset: FedDataset,
        init_func: Callable,
        number_of_miners: int,
        **kwargs,
    ):
        if number_of_miners < 1:
            raise ValueError("The number of nodes must be at least 1")

        # Split clients between miners
        actors = FlexActors(
            {actor_id: FlexRole.client for actor_id in fed_dataset.keys()}
        )

        actors, models = create_miners(actors, number_of_miners, CLIENT_CONNECTIONS)

        # Create pool and initialize servers
        pool = FlexPool(
            flex_data=fed_dataset,
            flex_actors=actors,
            flex_models=models,
        )

        pool.servers.map(init_func, **kwargs)

        bc = (
            BlockchainPoFL(genesis_block=BlockPoFL([]), **kwargs)
            if "blockchain" not in kwargs
            else kwargs["blockchain"]
        )

        config = PoolConfig(
            gossip_before_agg=False, aggregate_before_agg=True, gossip_on_agg=False
        )

        self.initialize_pool(
            bc,
            pool,
            config=config,
            **kwargs,
        )

    def pack_block(self, weights):
        return BlockPoFL(weights=weights)

    def consensus_mechanism(self, miners, **kwargs):
        eval_function = kwargs.get("eval_function")
        eval_dataset = kwargs.get("eval_dataset")
        accuracy = kwargs.get("accuracy")

        assert (
            eval_function is not None
        ), "eval_function must be provided to the aggregate method"
        assert isinstance(
            eval_dataset, Dataset
        ), "eval_dataset must be provided to the aggregate method"
        assert accuracy is not None, "accuracy must be provided"

        valid_miners = []

        for miner, model in miners.items():
            acc = eval_function(model, eval_dataset)
            if acc >= accuracy:
                if DEBUG >= 1:
                    print(f"[POFL] miner: {miner:10} acc: {acc} target: {accuracy}", flush=True)
                valid_miners.append((miner, acc))

        valid_miners.sort(key=lambda x: x[1])

        return valid_miners[-1][0] if len(valid_miners) > 0 else None


class PoSBlockchainPool(BlockchainPool):
    def __init__(
        self,
        fed_dataset: FedDataset,
        init_func: Callable,
        initial_stake: int = _INITIAL_STAKE,
        **kwargs,
    ):
        pool = FlexPool.p2p_pool(fed_dataset, init_func, **kwargs)
        bc = (
            BlockchainPoS(
                BlockPoS([]),
                **kwargs,
            )
            if "blockchain" not in kwargs
            else kwargs["blockchain"]
        )

        miners = list(pool.aggregators._models.values())
        for miner in miners:
            miner[_STAKE_BLOCKFED_TAG] = initial_stake

        config = PoolConfig()

        self.initialize_pool(bc, pool, config, **kwargs)

    def pack_block(self, weights):
        return BlockPoS(weights=weights)

    def consensus_mechanism(self, miners, **kwargs):
        key_stake = [(k, m.get(_STAKE_BLOCKFED_TAG)) for k, m in miners.items()]

        key_stake = list(filter(lambda x: x[1] and x[1] > 0, key_stake))
        selected_miner = random.choices(key_stake, [s for _, s in key_stake])[0][0]
        if DEBUG >= 1:
            print(
                f"[PoS] selected miner: {selected_miner: 10} stake: {self.aggregators._models[selected_miner][_STAKE_BLOCKFED_TAG]}"
            )
        self.aggregators._models[selected_miner][_STAKE_BLOCKFED_TAG] += 1

        return selected_miner
