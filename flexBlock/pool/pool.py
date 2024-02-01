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
from flexBlock.common import DEBUG
from flexBlock.pool.utils import create_miners

CLIENT_CONNECTIONS = "clients_connections"
_STAKE_BLOCKFED_TAG = "stake"
_INITIAL_STAKE = 32

_BlockchainType = TypeVar("_BlockchainType", bound=Blockchain)


@dataclass(frozen=True)
class PoolConfig:
    gossip_before_agg: bool = True
    gossip_on_agg: bool = True
    gossip_selected_only: bool = False
    aggregate_before_agg: bool = False

    def __post_init__(self):
        assert not (self.gossip_selected_only and self.gossip_before_agg), "Cannot gossip only to selected before aggregation"


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
                    continue # The miner did not collect any weights 

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

        if self._config.gossip_on_agg and not self._config.gossip_before_agg:
            self._gossip_to_miner(selected_miner) if self._config.gossip_selected_only else self._gossip()

        agg_function(self.aggregators._models[selected_miner], None)
        weights = deepcopy(
            self.aggregators._models[selected_miner]["aggregated_weights"]
        )
        self._blockchain.add_block(self.pack_block(weights=weights))
        for v in self.aggregators._models.values():
            v["aggregated_weights"] = deepcopy(weights)
            v["weights"] = []

        self.aggregators.map(set_weights, self.servers)

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

        config = PoolConfig(gossip_before_agg=False, gossip_on_agg=True, aggregate_before_agg=False, gossip_selected_only=True)

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

        if DEBUG >= 1: print(f"[PoW] selected miner: {miner_keys[selected_miner_index]:10} nonce: {nonce}")
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
                if DEBUG >= 1: print(f"[POFL] miner: {miner:10} acc: {acc}")
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
        
        config = PoolConfig(gossip_selected_only=True)

        self.initialize_pool(bc, pool, config, **kwargs)

    def pack_block(self, weights):
        return BlockPoS(weights=weights)

    def consensus_mechanism(self, miners, **kwargs):
        key_stake = [(k, m.get(_STAKE_BLOCKFED_TAG)) for k, m in miners.items()]

        key_stake = list(filter(lambda x: x[1] and x[1] > 0, key_stake))
        selected_miner = random.choices(key_stake, [s for _, s in key_stake])[0][0]
        if DEBUG >= 1: print(f"[PoS] selected miner: {selected_miner: 10} stake: {self.aggregators._models[selected_miner][_STAKE_BLOCKFED_TAG]}")
        self.aggregators._models[selected_miner][_STAKE_BLOCKFED_TAG] += 1

        return selected_miner
