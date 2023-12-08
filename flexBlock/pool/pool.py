from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy

import random
from typing import Callable, Generic, TypeVar, Hashable, Dict

import numpy as np
from numpy.random import permutation
from flex.actors.actors import FlexActors, FlexRole
from flex.data import FedDataset, Dataset
from flex.model.model import FlexModel
from flex.pool import FlexPool

import functools

from flexBlock.blockchain.blockchain import (
    Blockchain,
    BlockPoW,
    BlockchainPoS,
    BlockPoS,
    BlockchainPow,
    BlockPoFL,
    BlockchainPoFL,
)

CLIENT_CONNECTIONS = "clients_connections"
_STAKE_BLOCKFED_TAG = "stake"
_INITIAL_STAKE = 32

_BlockchainType = TypeVar("_BlockchainType", bound=Blockchain)


class BlockchainPool(ABC, Generic[_BlockchainType]):
    _pool: FlexPool
    _blockchain: _BlockchainType

    def initialize_pool(
        self,
        blockchain: _BlockchainType,
        pool: FlexPool,
        **kwargs,
    ):
        self._pool = pool
        self._blockchain = blockchain

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

    def gossip(self):
        """Gossiping mechanism for the pool. The miners will share the weights of their clients with each other."""
        miners = self._pool.aggregators
        total_weights = [
            weight for miner in miners._models.values() for weight in miner["weights"]
        ]

        for key in miners._models.keys():
            miners._models[key]["weights"] = total_weights

    def aggregate(
        self,
        agg_function: Callable,
        gossip: bool = True,
        *args,
        **kwargs,
    ):
        if gossip:
            self.gossip()

        selected_server = self.consensus_mechanism(
            miners=self._pool.aggregators._models, **kwargs
        )
        agg_function(self.aggregators._models[selected_server], None)
        weights = deepcopy(
            self.aggregators._models[selected_server]["aggregated_weights"]
        )
        self._blockchain.add_block(self.pack_block(weights=weights))
        for v in self.aggregators._models.values():
            v["aggregated_weights"] = [deepcopy(weights)]

    @abstractmethod
    def pack_block(self, weights):
        pass

    @abstractmethod
    def consensus_mechanism(self, miners: Dict[Hashable, FlexModel], **kwargs):
        pass

    def __len__(self):
        return len(self._pool)


class PoWBlockchainPool(BlockchainPool):
    def __init__(
        self,
        fed_dataset: FedDataset,
        number_of_miners: int,
        init_func: Callable,
        **kwargs,
    ):
        if number_of_miners < 1:
            raise ValueError("The number of nodes must be at least 1")

        # Split clients between miners
        actors = FlexActors(
            {actor_id: FlexRole.client for actor_id in fed_dataset.keys()}
        )

        actors, models = self._create_miners(actors, number_of_miners)

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

        self.initialize_pool(bc, pool, **kwargs)

    def _create_miners(self, actors: FlexActors, number_of_miners: int):
        # Create miners
        for i in range(number_of_miners):
            actors[f"server-{i+1}"] = FlexRole.server_aggregator

        # Populates actors with miners
        shuffled_actor_keys = permutation([key for key in actors.keys()])
        partition = np.array_split(shuffled_actor_keys, number_of_miners)

        models = {k: FlexModel() for k in actors}

        for i in range(number_of_miners):
            server = models.get(f"server-{i+1}")
            assert isinstance(server, FlexModel)
            server[CLIENT_CONNECTIONS] = partition[i]

        for k in models:
            # Store the key in the model so we can retrieve it later
            models[k].actor_id = k

        return actors, models

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

        return miner_keys[selected_miner_index]


class PoFLBlockchainPool(BlockchainPool):
    def __init__(
        self,
        fed_dataset: FedDataset,
        init_func: Callable,
        mining_dataset: Dataset,
        **kwargs,
    ):
        pool = FlexPool.p2p_architecture(fed_dataset, init_func, **kwargs)
        bc = (
            BlockchainPoFL(genesis_block=BlockPoFL([]), **kwargs)
            if "blockchain" not in kwargs
            else kwargs["blockchain"]
        )

        self._mining_dataset = mining_dataset
        self.initialize_pool(bc, pool, **kwargs)

    def pack_block(self, weights):
        return BlockPoFL(weights=weights)

    def consensus_mechanism(self, miners, **kwargs):
        eval_function = kwargs.get("eval_function")
        train_function = kwargs.get("train_function")

        assert (
            eval_function is not None
        ), "eval_function must be provided to the aggregate method"
        assert (
            train_function is not None
        ), "train_function must be provided to the aggregate method"

        miner_keys = list(miners.keys())
        previous_block = self._blockchain.get_last_block()
        assert isinstance(previous_block, BlockPoFL)
        selected_miner_index = 0

        while True:
            train_function(
                self.clients._models[miner_keys[selected_miner_index]],
                self._mining_dataset,
            )

            err = eval_function(
                self.clients._models[miner_keys[selected_miner_index]],
                self._mining_dataset,
            )
            if err <= previous_block.target_err:
                break

            selected_miner_index += 1
            selected_miner_index %= len(miner_keys)

        return miner_keys[selected_miner_index]


class PoSBlockchainPool(BlockchainPool):
    def __init__(
        self,
        fed_dataset: FedDataset,
        init_func: Callable,
        initial_stake: int = _INITIAL_STAKE,
        **kwargs,
    ):
        pool = FlexPool.p2p_architecture(fed_dataset, init_func, **kwargs)
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

        self.initialize_pool(bc, pool, **kwargs)

    def pack_block(self, weights):
        return BlockPoS(weights=weights)

    def consensus_mechanism(self, miners, **kwargs):
        key_stake = [(k, m.get(_STAKE_BLOCKFED_TAG)) for k, m in miners.items()]

        key_stake = list(filter(lambda x: x[1] and x[1] > 0, key_stake))
        selected_miner = random.choices(key_stake, [s for _, s in key_stake])[0][0]
        self.aggregators._models[selected_miner][_STAKE_BLOCKFED_TAG] += 1

        return selected_miner
