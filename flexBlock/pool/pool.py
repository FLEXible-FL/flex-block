from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy

import random
from typing import Callable

import numpy as np
from flex.actors.actors import FlexActors, FlexRole
from flex.data import FedDataset, Dataset
from flex.model.model import FlexModel
from flex.pool import FlexPool

import functools

from flexBlock.blockchain.blockchain import (
    _Blockchain,
    BlockPoW,
    BlockchainPoS,
    BlockPoS,
    BlockchainPow,
    BlockPoFL,
    BlockchainPoFL,
)

CLIENT_CONNS_BLOCKFED_TAG = "clients_connections"
STAKE_BLOCKFED_TAG = "stake"
INITIAL_STAKE = 32


class _BlockchainPool(ABC):
    def __init__(
        self,
        blockType: type,
        blockchain: _Blockchain,
        pool: FlexPool,
        **kwargs,
    ):
        self._pool = pool
        self._blockchain = blockchain
        self._blockchain.add_block(blockType(weights=[]))

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

    @abstractmethod
    def aggregate(
        self,
        agg_function: Callable,
        *args,
        **kwargs,
    ):
        pass

    def __len__(self):
        return len(self._pool)


class PoWBlockchainPool(_BlockchainPool):
    def __init__(
        self,
        fed_dataset: FedDataset,
        number_of_miners: int,
        init_func: Callable,
        **kwargs,
    ):
        # TODO: Separate this into other functions

        if "server" in fed_dataset.keys():
            # TODO: Change this to another message
            raise ValueError(
                "The name 'server' is reserved only for the server in a client-server architecture."
            )

        if number_of_miners < 1:
            raise ValueError("The number of nodes must be at least 1")

        # Split clients between miners
        actors = FlexActors(
            {actor_id: FlexRole.client for actor_id in fed_dataset.keys()}
        )
        actor_keys = [key for key in actors.keys()]
        random.shuffle(actor_keys)
        partition = np.array_split(actor_keys, number_of_miners)

        for i in range(number_of_miners):
            actors[f"server-{i+1}"] = FlexRole.server_aggregator

        models = {k: FlexModel() for k in actors}
        for i in range(number_of_miners):
            server = models.get(f"server-{i+1}")
            assert isinstance(server, FlexModel)
            server[CLIENT_CONNS_BLOCKFED_TAG] = partition[i]

        for k in models:
            models[k].actor_id = k

        # Create pool and initialize servers
        pool = FlexPool(
            flex_data=fed_dataset,
            flex_actors=actors,
            flex_models=models,
        )

        bc = BlockchainPow() if "blockchain" not in kwargs else kwargs["blockchain"]

        super().__init__(BlockPoW, bc, pool, **kwargs)
        self._pool.servers.map(init_func, **kwargs)

    def aggregate(self, agg_function, *args, **kwargs):
        selected_server = self._proof_of_work()
        agg_function(self.aggregators._models[selected_server], None)
        weights = deepcopy(
            self.aggregators._models[selected_server]["aggregated_weights"]
        )
        self._blockchain.add_block(BlockPoW(weights=weights))
        for v in self.aggregators._models.values():
            v["weights"] = [deepcopy(weights)]

    def _proof_of_work(self, *args, **kwargs):
        miners = list(self._pool.aggregators._models.keys())
        previous_block = self._blockchain.get_last_block()
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
            selected_miner_index %= len(miners)

        return miners[selected_miner_index]


class PoFLBlockchainPool(_BlockchainPool):
    def __init__(
        self,
        fed_dataset: FedDataset,
        init_func: Callable,
        mining_dataset: Dataset,
        **kwargs,
    ):
        pool = FlexPool.p2p_architecture(fed_dataset, init_func, **kwargs)
        bc = (
            BlockchainPoFL(
                max_err=kwargs.get("max_err"),
                min_err=kwargs.get("min_err"),
                seconds_to_mine=kwargs.get("seconds_to_mine"),
            )
            if "blockchain" not in kwargs
            else kwargs["blockchain"]
        )

        self._mining_dataset = mining_dataset
        super().__init__(BlockPoFL, bc, pool, **kwargs)

    def aggregate(
        self,
        agg_function: Callable,
        eval_function: Callable,
        train_function: Callable,
        *args,
        **kwargs,
    ):
        selected_server = self._proof_of_federated_learning(
            eval_function, train_function, *args, **kwargs
        )
        agg_function(self.aggregators._models[selected_server], None)
        weights = deepcopy(
            self.aggregators._models[selected_server]["aggregated_weights"]
        )
        self._blockchain.add_block(BlockPoFL(weights=weights))
        for v in self.aggregators._models.values():
            v["weights"] = [deepcopy(weights)]

    def _proof_of_federated_learning(
        self, eval_function: Callable, train_function: Callable, *args, **kwargs
    ):
        miners = list(self._pool.aggregators._models.keys())
        previous_block = self._blockchain.get_last_block()
        assert isinstance(previous_block, BlockPoFL)
        selected_miner_index = 0

        while True:
            train_function(
                self.clients._models[miners[selected_miner_index]],
                self._mining_dataset,
                *args,
                **kwargs,
            )

            err = eval_function(
                self.clients._models[miners[selected_miner_index]],
                self._mining_dataset,
                *args,
                **kwargs,
            )
            if err <= previous_block.target_err:
                break

            selected_miner_index += 1
            selected_miner_index %= len(miners)

        return miners[selected_miner_index]


class PoSBlockchainPool(_BlockchainPool):
    def __init__(
        self,
        fed_dataset: FedDataset,
        init_func: Callable,
        initial_stake: int = INITIAL_STAKE,
        **kwargs,
    ):
        pool = FlexPool.p2p_architecture(fed_dataset, init_func, **kwargs)
        bc = (
            BlockchainPoS(
                seconds_to_mine=kwargs.get("seconds_to_mine"),
            )
            if "blockchain" not in kwargs
            else kwargs["blockchain"]
        )

        miners = list(pool.aggregators._models.values())
        for miner in miners:
            miner[STAKE_BLOCKFED_TAG] = initial_stake

        super().__init__(BlockPoS, bc, pool, **kwargs)

    def aggregate(
        self,
        agg_function: Callable,
        *args,
        **kwargs,
    ):
        selected_miner = self._proof_of_stake(*args, **kwargs)
        agg_function(self.aggregators._models[selected_miner], None)
        weights = deepcopy(
            self.aggregators._models[selected_miner]["aggregated_weights"]
        )
        self._blockchain.add_block(BlockPoS(weights=weights))

        for v in self.aggregators._models.values():
            v["weights"] = [deepcopy(weights)]

        self.aggregators._models[selected_miner][STAKE_BLOCKFED_TAG] += 1

    def _proof_of_stake(self, *args, **kwargs):
        key_stake = [
            (k, m.get(STAKE_BLOCKFED_TAG))
            for k, m in self._pool.aggregators._models.items()
        ]

        key_stake = list(filter(lambda x: x[1] and x[1] > 0, key_stake))
        return random.choices(key_stake, [s for _, s in key_stake])[0][0]
