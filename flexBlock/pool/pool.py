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

from flexBlock.blockchain.blockchain import (
    Blockchain,
    BlockchainPoFL,
    BlockchainPoS,
    BlockchainPow,
    BlockPoFL,
    BlockPoS,
    BlockPoW,
)
from flexBlock.common import CLIENT_CONNECTIONS, DEBUG
from flexBlock.pool.utils import create_miners

_STAKE_BLOCKFED_TAG = "stake"
_INITIAL_STAKE = 32

_BlockchainType = TypeVar("_BlockchainType", bound=Blockchain)


@dataclass(frozen=True)
class PoolConfig:
    """
    Represents the configuration settings for a pool.

    Attributes:
        gossip_before_consensus: Boolean indicating if the gossiping mechanism should be executed before the consensus mechanism.
        gossip_on_agg: Boolean indicating if the gossiping mechanism should be executed before the aggregation.
        aggregate_before_consensus: Boolean indicating if the aggregation should be executed before the consensus mechanism.
    """

    gossip_before_consensus: bool = True
    gossip_on_agg: bool = True
    aggregate_before_consensus: bool = False


_default_pool_config = PoolConfig()


class BlockchainPool(ABC, Generic[_BlockchainType]):
    """
    A class representing a blockchain pool.

    Attributes:
        _pool (FlexPool): The underlying FlexPool object.
        _blockchain (_BlockchainType): The blockchain object.
        _config (PoolConfig): The configuration for the pool.

    Methods:
        initialize_pool: Initializes the pool with the given blockchain, pool, and configuration.
        blockchain: Returns the blockchain object.
        actor_ids: Returns the actor IDs in the pool.
        clients: Returns the clients in the pool.
        aggregators: Returns the aggregators in the pool.
        servers: Returns the servers in the pool.
        _gossip: Gossiping mechanism for the pool.
        _gossip_to_miner: Gossiping mechanism for the pool with a selected miner.
        aggregate: Aggregates weights in the pool.
        pack_block: Abstract method to pack a block with weights.
        consensus_mechanism: Abstract method to determine the consensus mechanism.

    """

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
        """
        Initialize the pool with the given blockchain, pool, and configuration. Should be called in the constructor of the subclass.

        Args:
        ----
            blockchain (_BlockchainType): The blockchain object.
            pool (FlexPool): The pool object.
            config (PoolConfig, optional): The configuration for the pool. Defaults to default pool configuration.
            **kwargs: Additional keyword arguments.

        """
        self._pool = pool
        self._blockchain = blockchain
        self._config = config

    @property
    def blockchain(self):
        """
        Returns the blockchain of the pool.

        Returns
        -------
            _BlockchainType: The blockchain object.

        """
        return self._blockchain

    @functools.cached_property
    def actor_ids(self):
        """
        Returns the actor IDs in the pool.

        Returns
        -------
            FlexPool: The pool containing all the actor IDs.

        """
        return self._pool.actor_ids

    @functools.cached_property
    def clients(self):
        """
        Returns the clients in the pool.

        Returns:
            FlexPool: The pool containing all the clients.

        """
        return self._pool.clients

    @functools.cached_property
    def aggregators(self):
        """
        Returns the aggregators in the pool.

        Returns:
            FlexPool: The pool containing all the aggregators.

        """
        return self._pool.aggregators

    @functools.cached_property
    def servers(self):
        """
        Returns the servers in the pool.

        Returns:
            FlexPool: The pool containing all the servers.

        """
        return self._pool.servers

    def _gossip(self):
        """
        Gossiping mechanism for the pool. The miners will share the weights of their clients with each other.

        """
        miners = self._pool.aggregators
        total_weights = [
            weight for miner in miners._models.values() for weight in miner["weights"]
        ]

        for key in miners._models.keys():
            miners._models[key]["weights"] = total_weights

    def _gossip_to_miner(self, selected_miner):
        """
        Gossiping mechanism for the pool. Only the selected miner will get all the weights. This is done for efficiency where a lot
        of miners are present in the pool (i.e. Proof of Stake).

        Args:
        ----
            selected_miner: The selected miner.

        """
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
        """
        Run the aggregation mechanism for the pool. This implies gossiping, consensus mechanism, and aggregation.

        Args:
        ----
            agg_function (Callable): The aggregation operator.
            set_weights (Callable): The function to set the weights for the server.
            **kwargs: Additional keyword arguments. For compulsory arguments, see the `consensus_mechanism` method for a given subclass.

        Returns:
        -------
            bool: True if the aggregation was performed, False otherwise. Some subclasses may always return True.

        """
        if self._config.gossip_before_consensus:
            self._gossip()

        if self._config.aggregate_before_consensus:
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

        gossip_after_consensus = (
            self._config.gossip_on_agg and not self._config.gossip_before_consensus
        )
        if gossip_after_consensus:
            self._gossip_to_miner(selected_miner)

        selected_model = self.aggregators._models[selected_miner]

        if not self._config.aggregate_before_consensus or gossip_after_consensus:
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
        """
        Abstract method to pack a block with weights.

        Args:
        ----
            weights: The weights to pack.

        Returns:
        -------
            The packed block.

        """
        pass

    @abstractmethod
    def consensus_mechanism(
        self, miners: Dict[Hashable, FlexModel], **kwargs
    ) -> Optional[Hashable]:
        """
        Abstract method to determine the consensus mechanism.

        Args:
        ----
        ----
            miners (Dict[Hashable, FlexModel]): The miners in the pool.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            Optional[Hashable]: The selected miner. If None is returned, no miner was selected.

        """
        pass

    def __len__(self):
        """
        Return the ammount of nodes in the pool.

        Returns
        -------
            int: The ammount of nodes in the pool.

        """
        return len(self._pool)


class PoWBlockchainPool(BlockchainPool):
    """
    A class representing a Proof-of-Work (PoW) blockchain pool. Miners and clients are disjoint sets.
    """

    def __init__(
        self,
        fed_dataset: FedDataset,
        init_func: Callable,
        number_of_miners: int,
        **kwargs,
    ):
        """
        Initialize the Pool object. A PoW blockchain is used.

        Args:
            fed_dataset (FedDataset): The federated dataset.
            init_func (Callable): The initialization function to be called on the servers.
            number_of_miners (int): The number of miners in the pool.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the number of miners is less than 1.
        """

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
            gossip_before_consensus=False,
            gossip_on_agg=True,
            aggregate_before_consensus=False,
        )

        self.initialize_pool(bc, pool, config, **kwargs)

    def pack_block(self, weights):
        return BlockPoW(weights=weights)

    def consensus_mechanism(self, miners, **kwargs):
        """
        Implement the consensus mechanism for selecting a miner to mine the next block.

        Args:
        ----
            miners (dict): A dictionary containing the miners and their respective keys.
            **kwargs: Additional keyword arguments. None is expected.

        Returns:
        -------
            str: The key of the selected miner.

        """
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
    """
    A class representing a Proof-of-Learning (PoFL) blockchain pool.

    The set of miners and clients are disjoint. The pool is initialized with a PoFL blockchain.
    """

    def __init__(
        self,
        fed_dataset: FedDataset,
        init_func: Callable,
        number_of_miners: int,
        **kwargs,
    ):
        """
        Initializes a Pool object.

        Args:
            fed_dataset (FedDataset): The federated dataset.
            init_func (Callable): The initialization function to be called on the pool's servers.
            number_of_miners (int): The number of miners in the pool.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the number of miners is less than 1.
        """

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
            gossip_before_consensus=False,
            aggregate_before_consensus=True,
            gossip_on_agg=False,
        )

        self.initialize_pool(
            bc,
            pool,
            config=config,
            **kwargs,
        )

    def pack_block(self, weights):
        """
        Packs the given weights into a PoFL block.

        Args:
        ----
            weights: The weights to be packed into the block.

        Returns:
        -------
            BlockPoFL: The PoFL block containing the weights.

        """
        return BlockPoFL(weights=weights)

    def consensus_mechanism(self, miners, **kwargs):
        """
        Implement the consensus mechanism for selecting the miner with the highest accuracy.

        Args:
        ----
            miners: A dictionary mapping miner IDs to their corresponding models.
            eval_function: The evaluation function to be used. It measures the accuracy of the models.
            eval_dataset: The dataset to be used for evaluation.
            accuracy: The minimum accuracy required for a miner to be selected.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            str: The ID of the miner with the highest accuracy, or None if no valid miners are found.

        Raises:
        ------
            AssertionError: If the evaluation function, evaluation dataset, or accuracy are not provided.

        """
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
                    print(
                        f"[POFL] miner: {miner:10} acc: {acc} target: {accuracy}",
                        flush=True,
                    )
                valid_miners.append((miner, acc))

        valid_miners.sort(key=lambda x: x[1])

        return valid_miners[-1][0] if len(valid_miners) > 0 else None


class PoSBlockchainPool(BlockchainPool):
    """
    A class representing a Proof-of-Stake (PoS) blockchain pool.

    The sets of miners and clients fully overlap. The pool is initialized with a PoS blockchain.
    """

    class Pool:
        def __init__(
            self,
            fed_dataset: FedDataset,
            init_func: Callable,
            initial_stake: int = _INITIAL_STAKE,
            **kwargs,
        ):
            """
            Initializes a Pool object.

            Args:
                fed_dataset (FedDataset): The federated dataset used by the pool.
                init_func (Callable): The initialization function used to create the pool.
                initial_stake (int, optional): The initial stake for each miner.
                **kwargs: Additional keyword arguments.
            """
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
        """
        Implements the consensus mechanism for selecting a miner.

        Args:
            miners: The list of miners.

        """
        key_stake = [(k, m.get(_STAKE_BLOCKFED_TAG)) for k, m in miners.items()]

        key_stake = list(filter(lambda x: x[1] and x[1] > 0, key_stake))
        selected_miner = random.choices(key_stake, [s for _, s in key_stake])[0][0]
        if DEBUG >= 1:
            print(
                f"[PoS] selected miner: {selected_miner: 10} stake: {self.aggregators._models[selected_miner][_STAKE_BLOCKFED_TAG]}"
            )
        self.aggregators._models[selected_miner][_STAKE_BLOCKFED_TAG] += 1

        return selected_miner
