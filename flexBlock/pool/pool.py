from __future__ import annotations

import random
from typing import Callable

import numpy as np
from flex.actors.actors import FlexActors, FlexRole
from flex.data import FedDataset
from flex.model.model import FlexModel
from flex.pool import FlexPool

CLIENT_CONNS_BLOCKFED_TAG = "clients_connections"


class BlockChainPool(FlexPool):
    """
    A decoupled blockFed architecture pool.

    Attributes
    ----------
        - flex_data (FedDataset): The federated dataset prepared to be used.
        - flex_actors (FlexActors): Actors with its roles.
        - flex_models (defaultdict): A dictionary containing the each actor id,
        and initialized to None. The model to train by each actor will be initialized
        using the map function following the communication constraints.


    --------------------------------------------------------------------------
    """

    @classmethod
    def decoupled_blockfed_architecture(
        cls,
        fed_dataset: FedDataset,
        number_of_nodes: int,
        init_func: Callable,
        **kwargs,
    ):
        """In a decoupled blockfed architecture, a device may only participate in a blockchain schema or in a federated learning schema.
        This function is used when you have a FlexDataset and you want to start the learning phase following a decoupled BlockFed architecture.

        This method will assing to each id from the FlexDataset the client-role, and will create n BlockChain nodes that will act as aggregators.
        Also each client will be assigned to a node.

        Args: fed_dataset (FedDataset): Federated dataset used to train a model.
            number_of_nodes (int): The number of different blockchain nodes that will take part in the schema.

        Returns:
            FlexPool: A FlexPool with the assigned roles for a client-server architecture.
        """
        # TODO: change this condition maybe
        if "server" in fed_dataset.keys():
            raise ValueError(
                "The name 'server' is reserved only for the server in a client-server architecture."
            )

        if number_of_nodes < 1:
            raise ValueError("The number of nodes must be at least 1")

        actors = FlexActors(
            {actor_id: FlexRole.client for actor_id in fed_dataset.keys()}
        )
        actor_keys = [key for key in actors.keys()]
        random.shuffle(actor_keys)
        partition = np.array_split(actor_keys, number_of_nodes)

        for i in range(number_of_nodes):
            actors[f"server-{i+1}"] = FlexRole.server_aggregator

        models = {k: FlexModel() for k in actors}
        for i in range(number_of_nodes):
            server = models.get(f"server-{i+1}")
            assert isinstance(server, FlexModel)
            server[CLIENT_CONNS_BLOCKFED_TAG] = partition[i]

        new_arch = cls(
            flex_data=fed_dataset,
            flex_actors=actors,
            flex_models=models,
        )
        new_arch.servers.map(init_func, **kwargs)
        return new_arch

    def run_concensus(self, concensus_mechanism, blockchain, *args, **kwargs):
        aggregator_pool = self.aggregators
        models = [aggregator_pool._models.get(i) for i in aggregator_pool._actors]
        chossen_model = concensus_mechanism(models, blockchain, *args, **kwargs)
        return chossen_model
