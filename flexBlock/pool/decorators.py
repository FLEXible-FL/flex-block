import functools
import random
from typing import List

import numpy as np
from flex.model.model import FlexModel

from flexBlock.blockchain.blockchain import Blockchain
from flexBlock.pool.pool import CLIENT_CONNS_BLOCKFED_TAG


def send_weights_to_miner(func):
    @functools.wraps(func)
    def _collect_weights_(aggregator_flex_model, clients_flex_models, *args, **kwargs):
        if "weights" not in aggregator_flex_model:
            aggregator_flex_model["weights"] = []
        for k in clients_flex_models:
            # Skip clients not connected to our blockchain node
            if (
                aggregator_flex_model[CLIENT_CONNS_BLOCKFED_TAG] is None
                or k in aggregator_flex_model[CLIENT_CONNS_BLOCKFED_TAG]
            ):
                client_weights = func(clients_flex_models[k], *args, **kwargs)
                aggregator_flex_model["weights"].append(client_weights)

    return _collect_weights_


def concensus(deterministic=True):
    def wrapper(func):
        def _concensus_(
            aggregators: List[FlexModel], blockchain: Blockchain, *args, **kwargs
        ):
            # return the index of the chosen aggregator
            puntuations = [
                func(model, blockchain, *args, **kwargs) for model in aggregators
            ]

            if deterministic:
                index_of_max = max(enumerate(puntuations), key=lambda x: x[1])[0]
                return index_of_max

            choice_weights = np.exp(puntuations) / np.sum(np.exp(puntuations))
            return random.choices(range(len(aggregators)), weights=choice_weights)[0]

        return _concensus_

    return wrapper
