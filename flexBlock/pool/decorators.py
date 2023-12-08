import functools
from flexBlock.pool.pool import _CLIENT_CONNS_BLOCKFED_TAG


def send_weights_to_miner(func):
    @functools.wraps(func)
    def _collect_weights_(aggregator_flex_model, clients_flex_models, *args, **kwargs):
        if "weights" not in aggregator_flex_model:
            aggregator_flex_model["weights"] = []
        if _CLIENT_CONNS_BLOCKFED_TAG not in aggregator_flex_model:
            # We assume that each server is a client and has
            # no other clients connected to it
            client_weights = func(aggregator_flex_model, *args, **kwargs)

            aggregator_flex_model["weights"].append(client_weights)
            return

        for k in clients_flex_models:
            # Skip clients not connected to our blockchain node
            if (
                aggregator_flex_model[_CLIENT_CONNS_BLOCKFED_TAG] is None
                or k in aggregator_flex_model[_CLIENT_CONNS_BLOCKFED_TAG]
            ):
                client_weights = func(clients_flex_models[k], *args, **kwargs)
                aggregator_flex_model["weights"].append(client_weights)

    return _collect_weights_
