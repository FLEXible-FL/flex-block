import functools
from flexBlock.pool.pool import CLIENT_CONNS_BLOCKFED_TAG


def collect_clients_weights(func):
    @functools.wraps(func)
    def _collect_weights_(aggregator_flex_model, clients_flex_models, *args, **kwargs):
        if "weights" not in aggregator_flex_model:
            aggregator_flex_model["weights"] = []
        for k in clients_flex_models:
            # Skip clients to connected to our blockchain node
            if (
                aggregator_flex_model[CLIENT_CONNS_BLOCKFED_TAG] is None
                or k in aggregator_flex_model[CLIENT_CONNS_BLOCKFED_TAG]
            ):
                client_weights = func(clients_flex_models[k], *args, **kwargs)
                aggregator_flex_model["weights"].append(client_weights)

    return _collect_weights_
