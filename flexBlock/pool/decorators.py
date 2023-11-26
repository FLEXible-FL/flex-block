import functools
from flexBlock.pool.pool import CLIENT_CONNS_BLOCKFED_TAG


def send_weights_to_miner(func):
    @functools.wraps(func)
    def _collect_weights_(aggregator_flex_model, clients_flex_models, *args, **kwargs):
        if "weights" not in aggregator_flex_model:
            aggregator_flex_model["weights"] = []
        if CLIENT_CONNS_BLOCKFED_TAG not in aggregator_flex_model:
            # We assume that each server is a client and has
            # no other clients connected to it
            client_weights = func(aggregator_flex_model, *args, **kwargs)

            aggregator_flex_model["weights"].append(client_weights)
            return

        for k in clients_flex_models:
            # Skip clients not connected to our blockchain node
            if (
                aggregator_flex_model[CLIENT_CONNS_BLOCKFED_TAG] is None
                or k in aggregator_flex_model[CLIENT_CONNS_BLOCKFED_TAG]
            ):
                client_weights = func(clients_flex_models[k], *args, **kwargs)
                aggregator_flex_model["weights"].append(client_weights)

    return _collect_weights_


def validate_models(func):
    @functools.wraps(func)
    def _validate_models_(aggregator_flex_model, clients_flex_models, *args, **kwargs):
        valid_models = []
        if "valid_models" not in aggregator_flex_model:
            aggregator_flex_model["valid_models"] = []
        if CLIENT_CONNS_BLOCKFED_TAG not in aggregator_flex_model:
            # We assume that each server is a client and has
            # no other clients connected to it
            # TODO: We need cross validation
            return
        for k in clients_flex_models:
            # Skip clients not connected to our blockchain node
            if (
                aggregator_flex_model[CLIENT_CONNS_BLOCKFED_TAG] is None
                or k in aggregator_flex_model[CLIENT_CONNS_BLOCKFED_TAG]
            ):
                if func(clients_flex_models[k], *args, **kwargs):
                    valid_models.append(clients_flex_models[k].actor_id)

    return _validate_models_
