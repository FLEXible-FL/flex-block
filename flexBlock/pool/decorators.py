import functools
from flexBlock.pool.pool import CLIENT_CONNECTIONS


def send_weights_to_miner(func):
    @functools.wraps(func)
    def _collect_weights_(aggregator_flex_model, clients_flex_models, *args, **kwargs):
        if "weights" not in aggregator_flex_model:
            aggregator_flex_model["weights"] = []
        if (
            CLIENT_CONNECTIONS not in aggregator_flex_model
            or aggregator_flex_model[CLIENT_CONNECTIONS] is None
        ):
            # If the tag is not present or stores none then we assume
            # that the server is itself his only client

            client_weights = func(aggregator_flex_model, *args, **kwargs)

            aggregator_flex_model["weights"].append(client_weights)
            return

        for k in clients_flex_models:
            # Skip clients not connected to our blockchain node
            if k in aggregator_flex_model[CLIENT_CONNECTIONS]:
                client_weights = func(clients_flex_models[k], *args, **kwargs)
                aggregator_flex_model["weights"].append(client_weights)

    return _collect_weights_
