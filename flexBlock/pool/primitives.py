from flex.model import FlexModel
from flexBlock.pool.decorators import deploy_miner_model, send_weights_to_miner


def collect_to_send_wrapper(func):
    """Allows to change a function wrapped by `collect_client_weights` to a compatible flex-block function with `send_weights_to_miner`.

    Args:
        func: Function to be wrapped.

    Returns:
        Function that can be used with blockchain methods.
    """

    @send_weights_to_miner
    def wrapped(model: FlexModel, *args, **kwargs):
        return func.__wrapped__(model, *args, **kwargs)

    return wrapped


def deploy_server_to_miner(func):
    """Allows to change a function wrapped by `deploy_server_model` to a compatible flex-block function with `deploy_miner_model`.

    Args:
        func: Function to be wrapped.

    Returns:
        Function that can be used with blockchain methods.
    """

    @deploy_miner_model
    def wrapped(model: FlexModel, *args, **kwargs):
        return func.__wrapped__(model, *args, **kwargs)

    return wrapped
