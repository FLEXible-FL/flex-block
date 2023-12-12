from flex.model import FlexModel

from flexBlock.pool.decorators import send_weights_to_miner


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
