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


def deploy_server_to_miner_wrapper(func):
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
