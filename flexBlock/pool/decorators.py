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
import functools
from typing import List

from flex.common.utils import check_min_arguments
from flex.model import FlexModel
from flex.pool.decorators import ERROR_MSG_MIN_ARG_GENERATOR

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


def deploy_miner_model(func):
    min_args = 1
    assert check_min_arguments(func, min_args), ERROR_MSG_MIN_ARG_GENERATOR(
        func, min_args
    )

    @functools.wraps(func)
    def _deploy_model(
        server_flex_model: FlexModel,
        clients_flex_models: List[FlexModel],
        *args,
        **kwargs,
    ):
        if (
            CLIENT_CONNECTIONS not in server_flex_model
            or server_flex_model[CLIENT_CONNECTIONS] is None
        ):
            # If the tag is not present or stores none then we assume
            # that the server is itself his only client
            server_flex_model.update(func(server_flex_model, *args, **kwargs))
            return

        for k in clients_flex_models:
            if k in server_flex_model[CLIENT_CONNECTIONS]:
                clients_flex_models[k].update(func(server_flex_model, *args, **kwargs))

    return _deploy_model
