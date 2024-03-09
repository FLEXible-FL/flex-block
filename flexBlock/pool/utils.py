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
from typing import Dict, Hashable, Tuple

import numpy as np
from flex.actors import FlexActors, FlexRole
from flex.model import FlexModel
from numpy.random import permutation

from flexBlock.common import CLIENT_CONNECTIONS


def create_miners(
    actors: FlexActors, number_of_miners: int, connection_tag: str = CLIENT_CONNECTIONS
) -> Tuple[FlexActors, Dict[Hashable, FlexModel]]:
    """Modify actors by creating a given ammount of miners and
    assinging the previous actors to them through the given connection_tag.

    Args:
        actors (FlexActors): The flex actors to be modified.
        number_of_miners (int): The ammount of miners to be created.
        connection_tag (str): The tag to be used to store the connection

    Returns:
    -------
        Tuple[FlexActors, Dict[Hashable, FlexModel]]: A tuple containing the modified actors
        and a dictionary containing the models of the miners.
    """
    for i in range(number_of_miners):
        actors[f"server-{i+1}"] = FlexRole.server_aggregator

    # Populates actors with miners
    shuffled_actor_keys = permutation(
        [
            key
            for key in actors.keys()
            if not (isinstance(key, str) and key.startswith("server-"))
        ]
    )
    partition = np.array_split(shuffled_actor_keys, number_of_miners)

    models = {k: FlexModel() for k in actors}
    for i in range(number_of_miners):
        server = models.get(f"server-{i+1}")
        assert isinstance(server, FlexModel)
        server[connection_tag] = partition[i]

    for k in models:
        # Store the key in the model so we can retrieve it later
        models[k].actor_id = k

    return actors, models
