"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI).

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

from copy import deepcopy
import unittest
from flexBlock.pool.utils import create_miners
from flex.actors import FlexActors, FlexRole


class TestUtils(unittest.TestCase):
    def test_when_creating_miners_then_servers_are_added_as_actors(self):
        actors = FlexActors(
            {
                "actor1": FlexRole.client,
                "actor2": FlexRole.client,
                "actor3": FlexRole.client,
            }
        )
        number_of_miners = 2
        connection_tag = "CLIENT_CONNECTIONS"

        modified_actors, _ = create_miners(
            deepcopy(actors), number_of_miners, connection_tag
        )

        # Check if the number of actors is increased by the number_of_miners
        self.assertEqual(len(modified_actors), len(actors) + number_of_miners)

        # Check if the miners are added with the correct roles
        for i in range(1, number_of_miners + 1):
            miner_key = f"server-{i}"
            self.assertEqual(modified_actors[miner_key], FlexRole.server_aggregator)

    def test_when_creating_miners_then_clients_are_splitted_to_miners(self):
        actors = FlexActors(
            {
                "actor1": FlexRole.client,
                "actor2": FlexRole.client,
                "actor3": FlexRole.client,
            }
        )
        number_of_miners = 2
        connection_tag = "CLIENT_CONNECTIONS"

        modified_actors, models = create_miners(
            deepcopy(actors), number_of_miners, connection_tag
        )

        # Check if the models dictionary is populated correctly
        self.assertEqual(len(models), len(modified_actors))

        # Check if the connection_tag is set correctly for each miner
        for i in range(1, number_of_miners + 1):
            miner_key = f"server-{i}"
            assert connection_tag in models[miner_key]

        connection_first = set(models["server-1"][connection_tag])
        connection_second = set(models["server-2"][connection_tag])
        assert len(connection_first.intersection(connection_second)) == 0
        self.assertEqual(connection_first.union(connection_second), set(actors.keys()))
