import copy
import pprint

import numpy as np
from flex.data import Dataset, FedDataDistribution
from flex.model import FlexModel
from flex.pool import aggregate_weights, deploy_server_model, init_server_model
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from flexBlock.pool import (
    PoSBlockchainPool,
    send_weights_to_miner,
)


@aggregate_weights
def aggregate(list_of_weights: list):
    return np.mean(np.asarray(list_of_weights, dtype=object), axis=0)


def train(client_flex_model: FlexModel, client_data: Dataset):
    client_flex_model["model"].fit(client_data.X_data, client_data.y_data)


@send_weights_to_miner
def get_clients_weights(client_flex_model: FlexModel):
    return [client_flex_model["model"].intercept_, client_flex_model["model"].coef_]


@init_server_model
def build_server_model(**kwargs):
    flex_model = FlexModel()
    flex_model["model"] = LinearRegression()
    return flex_model


@deploy_server_model
def copy_server_model_to_clients(server_flex_model: FlexModel):
    return copy.deepcopy(server_flex_model)


def evaluate_model(client_flex_model: FlexModel, client_data: Dataset):
    return client_flex_model["model"].score(client_data.X_data, client_data.y_data)


# Load the diabetes dataset
diabetes = load_diabetes()

# Generate train-test splits
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data[:, np.newaxis, 2], diabetes.target, test_size=0.33, random_state=42
)

# We are going to use the train dataset for mining since we are supossed to learn from it too
train_diabetes = Dataset.from_numpy(X_train, y_train)
test_diabetes = Dataset.from_numpy(X_test, y_test)
federated_diabetes = FedDataDistribution.iid_distribution(train_diabetes, n_clients=5)

# Create pool
p = PoSBlockchainPool(
    fed_dataset=federated_diabetes,
    init_func=build_server_model,
    seconds_to_mine=3,
)

servers = p.servers
aggregators = p.aggregators
clients = p.clients
print(
    f"Number of nodes in the pool {len(p.actor_ids)}. All of them are miners and clients."
)

for _ in range(10):
    servers.map(copy_server_model_to_clients, clients)
    clients.map(train)
    aggregators.map(get_clients_weights, clients)
    p.gossip()

    # Aggregate weights
    p.aggregate(aggregate, eval_function=evaluate_model, train_function=train)
    pprint.pprint(aggregators._models)

    block = p._blockchain.chain[-1]
    print(f"Block hash: {block.hash}")
    print(f"Block weights: {block.weights}")
    print(f"len of chain: {len(p._blockchain.chain)}")
