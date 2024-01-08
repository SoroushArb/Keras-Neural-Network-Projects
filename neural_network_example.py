import numpy as np

# Manual calculation of a simple neural network
weights = np.around(np.random.uniform(size=6), decimals=2)
biases = np.around(np.random.uniform(size=3), decimals=2)
x_1, x_2 = 0.5, 0.85

# Forward propagation for a simple neural network
z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
a_11 = 1.0 / (1.0 + np.exp(-z_11))
z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]
a_12 = 1.0 / (1.0 + np.exp(-z_12))
z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
a_2 = 1.0 / (1.0 + np.exp(-z_2))

# Generalizing the neural network
n = 2
num_hidden_layers = 2
m = [2, 2]
num_nodes_output = 1

# Initializing weights and biases for a generalized neural network
num_nodes_previous = n
network = {}

for layer in range(num_hidden_layers + 1):
    if layer == num_hidden_layers:
        layer_name = 'output'
        num_nodes = num_nodes_output
    else:
        layer_name = f'layer_{layer + 1}'
        num_nodes = m[layer]

    network[layer_name] = {}
    for node in range(num_nodes):
        node_name = f'node_{node + 1}'
        network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'bias': np.around(np.random.uniform(size=1), decimals=2),
        }

    num_nodes_previous = num_nodes

# Function to initialize a neural network with given parameters
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs
    network = {}

    for layer in range(num_hidden_layers + 1):
        if layer == num_hidden_layers:
            layer_name = 'output'
            num_nodes = num_nodes_output
        else:
            layer_name = f'layer_{layer + 1}'
            num_nodes = num_nodes_hidden[layer]

        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = f'node_{node + 1}'
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }

        num_nodes_previous = num_nodes

    return network

# Function to compute the weighted sum of inputs
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

# Function for node activation using sigmoid
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-weighted_sum))

# Function for forward propagation through the network
def forward_propagate(network, inputs):
    layer_inputs = list(inputs)

    for layer in network:
        layer_data = network[layer]
        layer_outputs = []

        for layer_node in layer_data:
            node_data = layer_data[layer_node]
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))

        if layer != 'output':
            print(f'The outputs of the nodes in hidden layer number {layer.split("_")[1]} is {layer_outputs}')

        layer_inputs = layer_outputs

    network_predictions = layer_outputs
    return network_predictions

# Example usage of the functions
my_network = initialize_network(5, 3, [2, 3, 2], 3)
inputs = np.around(np.random.uniform(size=5), decimals=2)
predictions = forward_propagate(my_network, inputs)
print(f'The predicted values by the network for the given input are {predictions}')

# Additional tasks as described in the comments
small_network = initialize_network(5, 3, [3, 2, 3], 1)
small_predictions = forward_propagate(small_network, inputs)
print(f'The predicted values by the small network are {small_predictions}')
output_of_first_node = node_activation(compute_weighted_sum(inputs, small_network['layer_1']['node_1']['weights'], small_network['layer_1']['node_1']['bias']))
print(f'The output of the first node in the first hidden layer is {np.around(output_of_first_node[0], decimals=4)}')
