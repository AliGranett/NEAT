from gene import Gene
import numpy as np
# import requests
# import time
import torch
import torch.nn as nn


class Chromosome:
    def __init__(self,
                 connections: list[Gene] = None,
                 nodes_in: int = None,
                 nodes_out: int = None):
        """
        Gene class for the connection between two nodes
        Parameters:
        connections (list[Connection]): List of connections
        """
        self.network = None
        self.fitness = 0
        self.adjusted_fitness = 0

        if connections is not None:
            # Initialize with a custom list of connections
            self.connections = connections
            # Extract unique nodes from connections
            self.nodes = set()
            for gene in connections:
                self.nodes.update([gene.in_node, gene.out_node])
            self.nodes_in = nodes_in if nodes_in is not None else  \
                max([gene.in_node for gene in connections if gene.in_node <= nodes_in])
            self.nodes_out = nodes_out if nodes_out is not None else \
                max([gene.out_node for gene in connections if gene.out_node >
                    nodes_in]) - self.nodes_in
        else:
            # Initialize with specified input/output sizes and random connections
            assert nodes_in is not None and nodes_out is not None, \
                "nodes_in and nodes_out are required for random initialization"
            self.nodes_in = nodes_in
            self.nodes_out = nodes_out
            self.connections = []  # Initialize with an empty list
            # Input + Output node IDs
            self.nodes = set(range(1, nodes_in + nodes_out + 1))
            self.initialize_connections()

        self.graph = '\n'.join([str(connection)
                               for connection in self.connections])
        self.build_network()

    def __str__(self):
        return f"Chromosome:\nFitness: {self.fitness}\n{self.graph}"

    def __repr__(self):
        return self.__str__()

    def build_network(self):
        """
        Build the network from the connections
        """

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Create a dictionary of nodes
        node_set = set()
        for gene in self.connections:
            if gene.active:
                node_set.update([gene.in_node, gene.out_node])

        # Prepare network structure
        network_structure = {node: [] for node in node_set}
        for gene in self.connections:
            if gene.active:
                network_structure[gene.in_node].append(gene)

        # Assuming a simple feedforward network for demonstration purposes
        # For each node, create a module (if necessary,
        # considering its type and connections)
        modules = nn.ModuleDict()
        for node in node_set:
            # This example assumes a simple structure; customize as needed
            for gene in network_structure.get(node, []):
                modules[f'{gene.in_node}->{gene.out_node}'] = \
                    nn.Sigmoid(1, 1, bias=False)
                modules[f'{gene.in_node}->{gene.out_node}'].weight \
                    .data.fill_(gene.weight)

        # Store the modules for later use in the forward pass
        self.network = modules

    def initialize_connections(self):
        """
        Initialize the connections with random connections
        """
        for i in range(self.nodes_in):
            for j in range(self.nodes_in, self.nodes_in + self.nodes_out):
                if np.random.rand() < 0.5:
                    # Randomly initialize the connection
                    self.connections.append(
                        Gene(i + 1, j + 1, np.random.rand(), True, len(self.connections) + 1))
