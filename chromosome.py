from gene import Gene
import numpy as np
import requests
import time

class Chromosome:
    def __init__(self, connections: list[Gene] = None, num_inputs: int = None, num_outputs: int = None):
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
            self.num_inputs = num_inputs if num_inputs is not None else max([gene.in_node for gene in connections if gene.in_node <= num_inputs])
            self.num_outputs = num_outputs if num_outputs is not None else max([gene.out_node for gene in connections if gene.out_node > num_inputs]) - self.num_inputs
        else:
            # Initialize with specified input/output sizes and random connections
            assert num_inputs is not None and num_outputs is not None, "num_inputs and num_outputs are required for random initialization"
            self.num_inputs = num_inputs
            self.num_outputs = num_outputs
            self.connections = []  # Initialize with an empty list
            self.nodes = set(range(1, num_inputs + num_outputs + 1))  # Input + Output node IDs
            self.initialize_connections()
        
            self.graph = self.dot_to_ascii(self.to_dot())


    def to_dot(self):
        dot_str = "digraph G {\n"
        dot_str += "    rankdir=LR;\n"  # Set graph direction left to right
        dot_str += "    node [shape=circle, style=filled];\n"
        
        # Input Nodes
        dot_str += "    subgraph cluster_input {\n"
        dot_str += "        node [color=lightblue];\n"
        for i in range(1, self.num_inputs + 1):
            dot_str += f"        {i} [label=\"Input {i}\"];\n"
        dot_str += "        label = \"Inputs\";\n"
        dot_str += "    }\n"
        
        # Output Nodes
        dot_str += "    subgraph cluster_output {\n"
        dot_str += "        node [color=lightcoral];\n"
        for i in range(self.num_inputs + 1, self.num_inputs + self.num_outputs + 1):
            dot_str += f"        {i} [label=\"Output {i-self.num_inputs}\"];\n"
        dot_str += "        label = \"Outputs\";\n"
        dot_str += "    }\n"

        # Assuming hidden nodes might exist, identify and add them
        all_nodes = {gene.in_node for gene in self.connections} | {gene.out_node for gene in self.connections}
        hidden_nodes = all_nodes - set(range(1, self.num_inputs + self.num_outputs + 1))
        for node in hidden_nodes:
            dot_str += f"    {node} [label=\"Hidden {node}\", color=gray];\n"

        # Connections
        for gene in self.connections:
            if gene.active:
                dot_str += f"    {gene.in_node} -> {gene.out_node} [label=\" {gene.weight:.2f}\"];\n"
        
        dot_str += "}\n"
        return dot_str
    
    def dot_to_ascii(self, dot: str, fancy: bool = True):

        url = 'https://dot-to-ascii.ggerganov.com/dot-to-ascii.php'
        boxart = 0

        # use nice box drawing char instead of + , | , -
        if fancy:
            boxart = 1

        params = {
            'boxart': boxart,
            'src': dot,
        }
        time.sleep(0.2) # Sleep for 0.2 seconds to avoid rate limiting
        response = requests.get(url, params=params).text

        if response == '':
            raise SyntaxError('DOT string is not formatted correctly')

        return response
    
    def __str__(self):
        connections = '\n'.join([str(connection) for connection in self.connections])

        return f"Chromosome:\nFitness: {self.fitness}\n{self.graph}"

    def __repr__(self):
        return self.__str__()
    
    def initialize_connections(self):
        """
        Initialize the connections with random connections
        """
        for i in range(self.num_inputs):
            for j in range(self.num_inputs, self.num_inputs + self.num_outputs):
                if np.random.rand() < 0.5:
                    # Randomly initialize the connection
                    self.connections.append(Gene(i + 1, j + 1, np.random.rand(), True, len(self.connections) + 1))