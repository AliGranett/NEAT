import itertools
import random
import copy

class Edge:
    """A gene object representing an edge in the neural network."""
    def __init__(self, weight):
        self.weight = weight
        self.enabled = True



class Node:
    """A gene object representing a node in the neural network."""
    def __init__(self, activation):
        self.output = 0
        self.bias = 0
        self.activation = activation


class Genome:
    """Base class for a standard genome used by the NEAT algorithm."""
    innov_edge = 0
    node_id = 0
    # map each (i,j) to an innovation number
    map_edge = {}
    
    # map each node to an edge used to when first created 
    map_node = {}
    
    def get_edge_key(innov):
        """Return the edge key"""
        for (i,j) in Genome.map_edge.keys():
            if Genome.map_edge[(i,j)] == innov:
                return (i,j)

    def get_node_id_by_edge(i, j):
        for k in Genome.map_node.keys():
            if Genome.map_node[k] == (i, j):
                return k
        return None

    def __init__(self, inputs, outputs, default_activation, distributions_parameters):
        # Nodes
        self._inputs = inputs
        self._outputs = outputs

        self._unhidden = inputs+outputs  # Number of unhidden nodes
        self._max_node = inputs+outputs  # Number of nodes (starting from i+o)

        # Structure
        self._edges = {}  # (i, j) : Edge
        self._nodes = {}  # NodeID : Node

        self._default_activation = default_activation

        # Performance
        self._fitness = 0
        self._adjusted_fitness = 0

        # range lim
        self.a = distributions_parameters['uniform_range']
        self.std = distributions_parameters['gauss_std']

    def generate(self):
        """
        Generate the neural network of this genome with minimal
        initial topology, i.e. (no hidden nodes). Call on genome
        creation.
        """
        # Minimum initial topology, no hidden layer

        # initialize nodes with default activation
        for n in range(self._max_node):
            self._nodes[n] = Node(self._default_activation)
            Genome.map_node[n] = None
        
        # connect all input nodes to all output nodes
        for i in range(self._inputs):
            for j in range(self._inputs, self._unhidden):
                self.add_edge(i, j, random.gauss(0, self.std))

    def forward(self, inputs):
        """
        Evaluate inputs and calculate the outputs of the
        neural network via the forward propagation algorithm.
        """

        # Check if the number of inputs is correct
        if len(inputs) != self._inputs:
            raise ValueError("Incorrect number of inputs.")

        # Set input values
        for i in range(self._inputs):
            self._nodes[i].output = inputs[i] #TODO: might want to modify this to self._nodes[i].output = activation(inputs[i])

        # Generate backward-adjacency list (from -> to) a list of all incoming edges to a node
        _from = {}
        # for n in range(self._max_node):
        #     _from[n] = []
        
        for n in self._nodes.keys():
            _from[n] = []

        for (i, j) in self._edges:
            if not self._edges[(i, j)].enabled:
                continue
            _from[j].append(i)

        # Calculate output values for each node in topological order
        # ordered_nodes = itertools.chain(
        #     range(self._unhidden, self._max_node),
        #     range(self._inputs, self._unhidden)
        # )
        ordered_nodes = itertools.chain(
            [n for n in self._nodes.keys() if n > self._unhidden ], # a list of all node ids that are hidden
            range(self._inputs, self._unhidden) 
        )

        # Calculate output values for each node in topological order
        for j in ordered_nodes:
            ax = 0  # Activation value
            for i in _from[j]:
                ax += self._edges[(i, j)].weight * self._nodes[i].output

            node = self._nodes[j]
            node.output = node.activation(ax + node.bias)

        return [self._nodes[n].output for n in range(self._inputs, self._unhidden)]

    def mutate(self, probabilities):
        """Randomly mutate the genome to initiate variation."""
        if self.is_disabled():
            self.add_enabled()

        # we extract probabilties use them to choose a key in acco
        population = list(probabilities.keys())
        weights = [probabilities[k] for k in population]
        choice = random.choices(population, weights=weights)[0]
        print(f"Mutation: {choice}")
        if choice == "node":
            self.add_node()
        elif choice == "edge":
            (i, j) = self.random_pair()
            self.add_edge(i, j, random.uniform(-self.a, self.a))
        elif choice == "weight_perturb" or choice == "weight_set":
            self.shift_weight(choice)
        elif choice == "bias_perturb" or choice == "bias_set":
            self.shift_bias(choice)

        

        self.reset()

    def add_node(self):
        """
        Add a new node between a randomly selected edge,
        disabling the parent edge.
        """
        enabled = [k for k in self._edges if self._edges[k].enabled]
        (i, j) = random.choice(enabled)
        edge = self._edges[(i, j)]
        edge.enabled = False

        # check if we already created a node for this edge
        # Genome.map_node : (node_id) : (i,j)
        if (i,j) in Genome.map_node.values():
            # if we already created a node for this edge, we use the same node_id
            new_node = Genome.get_node_id_by_edge(i, j)
        else:
            # if we haven't created a node for this edge
            # increment the node_id
            Genome.node_id += 1
            # use the new node_id as the node_id for the new node
            new_node = Genome.node_id
            Genome.map_node[new_node] = (i, j)

        self._nodes[new_node] = Node(self._default_activation)
        self.add_edge(i, new_node, 1.0)
        self.add_edge(new_node, j, edge.weight)
        return new_node

    def add_edge(self, i, j, weight):
        """Add a new connection between existing nodes."""
        if (i, j) in self._edges:
            self._edges[(i, j)].enabled = True
        else:
            self._edges[(i, j)] = Edge(weight)
            # ALI ADDED THIS TODAY
            # map each edge to an innovation number
            if (i, j) not in Genome.map_edge.keys():
                Genome.innov_edge += 1
                Genome.map_edge[(i, j)] = Genome.innov_edge

    def add_enabled(self):
        """Re-enable a random disabled edge."""
        disabled = [e for e in self._edges if not self._edges[e].enabled]

        if len(disabled) > 0:
            re_enabled = random.choice(disabled)
            self._edges[re_enabled].enabled = True
        

    def shift_weight(self, type):
        """Randomly shift, perturb, or set one of the edge weights."""
        e = random.choice(list(self._edges.keys()))
        if type == "weight_perturb":
            self._edges[e].weight += random.gauss(0, self.std)
        elif type == "weight_set":
            self._edges[e].weight = random.uniform(-self.a, self.a)

    def shift_bias(self, type):
        """Randomly shift, perturb, or set the bias of an incoming edge."""
        # Select only nodes in the hidden and output layer
        # n = random.choice(range(self._inputs, self._max_node))
        n = random.choice([n for n in self._nodes.keys() if n > self._inputs])
        if type == "bias_perturb":
            self._nodes[n].bias += random.gauss(0, self.std)
        elif type == "bias_set":
            self._nodes[n].bias = random.uniform(-self.a, self.a)

    def random_pair(self):
        """
        Generate random nodes (i, j) such that:
        1. i is not an output
        2. j is not an input
        3. i != j
        """
        # choose two random nodes 
        # i source node (not output)
        # j target node (not input)
        
        i = random.choice([n for n in self._nodes.keys() if not self.is_output(n)])
        j_list = [n for n in self._nodes.keys() if not self.is_input(n) and n != i]

        if not j_list:
            # j = self.node_id
            # self.add_node()
            j = self.add_node()
        else:
            j = random.choice(j_list)
        return (i, j)

    def is_input(self, n):
        """Determine if the node id is an input."""
        return 0 <= n < self._inputs

    def is_output(self, n):
        """Determine if the node id is an output."""
        return self._inputs <= n < self._unhidden

    def is_hidden(self, n):
        """Determine if the node id is hidden."""
        return self._unhidden <= n # TODO: double check : < self._max_node?

    def is_disabled(self):
        """Determine if all of its genes are disabled."""
        return all(self._edges[i].enabled == False for i in self._edges)
    
    def is_enabled(self):
        """Determine if all of its genes are enabled."""
        return all(self._edges[i].enabled == True for i in self._edges)

    def get_fitness(self):
        """Return the fitness of the genome."""
        return self._fitness

    def get_nodes(self):
        """Get the nodes of the network."""
        return self._nodes.copy()

    def get_edges(self):
        """Get the network's edges."""
        return self._edges.copy()

    # def get_num_nodes(self):
    #     """Get the number of nodes in the network."""
    #     return self._max_node

    def set_fitness(self, score):
        """Set the fitness score of this genome."""
        self._fitness = score

    def reset(self):
        """Reset the genome's internal state."""
        # for n in range(self._max_node):
        #     self._nodes[n].output = 0
        for n in self._nodes.values(): # {node_id: node}
            n.output = 0
        # TODO : REMOVE test print 
        for n in self._nodes.values():
            if n.output != 0:
                print(f"eddie is to blame")
        self._fitness = 0

    def clone(self):
        """Return a clone of the genome.
        """
        return copy.deepcopy(self)
    
    def get_edges_innov(self):
        """Return the enovation number of the edges"""
        return [Genome.map_edge[(i,j)] for (i,j) in self._edges.keys()]
        
    