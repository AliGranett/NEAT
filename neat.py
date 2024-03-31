import random
import pickle
import math
import multiprocessing as mp
import activations

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
    node_id = 2
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
            self._nodes[i].output = inputs[i] # TODO might want to call activation function here

        # Generate backward-adjacency list (from -> to) a list of all incoming edges to a node
        
        _from = {}
        for n in self._nodes.keys():
            _from[n] = []

        for (i, j) in self._edges:
            if not self._edges[(i, j)].enabled:
                continue
            _from[j].append(i)

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
        # population = list(probabilities.keys())
        # weights = [probabilities[k] for k in population]
        # choice = random.choices(population, weights=weights)[0]

        if (random.random() < probabilities["node"]) | (len(self._nodes.keys())==self._unhidden):#"node": 
            self.add_node()
        if random.random() < probabilities["edge"]:
            (i, j) = self.random_pair()
            self.add_edge(i, j, random.uniform(-self.a, self.a))
        chance = random.random()
        if chance < (probabilities["weight_perturb"]+probabilities["weight_set"]): #choice == "weight_perturb" or choice == "weight_set":
            if chance < probabilities["weight_perturb"]:
                self.shift_weight("weight_perturb")
            else:
                self.shift_weight("weight_set")
        chance = random.random()
        if chance < (probabilities["bias_perturb"]+probabilities["bias_set"]):
            if chance < probabilities["bias_perturb"]:
                self.shift_bias("bias_perturb")
            else:
                self.shift_bias("bias_set")    
        # print(self._nodes.keys())
        
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
            self._edges[e].weight += random.uniform(-self.a, self.a)
        elif type == "weight_set":
            self._edges[e].weight = random.uniform(-self.a, self.a)

    def shift_bias(self, type):
        """Randomly shift, perturb, or set the bias of an incoming edge."""
        # Select only nodes in the hidden and output layer
        # n = random.choice(range(self._inputs, self._max_node))
        # print([n for n in self._nodes.keys() if n >= self._inputs])
        # print(self._nodes.keys())
        n = random.choice([n for n in self._nodes.keys() if n >= self._inputs])
        if type == "bias_perturb":
            self._nodes[n].bias += random.uniform(-self.a, self.a)
        elif type == "bias_set":
            self._nodes[n].bias = random.uniform(-self.a, self.a)

    def random_pair(self):
        """
        Generate random nodes (i, j) such that:
        1. i is not an output
        2. j is not an input
        3. i != j
        """
        
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
        return self._unhidden <= n 

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
        self._fitness = 0
    
    def create_nodes_from_edges(self):
        """Create nodes from edges"""
        # extract all nodes from edges
        node_ids= set([i for (i,j)in self._edges.keys()] + [j for (i,j) in self._edges.keys()])
        for node_id in node_ids:
            self._nodes[node_id] = Node(self._default_activation)
        
    def clone(self):
        """Return a clone of the genome.
        """
        return copy.deepcopy(self)
    
    def get_edges_innov(self):
        """Return the enovation number of the edges"""
        return [Genome.map_edge[(i,j)] for (i,j) in self._edges.keys()]
    
    def disable_edge(self, i, j):
        """Disable an edge"""
        self._edges[(i,j)].enabled = False
        
    def enable_edge(self, i, j):
        """Enable an edge"""
        self._edges[(i,j)].enabled = True
    
    def get_disabled_edges(self):
        """Return a list of disabled edges"""
        return [(i,j) for (i,j) in self._edges.keys() if not self._edges[(i,j)].enabled]
        
def genomic_distance(a :Genome, b:Genome, distance_weights): # distance_weights === c vector
    """Calculate the genomic distance between two genomes."""
    
    c1, c2, c3 = distance_weights['excess'], distance_weights['disjoint'] , distance_weights['weight']
    
    a_edges_innov = set(a.get_edges_innov())
    b_edges_innov = set(b.get_edges_innov())
    
    max_innov_a = max(a_edges_innov)
    max_innov_b = max(b_edges_innov)
    
    cut_off = min(max_innov_a, max_innov_b)
    
    Excess = 0
    Disjoint = 0
    for innov in a_edges_innov ^ b_edges_innov:
        if innov <= cut_off:
            Disjoint += 1
        else:
            Excess += 1      
    N = max(len(a_edges_innov), len(b_edges_innov))
    matching =len(a_edges_innov & b_edges_innov)
    
    W_avg = 0
    for i in a_edges_innov & b_edges_innov:
        key = Genome.get_edge_key(i)
        W_avg += abs(a._edges[key].weight - b._edges[key].weight)
    W_avg /= matching
    dist= (c1 * Excess/N + c2 * Disjoint/N + c3 * W_avg)
    # print(f'a_nodes: {a._nodes.keys()} , b_nodes: {b._nodes.keys()} , Excess: {Excess} , Disjoint: {Disjoint} , N: {N} , matching: {matching} , W_avg: {W_avg} , dist: {dist}')
    return dist
    

def genomic_crossover(a:Genome, b :Genome):
    """Breed two genomes and return the child. Matching genes
    are inherited randomly, while disjoint genes are inherited
    from the fitter parent.
    """
    # Template genome for child
    distributions_parameters = {
        "uniform_range" : a.a,
        "gauss_std": a.std
    }
    child = Genome(a._inputs, a._outputs, a._default_activation, distributions_parameters)
    
    # extract edges (by innov number) from each parent
    a_edges_innov =set(a.get_edges_innov())
    b_edges_innov = set(b.get_edges_innov())
    
    disabled_a = a.get_disabled_edges()
    disabled_b = b.get_disabled_edges()
    
    # disabled edges in both parents
    disabled_edges = set(disabled_a) ^ set(disabled_b)
    
    # Inherit homologous gene from a random parent
    for innov in a_edges_innov & b_edges_innov:
        parent = random.choice([a, b])
        key = Genome.get_edge_key(innov)
        if key in disabled_edges:
            child._edges[key] = copy.deepcopy(parent._edges[key])
            is_disabled = random.random()<0.75 #choice([True, False], p=[0.75, 0.25])
            if is_disabled:
                child._edges[key].enabled = False
            else:
                child._edges[key].enabled = True
        child._edges[key] = copy.deepcopy(parent._edges[key])
        
        
    # inherit disjoint genes from either parent
    max_innov_a = max(a_edges_innov)
    max_innov_b = max(b_edges_innov)
    
    cut_off = min(max_innov_a, max_innov_b)
    
    # Inherit disjoint genes from the parent that has them
    for innov in a_edges_innov ^ b_edges_innov:
        if innov <= cut_off:
            key = Genome.get_edge_key(innov)
            if innov in a_edges_innov:
                child._edges[key] = copy.deepcopy(a._edges[key])
            else:
                child._edges[key] = copy.deepcopy(b._edges[key])
    
    
    # inherit excess genes from the fitter parent
    fitter = a if a._fitness > b._fitness else b
    for innov in fitter.get_edges_innov():
        if innov > cut_off:
            key = Genome.get_edge_key(innov)
            child._edges[key] = copy.deepcopy(fitter._edges[key])
    child.create_nodes_from_edges()
    child.reset()
    return child
 

class Hyperparameters(object):
    """Hyperparameter settings for the FeedForwardNet object."""

    def __init__(self, **kwargs):
        # distance threshold for speciation
        self.delta_threshold = kwargs.get('delta_threshold', 1.5)
        self.distance_weights = kwargs.get('distance_weights', {
            'excess': 1.0,
            'disjoint': 1.0,
            'weight': 0.4
        })
        self.default_activation = kwargs.get('default_activation', activations.sigmoid)
        self.max_fitness = kwargs.get('max_fitness', float('inf'))
        self.max_generations = kwargs.get('max_generations', float('inf'))
        self.max_fitness_history = kwargs.get('max_fitness_history', 30)
        self.breed_probabilities = kwargs.get('breed_probabilities', {
            'asexual': 0.5,
            'sexual': 0.5
        })
        self.mutation_probabilities = kwargs.get('mutation_probabilities', {
            'node': 0.01,
            'edge': 0.09,
            'weight_perturb': 0.4,
            'weight_set': 0.1,
            'bias_perturb': 0.3,
            'bias_set': 0.1
        })
        self.distributions_parameters = kwargs.get('distributions_parameters', {
            'uniform_range': 1,
            'gauss_sigma': 1
        })
        self.interspecies_mating = kwargs.get('interspecies_mating', 0.001)
        self.patience = kwargs.get('patience', float('inf'))
        self.champion_population_threshold = kwargs.get('champion_threshold', 0)


class Specie(object):
    """
    A specie represents a set of genomes whose genomic distances 
    between them fall under the FeedForwardNet's delta threshold.
    """

    def __init__(self, max_fitness_history, *members):
        self._members :list[Genome] = list(members)
        self._fitness_history = []
        self._fitness_sum = 0
        self._max_fitness_history = max_fitness_history

    def breed(self, mutation_probabilities, breed_probabilities,other_specie=None):
        """Return a child as a result of either a mutated clone
        or crossover between two parent genomes.
        """
        # Either mutate a clone or breed two random genomes
        population = list(breed_probabilities.keys())
        probabilities = [breed_probabilities[k] for k in population]
        choice = random.choices(population, weights=probabilities)[0]

        if choice == "asexual" or len(self._members) == 1:
            child = random.choice(self._members).clone()
            child.mutate(mutation_probabilities)
        elif choice == "sexual":
            if other_specie!=None:
                # Interspecies mating
                mom = random.choice(self._members)
                dad = random.choice(other_specie._members)
                child = genomic_crossover(mom, dad)
            else:
                (mom, dad) = random.sample(self._members, 2)
                child = genomic_crossover(mom, dad)
        return child

    def update_fitness(self):
        """Update the adjusted fitness values of each genome 
        and the historical fitness."""
        for g in self._members:
            g._adjusted_fitness = g._fitness/len(self._members)
        self._fitness_sum = sum([g._adjusted_fitness for g in self._members])
        self._fitness_history.append(self._fitness_sum)
        # TODO: check if it pops the oldest fitness and not the newest
        if len(self._fitness_history) > self._max_fitness_history:
            self._fitness_history.pop(0)

    def cull_genomes(self, fittest_only):
        """Exterminate the weakest genomes per specie."""
        self._members.sort(key=lambda g: g._fitness, reverse=True)
        if fittest_only:
            # Only keep the winning genome
            remaining = 1
        else:
            # Keep top 25%
            # TODO: try to change to 50%
            # TODO: check what they did in original paper
            remaining = int(math.ceil(0.5*len(self._members)))

        self._members = self._members[:remaining]

    def get_best(self):
        """Get the member with the highest fitness score."""
        return max(self._members, key=lambda g: g._fitness)

    def can_progress(self):
        """Determine whether species should survive the culling."""
        n = len(self._fitness_history)
        avg = sum(self._fitness_history) / n
        # TODO: patience = self._max_fitness_history
        return avg > self._fitness_history[0] or n < self._max_fitness_history
    

class NEAT(object):
    """Base class for 'NEAT' algorithm that learns through the evolution
    of a population of genomes.
    """

    def __init__(self, inputs, outputs, population=100, hyperparams=Hyperparameters()):
        self._inputs = inputs
        self._outputs = outputs

        self._species :list[Specie] = []
        self._population = population

        # Hyper-parameters
        self._hyperparams = hyperparams

        self._generation = 0
        self._current_species = 0
        self._current_genome = 0

        self._global_best = None
        self._champions = []

    def generate(self):
        """Generate the initial population of genomes."""
        for i in range(self._population):
            g = Genome(self._inputs, self._outputs,
                       self._hyperparams.default_activation,
                       self._hyperparams.distributions_parameters
                       )
            g.generate()
            self.classify_genome(g)

        # Set the initial best genome
        self._global_best = self._species[0]._members[0]

    def classify_genome(self, genome):
        """Classify genomes into species via the genomic
        distance algorithm.
        """
        if not self._species:
            # Empty population
            self._species.append(Specie(
                self._hyperparams.max_fitness_history, genome
            )
            )
        else:
            # Compare genome against representative s[0] in each specie
            for s in self._species:
                rep = s._members[0]
                dist = genomic_distance(
                    genome, rep, self._hyperparams.distance_weights
                )
                if dist <= self._hyperparams.delta_threshold:
                    s._members.append(genome)
                    return
            # Doesn't fit with any other specie, create a new one
            self._species.append(Specie(
                self._hyperparams.max_fitness_history, genome
            )
            )

    def update_fittest(self):
        """Update the highest fitness score of the whole population."""
        top_performers = [s.get_best() for s in self._species]
        current_top = max(top_performers, key=lambda g: g._fitness)
        self._champions = [s.get_best() for s in self._species if len(s._members) > self._hyperparams.champion_population_threshold]

        # TODO: roll over the champion automatically to the next generation if specie has enough members
        if current_top._fitness > self._global_best._fitness:
            self._global_best = current_top.clone()
        

    def evolve(self):
        """
        Evolve the population by eliminating the poorest performing
        genomes and repopulating with mutated children, prioritizing
        the most promising species.
        """
        global_fitness_sum = 0
        for s in self._species:
            s.update_fitness()
            global_fitness_sum += s._fitness_sum
        
        if global_fitness_sum == 0:
            # No progress, mutate everybody
            for s in self._species:
                for g in s._members:
                    g.mutate(self._hyperparams.mutation_probabilities)
        else:
            # Only keep the species with potential to improve
            surviving_species = []
            for s in self._species:
                if s.can_progress():
                    surviving_species.append(s)
            self._species = surviving_species

            # Eliminate lowest performing genomes per specie
            for s in self._species:
                s.cull_genomes(False)
            
            for c in self._champions:
                c_clone = c.clone()
                self.classify_genome(c_clone)
        

            # Repopulate
            for i, s in enumerate(self._species):
                ratio = s._fitness_sum/global_fitness_sum
                diff = self._population - self.get_population()
                offspring = int(round(ratio * diff))
                
                # Repopulate : each species gets to mutate and breed in proportion to its fitness
                for j in range(offspring):
                    s1= None
                    if len(self._species)>1 and random.random() < self._hyperparams.interspecies_mating:
                        # Interspecies mating
                        s1 = random.choice(list(set(self._species)-set([s])))
                    self.classify_genome(
                        s.breed(
                            self._hyperparams.mutation_probabilities,
                            self._hyperparams.breed_probabilities,
                            s1
                        )
                    )
            
            # TODO: avoid death of all species
            # No species survived
            # Repopulate using mutated minimal structures and global best
            if not self._species:
                print("No species survived, repopulating...")
                for i in range(self._population):
                    
                    if i % 3 == 0:
                        g = self._global_best.clone()
                    else:
                        g = Genome(self._inputs, self._outputs,
                                   self._hyperparams.default_activation, 
                                   self._hyperparams.distributions_parameters)
                        g.generate()
                    g.mutate(self._hyperparams.mutation_probabilities)
                    self.classify_genome(g)
        
        # print(f"Generation {self._generation}, #Species: {len(self._species)}, #Population: {self.get_population()}")
        self._generation += 1

    def should_evolve(self):
        """Determine if the system should continue to evolve
        based on the maximum fitness and generation count.
        """
        self.update_fittest()
        fit = self._global_best._fitness <= self._hyperparams.max_fitness
        end = self._generation != self._hyperparams.max_generations

        return fit and end

    def next_iteration(self):
        """Call after every evaluation of individual genomes to
        progress training.
        """
        s = self._species[self._current_species]
        if self._current_genome < len(s._members)-1:
            self._current_genome += 1
        else:
            if self._current_species < len(self._species)-1:
                self._current_species += 1
                self._current_genome = 0
            else:
                # Evolve to the next generation
                self.evolve()
                self._current_species = 0
                self._current_genome = 0
    
    def evaluate_parallel(self, evaluator, *args, **kwargs):
        """Evaluate the entire population on separate processes
        to progress training. The evaluator function must take a Genome
        as its first parameter and return a numerical fitness score.

        Any global state passed to the evaluator is copied and will not
        be modified at the parent process.
        """
        max_proc = max(mp.cpu_count()-1, 1)
        pool = mp.Pool(processes=max_proc)

        results = {}
        for i in range(len(self._species)):
            for j in range(len(self._species[i]._members)):
                results[(i, j)] = pool.apply_async(
                    evaluator,
                    args=[self._species[i]._members[j]]+list(args),
                    kwds=kwargs
                )

        for key in results:
            genome = self._species[key[0]]._members[key[1]]
            genome.set_fitness(results[key].get())

        pool.close()
        pool.join()
        self.evolve()

    def get_fittest(self):
        """Return the genome with the highest global fitness score."""
        return self._global_best

    def get_population(self):
        """Return the true (calculated) population size."""
        return sum([len(s._members) for s in self._species])

    def get_current(self):
        """Get the current genome for evaluation."""
        s = self._species[self._current_species]
        return s._members[self._current_genome]

    def get_current_species(self):
        """Get index of current species being evaluated."""
        return self._current_species

    def get_current_genome(self):
        """Get index of current genome being evaluated."""
        return self._current_genome

    def get_generation(self):
        """Get the current generation number of this population."""
        return self._generation

    def get_species(self):
        """Get the list of species and their respective member genomes."""
        return self._species

    def save(self, filename):
        """Save an instance of the population to disk."""
        with open(filename+'.neat', 'wb') as _out:
            pickle.dump(self, _out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        """Return an instance of a population from disk."""
        with open(filename+'.neat', 'rb') as _in:
            return pickle.load(_in)
