from importlib.metadata import distribution
import random
import copy

from genotype import Genome

def genomic_distance(a, b, distance_weights): # distance_weights === c vector
    """Calculate the genomic distance between two genomes."""
    a_edges = set(a._edges)
    b_edges = set(b._edges)

    # Does not distinguish between disjoint and excess
    matching_edges = a_edges & b_edges
    disjoint_edges = (a_edges - b_edges) | (b_edges - a_edges)

    max_innov_a = max(a_edges, key=lambda e: a._edges[e].innov)
    max_innov_b = max(b_edges, key=lambda e: b._edges[e].innov)
    
    # calculate excess genes (edges)
    smaller, larger = (a, b) if max_innov_a < max_innov_b else (b, a)
    excess_edges = {e for e in larger._edges if larger._edges[e].innov > smaller._edges[max_innov_a].innov}
    
    # check which parent is fittest

    # if parent x is fitter than y and also x is longer than y add all excess genes from x to child

    # else discard excess genes

    # check non matching genes that also have innov number smaller than min(max_innov_a, max_innov_b)

    # copy all non matching genes that have innov number smaller than min(max_innov_a, max_innov_b) to child
     


    

    N_edges = len(max(a_edges, b_edges, key=len))
    N_nodes = min(a._max_node, b._max_node)

    weight_diff = 0
    for i in matching_edges:
        weight_diff += abs(a._edges[i].weight - b._edges[i].weight)

    bias_diff = 0
    for i in range(N_nodes):
        bias_diff += abs(a._nodes[i].bias - b._nodes[i].bias)

    t1 = distance_weights['edge'] * len(disjoint_edges)/N_edges
    t2 = distance_weights['weight'] * weight_diff/len(matching_edges)
    t3 = distance_weights['bias'] * bias_diff/N_nodes
    return t1 + t2 + t3


def excess_genes(a, b):
    """Return the excess genes between two genomes."""
    a_edges = set(a._edges)
    b_edges = set(b._edges)
    max_innov_a = max(a_edges, key=lambda e: a._edges[e].innov)
    max_innov_b = max(b_edges, key=lambda e: b._edges[e].innov)
    smaller, larger = (a, b) if max_innov_a < max_innov_b else (b, a)
    return {e for e in larger._edges if larger._edges[e].innov > smaller._edges[max_innov_a].innov}


def disjoint_genes(a, b):
    """Return the disjoint without excess genes between two genomes."""
    a_edges = set(a._edges)
    b_edges = set(b._edges)
    max_innov_a = max(a_edges, key=lambda e: a._edges[e].innov)
    max_innov_b = max(b_edges, key=lambda e: b._edges[e].innov)
    smaller, larger = (a, b) if max_innov_a < max_innov_b else (b, a)
    return {e for e in larger._edges if larger._edges[e].innov < smaller._edges[max_innov_a].innov}


def genomic_crossover(a, b):
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
    a_in = set(a._edges)
    b_in = set(b._edges)

    # Inherit homologous gene from a random parent
    for i in a_in & b_in:
        parent = random.choice([a, b])
        child._edges[i] = copy.deepcopy(parent._edges[i])

    # Inherit disjoint/excess genes from fitter parent
    if a._fitness > b._fitness:
        for i in a_in - b_in:
            child._edges[i] = copy.deepcopy(a._edges[i])
    else:
        for i in b_in - a_in:
            child._edges[i] = copy.deepcopy(b._edges[i])
    
    # Calculate max node
    child._max_node = 0
    for (i, j) in child._edges:
        current_max = max(i, j)
        child._max_node = max(child._max_node, current_max)
    child._max_node += 1

    # Inherit nodes
    for n in range(child._max_node):
        inherit_from = []
        if n in a._nodes:
            inherit_from.append(a)
        if n in b._nodes:
            inherit_from.append(b)

        random.shuffle(inherit_from)
        parent = max(inherit_from, key=lambda p: p._fitness)
        child._nodes[n] = copy.deepcopy(parent._nodes[n])

    child.reset()
    return child
