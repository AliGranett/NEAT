from importlib.metadata import distribution
import random
import copy

from genotype import Genome

def genomic_distance(a, b, distance_weights): # distance_weights === c vector
    """Calculate the genomic distance between two genomes."""
    
    c1, c2, c3 = distance_weights['edge'], distance_weights['bias'] , distance_weights['weight']
    
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
    
    return c1 * Excess/N + c2 * Disjoint/N + c3 * W_avg
    
    # Does not distinguish between disjoint and excess
    matching_edges = a_edges & b_edges
    disjoint_edges = (a_edges - b_edges) | (b_edges - a_edges)

    max_innov_a = max(a_edges, key=lambda e: a._edges[e].innov)
    max_innov_b = max(b_edges, key=lambda e: b._edges[e].innov)
    
    # calculate excess genes (edges)
    smaller, larger = (a, b) if max_innov_a < max_innov_b else (b, a)
    excess_edges = {e for e in larger._edges if larger._edges[e].innov > smaller._edges[max_innov_a].innov}
     

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


# def excess_genes(a, b):
#     """Return the excess genes between two genomes."""
#     a_edges = set(a._edges)
#     b_edges = set(b._edges)
#     max_innov_a = max(a_edges, key=lambda e: a._edges[e].innov)
#     max_innov_b = max(b_edges, key=lambda e: b._edges[e].innov)
#     smaller, larger = (a, b) if max_innov_a < max_innov_b else (b, a)
#     return {e for e in larger._edges if larger._edges[e].innov > smaller._edges[max_innov_a].innov}


# def disjoint_genes(a, b):
#     """Return the disjoint without excess genes between two genomes."""
#     a_edges = set(a._edges)
#     b_edges = set(b._edges)
#     max_innov_a = max(a_edges, key=lambda e: a._edges[e].innov)
#     max_innov_b = max(b_edges, key=lambda e: b._edges[e].innov)
#     smaller, larger = (a, b) if max_innov_a < max_innov_b else (b, a)
#     return {e for e in larger._edges if larger._edges[e].innov < smaller._edges[max_innov_a].innov}


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
    
    
    # a_in = set(a._edges)
    # b_in = set(b._edges)
    
    # extract edges (by innov number) from each parent
    a_edges_innov =set(a.get_edges_innov())
    b_edges_innov = set(b.get_edges_innov())
    
    #     # Inherit homologous gene from a random parent
    # for i in a_in & b_in:
    #     parent = random.choice([a, b])
    #     child._edges[i] = copy.deepcopy(parent._edges[i])
    
    # Inherit homologous gene from a random parent
    for innov in a_edges_innov & b_edges_innov:
        parent = random.choice([a, b])
        key = Genome.get_edge_key(innov)
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
    # # Inherit disjoint/excess genes from fitter parent
    # if a._fitness > b._fitness:
    #     for i in a_in - b_in:
    #         child._edges[i] = copy.deepcopy(a._edges[i])
    # else:
    #     for i in b_in - a_in:
    #         child._edges[i] = copy.deepcopy(b._edges[i])
    
    # # Calculate max node
    # child._max_node = 0
    # for (i, j) in child._edges:
    #     current_max = max(i, j)
    #     child._max_node = max(child._max_node, current_max)
    # child._max_node += 1

    # # Inherit nodes
    # for n in range(child._max_node):
    #     inherit_from = []
    #     if n in a._nodes:
    #         inherit_from.append(a)
    #     if n in b._nodes:
    #         inherit_from.append(b)

    #     random.shuffle(inherit_from)
    #     parent = max(inherit_from, key=lambda p: p._fitness)
    #     child._nodes[n] = copy.deepcopy(parent._nodes[n])
    child.create_nodes_from_edges()
    child.reset()
    return child
