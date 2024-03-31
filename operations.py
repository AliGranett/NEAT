from importlib.metadata import distribution
import random
import copy

from genotype import Genome

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
    print(f'a_nodes: {a._nodes.keys()} , b_nodes: {b._nodes.keys()} , Excess: {Excess} , Disjoint: {Disjoint} , N: {N} , matching: {matching} , W_avg: {W_avg} , dist: {dist}')
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
