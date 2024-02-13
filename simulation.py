from chromosome import Chromosome
import numpy as np
if __name__ == '__main__':
    print('This is the main program')
    # Create a cgromosome
    np.random.seed(0)
    chromosomes = [Chromosome(nodes_in=3, nodes_out=2) for _ in range(10)]

    for chromosome in chromosomes:
        print(chromosome.network)
