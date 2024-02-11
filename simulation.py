from chromosome import Chromosome
from gene import Gene
import numpy as np
if __name__ == '__main__':
    print('This is the main program')
    # Create a cgromosome
    np.random.seed(0)
    chromosomes = [Chromosome(num_inputs=3, num_outputs=2) for _ in range(10)]
    for chromosome in chromosomes:
        print(chromosome)