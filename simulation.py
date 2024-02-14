from chromosome import Chromosome
import numpy as np
import time
# from tqdm import tqdm

if __name__ == '__main__':
    start = time.time()
    # Create a cgromosome
    np.random.seed(0)

    chromosomes = [Chromosome(nodes_in=3, nodes_out=2) for _ in range(10)]

    for chromosome in chromosomes:
        print(chromosome)
        print(chromosome.network)
    end = time.time()

    # print(f"Time took: {end - start:.2} seconds.")
